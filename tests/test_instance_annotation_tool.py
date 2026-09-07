from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from instance_annotation_web.store import AnnotationStore, discover_cases


class InstanceAnnotationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "imagesTr").mkdir(parents=True)
        (self.dataset / "labelsTr").mkdir()
        self.case_id = "mats_overview_test_1"
        raw = np.arange(48 * 56, dtype=np.uint16).reshape(48, 56)
        semantic = np.zeros(raw.shape, dtype=np.uint8)
        semantic[5:9, 5:9] = 2
        semantic[8:25, 7] = 1
        semantic[35:39, 42:46] = 2
        semantic[20:37, 43] = 1
        semantic[20, 7:44] = 1
        tifffile.imwrite(self.dataset / "imagesTr" / f"{self.case_id}_0000.tif", raw)
        tifffile.imwrite(self.dataset / "labelsTr" / f"{self.case_id}.tif", semantic)
        (self.dataset / "splits_final_overview.json").write_text(
            json.dumps([{"train": [self.case_id], "val": []}]), encoding="utf-8"
        )
        self.output = self.root / "annotations"
        self.store = AnnotationStore(self.dataset, self.output)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create_cell(self) -> int:
        details = self.store.create_cell(self.case_id)
        return int(details["cells"][-1]["id"])

    def test_discovers_case_and_preserves_dataset_role(self) -> None:
        records = discover_cases(self.dataset)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].role, "training")
        self.assertEqual((records[0].height, records[0].width), (48, 56))

    def test_fill_and_stroke_assign_only_existing_semantic_pixels(self) -> None:
        cell_id = self.create_cell()
        self.store.apply_fill(
            self.case_id, x=6, y=6, operation="assign", target="soma", cell_id=cell_id
        )
        self.store.apply_stroke(
            self.case_id,
            points=[(7, 8), (7, 24)],
            radius=1,
            operation="assign",
            target="skeleton",
            cell_id=cell_id,
        )
        details = self.store.case_details(self.case_id)
        cell = details["cells"][0]
        self.assertEqual(cell["soma_pixels"], 15)
        self.assertGreaterEqual(cell["skeleton_pixels"], 17)
        self.assertTrue(cell["valid"])

        primary = np.load(
            self.output / "working" / self.case_id / "instance_primary.npy"
        )
        self.assertEqual(int(primary[0, 0]), 0)
        self.assertEqual(int(primary[6, 6]), cell_id)

    def test_normal_assignment_does_not_overwrite_but_overlap_uses_second_slot(self) -> None:
        first = self.create_cell()
        second = self.create_cell()
        self.store.apply_stroke(
            self.case_id,
            points=[(7, 20)],
            radius=0,
            operation="assign",
            target="skeleton",
            cell_id=first,
        )
        skipped = self.store.apply_stroke(
            self.case_id,
            points=[(7, 20)],
            radius=0,
            operation="assign",
            target="skeleton",
            cell_id=second,
        )
        self.assertEqual(skipped["change"]["skipped"], 1)
        self.store.apply_stroke(
            self.case_id,
            points=[(7, 20)],
            radius=0,
            operation="overlap",
            target="skeleton",
            cell_id=second,
        )
        primary = np.load(self.output / "working" / self.case_id / "instance_primary.npy")
        secondary = np.load(self.output / "working" / self.case_id / "instance_secondary.npy")
        self.assertEqual(int(primary[20, 7]), first)
        self.assertEqual(int(secondary[20, 7]), second)

    def test_ambiguous_pixels_can_be_marked_and_cleared(self) -> None:
        self.store.apply_stroke(
            self.case_id,
            points=[(7, 20)],
            radius=0,
            operation="ambiguous",
            target="foreground",
            cell_id=None,
        )
        self.assertEqual(self.store.case_details(self.case_id)["coverage"]["ambiguous"], 1)
        self.store.apply_stroke(
            self.case_id,
            points=[(7, 20)],
            radius=0,
            operation="clear_ambiguous",
            target="foreground",
            cell_id=None,
        )
        self.assertEqual(self.store.case_details(self.case_id)["coverage"]["ambiguous"], 0)

    def test_annotations_survive_store_restart(self) -> None:
        cell_id = self.create_cell()
        self.store.apply_fill(
            self.case_id, x=6, y=6, operation="assign", target="auto", cell_id=cell_id
        )
        reopened = AnnotationStore(self.dataset, self.output)
        details = reopened.case_details(self.case_id)
        self.assertEqual(details["cells_count"], 1)
        self.assertEqual(details["cells"][0]["soma_pixels"], 15)

    def test_export_writes_per_cell_training_artifacts_and_overlap(self) -> None:
        first = self.create_cell()
        second = self.create_cell()
        self.store.apply_fill(
            self.case_id, x=6, y=6, operation="assign", target="soma", cell_id=first
        )
        self.store.apply_fill(
            self.case_id, x=44, y=36, operation="assign", target="soma", cell_id=second
        )
        self.store.apply_stroke(
            self.case_id,
            points=[(7, 8), (7, 20)],
            radius=1,
            operation="assign",
            target="skeleton",
            cell_id=first,
        )
        self.store.apply_stroke(
            self.case_id,
            points=[(43, 20), (43, 36)],
            radius=1,
            operation="assign",
            target="skeleton",
            cell_id=second,
        )
        self.store.apply_stroke(
            self.case_id,
            points=[(7, 20)],
            radius=0,
            operation="overlap",
            target="skeleton",
            cell_id=second,
        )
        manifest = self.store.export(self.case_id)
        self.assertEqual(manifest["format"], "evo-instance-annotations-v1")
        self.assertEqual(manifest["valid_cells"], 2)
        latest = json.loads(
            (self.output / "exports" / self.case_id / "latest.json").read_text(encoding="utf-8")
        )
        revision_root = Path(latest["path"])
        self.assertTrue((revision_root / "instance_primary.tif").is_file())
        self.assertTrue((revision_root / "instance_secondary.tif").is_file())
        for cell_id in (first, second):
            cell_root = revision_root / "cells" / f"cell{cell_id:04d}"
            self.assertTrue((cell_root / "raw.tif").is_file())
            self.assertTrue((cell_root / "label.tif").is_file())
            self.assertTrue((cell_root / "skeleton.tif").is_file())
            self.assertTrue((cell_root / "soma.tif").is_file())
            self.assertTrue((cell_root / "ignore.tif").is_file())
            self.assertTrue((cell_root / "overlap.tif").is_file())
        second_overlap = tifffile.imread(
            revision_root / "cells" / f"cell{second:04d}" / "overlap.tif"
        )
        self.assertGreater(int(np.count_nonzero(second_overlap)), 0)


if __name__ == "__main__":
    unittest.main()
