from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from build_dataset143_synthetic_multicell import (
    add_instance,
    build_dataset,
    merge_semantic,
    read_2d,
)


class SyntheticMultiCellBuilderTests(unittest.TestCase):
    def test_semantic_merge_keeps_soma_priority(self) -> None:
        existing = np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
        incoming = np.asarray([[1, 2, 1], [2, 1, 0]], dtype=np.uint8)

        merged = merge_semantic(existing, incoming)

        np.testing.assert_array_equal(
            merged,
            np.asarray([[1, 2, 2], [2, 2, 0]], dtype=np.uint8),
        )

    def test_instance_overlap_is_preserved_and_triples_are_rejected(self) -> None:
        primary = np.zeros((4, 4), dtype=np.uint16)
        secondary = np.zeros((4, 4), dtype=np.uint16)
        first = np.zeros((4, 4), dtype=bool)
        second = np.zeros((4, 4), dtype=bool)
        first[1, 1:3] = True
        second[1:3, 2] = True

        add_instance(primary, secondary, first, 1)
        add_instance(primary, secondary, second, 2)

        self.assertEqual(primary[1, 2], 1)
        self.assertEqual(secondary[1, 2], 2)
        third = np.zeros((4, 4), dtype=bool)
        third[1, 2] = True
        with self.assertRaises(ValueError):
            add_instance(primary, secondary, third, 3)

    def test_small_end_to_end_dataset_is_reproducible_and_split_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "Dataset141_test"
            (base / "imagesTr").mkdir(parents=True)
            (base / "labelsTr").mkdir()
            train_ids = [f"cell_{index:03d}" for index in range(4)]
            validation_ids = [f"cell_{index:03d}" for index in range(4, 6)]
            all_ids = train_ids + validation_ids
            for index, case_id in enumerate(all_ids):
                raw = np.full((64, 64), 100 + index, dtype=np.uint16)
                label = np.zeros((64, 64), dtype=np.uint8)
                label[30, 12:38] = 1
                label[27:34, 38:45] = 2
                raw[label == 1] += 180
                raw[label == 2] += 320
                tifffile.imwrite(base / "imagesTr" / f"{case_id}_0000.tif", raw)
                tifffile.imwrite(base / "labelsTr" / f"{case_id}.tif", label)
            (base / "dataset.json").write_text(
                json.dumps(
                    {
                        "channel_names": {"0": "microscopy"},
                        "labels": {"background": 0, "skeleton": 1, "soma": 2},
                        "numTraining": len(all_ids),
                        "file_ending": ".tif",
                    }
                ),
                encoding="utf-8",
            )
            split_path = root / "splits_final.json"
            split_path.write_text(
                json.dumps([{"train": train_ids, "val": validation_ids}]),
                encoding="utf-8",
            )
            output = root / "Dataset143_test"
            args = argparse.Namespace(
                base_dataset=base,
                base_splits=split_path,
                output_dataset=output,
                fold=0,
                canvas_size=128,
                min_cells=2,
                max_cells=2,
                edge_margin=8,
                context_radius=12,
                signal_radius=6,
                min_soma_pixels=20,
                min_skeleton_pixels=8,
                preview_count=3,
                seed=143,
            )

            summary = build_dataset(args)

            self.assertEqual(summary["synthetic_training_scenes"], 2)
            self.assertEqual(summary["synthetic_validation_scenes"], 1)
            self.assertEqual(summary["total_semantic_cases"], 9)
            manifest = json.loads(
                (output / "synthetic_manifest.json").read_text(encoding="utf-8")
            )
            used = [source for scene in manifest for source in scene["source_cases"]]
            self.assertCountEqual(used, all_ids)
            for scene in manifest:
                allowed = train_ids if scene["split"] == "training" else validation_ids
                self.assertTrue(set(scene["source_cases"]).issubset(allowed))
                semantic = read_2d(output / "labelsTr" / f"{scene['case_id']}.tif")
                primary = read_2d(
                    output / "instance_primaryTr" / f"{scene['case_id']}.tif"
                )
                secondary = read_2d(
                    output / "instance_secondaryTr" / f"{scene['case_id']}.tif"
                )
                self.assertEqual(semantic.shape, (128, 128))
                self.assertTrue(set(np.unique(semantic)).issubset({0, 1, 2}))
                self.assertEqual(int(primary.max()), 2)
                self.assertTrue(np.all(secondary[secondary > 0] <= 2))
            self.assertTrue((output / "synthetic_multicell_review.html").is_file())


if __name__ == "__main__":
    unittest.main()
