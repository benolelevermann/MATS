from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from build_dataset139_from_dataset138_and_reviewed import build_dataset
from cell_pipeline_web.training_review_import import TrainingSource, import_training_review


class TrainingReviewImportTests(unittest.TestCase):
    def test_import_creates_completed_review_job_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "tracings"
            cell = source_root / "cell7"
            cell.mkdir(parents=True)
            raw = np.arange(80 * 96, dtype=np.uint16).reshape(80, 96)
            tifffile.imwrite(cell / "raw.tif", raw)
            (cell / "seg.swc").write_text("dummy", encoding="utf-8")
            (cell / "soma.zip").write_bytes(b"dummy")
            (cell / "source_fov.txt").write_text("1234567\n", encoding="utf-8")

            def labels(_cell, shape):
                skeleton = np.zeros(shape, dtype=bool)
                skeleton[40, 25:70] = True
                soma = np.zeros(shape, dtype=bool)
                soma[36:45, 20:30] = True
                return skeleton, soma, 0.40625, 0.40625, 1

            status = import_training_review(
                (TrainingSource("test_cc", "Test CC", source_root),),
                root / "runs",
                "training_review_test",
                label_loader=labels,
            )
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["review_kind"], "manual_training_data")
            self.assertEqual(len(status["crops"]), 1)
            crop = status["crops"][0]
            self.assertIn("fov1234567_cell7", crop["case_id"])
            output = root / "runs" / "training_review_test" / "04_evo_single_cells" / "cell0001"
            for name in (
                "raw.tif",
                "seg.tif",
                "skeleton.tif",
                "soma.tif",
                "raw_preview.png",
                "prediction_preview.png",
                "preview.png",
                "metadata.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_cell"], "cell7")


class Dataset139BuildTests(unittest.TestCase):
    @staticmethod
    def _write_pair(root: Path, case_id: str, offset: int) -> None:
        (root / "imagesTr").mkdir(parents=True, exist_ok=True)
        (root / "labelsTr").mkdir(parents=True, exist_ok=True)
        raw = np.full((48, 52), 100 + offset, dtype=np.uint16)
        label = np.zeros(raw.shape, dtype=np.uint8)
        label[24, 10:42] = 1
        label[20:29, 8:17] = 2
        tifffile.imwrite(root / "imagesTr" / f"{case_id}_0000.tif", raw)
        tifffile.imwrite(root / "labelsTr" / f"{case_id}.tif", label)

    def test_build_preserves_base_validation_and_adds_review_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "Dataset138"
            approved = root / "approved"
            output = root / "Dataset139"
            self._write_pair(base, "100", 0)
            self._write_pair(base, "101", 1)
            self._write_pair(base, "102", 0)
            self._write_pair(approved, "rocki_case", 2)
            (base / "dataset.json").write_text(
                json.dumps(
                    {
                        "channel_names": {"0": "image"},
                        "labels": {"background": 0, "skeleton": 1, "soma": 2},
                        "numTraining": 3,
                        "file_ending": ".tif",
                    }
                ),
                encoding="utf-8",
            )
            splits = root / "splits.json"
            splits.write_text(
                json.dumps([{"train": ["100", "102"], "val": ["101"]}]), encoding="utf-8"
            )

            summary = build_dataset(base, approved, output, splits)
            self.assertEqual(summary["total_training"], 4)
            self.assertEqual(summary["approved_pairs_added"], 1)
            dataset = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset["numTraining"], 4)
            generated = json.loads(
                (output / "splits_final_preserve_dataset138.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(generated["val"], ["101"])
            self.assertIn("100", generated["train"])
            self.assertIn("102", generated["train"])
            self.assertIn("rv_rocki_case", generated["train"])


if __name__ == "__main__":
    unittest.main()
