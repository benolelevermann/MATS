from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from sweep_hysteresis_tlow import threshold_name, validate_thresholds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ThresholdSweepTests(unittest.TestCase):
    def test_threshold_validation_and_names(self) -> None:
        self.assertEqual(validate_thresholds(0.6, [0.3, 0.2, 0.3]), [0.3, 0.2])
        self.assertEqual(threshold_name(0.25), "tlow_0p250")
        with self.assertRaises(ValueError):
            validate_thresholds(0.6, [0.6])

    def test_end_to_end_sweep_writes_variants_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (64, 64)
            probabilities = np.zeros((3, *shape), dtype=np.float32)
            probabilities[0] = 0.94
            probabilities[1] = 0.03
            probabilities[2] = 0.03

            rows, cols = np.ogrid[: shape[0], : shape[1]]
            soma = (rows - 32) ** 2 + (cols - 12) ** 2 <= 4**2
            probabilities[:, soma] = np.asarray([0.02, 0.03, 0.95], dtype=np.float32)[:, None]
            probabilities[0, 32, 16:51] = 0.15
            probabilities[1, 32, 16:51] = 0.82
            probabilities[2, 32, 16:51] = 0.03
            probabilities[0, 32, 30:35] = 0.75
            probabilities[1, 32, 30:35] = 0.22
            probabilities[2, 32, 30:35] = 0.03

            probability_path = root / "synthetic.npz"
            raw_path = root / "raw.tif"
            output_root = root / "sweep"
            np.savez_compressed(probability_path, probabilities=probabilities)
            tifffile.imwrite(raw_path, np.arange(64 * 64, dtype=np.uint16).reshape(shape))

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "sweep_hysteresis_tlow.py"),
                    "--probabilities",
                    str(probability_path),
                    "--raw",
                    str(raw_path),
                    "--output-dir",
                    str(output_root),
                    "--t-high",
                    "0.60",
                    "--t-low",
                    "0.30",
                    "0.20",
                    "--min-soma-area",
                    "5",
                    "--min-skeleton-px",
                    "3",
                    "--cell-crop-size",
                    "48",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for folder in ("tlow_0p300", "tlow_0p200"):
                for name in (
                    "semantic_0-1-2.tif",
                    "skeleton.tif",
                    "added_vs_argmax.tif",
                    "removed_vs_argmax.tif",
                    "preview.png",
                ):
                    self.assertTrue((output_root / folder / name).is_file(), f"{folder}/{name}")
            with (output_root / "tlow_sweep.csv").open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 2)
            self.assertGreater(int(metrics[1]["skeleton_px"]), int(metrics[0]["skeleton_px"]))
            self.assertLess(int(metrics[1]["skeleton_components"]), int(metrics[0]["skeleton_components"]))
            semantic = tifffile.imread(output_root / "tlow_0p200" / "semantic_0-1-2.tif")
            self.assertLessEqual(set(np.unique(semantic).tolist()), {0, 1, 2})
            self.assertTrue((output_root / "tlow_sweep_summary.json").is_file())
            self.assertTrue((output_root / "tlow_sweep_report.html").is_file())
            self.assertTrue((output_root / "tlow_cell_comparison.csv").is_file())
            cell_root = output_root / "cell_crops" / "soma_0001"
            self.assertTrue((cell_root / "raw.png").is_file())
            self.assertTrue((cell_root / "tlow_0p300.png").is_file())
            self.assertTrue((cell_root / "tlow_0p200.png").is_file())
            with (output_root / "tlow_cell_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                cell_metrics = list(csv.DictReader(stream))
            self.assertEqual(len(cell_metrics), 2)
            self.assertEqual({row["soma_id"] for row in cell_metrics}, {"1"})
            report = (output_root / "tlow_sweep_report.html").read_text(encoding="utf-8")
            self.assertIn("Soma 0001", report)
            self.assertIn("dieselbe Soma-Position", report)


if __name__ == "__main__":
    unittest.main()
