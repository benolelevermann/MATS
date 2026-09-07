from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import tifffile

from make_skeleton_calibration_review import build_report, write_report
from skeleton_connectivity_experiments import generate_hysteresis_variants


class SkeletonCalibrationReviewTests(unittest.TestCase):
    def test_r2_candidates_feed_named_offline_calibration_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            labels_dir = root / "labels"
            images_dir = root / "images"
            probabilities_dir = root / "probabilities"
            for directory in (
                baseline_dir,
                labels_dir,
                images_dir,
                probabilities_dir,
            ):
                directory.mkdir()

            case_id = "case001"
            label = np.zeros((16, 16), dtype=np.uint8)
            label[8, 2:12] = 1
            label[6:11, 12:15] = 2
            baseline = label.copy()
            baseline[8, 7] = 0
            image = (label > 0).astype(np.uint16) * 1000
            probabilities = np.zeros((3, 16, 16), dtype=np.float32)
            probabilities[0] = 0.05
            probabilities[1, baseline == 1] = 0.95
            probabilities[1, 8, 7] = 0.25
            probabilities[2, baseline == 2] = 0.95

            tifffile.imwrite(baseline_dir / f"{case_id}.tif", baseline)
            tifffile.imwrite(labels_dir / f"{case_id}.tif", label)
            tifffile.imwrite(images_dir / f"{case_id}_0000.tif", image)
            np.savez_compressed(
                probabilities_dir / f"{case_id}.npz",
                probabilities=probabilities,
            )
            manifest = root / "calibration.csv"
            with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=["case_id"])
                writer.writeheader()
                writer.writerow({"case_id": case_id})

            output_root = root / "r2"
            variants = generate_hysteresis_variants(
                Namespace(
                    baseline_dir=baseline_dir,
                    labels_dir=labels_dir,
                    probabilities_dir=probabilities_dir,
                    output_root=output_root,
                    t_low=[0.30, 0.20, 0.10],
                    overwrite=False,
                )
            )
            review_args = Namespace(
                project_root=root,
                images_dir=images_dir,
                labels_dir=labels_dir,
                base_dir=baseline_dir,
                base_name="R0",
                candidate=[
                    f"{variant['label']}={variant['directory']}" for variant in variants
                ],
                manifest=manifest,
                output_dir=output_root / "calibration_review",
                overwrite=False,
            )
            report = build_report(review_args)
            html_path = write_report(review_args, report)

            self.assertEqual(len(report["cases"]), 1)
            self.assertEqual(len(report["candidate_order"]), 3)
            self.assertTrue(html_path.is_file())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Skeleton-Kalibrierung", html)
            self.assertIn("Fehlverbindung", html)
            summary = json.loads(
                (output_root / "tlow_0p200" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["parameters"]["t_low"], 0.2)
            self.assertTrue((output_root / "tlow_0p200" / f"{case_id}.tif").is_file())


if __name__ == "__main__":
    unittest.main()
