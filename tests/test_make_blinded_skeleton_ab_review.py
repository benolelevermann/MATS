from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import tifffile

from make_blinded_skeleton_ab_review import (
    build_report,
    load_or_create_manifest,
    resolve_paths,
    validate_output,
    write_outputs,
)


def write_case(root: Path, case_id: str, prediction_shift: int = 0) -> None:
    images = root / "images"
    labels = root / "labels"
    predictions = root / "predictions"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)

    image = np.zeros((16, 16), dtype=np.uint16)
    image[7:10, 2:14] = 1000
    label = np.zeros((16, 16), dtype=np.uint8)
    label[8, 2:12] = 1
    label[6:11, 11:15] = 2
    prediction = np.zeros_like(label)
    row = 8 + prediction_shift
    prediction[row, 2:12] = 1
    prediction[6:11, 11:15] = 2

    tifffile.imwrite(images / f"{case_id}_0000.tif", image)
    tifffile.imwrite(labels / f"{case_id}.tif", label)
    tifffile.imwrite(predictions / f"{case_id}.tif", prediction)


def make_paths(tmp_path: Path, output_name: str = "report"):
    pred_a = tmp_path / "model_a" / "predictions"
    pred_b = tmp_path / "model_b" / "predictions"
    for index in range(8):
        case_id = str(100 + index)
        write_case(tmp_path / "source", case_id)
        write_case(tmp_path / "model_a", case_id)
        write_case(tmp_path / "model_b", case_id, prediction_shift=index % 2)

    split_file = tmp_path / "splits_final.json"
    split_file.write_text(
        json.dumps(
            [
                {
                    "train": ["100", "101"],
                    "val": [str(100 + index) for index in range(2, 8)],
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        project_root=tmp_path,
        images_dir=tmp_path / "source" / "images",
        labels_dir=tmp_path / "source" / "labels",
        split_file=split_file,
        manifest=tmp_path / "panel_manifest.csv",
        predictions_a=pred_a,
        predictions_b=pred_b,
        output_dir=tmp_path / output_name,
    )
    return resolve_paths(args)


class BlindedSkeletonReviewTests(unittest.TestCase):
    def test_fixed_manifest_uses_validation_cases_and_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            manifest = tmp_path / "panel_manifest.csv"
            validation = ["102", "103", "104", "105", "106", "107"]
            first = load_or_create_manifest(manifest, validation, 4, 42, 0)
            second = load_or_create_manifest(
                manifest, list(reversed(validation)), 2, 999, 0
            )

            self.assertEqual(len(first), 4)
            self.assertLessEqual(set(first), set(validation))
            self.assertEqual(second, first)
            with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["case_id"] for row in rows], first)

    def test_builds_offline_blinded_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            paths = make_paths(tmp_path)
            validation = ["102", "103", "104", "105", "106", "107"]
            case_ids = load_or_create_manifest(paths.manifest, validation, 4, 42, 0)
            validate_output(paths.output_dir, overwrite=False)
            report = build_report(
                paths,
                case_ids,
                model_a="Current baseline",
                model_b="Full resolution",
                blind_seed=17,
                fold=0,
                selection_seed=42,
            )
            write_outputs(paths, report)

            html = (paths.output_dir / "review.html").read_text(encoding="utf-8")
            self.assertIn("Geblendeter Skeleton-A/B-Vergleich", html)
            self.assertIn("Bewertung abschließen &amp; entblinden", html)
            self.assertIn("localStorage", html)
            self.assertLess(html.index('key: "continuity"'), html.index('key: "overall"'))
            self.assertEqual(len(report["cases"]), 4)
            self.assertTrue((paths.output_dir / "blind_key.json").is_file())
            for case_id in case_ids:
                case_assets = paths.output_dir / "assets" / case_id
                self.assertTrue((case_assets / "original.png").is_file())
                self.assertTrue((case_assets / "ground_truth_overlay.png").is_file())
                self.assertTrue((case_assets / "prediction_a_overlay.png").is_file())
                self.assertTrue((case_assets / "prediction_b_overlay.png").is_file())
                self.assertTrue((case_assets / "difference.png").is_file())

    def test_refuses_partial_fixed_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            paths = make_paths(tmp_path)
            missing_prediction = paths.predictions_b / "103.tif"
            missing_prediction.unlink()
            paths.output_dir.mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "No partial report"):
                build_report(
                    paths,
                    ["102", "103"],
                    model_a="A",
                    model_b="B",
                    blind_seed=1,
                    fold=0,
                    selection_seed=42,
                )

    def test_existing_report_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            output.mkdir()
            (output / "review.html").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                validate_output(output, overwrite=False)
            validate_output(output, overwrite=True)


if __name__ == "__main__":
    unittest.main()
