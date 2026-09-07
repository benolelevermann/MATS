from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from sweep_hysteresis_settings import build_variants, parse_setting, select_cells, threshold_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_namespace(**overrides: object) -> argparse.Namespace:
    defaults = {
        "t_high": [0.60, 0.40],
        "t_low": [0.30, 0.20],
        "setting": [],
        "alpha": [1.0 / 3.0],
        "no_argmax": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def synthetic_probabilities(shape: tuple[int, int]) -> np.ndarray:
    """Two somas, each with a neurite whose middle stretch is weak."""
    probabilities = np.zeros((3, *shape), dtype=np.float32)
    probabilities[0] = 0.94
    probabilities[1] = 0.03
    probabilities[2] = 0.03
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    for center_row, center_col in ((20, 10), (44, 10)):
        soma = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= 4**2
        probabilities[:, soma] = np.asarray([0.02, 0.03, 0.95], dtype=np.float32)[:, None]
        probabilities[0, center_row, 14:52] = 0.15
        probabilities[1, center_row, 14:52] = 0.82
        probabilities[2, center_row, 14:52] = 0.03
        # weak stretch: below argmax, above a low T_low, below a high T_low
        probabilities[0, center_row, 28:33] = 0.72
        probabilities[1, center_row, 28:33] = 0.25
        probabilities[2, center_row, 28:33] = 0.03
    return probabilities


class VariantBuildingTests(unittest.TestCase):
    def test_grid_skips_invalid_pairs_and_keeps_references_first(self) -> None:
        variants = build_variants(make_namespace(t_high=[0.60, 0.20], t_low=[0.30, 0.10]))
        keys = [variant["key"] for variant in variants]
        self.assertEqual(keys[0], "argmax")
        self.assertEqual(keys[1], "adaptive_a0p333")
        # 0.20/0.30 is invalid and must not appear
        self.assertEqual(keys[2:], ["th0p600_tl0p300", "th0p600_tl0p100", "th0p200_tl0p100"])

    def test_explicit_settings_and_deduplication(self) -> None:
        variants = build_variants(
            make_namespace(t_high=[0.60], t_low=[0.30], setting=["0.60:0.30", "0.50:0.05"], alpha=[], no_argmax=True)
        )
        self.assertEqual([variant["key"] for variant in variants], ["th0p600_tl0p300", "th0p500_tl0p050"])

    def test_rejects_impossible_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            build_variants(make_namespace(t_high=[0.5], t_low=[0.9], setting=["0.5:0.9"], alpha=[], no_argmax=True))
        with self.assertRaises(ValueError):
            build_variants(make_namespace(alpha=[1.5]))

    def test_setting_parsing_and_tokens(self) -> None:
        self.assertEqual(parse_setting("0.6:0.15"), (0.6, 0.15))
        self.assertEqual(parse_setting("0.6,0.15"), (0.6, 0.15))
        self.assertEqual(threshold_token(0.15), "0p150")
        with self.assertRaises(ValueError):
            parse_setting("0.6")


class CellSelectionTests(unittest.TestCase):
    def test_explicit_ids_win_and_stay_sorted(self) -> None:
        valid = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
        self.assertEqual(select_cells(valid, [4, 2], 1).tolist(), [2, 4])

    def test_even_spread_and_unlimited(self) -> None:
        valid = np.arange(1, 21, dtype=np.int64)
        self.assertEqual(select_cells(valid, None, 0).tolist(), valid.tolist())
        spread = select_cells(valid, None, 5)
        self.assertEqual(spread.tolist(), [1, 6, 11, 15, 20])

    def test_unknown_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_cells(np.asarray([1, 2], dtype=np.int64), [3], 0)


class EndToEndTests(unittest.TestCase):
    def test_sweep_writes_identical_crops_for_every_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (64, 64)
            probability_path = root / "synthetic.npz"
            raw_path = root / "raw.tif"
            output_root = root / "sweep"
            np.savez_compressed(probability_path, probabilities=synthetic_probabilities(shape))
            tifffile.imwrite(raw_path, np.arange(64 * 64, dtype=np.uint16).reshape(shape))

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "sweep_hysteresis_settings.py"),
                    "--probabilities", str(probability_path),
                    "--raw", str(raw_path),
                    "--output-dir", str(output_root),
                    "--t-high", "0.60", "0.40",
                    "--t-low", "0.30", "0.20",
                    "--min-soma-area", "5",
                    "--min-skeleton-px", "3",
                    "--cell-crop-size", "48",
                    "--max-cells", "0",
                    "--write-full-tifs",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            with (output_root / "settings_sweep.csv").open(newline="", encoding="utf-8") as stream:
                metrics = {row["key"]: row for row in csv.DictReader(stream)}
            expected = {
                "argmax",
                "adaptive_a0p333",
                "th0p600_tl0p300",
                "th0p600_tl0p200",
                "th0p400_tl0p300",
                "th0p400_tl0p200",
            }
            self.assertEqual(set(metrics), expected)

            # The weak stretch sits at p=0.25, so only T_low=0.20 grows through it.
            self.assertGreater(
                int(metrics["th0p600_tl0p200"]["skeleton_px"]), int(metrics["th0p600_tl0p300"]["skeleton_px"])
            )
            self.assertLess(
                int(metrics["th0p600_tl0p200"]["skeleton_components"]),
                int(metrics["th0p600_tl0p300"]["skeleton_components"]),
            )
            self.assertEqual(int(metrics["argmax"]["added_px"]), 0)
            self.assertEqual(int(metrics["argmax"]["removed_px"]), 0)

            for key in expected:
                self.assertTrue((output_root / key / "preview.png").is_file(), key)
            for name in ("semantic_0-1-2.tif", "skeleton.tif", "added_vs_argmax.tif", "removed_vs_argmax.tif"):
                self.assertTrue((output_root / "th0p600_tl0p200" / name).is_file(), name)
            semantic = tifffile.imread(output_root / "th0p600_tl0p200" / "semantic_0-1-2.tif")
            self.assertLessEqual(set(np.unique(semantic).tolist()), {0, 1, 2})

            # Both somas are compared, and every crop of one cell covers the identical window.
            for soma_id in (1, 2):
                cell_root = output_root / "cell_crops" / f"soma_{soma_id:04d}"
                self.assertTrue((cell_root / "raw.png").is_file())
                windows = set()
                for key in ("raw", *expected):
                    image = cell_root / f"{key}.png"
                    self.assertTrue(image.is_file(), f"{soma_id}/{key}")
                    with Image.open(image) as handle:
                        windows.add(handle.size)
                self.assertEqual(windows, {(48, 48)}, f"soma {soma_id} crops differ in size")

            with (output_root / "settings_cell_comparison.csv").open(newline="", encoding="utf-8") as stream:
                cell_rows = list(csv.DictReader(stream))
            self.assertEqual(len(cell_rows), 2 * len(expected))
            self.assertEqual({row["soma_id"] for row in cell_rows}, {"1", "2"})
            reference_rows = [row for row in cell_rows if row["setting"] == "adaptive_a0p333"]
            self.assertTrue(all(int(row["delta_vs_reference"]) == 0 for row in reference_rows))
            low_rows = [row for row in cell_rows if row["setting"] == "th0p600_tl0p200"]
            self.assertTrue(all(row["status"] == "isolated" for row in low_rows))

            summary = json.loads((output_root / "settings_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reference_setting"], "adaptive_a0p333")
            self.assertEqual(summary["compared_soma_ids"], [1, 2])

            report = (output_root / "settings_sweep_report.html").read_text(encoding="utf-8")
            self.assertIn("Soma 0001", report)
            self.assertIn("Soma 0002", report)
            self.assertIn("dieselbe Zelle", report)
            for key in expected:
                self.assertIn(f'value="{key}"', report)

    def test_pinned_soma_ids_restrict_the_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probability_path = root / "synthetic.npz"
            output_root = root / "sweep"
            np.savez_compressed(probability_path, probabilities=synthetic_probabilities((64, 64)))

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "sweep_hysteresis_settings.py"),
                    "--probabilities", str(probability_path),
                    "--output-dir", str(output_root),
                    "--t-high", "0.60",
                    "--t-low", "0.20",
                    "--alpha",
                    "--no-argmax",
                    "--min-soma-area", "5",
                    "--min-skeleton-px", "3",
                    "--cell-crop-size", "48",
                    "--soma-ids", "2",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output_root / "cell_crops" / "soma_0002").is_dir())
            self.assertFalse((output_root / "cell_crops" / "soma_0001").exists())
            summary = json.loads((output_root / "settings_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["compared_soma_ids"], [2])
            self.assertEqual(summary["reference_setting"], "th0p600_tl0p200")


if __name__ == "__main__":
    unittest.main()
