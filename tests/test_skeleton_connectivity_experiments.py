from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from evaluate_skeleton_ab_acceptance import evaluate
from skeleton_connectivity_experiments import (
    aggregate_metrics,
    grow_hysteresis_from_baseline,
    metrics_row,
    reconnect_semantic,
    select_calibration_cases,
    union_components,
)


class ConnectivityCandidateTests(unittest.TestCase):
    def test_baseline_seeded_hysteresis_closes_weak_gap_and_locks_soma(self) -> None:
        baseline = np.zeros((15, 15), dtype=np.uint8)
        baseline[7, 1:6] = 1
        baseline[7, 7:11] = 1
        baseline[5:10, 11:14] = 2
        probability = np.zeros(baseline.shape, dtype=np.float32)
        probability[baseline == 1] = 0.9
        probability[7, 6] = 0.25
        probability[baseline == 2] = 0.8

        result = grow_hysteresis_from_baseline(baseline, probability, 0.20)

        self.assertEqual(result[7, 6], 1)
        np.testing.assert_array_equal(result == 2, baseline == 2)
        self.assertEqual(union_components(result), 1)

    def test_reconnection_adds_straight_one_pixel_bridge_without_deletion(self) -> None:
        baseline = np.zeros((15, 15), dtype=np.uint8)
        baseline[7, 1:5] = 1
        baseline[7, 7:11] = 1

        result, records = reconnect_semantic(baseline, max_gap=4)

        self.assertEqual(union_components(baseline), 2)
        self.assertEqual(union_components(result), 1)
        self.assertTrue(np.all(result[baseline == 1] == 1))
        self.assertTrue(any(record["accepted"] for record in records))
        self.assertTrue(np.all(np.count_nonzero(result == 1, axis=0) <= 1))

    def test_reconnection_does_not_merge_two_soma_attached_components(self) -> None:
        baseline = np.zeros((17, 21), dtype=np.uint8)
        baseline[8, 2:8] = 1
        baseline[8, 11:17] = 1
        baseline[7:10, 0:2] = 2
        baseline[7:10, 17:20] = 2

        result, records = reconnect_semantic(baseline, max_gap=5)

        np.testing.assert_array_equal(result, baseline)
        self.assertTrue(
            any(
                record["reason"] == "would_merge_two_soma_attached_components"
                for record in records
            )
        )

    def test_automatic_gate_detects_skeleton_overgrowth(self) -> None:
        target = np.zeros((20, 20), dtype=np.uint8)
        target[10, 2:12] = 1
        target[8:13, 12:16] = 2
        baseline = target.copy()
        candidate = target.copy()
        candidate[2:18, 3:17] = 1
        candidate[target == 2] = 2
        rows = [metrics_row("1", baseline, candidate, target)]

        summary = aggregate_metrics(rows)

        self.assertFalse(summary["automatic_gate_pass"])
        self.assertFalse(
            summary["automatic_gates"]["skeleton_pixel_growth_at_most_25_percent"]
        )

    def test_empty_baseline_growth_summary_is_strict_json(self) -> None:
        target = np.zeros((12, 12), dtype=np.uint8)
        target[6, 2:9] = 1
        baseline = np.zeros_like(target)
        candidate = target.copy()

        summary = aggregate_metrics(
            [metrics_row("empty-baseline", baseline, candidate, target)]
        )

        self.assertFalse(
            summary["automatic_gates"]["skeleton_pixel_growth_at_most_25_percent"]
        )
        json.dumps(summary, allow_nan=False)

    def test_calibration_manifest_excludes_formal_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            baseline.mkdir()
            validation = [str(index) for index in range(30)]
            for index, case_id in enumerate(validation):
                label = np.zeros((24, 24), dtype=np.uint8)
                for component in range(index % 7 + 1):
                    label[2 + component * 3, 2:8] = 1
                tifffile.imwrite(baseline / f"{case_id}.tif", label)
            split = root / "splits_final.json"
            split.write_text(
                json.dumps([{"train": [], "val": validation}]), encoding="utf-8"
            )
            formal = root / "formal.csv"
            with formal.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["case_id"])
                writer.writeheader()
                for case_id in validation[:10]:
                    writer.writerow({"case_id": case_id})
            output = root / "calibration.csv"

            rows = select_calibration_cases(baseline, split, formal, output)

            self.assertEqual(len(rows), 12)
            self.assertTrue(set(row["case_id"] for row in rows).isdisjoint(validation[:10]))
            self.assertEqual(
                {row["selection_group"] for row in rows},
                {"fragmented", "middle", "clean"},
            )


class AcceptanceTests(unittest.TestCase):
    def test_manual_acceptance_requires_eight_net_continuity_wins(self) -> None:
        rows = []
        for index in range(20):
            winner = "candidate" if index < 12 else ("base" if index < 16 else "tie")
            row = {"complete": "yes"}
            for field in (
                "continuity",
                "false_connections",
                "soma_attachment",
                "one_pixel_width",
                "overall",
            ):
                row[f"{field}_actual_winner"] = winner
            rows.append(row)

        result = evaluate(rows, "candidate", "base", None)

        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["counts"]["continuity"]["candidate"]
            - result["counts"]["continuity"]["base"],
            8,
        )


if __name__ == "__main__":
    unittest.main()
