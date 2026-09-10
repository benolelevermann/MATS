from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evo_test_holdout import find_overlaps, load_registry


class EvoTestHoldoutTests(unittest.TestCase):
    def test_finds_direct_and_synthetic_donor_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "holdout.csv"
            with registry.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["case_id", "raw_sha256"]
                )
                writer.writeheader()
                writer.writerows(
                    [{"case_id": "evotest_div10cc_cell7", "raw_sha256": ""}]
                )

            dataset = root / "Dataset999_test"
            images = dataset / "imagesTr"
            images.mkdir(parents=True)
            (images / "evotest_div10cc_cell7_0000.tif").write_bytes(b"test")
            donor_audit = dataset / "synthetic_donor_audit.csv"
            with donor_audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["case_id", "synthetic_case_id"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "evotest_div10cc_cell7",
                        "synthetic_case_id": "scene1",
                    }
                )

            overlaps = find_overlaps(dataset, load_registry(registry))

        self.assertEqual(
            {row["leakage_kind"] for row in overlaps},
            {"direct_training_case", "synthetic_donor"},
        )


if __name__ == "__main__":
    unittest.main()
