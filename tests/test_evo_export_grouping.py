from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from organize_evo_cells_by_source_image import group_cells


class EvoExportGroupingTests(unittest.TestCase):
    def test_flat_cells_are_grouped_by_exact_job_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "flat"
            runs = root / "runs"
            output = root / "grouped"
            for index, (job_id, filename) in enumerate(
                (
                    ("job_a", "10x_zoom1.6_ERplate_E8_DMSO.tif"),
                    ("job_b", "plate#02_PBS_0000.tif"),
                ),
                start=1,
            ):
                cell = source / f"cell{index:06d}"
                cell.mkdir(parents=True)
                (cell / "raw.tif").write_bytes(f"raw-{index}".encode())
                (cell / "review.json").write_text(
                    json.dumps({"job_id": job_id, "case": "truncated"}),
                    encoding="utf-8",
                )
                status = runs / job_id / "status.json"
                status.parent.mkdir(parents=True)
                status.write_text(json.dumps({"filename": filename}), encoding="utf-8")

            summary = group_cells(source, output, runs)

            self.assertEqual(summary["cells"], 2)
            self.assertEqual(summary["source_images"], 2)
            first = output / "10x_zoom1.6_ERplate_E8_DMSO" / "cell000001" / "raw.tif"
            second = output / "plate_02_PBS" / "cell000002" / "raw.tif"
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            self.assertTrue(os.path.samefile(source / "cell000001" / "raw.tif", first))
            with self.assertRaises(FileExistsError):
                group_cells(source, output, runs)


if __name__ == "__main__":
    unittest.main()
