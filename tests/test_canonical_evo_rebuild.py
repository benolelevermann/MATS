from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from canonical_skeleton import is_canonical_one_pixel
from rebuild_evo_cells_canonical_1px import rebuild_cell


def write_source_cell(root: Path, gap_start: int) -> Path:
    cell = root / "cell000001"
    cell.mkdir()
    shape = (64, 64)
    raw = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape)
    soma = np.zeros(shape, dtype=np.uint8)
    soma[25:39, 8:22] = 1
    skeleton = np.zeros(shape, dtype=np.uint8)
    skeleton[30:33, gap_start:55] = 1
    semantic = np.zeros(shape, dtype=np.uint8)
    semantic[skeleton > 0] = 1
    semantic[soma > 0] = 2
    for name, array in (
        ("raw.tif", raw),
        ("skeleton.tif", skeleton),
        ("soma.tif", soma),
        ("seg.tif", semantic),
        ("postprocessed.tif", semantic),
        ("prediction.tif", semantic),
    ):
        tifffile.imwrite(cell / name, array)
    (cell / "metadata.json").write_text(
        json.dumps({"folder": cell.name}), encoding="utf-8"
    )
    (cell / "bounds.json").write_text(
        json.dumps(
            {
                "x_min": 100,
                "y_min": 200,
                "width": 64,
                "height": 64,
            }
        ),
        encoding="utf-8",
    )
    (cell / "location.json").write_text(
        json.dumps({"global_x": 110, "global_y": 210}), encoding="utf-8"
    )
    return cell


class CanonicalEvoRebuildTests(unittest.TestCase):
    def test_safe_cell_uses_the_same_one_pixel_mask_in_tiff_and_swc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_source_cell(root, gap_start=23)
            output = root / "output"
            row = rebuild_cell(source, output, max_soma_gap_px=3.0, copy_files=True)

            self.assertEqual(row["status"], "safe")
            skeleton = np.squeeze(tifffile.imread(output / "skeleton.tif")) > 0
            semantic = np.squeeze(tifffile.imread(output / "seg.tif"))
            np.testing.assert_array_equal(skeleton, semantic == 1)
            self.assertTrue(is_canonical_one_pixel(skeleton))
            self.assertTrue((output / "seg-000.swc").is_file())
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "accepted_single_cell_canonical_1px")

    def test_long_detached_component_is_routed_to_review_without_swc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_source_cell(root, gap_start=35)
            output = root / "output"
            row = rebuild_cell(source, output, max_soma_gap_px=3.0, copy_files=True)

            self.assertEqual(row["status"], "review")
            self.assertFalse((output / "seg-000.swc").exists())


if __name__ == "__main__":
    unittest.main()
