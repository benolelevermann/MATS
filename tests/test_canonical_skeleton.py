from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from skimage.draw import line

from canonical_skeleton import (
    canonicalize_semantic,
    canonicalize_skeleton,
    detached_component_count,
    is_canonical_one_pixel,
)
from r_pipeline.export_cells_for_r_pipeline import write_swc


class CanonicalSkeletonTests(unittest.TestCase):
    def test_thick_curve_becomes_idempotent_one_pixel_centerline(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        for offset in (-1, 0, 1):
            rows, columns = line(10 + offset, 8, 48 + offset, 45)
            mask[rows, columns] = True

        thin, report = canonicalize_skeleton(mask)

        self.assertTrue(is_canonical_one_pixel(thin))
        self.assertLess(int(thin.sum()), int(mask.sum()))
        self.assertGreater(report.pixels_removed_by_thinning, 0)

    def test_short_soma_gap_is_written_into_the_mask(self) -> None:
        soma = np.zeros((32, 32), dtype=bool)
        soma[13:19, 5:11] = True
        skeleton = np.zeros_like(soma)
        skeleton[16, 12:26] = True

        thin, report = canonicalize_skeleton(
            skeleton,
            soma=soma,
            max_soma_gap_px=3.0,
        )

        self.assertGreater(report.soma_gap_pixels_added, 0)
        self.assertEqual(detached_component_count(thin, soma), 0)
        self.assertTrue(is_canonical_one_pixel(thin))

    def test_long_gap_remains_explicitly_detached(self) -> None:
        soma = np.zeros((32, 32), dtype=bool)
        soma[13:19, 5:11] = True
        skeleton = np.zeros_like(soma)
        skeleton[16, 18:26] = True

        thin, report = canonicalize_skeleton(
            skeleton,
            soma=soma,
            max_soma_gap_px=3.0,
        )

        self.assertEqual(report.soma_gap_pixels_added, 0)
        self.assertEqual(detached_component_count(thin, soma), 1)

    def test_semantic_keeps_soma_and_supported_labels(self) -> None:
        semantic = np.zeros((24, 24), dtype=np.uint8)
        semantic[8:16, 3:9] = 2
        semantic[10:13, 8:21] = 1
        expected_soma = semantic == 2

        canonical, _ = canonicalize_semantic(semantic, max_soma_gap_px=3.0)

        np.testing.assert_array_equal(canonical == 2, expected_soma)
        self.assertLessEqual(set(np.unique(canonical).tolist()), {0, 1, 2})
        self.assertTrue(is_canonical_one_pixel(canonical == 1))

    def test_swc_uses_only_explicit_adjacent_edges(self) -> None:
        soma = np.zeros((48, 48), dtype=bool)
        soma[18:30, 18:30] = True
        skeleton = np.zeros_like(soma)
        skeleton[23, 7:18] = True
        skeleton[24, 30:43] = True

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.swc"
            report = write_swc(path, skeleton, soma)
            nodes = {}
            for line_text in path.read_text(encoding="utf-8").splitlines():
                if not line_text or line_text.startswith("#"):
                    continue
                fields = line_text.split()
                nodes[int(fields[0])] = (
                    float(fields[2]),
                    float(fields[3]),
                    int(fields[6]),
                )

        edge_lengths = []
        for node_id, (x, y, parent_id) in nodes.items():
            if parent_id < 0:
                continue
            parent_x, parent_y, _ = nodes[parent_id]
            edge_lengths.append(np.hypot(x - parent_x, y - parent_y))
        self.assertLessEqual(max(edge_lengths), np.sqrt(2) + 1e-6)
        self.assertEqual(report["skeleton_nodes"], int(skeleton.sum()))
        self.assertGreater(report["soma_connector_nodes"], 1)

    def test_swc_rejects_a_hidden_long_background_connection(self) -> None:
        soma = np.zeros((48, 48), dtype=bool)
        soma[18:30, 18:30] = True
        skeleton = np.zeros_like(soma)
        skeleton[23, 2:8] = True
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "not attached"):
                write_swc(Path(temporary) / "cell.swc", skeleton, soma)


if __name__ == "__main__":
    unittest.main()
