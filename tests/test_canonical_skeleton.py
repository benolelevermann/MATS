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

    def test_swc_uses_manual_style_soma_root_links(self) -> None:
        soma = np.zeros((48, 48), dtype=bool)
        soma[18:30, 18:30] = True
        skeleton = np.zeros_like(soma)
        skeleton[19, 7:18] = True
        skeleton[21, 30:43] = True

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.swc"
            report = write_swc(path, skeleton, soma)
            nodes = {}
            for line_text in path.read_text(encoding="utf-8").splitlines():
                if not line_text or line_text.startswith("#"):
                    continue
                fields = line_text.split()
                nodes[int(fields[0])] = (
                    int(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    int(fields[6]),
                )

        root_x = nodes[1][1]
        root_y = nodes[1][2]
        path_start_ids = []
        for node_id, (node_type, x, y, parent_id) in nodes.items():
            if parent_id == 1:
                path_start_ids.append(node_id)
                self.assertEqual(node_type, 3)
                self.assertAlmostEqual(x, root_x)
                self.assertAlmostEqual(y, root_y)
        self.assertEqual(nodes[1][0], 1)
        self.assertEqual(len(path_start_ids), 2)

        attachment_ids = []
        connector_edges = set()
        for path_start_id in path_start_ids:
            children = [
                node_id
                for node_id, (_node_type, _x, _y, parent_id) in nodes.items()
                if parent_id == path_start_id
            ]
            self.assertEqual(len(children), 1)
            attachment_ids.append(children[0])
            connector_edges.add((children[0], path_start_id))
        attachments = [(nodes[node_id][1], nodes[node_id][2]) for node_id in attachment_ids]
        self.assertAlmostEqual(
            root_x,
            sum(point[0] for point in attachments) / len(attachments),
        )
        self.assertAlmostEqual(
            root_y,
            sum(point[1] for point in attachments) / len(attachments),
        )
        self.assertNotAlmostEqual(root_y, 23.5)

        edge_lengths = []
        for node_id, (_node_type, x, y, parent_id) in nodes.items():
            if parent_id < 0 or parent_id == 1:
                continue
            if (node_id, parent_id) in connector_edges:
                continue
            _, parent_x, parent_y, _ = nodes[parent_id]
            edge_lengths.append(np.hypot(x - parent_x, y - parent_y))
        self.assertLessEqual(max(edge_lengths), np.sqrt(2) + 1e-6)
        self.assertEqual(report["skeleton_nodes"], int(skeleton.sum()))
        self.assertEqual(report["soma_connector_nodes"], 3)
        self.assertEqual(report["shared_path_start_nodes"], 2)
        self.assertEqual(report["soma_center_connector_nodes"], 0)

    def test_single_primary_path_adds_manual_style_soma_center_path(self) -> None:
        soma = np.zeros((48, 48), dtype=bool)
        soma[18:30, 18:30] = True
        skeleton = np.zeros_like(soma)
        skeleton[23, 30:43] = True

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.swc"
            report = write_swc(path, skeleton, soma)
            nodes = {}
            for line_text in path.read_text(encoding="utf-8").splitlines():
                if not line_text or line_text.startswith("#"):
                    continue
                fields = line_text.split()
                nodes[int(fields[0])] = (
                    int(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    int(fields[6]),
                )

        # The root is the soma-adjacent skeleton attachment, not the soma center.
        self.assertEqual(nodes[1][0], 1)
        self.assertAlmostEqual(nodes[1][1], 30.0)
        self.assertAlmostEqual(nodes[1][2], 23.0)

        root_children = [
            node_id
            for node_id, (_node_type, _x, _y, parent_id) in nodes.items()
            if parent_id == 1
        ]
        self.assertEqual(len(root_children), 2)
        process_children = [node_id for node_id in root_children if nodes[node_id][1] > 30.0]
        connector_starts = [
            node_id
            for node_id in root_children
            if nodes[node_id][1] == 30.0 and nodes[node_id][2] == 23.0
        ]
        self.assertEqual(len(process_children), 1)
        self.assertEqual(len(connector_starts), 1)

        connector_end = [
            node_id
            for node_id, (_node_type, _x, _y, parent_id) in nodes.items()
            if parent_id == connector_starts[0]
        ]
        self.assertEqual(len(connector_end), 1)
        self.assertAlmostEqual(nodes[connector_end[0]][1], 23.5)
        self.assertAlmostEqual(nodes[connector_end[0]][2], 23.5)
        self.assertEqual(report["skeleton_nodes"], int(skeleton.sum()))
        self.assertEqual(report["soma_connector_nodes"], 3)
        self.assertEqual(report["shared_path_start_nodes"], 0)
        self.assertEqual(report["soma_center_connector_nodes"], 2)

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
