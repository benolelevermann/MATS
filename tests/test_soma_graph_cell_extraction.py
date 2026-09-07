from __future__ import annotations

import unittest

import numpy as np

from soma_graph_cell_extraction import (
    GraphSplitSettings,
    assign_fragments_to_somata,
    build_fragment_graph,
    split_soma_mask,
)


def disk(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.indices(shape)
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


class SomaSplitTests(unittest.TestCase):
    def test_oversized_touching_somata_are_split(self) -> None:
        shape = (120, 160)
        mask = np.zeros(shape, dtype=bool)
        for center in ((15, 15), (15, 50), (15, 85), (15, 120)):
            mask |= disk(shape, center, 8)
        mask |= disk(shape, (75, 65), 13)
        mask |= disk(shape, (75, 86), 13)

        instances, report, info, reference_area = split_soma_mask(
            mask,
            GraphSplitSettings(large_soma_factor=1.55, h_maxima_radius_factor=0.10),
        )

        self.assertGreater(reference_area, 150)
        self.assertEqual(int(instances.max()), 6)
        split_rows = [row for row in report if row["reason"] == "watershed_split"]
        self.assertEqual(len(split_rows), 1)
        split_ids = [int(value) for value in str(split_rows[0]["output_instances"]).split(";")]
        self.assertEqual(len(split_ids), 2)
        self.assertTrue(all(info[value]["split_count"] == 2 for value in split_ids))

    def test_normal_somata_remain_single_instances(self) -> None:
        shape = (80, 120)
        mask = disk(shape, (25, 25), 9) | disk(shape, (55, 90), 10)
        instances, report, _, _ = split_soma_mask(mask, GraphSplitSettings())
        self.assertEqual(int(instances.max()), 2)
        self.assertTrue(all(row["reason"] == "kept" for row in report))


class DirectionalGraphTests(unittest.TestCase):
    def test_crossing_lines_continue_toward_the_matching_soma(self) -> None:
        shape = (81, 81)
        skeleton = np.zeros(shape, dtype=bool)
        skeleton[40, 10:71] = True
        skeleton[10:71, 40] = True
        soma = np.zeros(shape, dtype=np.uint32)
        soma[disk(shape, (40, 6), 3)] = 1
        soma[disk(shape, (6, 40), 3)] = 2
        probability = np.ones(shape, dtype=np.float32)
        raw = np.ones(shape, dtype=np.float32)
        settings = GraphSplitSettings(
            soma_attach_radius=3,
            direction_weight=40.0,
            fragment_margin_threshold=0.02,
        )

        graph = build_fragment_graph(skeleton, probability, raw, settings)
        assignment, ambiguous, shared, _, _ = assign_fragments_to_somata(graph, soma, settings)

        self.assertEqual(int(assignment[40, 65]), 1)
        self.assertEqual(int(assignment[65, 40]), 2)
        self.assertFalse(bool(ambiguous[40, 65]))
        self.assertFalse(bool(ambiguous[65, 40]))
        self.assertTrue(np.all(np.isfinite(graph.fragment_costs)))
        self.assertGreaterEqual(int(shared.sum()), 1)

    def test_empty_skeleton_is_finite(self) -> None:
        shape = (32, 32)
        skeleton = np.zeros(shape, dtype=bool)
        soma = np.zeros(shape, dtype=np.uint32)
        soma[disk(shape, (16, 16), 4)] = 1
        settings = GraphSplitSettings()
        graph = build_fragment_graph(
            skeleton,
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            settings,
        )
        assignment, ambiguous, shared, details, report = assign_fragments_to_somata(
            graph, soma, settings
        )
        self.assertEqual(graph.count, 0)
        self.assertEqual(int(assignment.sum()), 0)
        self.assertFalse(np.any(ambiguous))
        self.assertFalse(np.any(shared))
        self.assertEqual(details, {})
        self.assertEqual(report, [])


if __name__ == "__main__":
    unittest.main()
