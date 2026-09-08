from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from instance_separator import (
    SomaSeededSeparationLoss,
    SomaSeededSeparationUNet,
    assign_memberships_to_instances,
    build_membership_target,
    exact_spatial_augmentation,
    make_model_input,
    soma_guided_crop,
)
from apply_instance_separator import export_cells


def two_cell_labels(size: int = 64) -> tuple[np.ndarray, ...]:
    semantic = np.zeros((size, size), dtype=np.uint8)
    primary = np.zeros((size, size), dtype=np.uint16)
    secondary = np.zeros((size, size), dtype=np.uint16)
    ambiguous = np.zeros((size, size), dtype=np.uint8)
    semantic[28:34, 7:13] = 2
    primary[28:34, 7:13] = 7
    semantic[30, 13:52] = 1
    primary[30, 13:52] = 7
    semantic[36:42, 51:57] = 2
    primary[36:42, 51:57] = 42
    semantic[38, 12:51] = 1
    primary[38, 12:51] = 42
    semantic[31:38, 31] = 1
    primary[31:38, 31] = 7
    secondary[31:38, 31] = 42
    return semantic, primary, secondary, ambiguous


class InstanceSeparatorTests(unittest.TestCase):
    def test_query_target_uses_id_as_selector_not_as_class_value(self) -> None:
        semantic, primary, secondary, ambiguous = two_cell_labels()
        target_7 = build_membership_target(
            semantic, primary, secondary, 7, ambiguous
        )
        target_42 = build_membership_target(
            semantic, primary, secondary, 42, ambiguous
        )
        self.assertEqual(set(np.unique(target_7.membership)), {False, True})
        self.assertTrue(target_7.membership[34, 31])
        self.assertTrue(target_42.membership[34, 31])
        self.assertTrue(target_7.query_soma[30, 9])
        self.assertFalse(target_42.query_soma[30, 9])

    def test_instance_without_visible_soma_is_rejected(self) -> None:
        semantic = np.zeros((16, 16), dtype=np.uint8)
        primary = np.zeros_like(semantic, dtype=np.uint16)
        secondary = np.zeros_like(primary)
        semantic[8, 2:14] = 1
        primary[8, 2:14] = 9
        with self.assertRaises(ValueError):
            build_membership_target(semantic, primary, secondary, 9)

    def test_soma_guided_crop_keeps_query_soma_and_one_pixel_labels(self) -> None:
        semantic, primary, secondary, ambiguous = two_cell_labels()
        raw = semantic.astype(np.float32) * 17.0
        cropped = soma_guided_crop(
            [raw, semantic, primary, secondary, ambiguous],
            7,
            32,
            np.random.default_rng(8),
        )
        self.assertEqual(cropped[0].shape, (32, 32))
        self.assertTrue(np.any((cropped[2] == 7) & (cropped[1] == 2)))
        self.assertTrue(np.array_equal(cropped[0] > 0, cropped[1] > 0))

    def test_exact_augmentation_keeps_all_maps_aligned(self) -> None:
        semantic, primary, secondary, ambiguous = two_cell_labels()
        raw = semantic.astype(np.float32)
        transformed = exact_spatial_augmentation(
            [raw, semantic, primary, secondary, ambiguous],
            np.random.default_rng(91),
        )
        self.assertTrue(np.array_equal(transformed[0] > 0, transformed[1] > 0))
        self.assertEqual(np.count_nonzero(transformed[1] == 1), np.count_nonzero(semantic == 1))
        self.assertEqual(np.count_nonzero(transformed[3]), np.count_nonzero(secondary))

    def test_model_and_balanced_loss_are_finite_and_differentiable(self) -> None:
        model = SomaSeededSeparationUNet(base_channels=4)
        output = model(torch.randn(2, 4, 64, 64))
        membership = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
        membership[:, :, 20:45, 20:45] = True
        valid = torch.ones_like(membership)
        semantic = torch.ones(2, 1, 64, 64, dtype=torch.long)
        losses = SomaSeededSeparationLoss()(output, membership, valid, semantic)
        losses["loss"].backward()
        self.assertEqual(output.shape, (2, 1, 64, 64))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        )

    def test_assignment_retains_two_ids_only_where_both_queries_are_positive(self) -> None:
        semantic = np.zeros((8, 12), dtype=np.uint8)
        semantic[3, 1:11] = 1
        semantic[2:5, 1:3] = 2
        semantic[2:5, 9:11] = 2
        somas = np.zeros_like(semantic, dtype=np.uint16)
        somas[2:5, 1:3] = 7
        somas[2:5, 9:11] = 42
        probabilities = np.zeros((2, 8, 12), dtype=np.float32)
        probabilities[0, semantic > 0] = 0.2
        probabilities[1, semantic > 0] = 0.2
        probabilities[0, :, :6] = 0.9
        probabilities[1, :, 6:] = 0.9
        probabilities[:, 3, 5:7] = 0.8
        primary, secondary, confidence = assign_memberships_to_instances(
            probabilities, np.asarray([7, 42]), semantic, somas, membership_threshold=0.5
        )
        self.assertEqual(primary[3, 3], 7)
        self.assertEqual(primary[3, 8], 42)
        self.assertGreater(secondary[3, 5], 0)
        self.assertEqual(secondary[3, 3], 0)
        self.assertTrue(np.all(confidence[semantic > 0] > 0))

    def test_model_input_highlights_only_selected_soma(self) -> None:
        raw = np.ones((16, 16), dtype=np.float32)
        semantic = np.zeros((16, 16), dtype=np.uint8)
        semantic[5:8, 5:8] = 2
        query = semantic == 2
        model_input = make_model_input(raw, semantic, query)
        self.assertEqual(model_input.shape, (1, 4, 16, 16))
        self.assertEqual(int(model_input[0, 3].sum()), 9)

    def test_one_unsafe_member_rejects_the_complete_conflict_group(self) -> None:
        raw = np.zeros((48, 48), dtype=np.uint16)
        semantic = np.zeros((48, 48), dtype=np.uint8)
        primary = np.zeros((48, 48), dtype=np.uint16)
        secondary = np.zeros_like(primary)
        somas = np.zeros_like(primary)
        confidence = np.ones((48, 48), dtype=np.float32)
        semantic[20:25, 5:10] = 2
        semantic[20:25, 38:43] = 2
        somas[20:25, 5:10] = 1
        somas[20:25, 38:43] = 2
        primary[20:25, 5:10] = 1
        primary[20:25, 38:43] = 2
        semantic[22, 10:38] = 1
        primary[22, 10:24] = 1
        primary[22, 24:38] = 2
        semantic[12:22, 15] = 1
        primary[21, 15] = 1
        primary[12:21, 15] = 2  # disconnected from soma 2 despite material contact
        conflicts: list[dict[str, object]] = [
            {
                "component_id": 1,
                "soma_ids": [1, 2],
                "crop_yxyx": [0, 48, 0, 48],
                "status": "separated",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            accepted, rejected = export_cells(
                raw=raw,
                semantic=semantic,
                primary=primary,
                secondary=secondary,
                soma_instances=somas,
                confidence=confidence,
                conflicts=conflicts,
                output_dir=Path(temporary_directory) / "cells",
                margin=2,
                minimum_size=16,
                min_skeleton_pixels=8,
            )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 2)
        self.assertEqual(conflicts[0]["status"], "rejected_uncertain")


if __name__ == "__main__":
    unittest.main()
