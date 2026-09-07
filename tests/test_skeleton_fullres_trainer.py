from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from custom_trainers.skeleton_recall.nnUNetTrainerSkeletonRecallCells import (
    deep_supervision_loss_weights,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnlyDebug50,
)


class FullResolutionTrainerTests(unittest.TestCase):
    def test_baseline_weights_are_unchanged(self) -> None:
        weights = deep_supervision_loss_weights(
            4,
            full_resolution_only=False,
        )
        np.testing.assert_allclose(
            weights,
            np.array([4 / 7, 2 / 7, 1 / 7, 0], dtype=np.float64),
        )

    def test_full_resolution_variant_uses_only_first_output(self) -> None:
        weights = deep_supervision_loss_weights(
            4,
            full_resolution_only=True,
        )
        np.testing.assert_array_equal(weights, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_experiment_inherits_baseline_class_weighting(self) -> None:
        self.assertTrue(
            issubclass(
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly,
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x,
            )
        )
        self.assertEqual(
            nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly.skeleton_recall_class_weights,
            (2.0, 1.0),
        )
        self.assertTrue(
            nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly.full_resolution_loss_only
        )
        self.assertTrue(
            issubclass(
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnlyDebug50,
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly,
            )
        )

    def test_debug_variant_stops_after_50_epochs(self) -> None:
        parent = nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly

        def initialize_parent(instance, *args, **kwargs) -> None:
            instance.num_epochs = 1000

        with patch.object(parent, "__init__", initialize_parent):
            trainer = (
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnlyDebug50(
                    {}, "2d", 0, {}, None
                )
            )
        self.assertEqual(trainer.num_epochs, 50)

    def test_debug_constructor_keeps_nnunet_parameter_names(self) -> None:
        parameters = list(
            inspect.signature(
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnlyDebug50.__init__
            ).parameters
        )
        self.assertEqual(
            parameters,
            ["self", "plans", "configuration", "fold", "dataset_json", "device"],
        )


if __name__ == "__main__":
    unittest.main()
