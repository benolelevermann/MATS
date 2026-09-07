from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import tifffile
from torch import nn

from custom_trainers.cldice.nnUNetTrainerClDiceCells import (
    FullResolutionClDiceLoss,
    SkeletonClassSoftClDiceLoss,
    nnUNetTrainerClDiceCellsAlpha01,
    nnUNetTrainerClDiceCellsAlpha01Debug50,
    soft_cldice_components,
    soft_skeletonize_2d,
)
from evaluate_cldice_validation import (
    _write_report,
    compare_directories,
    hard_cldice,
)


def _line() -> torch.Tensor:
    value = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    value[0, 0, 4, 1:8] = 1
    return value


class _ZeroDeepSupervisionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_output_count = 0

    def forward(self, outputs, targets):
        output_list = outputs if isinstance(outputs, (tuple, list)) else [outputs]
        target_list = targets if isinstance(targets, (tuple, list)) else [targets]
        self.received_output_count = len(output_list)
        return sum(output.sum() * 0.0 for output in output_list) + sum(
            target.sum() * 0.0 for target in target_list
        )


class ClDiceMathTests(unittest.TestCase):
    def test_perfect_prediction_has_score_one(self) -> None:
        line = _line()
        precision, sensitivity, score, valid = soft_cldice_components(line, line)
        self.assertTrue(valid.item())
        self.assertAlmostEqual(float(precision), 1.0, places=5)
        self.assertAlmostEqual(float(sensitivity), 1.0, places=5)
        self.assertAlmostEqual(float(score), 1.0, places=5)

    def test_gap_in_line_reduces_score(self) -> None:
        target = _line()
        prediction = target.clone()
        prediction[0, 0, 4, 4] = 0
        perfect = float(soft_cldice_components(target, target)[2])
        broken = float(soft_cldice_components(prediction, target)[2])
        self.assertLess(broken, perfect)

    def test_false_branch_reduces_topology_precision(self) -> None:
        target = _line()
        prediction = target.clone()
        prediction[0, 0, 1:5, 6] = 1
        precision = float(soft_cldice_components(prediction, target)[0])
        self.assertLess(precision, 1.0)

    def test_one_pixel_shift_is_penalized_less_by_tolerant_score(self) -> None:
        target = np.zeros((9, 9), dtype=bool)
        target[4, 1:8] = True
        prediction = np.zeros_like(target)
        prediction[3, 1:8] = True
        strict = hard_cldice(prediction, target, tolerance=0)[2]
        tolerant = hard_cldice(prediction, target, tolerance=1)[2]
        self.assertLess(strict, tolerant)
        self.assertAlmostEqual(tolerant, 1.0)

    def test_soma_channel_is_not_part_of_cldice_components(self) -> None:
        skeleton_probability = _line() * 0.8
        skeleton_target = _line()
        dummy_soma_a = torch.zeros_like(skeleton_probability)
        dummy_soma_b = torch.rand_like(skeleton_probability)
        score_a = soft_cldice_components(
            skeleton_probability + dummy_soma_a * 0.0, skeleton_target
        )[2]
        score_b = soft_cldice_components(
            skeleton_probability + dummy_soma_b * 0.0, skeleton_target
        )[2]
        torch.testing.assert_close(score_a, score_b)

    def test_empty_single_pixel_and_line_are_finite_and_differentiable(self) -> None:
        for initial in (torch.zeros((1, 1, 9, 9)), _line(), _line()):
            value = initial.clone()
            if initial is not None and torch.count_nonzero(initial) > 1:
                value = initial
            value.requires_grad_(True)
            skeleton = soft_skeletonize_2d(value, iterations=5)
            self.assertTrue(torch.isfinite(skeleton).all())
            skeleton.sum().backward()
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

        single_pixel = torch.zeros((1, 1, 9, 9), requires_grad=True)
        with torch.no_grad():
            single_pixel[0, 0, 4, 4] = 1
        result = soft_skeletonize_2d(single_pixel, iterations=5)
        result.sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(single_pixel.grad).all())

    def test_empty_ground_truth_is_ignored_with_zero_gradient(self) -> None:
        logits = torch.randn((2, 3, 9, 9), requires_grad=True)
        target = torch.zeros((2, 1, 9, 9), dtype=torch.long)
        loss = SkeletonClassSoftClDiceLoss()(logits, target)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


class ClDiceIntegrationTests(unittest.TestCase):
    def test_only_full_resolution_output_gets_cldice_gradient(self) -> None:
        base = _ZeroDeepSupervisionLoss()
        loss = FullResolutionClDiceLoss(base, alpha=0.1, iterations=5)
        full = torch.randn((1, 3, 9, 9), requires_grad=True)
        low = torch.randn((1, 3, 5, 5), requires_grad=True)
        full_target = torch.zeros((1, 1, 9, 9), dtype=torch.long)
        full_target[0, 0, 4, 1:8] = 1
        low_target = torch.zeros((1, 1, 5, 5), dtype=torch.long)
        result = loss([full, low], [full_target, low_target])
        result.backward()

        self.assertEqual(base.received_output_count, 2)
        self.assertGreater(float(full.grad.abs().sum()), 0.0)
        self.assertEqual(float(low.grad.abs().sum()), 0.0)

    def test_trainer_configuration_is_isolated_and_conservative(self) -> None:
        self.assertEqual(nnUNetTrainerClDiceCellsAlpha01.cldice_alpha, 0.1)
        self.assertEqual(nnUNetTrainerClDiceCellsAlpha01.cldice_iterations, 5)
        self.assertTrue(
            issubclass(
                nnUNetTrainerClDiceCellsAlpha01Debug50,
                nnUNetTrainerClDiceCellsAlpha01,
            )
        )

    def test_debug_variant_stops_after_50_epochs(self) -> None:
        def initialize_parent(instance, *args, **kwargs) -> None:
            instance.num_epochs = 1000

        with patch.object(
            nnUNetTrainerClDiceCellsAlpha01, "__init__", initialize_parent
        ):
            trainer = nnUNetTrainerClDiceCellsAlpha01Debug50(
                {}, "2d", 0, {}, None
            )
        self.assertEqual(trainer.num_epochs, 50)

    def test_debug_constructor_keeps_nnunet_parameter_names(self) -> None:
        parameters = list(
            inspect.signature(
                nnUNetTrainerClDiceCellsAlpha01Debug50.__init__
            ).parameters
        )
        self.assertEqual(
            parameters,
            ["self", "plans", "configuration", "fold", "dataset_json", "device"],
        )

    def test_debug_and_full_trainers_have_separate_result_names(self) -> None:
        self.assertNotEqual(
            nnUNetTrainerClDiceCellsAlpha01.__name__,
            nnUNetTrainerClDiceCellsAlpha01Debug50.__name__,
        )

    def test_evaluator_writes_strict_json_and_all_acceptance_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference_dir = root / "reference"
            baseline_dir = root / "baseline"
            candidate_dir = root / "candidate"
            output_dir = root / "output"
            for directory in (reference_dir, baseline_dir, candidate_dir):
                directory.mkdir()

            reference = np.zeros((9, 9), dtype=np.uint8)
            reference[4, 1:8] = 1
            reference[1:3, 1:3] = 2
            baseline = reference.copy()
            baseline[4, 4] = 0
            candidate = reference.copy()
            for directory, labels in (
                (reference_dir, reference),
                (baseline_dir, baseline),
                (candidate_dir, candidate),
            ):
                tifffile.imwrite(directory / "case001.tif", labels)

            rows, summary = compare_directories(
                reference_dir, baseline_dir, candidate_dir
            )
            _write_report(output_dir, rows, summary)
            parsed = json.loads((output_dir / "summary.json").read_text())

            self.assertEqual(parsed["case_count"], 1)
            self.assertIn("strict_cldice", parsed["candidate"])
            self.assertIn("tolerant_cldice", parsed["candidate"])
            self.assertIn("false_split_excess", parsed["candidate"])
            self.assertIn("attached_skeleton_fraction", parsed["candidate"])
            self.assertTrue((output_dir / "per_case.csv").is_file())
            self.assertTrue((output_dir / "comparison.md").is_file())


if __name__ == "__main__":
    unittest.main()
