from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import nn
from torch.nn import functional as F


def _soft_erode_2d(image: torch.Tensor) -> torch.Tensor:
    """Differentiable 3x3 min-pooling used by Algorithm 1 of clDice."""

    if image.ndim != 4:
        raise ValueError(f"Expected a 4-D BCHW tensor, got shape {tuple(image.shape)}")
    vertical = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate_2d(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected a 4-D BCHW tensor, got shape {tuple(image.shape)}")
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def _soft_open_2d(image: torch.Tensor) -> torch.Tensor:
    return _soft_dilate_2d(_soft_erode_2d(image))


def soft_skeletonize_2d(image: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Soft-skeletonize a BCHW probability mask without new dependencies."""

    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    eroded = image
    skeleton = F.relu(eroded - _soft_open_2d(eroded))
    for _ in range(iterations):
        eroded = _soft_erode_2d(eroded)
        opened = _soft_open_2d(eroded)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + (1.0 - skeleton) * delta
    return skeleton


def soft_cldice_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int = 5,
    smooth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample topology precision, sensitivity, clDice and validity."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("prediction and target must be BCHW tensors with one channel")

    prediction = prediction.float()
    target = target.float()
    skeleton_prediction = soft_skeletonize_2d(prediction, iterations)
    with torch.no_grad():
        skeleton_target = soft_skeletonize_2d(target, iterations)

    spatial_axes = tuple(range(2, prediction.ndim))
    topology_precision = (
        (skeleton_prediction * target).sum(spatial_axes) + smooth
    ) / (skeleton_prediction.sum(spatial_axes) + smooth)
    topology_sensitivity = (
        (skeleton_target * prediction).sum(spatial_axes) + smooth
    ) / (skeleton_target.sum(spatial_axes) + smooth)
    cldice = (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity + smooth)
    )
    valid = target.sum(spatial_axes) > 0
    return topology_precision, topology_sensitivity, cldice, valid


class SkeletonClassSoftClDiceLoss(nn.Module):
    """Soft-clDice loss for semantic class 1 only.

    The multiclass softmax probability of class 1 is used. Samples with no
    class-1 ground truth contribute a differentiable zero to this term; the
    unchanged Dice/CE loss still supervises them.
    """

    def __init__(self, iterations: int = 5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def score_from_probabilities(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return soft_cldice_components(
            prediction,
            target,
            iterations=self.iterations,
            smooth=self.smooth,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("The first clDice experiment supports 2-D nnU-Net outputs only")
        if logits.shape[1] <= 1:
            raise ValueError("Class-1 skeleton logits are missing")
        if target.ndim == logits.ndim - 1:
            target = target[:, None]
        if target.ndim != logits.ndim or target.shape[1] != 1:
            raise ValueError("Expected a single-channel semantic target")

        skeleton_probability = torch.softmax(logits.float(), dim=1)[:, 1:2]
        skeleton_target = (target == 1).to(dtype=skeleton_probability.dtype)
        _, _, scores, valid = self.score_from_probabilities(
            skeleton_probability, skeleton_target
        )
        if torch.any(valid):
            return 1.0 - scores[valid].mean()
        return skeleton_probability.sum() * 0.0


class FullResolutionClDiceLoss(nn.Module):
    """Add class-1 clDice to nnU-Net's unmodified Dice/CE loss stack."""

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        alpha: float = 0.1,
        iterations: int = 5,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.base_loss = base_loss
        self.alpha = float(alpha)
        self.cldice_loss = SkeletonClassSoftClDiceLoss(iterations=iterations)
        self.last_base_loss = float("nan")
        self.last_cldice_loss = float("nan")

    @staticmethod
    def _full_resolution(value: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        return value[0] if isinstance(value, (tuple, list)) else value

    def forward(self, output, target) -> torch.Tensor:
        base = self.base_loss(output, target)
        full_output = self._full_resolution(output)
        full_target = self._full_resolution(target)
        topology = self.cldice_loss(full_output, full_target)
        self.last_base_loss = float(base.detach().cpu())
        self.last_cldice_loss = float(topology.detach().cpu())
        return base + self.alpha * topology


class nnUNetTrainerClDiceCellsAlpha01(nnUNetTrainer):
    """Dataset138 trainer: standard Dice/CE plus full-resolution class-1 clDice."""

    cldice_alpha = 0.1
    cldice_iterations = 5

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.label_manager.has_regions:
            raise NotImplementedError("Region-based training is not supported here")
        local_log = self.logger.local_logger.my_fantastic_logging
        local_log.setdefault("train_cldice_losses", [])
        local_log.setdefault("val_cldice_losses", [])

    def _build_loss(self):
        return FullResolutionClDiceLoss(
            super()._build_loss(),
            alpha=self.cldice_alpha,
            iterations=self.cldice_iterations,
        )

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        result["cldice_loss"] = np.asarray(self.loss.last_cldice_loss)
        return result

    def validation_step(self, batch: dict) -> dict:
        result = super().validation_step(batch)
        result["cldice_loss"] = np.asarray(self.loss.last_cldice_loss)
        return result

    def _mean_across_workers(self, values: np.ndarray) -> float:
        local_mean = float(np.nanmean(values))
        if not self.is_ddp:
            return local_mean
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_mean)
        return float(np.nanmean(gathered))

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        values = collate_outputs(train_outputs)["cldice_loss"]
        mean_value = self._mean_across_workers(values)
        self.logger.log("train_cldice_losses", mean_value, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        values = collate_outputs(val_outputs)["cldice_loss"]
        mean_value = self._mean_across_workers(values)
        self.logger.log("val_cldice_losses", mean_value, self.current_epoch)

    def on_epoch_end(self):
        self.print_to_log_file(
            "clDice loss train/val",
            np.round(self.logger.get_value("train_cldice_losses", step=-1), 4),
            np.round(self.logger.get_value("val_cldice_losses", step=-1), 4),
        )
        super().on_epoch_end()


class nnUNetTrainerClDiceCellsAlpha01Debug50(nnUNetTrainerClDiceCellsAlpha01):
    """Separate 50-epoch technical test for the clDice trainer."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
