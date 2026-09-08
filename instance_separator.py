from __future__ import annotations

"""Soma-seeded instance separation for neuronal microscopy images.

Network 141 remains the semantic segmenter. The model in this module receives
its raw image, skeleton/soma masks and one highlighted soma. It predicts which
foreground pixels belong to that soma. Running it once per soma creates local
instance IDs; pixels may belong to two cells at a true overlap.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


MODEL_VERSION = "soma-seeded-membership-v1-2026-09-08"
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def read_2d(path: Path) -> np.ndarray:
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional TIFF, got {array.shape}: {path}")
    return array


def robust_zscore(raw: np.ndarray) -> np.ndarray:
    values = raw.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    mean = float(finite.mean())
    standard_deviation = max(float(finite.std()), 1e-6)
    return np.clip((values - mean) / standard_deviation, -6.0, 6.0).astype(
        np.float32
    )


@dataclass(frozen=True)
class MembershipTarget:
    membership: np.ndarray
    valid: np.ndarray
    query_soma: np.ndarray
    semantic: np.ndarray


def build_membership_target(
    semantic: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    query_instance_id: int,
    ambiguous: np.ndarray | None = None,
) -> MembershipTarget:
    """Create a binary target for one soma without encoding the numeric ID."""

    semantic = np.asarray(semantic, dtype=np.uint8)
    primary = np.asarray(primary, dtype=np.uint16)
    secondary = np.asarray(secondary, dtype=np.uint16)
    if semantic.shape != primary.shape or semantic.shape != secondary.shape:
        raise ValueError("Semantic and instance maps must have identical shapes")
    if ambiguous is None:
        ambiguous = np.zeros(semantic.shape, dtype=np.uint8)
    ambiguous = np.asarray(ambiguous, dtype=np.uint8)
    if ambiguous.shape != semantic.shape:
        raise ValueError("Ambiguous mask must have the same shape as semantic labels")
    if query_instance_id <= 0:
        raise ValueError("Query instance ID must be positive")
    if np.any((secondary > 0) & (primary == 0)):
        raise ValueError("A secondary instance membership requires a primary membership")
    foreground = semantic > 0
    assigned = (primary > 0) | (secondary > 0)
    valid = foreground & assigned & (ambiguous == 0)
    membership = ((primary == query_instance_id) | (secondary == query_instance_id)) & valid
    query_soma = membership & (semantic == 2)
    if not query_soma.any():
        raise ValueError(f"Instance {query_instance_id} has no soma in this sample")
    return MembershipTarget(
        membership=membership,
        valid=valid,
        query_soma=query_soma,
        semantic=semantic,
    )


def exact_spatial_augmentation(
    arrays: Sequence[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    rotation = int(rng.integers(0, 4))
    flip_y = bool(rng.integers(0, 2))
    flip_x = bool(rng.integers(0, 2))
    transformed: list[np.ndarray] = []
    for array in arrays:
        result = np.rot90(array, rotation)
        if flip_y:
            result = np.flipud(result)
        if flip_x:
            result = np.fliplr(result)
        transformed.append(np.ascontiguousarray(result))
    return transformed


def soma_guided_crop(
    arrays: Sequence[np.ndarray],
    query_instance_id: int,
    patch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Crop around the queried soma and, often, one distant/overlap pixel."""

    if len(arrays) < 4:
        raise ValueError("Expected raw, semantic, primary and secondary arrays")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("All arrays must have the same shape")
    semantic, primary, secondary = arrays[1:4]
    query_membership = (primary == query_instance_id) | (secondary == query_instance_id)
    soma_pixels = np.argwhere(query_membership & (semantic == 2))
    if soma_pixels.size == 0:
        raise ValueError(f"Instance {query_instance_id} has no soma")
    soma_center = soma_pixels.mean(axis=0)
    overlap_pixels = np.argwhere((secondary > 0) & query_membership)
    member_pixels = np.argwhere(query_membership)
    candidates = overlap_pixels if overlap_pixels.size else member_pixels
    focus = candidates[int(rng.integers(0, len(candidates)))]
    center = soma_center if rng.random() < 0.35 else (soma_center + focus) / 2.0
    jitter = max(1, patch_size // 12)
    center += rng.integers(-jitter, jitter + 1, size=2)

    height, width = shape
    y0 = int(np.clip(round(center[0] - patch_size / 2), 0, max(0, height - patch_size)))
    x0 = int(np.clip(round(center[1] - patch_size / 2), 0, max(0, width - patch_size)))
    y0 = min(y0, int(soma_center[0]))
    x0 = min(x0, int(soma_center[1]))
    y0 = max(y0, int(soma_center[0]) - patch_size + 1)
    x0 = max(x0, int(soma_center[1]) - patch_size + 1)
    y0 = int(np.clip(y0, 0, max(0, height - patch_size)))
    x0 = int(np.clip(x0, 0, max(0, width - patch_size)))
    y1 = min(height, y0 + patch_size)
    x1 = min(width, x0 + patch_size)
    cropped = [array[y0:y1, x0:x1] for array in arrays]
    pad_y = patch_size - cropped[0].shape[0]
    pad_x = patch_size - cropped[0].shape[1]
    if pad_y or pad_x:
        cropped = [
            np.pad(array, ((0, pad_y), (0, pad_x)), mode="constant")
            for array in cropped
        ]
    return [np.ascontiguousarray(array) for array in cropped]


class SyntheticInstanceDataset(Dataset):
    """One training item is one (multi-cell scene, queried soma) pair."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        *,
        patch_size: int = 320,
        augment: bool = False,
        seed: int = 144,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        manifest = json.loads(
            (self.dataset_root / "synthetic_manifest.json").read_text(encoding="utf-8")
        )
        records = [row for row in manifest if row["split"] == split]
        self.samples = [
            (row, instance_id)
            for row in records
            for instance_id in range(1, int(row["cell_count"]) + 1)
        ]
        self.scene_count = len(records)
        if not self.samples:
            raise ValueError(f"No synthetic records for split {split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row, query_instance_id = self.samples[index]
        case_id = str(row["case_id"])
        raw = read_2d(self.dataset_root / "imagesTr" / f"{case_id}_0000.tif")
        semantic = read_2d(self.dataset_root / "labelsTr" / f"{case_id}.tif")
        primary = read_2d(self.dataset_root / "instance_primaryTr" / f"{case_id}.tif")
        secondary = read_2d(self.dataset_root / "instance_secondaryTr" / f"{case_id}.tif")
        ambiguous = read_2d(self.dataset_root / "ambiguousTr" / f"{case_id}.tif")
        dynamic_seed = int(np.random.randint(0, np.iinfo(np.int32).max)) if self.augment else 0
        rng = np.random.default_rng(self.seed + 104729 * index + dynamic_seed)
        arrays: list[np.ndarray] = [raw, semantic, primary, secondary, ambiguous]
        if self.patch_size > 0:
            arrays = soma_guided_crop(arrays, query_instance_id, self.patch_size, rng)
        if self.augment:
            arrays = exact_spatial_augmentation(arrays, rng)
        raw, semantic, primary, secondary, ambiguous = arrays
        target = build_membership_target(
            semantic, primary, secondary, query_instance_id, ambiguous
        )
        normalized_raw = robust_zscore(raw)
        if self.augment:
            normalized_raw = np.clip(
                rng.uniform(0.85, 1.15) * normalized_raw
                + rng.uniform(-0.15, 0.15)
                + rng.normal(0.0, 0.04, normalized_raw.shape),
                -6.0,
                6.0,
            ).astype(np.float32)
        model_input = np.stack(
            (
                normalized_raw,
                (semantic == 1).astype(np.float32),
                (semantic == 2).astype(np.float32),
                target.query_soma.astype(np.float32),
            ),
            axis=0,
        )
        return {
            "input": torch.from_numpy(np.ascontiguousarray(model_input)),
            "membership": torch.from_numpy(target.membership[None]),
            "valid": torch.from_numpy(target.valid[None]),
            "semantic": torch.from_numpy(target.semantic[None].astype(np.int64)),
            "case_id": case_id,
            "query_instance_id": torch.tensor(query_instance_id, dtype=torch.int64),
        }


def _group_count(channels: int) -> int:
    for count in (8, 4, 2, 1):
        if channels % count == 0:
            return count
    return 1


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = _group_count(output_channels)
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SomaSeededSeparationUNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        features = [
            self.base_channels,
            self.base_channels * 2,
            self.base_channels * 4,
            self.base_channels * 8,
            self.base_channels * 12,
            self.base_channels * 16,
        ]
        self.encoders = nn.ModuleList()
        input_channels = 4
        for output_channels in features[:-1]:
            self.encoders.append(DoubleConv(input_channels, output_channels))
            input_channels = output_channels
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-2], features[-1])
        self.decoders = nn.ModuleList()
        current_channels = features[-1]
        for skip_channels in reversed(features[:-1]):
            self.decoders.append(DoubleConv(current_channels + skip_channels, skip_channels))
            current_channels = skip_channels
        self.output_head = nn.Conv2d(features[0], 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError(f"Expected Bx4xHxW input, got {tuple(inputs.shape)}")
        output = inputs
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            output = encoder(output)
            skips.append(output)
            output = self.pool(output)
        output = self.bottleneck(output)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            output = F.interpolate(
                output, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            output = decoder(torch.cat((output, skip), dim=1))
        return self.output_head(output)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class SomaSeededSeparationLoss(nn.Module):
    def __init__(self, skeleton_weight: float = 4.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.skeleton_weight = float(skeleton_weight)
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        output: torch.Tensor,
        membership: torch.Tensor,
        valid: torch.Tensor,
        semantic: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = output[:, 0]
        target = membership[:, 0].to(dtype=output.dtype)
        valid_mask = valid[:, 0].bool()
        positive = valid_mask & (target > 0.5)
        negative = valid_mask & ~positive
        class_weights = torch.where(
            semantic[:, 0] == 1,
            torch.as_tensor(self.skeleton_weight, device=output.device, dtype=output.dtype),
            torch.ones((), device=output.device, dtype=output.dtype),
        )
        positive_loss = _masked_mean(F.softplus(-logits) * class_weights, positive)
        negative_loss = _masked_mean(F.softplus(logits) * class_weights, negative)
        probability = torch.sigmoid(logits) * valid_mask
        weighted_target = target * class_weights * valid_mask
        weighted_probability = probability * class_weights
        intersection = (weighted_probability * target).sum(dim=(1, 2))
        denominator = weighted_probability.sum(dim=(1, 2)) + weighted_target.sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        total = positive_loss + negative_loss + self.dice_weight * dice_loss
        return {
            "loss": total,
            "positive_bce": positive_loss.detach(),
            "negative_bce": negative_loss.detach(),
            "dice_loss": dice_loss.detach(),
        }


@torch.no_grad()
def membership_metrics(
    output: torch.Tensor,
    membership: torch.Tensor,
    valid: torch.Tensor,
    semantic: torch.Tensor,
) -> dict[str, float]:
    prediction = torch.sigmoid(output[:, 0]) >= 0.5
    target = membership[:, 0].bool()
    valid_mask = valid[:, 0].bool()
    prediction &= valid_mask
    target &= valid_mask
    intersection = int((prediction & target).sum().item())
    prediction_count = int(prediction.sum().item())
    target_count = int(target.sum().item())
    skeleton = (semantic[:, 0] == 1) & valid_mask
    skeleton_target = target & skeleton
    skeleton_prediction = prediction & skeleton
    skeleton_intersection = int((skeleton_prediction & skeleton_target).sum().item())
    skeleton_false_positive = int((skeleton_prediction & ~target).sum().item())
    return {
        "membership_dice": (2 * intersection) / max(1, prediction_count + target_count),
        "membership_precision": intersection / max(1, prediction_count),
        "membership_recall": intersection / max(1, target_count),
        "skeleton_recall": skeleton_intersection / max(1, int(skeleton_target.sum().item())),
        "other_skeleton_false_positive_rate": skeleton_false_positive
        / max(1, int((skeleton & ~target).sum().item())),
    }


def soma_centers_from_instances(soma_instances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    instance_ids = np.asarray(
        [int(value) for value in np.unique(soma_instances) if value > 0], dtype=np.int64
    )
    centers = np.asarray(
        [np.argwhere(soma_instances == instance_id).mean(axis=0) for instance_id in instance_ids],
        dtype=np.float32,
    )
    return instance_ids, centers


def make_model_input(
    raw: np.ndarray, semantic: np.ndarray, query_soma: np.ndarray
) -> torch.Tensor:
    model_input = np.stack(
        (
            robust_zscore(raw),
            (semantic == 1).astype(np.float32),
            (semantic == 2).astype(np.float32),
            np.asarray(query_soma, dtype=np.float32),
        ),
        axis=0,
    )
    return torch.from_numpy(np.ascontiguousarray(model_input))[None]


@torch.no_grad()
def predict_memberships(
    model: SomaSeededSeparationUNet,
    raw: np.ndarray,
    semantic: np.ndarray,
    soma_instances: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    instance_ids, _centers = soma_centers_from_instances(soma_instances)
    probabilities: list[np.ndarray] = []
    for instance_id in instance_ids:
        query_soma = soma_instances == instance_id
        model_input = make_model_input(raw, semantic, query_soma).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(model_input)
        probability = torch.sigmoid(output[0, 0]).float().cpu().numpy()
        probabilities.append(probability)
    if not probabilities:
        return instance_ids, np.zeros((0, *semantic.shape), dtype=np.float32)
    return instance_ids, np.stack(probabilities, axis=0)


def assign_memberships_to_instances(
    probabilities: np.ndarray,
    instance_ids: np.ndarray,
    semantic: np.ndarray,
    soma_instances: np.ndarray,
    *,
    membership_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign best membership and retain a second ID at predicted overlaps."""

    if probabilities.shape != (len(instance_ids), *semantic.shape):
        raise ValueError("Probability stack shape does not match IDs and semantic image")
    primary = np.zeros(semantic.shape, dtype=np.uint16)
    secondary = np.zeros_like(primary)
    confidence = np.zeros(semantic.shape, dtype=np.float32)
    foreground = semantic > 0
    if not foreground.any() or not len(instance_ids):
        return primary, secondary, confidence
    values = probabilities[:, foreground].T
    order = np.argsort(values, axis=1)[:, ::-1]
    primary_choices = order[:, 0]
    primary[foreground] = instance_ids[primary_choices].astype(np.uint16)
    best = np.take_along_axis(values, primary_choices[:, None], axis=1)[:, 0]
    confidence[foreground] = best
    if len(instance_ids) > 1:
        secondary_choices = order[:, 1]
        second = np.take_along_axis(values, secondary_choices[:, None], axis=1)[:, 0]
        overlap_foreground = second >= membership_threshold
        foreground_pixels = np.argwhere(foreground)
        overlap_pixels = foreground_pixels[overlap_foreground]
        secondary[overlap_pixels[:, 0], overlap_pixels[:, 1]] = instance_ids[
            secondary_choices[overlap_foreground]
        ].astype(np.uint16)
    soma_pixels = np.argwhere(soma_instances > 0)
    primary[soma_pixels[:, 0], soma_pixels[:, 1]] = soma_instances[
        soma_pixels[:, 0], soma_pixels[:, 1]
    ].astype(np.uint16)
    secondary[soma_pixels[:, 0], soma_pixels[:, 1]] = 0
    confidence[soma_pixels[:, 0], soma_pixels[:, 1]] = 1.0
    return primary, secondary, confidence
