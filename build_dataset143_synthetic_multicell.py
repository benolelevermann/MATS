from __future__ import annotations

"""Build Dataset143 from Dataset141 plus synthetic multi-cell microscopy scenes.

Every qualified single-cell case is used exactly once in a synthetic scene.
Only exact 90 degree rotations and flips are used, so one-pixel skeleton labels
remain one pixel wide. In addition to nnU-Net compatible semantic labels, each
synthetic scene stores primary and secondary instance IDs at real overlaps.
"""

import argparse
import csv
import html
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

SCRIPT_VERSION = "dataset143-synthetic-multicell-v2-2026-09-08"
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)
SEMANTIC_COLORS = np.asarray(
    [
        [0, 0, 0],
        [0, 224, 235],
        [244, 68, 168],
    ],
    dtype=np.uint8,
)
INSTANCE_COLORS = np.asarray(
    [
        [74, 222, 128],
        [255, 157, 84],
        [112, 168, 255],
        [229, 126, 255],
        [255, 211, 92],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class DonorInfo:
    case_id: str
    image_path: str
    label_path: str
    split: str
    height: int
    width: int
    bbox_y0: int
    bbox_y1: int
    bbox_x0: int
    bbox_x1: int
    skeleton_pixels: int
    soma_pixels: int
    soma_components: int
    background_median: float
    background_std: float
    raw_low: float
    raw_high: float
    eligible: bool
    exclusion_reason: str

    @property
    def foreground_height(self) -> int:
        return self.bbox_y1 - self.bbox_y0

    @property
    def foreground_width(self) -> int:
        return self.bbox_x1 - self.bbox_x0


@dataclass(frozen=True)
class PreparedDonor:
    case_id: str
    split: str
    signal: np.ndarray
    semantic: np.ndarray
    rotation_degrees: int
    flip_x: bool
    flip_y: bool
    source_bbox_yxyx: tuple[int, int, int, int]


def read_2d(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional TIFF, got {array.shape}: {path}")
    return array


def case_id_from_image(path: Path) -> str:
    if not path.stem.endswith("_0000"):
        raise ValueError(f"nnU-Net input file needs the _0000 suffix: {path.name}")
    return path.stem[:-5]


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def analyze_case(
    image_path: Path,
    label_path: Path,
    split: str,
    canvas_size: int,
    edge_margin: int,
    min_soma_pixels: int,
    min_skeleton_pixels: int,
) -> DonorInfo:
    raw = read_2d(image_path)
    semantic = read_2d(label_path)
    case_id = case_id_from_image(image_path)
    if raw.shape != semantic.shape:
        raise ValueError(
            f"Raw/label shape mismatch for {case_id}: {raw.shape} != {semantic.shape}"
        )

    unique_values = set(int(v) for v in np.unique(semantic))
    invalid_values = sorted(unique_values.difference({0, 1, 2}))
    foreground = semantic > 0
    bbox = foreground_bbox(foreground)
    soma_mask = semantic == 2
    soma_components = int(ndimage.label(soma_mask, structure=CONNECTIVITY_8)[1])
    skeleton_pixels = int(np.count_nonzero(semantic == 1))
    soma_pixels = int(np.count_nonzero(soma_mask))

    background_values = raw[semantic == 0].astype(np.float64, copy=False)
    if background_values.size == 0:
        background_values = raw.reshape(-1).astype(np.float64, copy=False)
    background_median = float(np.median(background_values))
    background_std = float(np.std(background_values))
    raw_low, raw_high = np.percentile(raw.astype(np.float64, copy=False), [0.5, 99.5])

    reasons: list[str] = []
    if invalid_values:
        reasons.append("invalid_label_values=" + ",".join(map(str, invalid_values)))
    if bbox is None:
        reasons.append("empty_foreground")
        bbox = (0, 0, 0, 0)
    if soma_components != 1:
        reasons.append(f"soma_components={soma_components}")
    if soma_pixels < min_soma_pixels:
        reasons.append(f"soma_pixels<{min_soma_pixels}")
    if soma_mask.size and (
        np.any(soma_mask[0])
        or np.any(soma_mask[-1])
        or np.any(soma_mask[:, 0])
        or np.any(soma_mask[:, -1])
    ):
        reasons.append("soma_touches_image_border")
    if skeleton_pixels < min_skeleton_pixels:
        reasons.append(f"skeleton_pixels<{min_skeleton_pixels}")
    max_foreground_side = canvas_size - 2 * edge_margin
    if bbox[1] - bbox[0] > max_foreground_side:
        reasons.append(f"foreground_height>{max_foreground_side}")
    if bbox[3] - bbox[2] > max_foreground_side:
        reasons.append(f"foreground_width>{max_foreground_side}")

    return DonorInfo(
        case_id=case_id,
        image_path=str(image_path.resolve()),
        label_path=str(label_path.resolve()),
        split=split,
        height=int(raw.shape[0]),
        width=int(raw.shape[1]),
        bbox_y0=bbox[0],
        bbox_y1=bbox[1],
        bbox_x0=bbox[2],
        bbox_x1=bbox[3],
        skeleton_pixels=skeleton_pixels,
        soma_pixels=soma_pixels,
        soma_components=soma_components,
        background_median=background_median,
        background_std=background_std,
        raw_low=float(raw_low),
        raw_high=float(raw_high),
        eligible=not reasons,
        exclusion_reason=";".join(reasons),
    )


def context_bounds(
    info: DonorInfo, context_radius: int, maximum_side: int
) -> tuple[int, int, int, int]:
    def one_axis(start: int, stop: int, limit: int) -> tuple[int, int]:
        foreground_length = stop - start
        target_length = min(
            limit, max(foreground_length, foreground_length + 2 * context_radius)
        )
        extra = target_length - foreground_length
        before = min(start, extra // 2)
        after = min(limit - stop, extra - before)
        missing = extra - before - after
        if missing:
            take_before = min(start - before, missing)
            before += take_before
            missing -= take_before
            after += min(limit - stop - after, missing)
        return start - before, stop + after

    y0, y1 = one_axis(info.bbox_y0, info.bbox_y1, info.height)
    x0, x1 = one_axis(info.bbox_x0, info.bbox_x1, info.width)

    if y1 - y0 > maximum_side:
        center = (info.bbox_y0 + info.bbox_y1) // 2
        y0 = max(0, min(info.height - maximum_side, center - maximum_side // 2))
        y1 = y0 + maximum_side
    if x1 - x0 > maximum_side:
        center = (info.bbox_x0 + info.bbox_x1) // 2
        x0 = max(0, min(info.width - maximum_side, center - maximum_side // 2))
        x1 = x0 + maximum_side
    return y0, y1, x0, x1


def prepare_donor(
    info: DonorInfo,
    rng: np.random.Generator,
    context_radius: int,
    signal_radius: int,
    maximum_side: int,
) -> PreparedDonor:
    raw = read_2d(Path(info.image_path)).astype(np.float32)
    semantic = read_2d(Path(info.label_path)).astype(np.uint8)
    y0, y1, x0, x1 = context_bounds(info, context_radius, maximum_side)
    raw = raw[y0:y1, x0:x1]
    semantic = semantic[y0:y1, x0:x1]

    rotation = int(rng.integers(0, 4))
    flip_x = bool(rng.integers(0, 2))
    flip_y = bool(rng.integers(0, 2))
    raw = np.rot90(raw, rotation)
    semantic = np.rot90(semantic, rotation)
    if flip_x:
        raw = np.fliplr(raw)
        semantic = np.fliplr(semantic)
    if flip_y:
        raw = np.flipud(raw)
        semantic = np.flipud(semantic)

    foreground = semantic > 0
    distance = ndimage.distance_transform_edt(~foreground)
    # Raw single-cell crops are noisy. Copying every positive noise deviation
    # throughout the wider context produces visible rectangular donor patches.
    # A small, feathered support around the annotated cell retains the real
    # neurite halo but lets the destination background remain homogeneous.
    alpha = np.clip(1.0 - distance / max(1.0, float(signal_radius)), 0.0, 1.0)
    alpha[foreground] = 1.0
    denoised = ndimage.gaussian_filter(raw, sigma=0.7)
    positive_signal = np.maximum(
        denoised - np.float32(info.background_median),
        0.0,
    )
    signal = positive_signal * alpha.astype(np.float32)
    return PreparedDonor(
        case_id=info.case_id,
        split=info.split,
        signal=np.ascontiguousarray(signal),
        semantic=np.ascontiguousarray(semantic),
        rotation_degrees=rotation * 90,
        flip_x=flip_x,
        flip_y=flip_y,
        source_bbox_yxyx=(y0, y1, x0, x1),
    )


def group_donors(
    donors: Sequence[DonorInfo],
    rng: np.random.Generator,
    min_cells: int,
    max_cells: int,
    intensity_block_size: int = 64,
) -> list[list[DonorInfo]]:
    if min_cells < 2 or max_cells < min_cells:
        raise ValueError("Need 2 <= min_cells <= max_cells")
    if len(donors) < min_cells:
        return []

    ordered = sorted(donors, key=lambda donor: donor.background_median)
    intensity_matched: list[DonorInfo] = []
    for start in range(0, len(ordered), intensity_block_size):
        block = ordered[start : start + intensity_block_size]
        indices = rng.permutation(len(block))
        intensity_matched.extend(block[int(i)] for i in indices)

    sizes: list[int] = []
    remaining = len(intensity_matched)
    while remaining:
        if min_cells <= remaining <= max_cells:
            sizes.append(remaining)
            break
        choices = [
            size
            for size in range(min_cells, max_cells + 1)
            if remaining - size == 0 or remaining - size >= min_cells
        ]
        if not choices:
            raise RuntimeError(
                f"Cannot partition {len(donors)} donors into valid groups"
            )
        weights = np.asarray(
            [2.0 if size == min(3, max_cells) else 1.0 for size in choices],
            dtype=np.float64,
        )
        weights /= weights.sum()
        size = int(rng.choice(np.asarray(choices), p=weights))
        sizes.append(size)
        remaining -= size

    groups: list[list[DonorInfo]] = []
    cursor = 0
    for size in sizes:
        groups.append(intensity_matched[cursor : cursor + size])
        cursor += size
    assert cursor == len(intensity_matched)
    return groups


def synthetic_background(
    donors: Sequence[DonorInfo], canvas_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.dtype]:
    sample = read_2d(Path(donors[0].image_path))
    dtype = sample.dtype
    median = float(np.median([donor.background_median for donor in donors]))
    std = float(np.median([max(1.0, donor.background_std) for donor in donors]))
    low = float(np.median([donor.raw_low for donor in donors]))
    high = float(np.median([donor.raw_high for donor in donors]))

    white = rng.normal(0.0, std, size=(canvas_size, canvas_size)).astype(np.float32)
    low_frequency = ndimage.gaussian_filter(
        rng.normal(0.0, 1.0, size=(canvas_size, canvas_size)).astype(np.float32),
        sigma=max(8.0, canvas_size / 18.0),
    )
    low_frequency /= max(float(low_frequency.std()), 1e-6)
    canvas = median + white + low_frequency * (0.12 * std)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        clip_low = max(float(limits.min), min(low, median - 5.0 * std))
        clip_high = min(float(limits.max), max(high, median + 5.0 * std))
    else:
        clip_low = min(low, median - 5.0 * std)
        clip_high = max(high, median + 5.0 * std)
    return np.clip(canvas, clip_low, clip_high).astype(np.float32), dtype


def interaction_classes(mode: str) -> tuple[int, int]:
    if mode == "skeleton_skeleton":
        return 1, 1
    if mode == "skeleton_soma":
        return 1, 2
    if mode == "soma_skeleton":
        return 2, 1
    if mode == "soma_soma":
        return 2, 2
    return 0, 0


def candidate_position(
    prepared: PreparedDonor,
    existing_semantic: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    mode: str,
    edge_margin: int,
    rng: np.random.Generator,
    attempts: int = 180,
) -> tuple[int, int, int]:
    canvas_height, canvas_width = existing_semantic.shape
    height, width = prepared.semantic.shape
    min_y = edge_margin
    min_x = edge_margin
    max_y = canvas_height - edge_margin - height
    max_x = canvas_width - edge_margin - width
    if max_y < min_y or max_x < min_x:
        raise ValueError(
            f"Prepared donor {prepared.case_id} does not fit the canvas: "
            f"{prepared.semantic.shape} versus {existing_semantic.shape}"
        )

    donor_foreground = prepared.semantic > 0
    donor_area = int(np.count_nonzero(donor_foreground))
    existing_foreground = existing_semantic > 0

    if not np.any(existing_foreground):
        return (
            int(rng.integers(min_y, max_y + 1)),
            int(rng.integers(min_x, max_x + 1)),
            0,
        )

    donor_class, target_class = interaction_classes(mode)
    donor_points = np.argwhere(
        prepared.semantic == donor_class if donor_class else donor_foreground
    )
    target_points = np.argwhere(
        existing_semantic == target_class if target_class else existing_foreground
    )
    if donor_points.size == 0:
        donor_points = np.argwhere(donor_foreground)
    if target_points.size == 0:
        target_points = np.argwhere(existing_foreground)

    best: tuple[float, int, int, int] | None = None
    for attempt in range(attempts):
        if mode == "separate":
            y = int(rng.integers(min_y, max_y + 1))
            x = int(rng.integers(min_x, max_x + 1))
        else:
            donor_y, donor_x = donor_points[int(rng.integers(0, len(donor_points)))]
            target_y, target_x = target_points[int(rng.integers(0, len(target_points)))]
            jitter = 0 if attempt < attempts // 2 else int(rng.integers(-3, 4))
            y = int(np.clip(target_y - donor_y + jitter, min_y, max_y))
            x = int(np.clip(target_x - donor_x - jitter, min_x, max_x))

        region_existing = existing_foreground[y : y + height, x : x + width]
        overlap = donor_foreground & region_existing
        overlap_pixels = int(np.count_nonzero(overlap))
        triple = overlap & (secondary[y : y + height, x : x + width] > 0)
        if np.any(triple):
            continue

        if mode == "separate":
            if overlap_pixels == 0:
                return y, x, 0
            score = float(overlap_pixels)
        else:
            existing_area = int(np.count_nonzero(existing_foreground))
            maximum_overlap = max(8, int(0.18 * min(donor_area, existing_area)))
            if 1 <= overlap_pixels <= maximum_overlap:
                return y, x, overlap_pixels
            score = abs(overlap_pixels - min(8, maximum_overlap))
        if best is None or score < best[0]:
            best = (score, y, x, overlap_pixels)

    if best is not None:
        return best[1], best[2], best[3]
    raise RuntimeError(f"No non-tertiary placement found for {prepared.case_id}")


def merge_semantic(existing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    merged = existing.copy()
    merged[(incoming == 1) & (merged == 0)] = 1
    merged[incoming == 2] = 2
    return merged


def add_instance(
    primary: np.ndarray,
    secondary: np.ndarray,
    foreground: np.ndarray,
    cell_id: int,
) -> None:
    occupied = foreground & (primary > 0)
    if np.any(occupied & (secondary > 0)):
        raise ValueError("A synthetic pixel would belong to more than two cells")
    primary[foreground & ~occupied] = cell_id
    secondary[occupied] = cell_id


def compose_scene(
    donors: Sequence[DonorInfo],
    canvas_size: int,
    edge_margin: int,
    context_radius: int,
    signal_radius: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    canvas, raw_dtype = synthetic_background(donors, canvas_size, rng)
    semantic = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    primary = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
    secondary = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
    placements: list[dict] = []
    interaction_options = np.asarray(
        [
            "skeleton_skeleton",
            "skeleton_skeleton",
            "skeleton_skeleton",
            "skeleton_soma",
            "soma_skeleton",
            "soma_soma",
            "separate",
        ]
    )

    for index, donor in enumerate(donors, start=1):
        prepared = prepare_donor(
            donor,
            rng,
            context_radius=context_radius,
            signal_radius=signal_radius,
            maximum_side=canvas_size - 2 * edge_margin,
        )
        mode = "first" if index == 1 else str(rng.choice(interaction_options))
        if index == 1:
            y, x, overlap_pixels = candidate_position(
                prepared,
                semantic,
                primary,
                secondary,
                mode="separate",
                edge_margin=edge_margin,
                rng=rng,
            )
        else:
            try:
                y, x, overlap_pixels = candidate_position(
                    prepared,
                    semantic,
                    primary,
                    secondary,
                    mode=mode,
                    edge_margin=edge_margin,
                    rng=rng,
                )
            except RuntimeError:
                mode = "fallback_separate"
                y, x, overlap_pixels = candidate_position(
                    prepared,
                    semantic,
                    primary,
                    secondary,
                    mode="separate",
                    edge_margin=edge_margin,
                    rng=rng,
                )

        height, width = prepared.semantic.shape
        raw_region = canvas[y : y + height, x : x + width]
        raw_region += prepared.signal
        semantic_region = semantic[y : y + height, x : x + width]
        semantic[y : y + height, x : x + width] = merge_semantic(
            semantic_region, prepared.semantic
        )
        add_instance(
            primary[y : y + height, x : x + width],
            secondary[y : y + height, x : x + width],
            prepared.semantic > 0,
            index,
        )
        placements.append(
            {
                "cell_id": index,
                "source_case": donor.case_id,
                "interaction": mode,
                "overlap_pixels_at_placement": overlap_pixels,
                "rotation_degrees": prepared.rotation_degrees,
                "flip_x": prepared.flip_x,
                "flip_y": prepared.flip_y,
                "source_bbox_yxyx": list(prepared.source_bbox_yxyx),
                "target_yx": [y, x],
                "prepared_shape": [height, width],
            }
        )

    if np.issubdtype(raw_dtype, np.integer):
        limits = np.iinfo(raw_dtype)
        raw = np.clip(canvas, limits.min, limits.max).astype(raw_dtype)
    else:
        raw = canvas.astype(raw_dtype)
    metadata = {
        "source_cases": [donor.case_id for donor in donors],
        "cell_count": len(donors),
        "overlap_pixels": int(np.count_nonzero(secondary)),
        "placements": placements,
    }
    return raw, semantic, primary, secondary, metadata


def write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array, compression="zlib")


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def normalize_raw(raw: np.ndarray) -> np.ndarray:
    values = raw.astype(np.float32)
    low, high = np.percentile(values, [0.5, 99.7])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def semantic_preview(raw: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    gray = normalize_raw(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for label_value in (1, 2):
        mask = semantic == label_value
        rgb[mask] = np.round(
            0.28 * rgb[mask].astype(np.float32)
            + 0.72 * SEMANTIC_COLORS[label_value].astype(np.float32)
        ).astype(np.uint8)
    return rgb


def instance_preview(
    raw: np.ndarray, primary: np.ndarray, secondary: np.ndarray
) -> np.ndarray:
    gray = normalize_raw(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for cell_id in sorted(int(v) for v in np.unique(primary) if v > 0):
        mask = primary == cell_id
        color = INSTANCE_COLORS[(cell_id - 1) % len(INSTANCE_COLORS)]
        rgb[mask] = np.round(
            0.22 * rgb[mask].astype(np.float32) + 0.78 * color.astype(np.float32)
        ).astype(np.uint8)
    overlap = secondary > 0
    rgb[overlap] = np.asarray([255, 255, 255], dtype=np.uint8)
    return rgb


def save_previews(
    dataset_root: Path,
    scene_records: Sequence[dict],
    preview_count: int,
) -> Path:
    preview_root = dataset_root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    if not scene_records:
        raise ValueError("No synthetic scenes available for the preview")

    validation = [row for row in scene_records if row["split"] == "validation"]
    training = [row for row in scene_records if row["split"] == "training"]
    validation_slots = min(len(validation), max(1, preview_count // 4))
    training_slots = min(len(training), max(0, preview_count - validation_slots))

    def spaced(rows: Sequence[dict], count: int) -> list[dict]:
        if count <= 0:
            return []
        if count >= len(rows):
            return list(rows)
        indices = np.linspace(0, len(rows) - 1, count, dtype=int)
        return [rows[int(index)] for index in indices]

    selected = spaced(training, training_slots) + spaced(validation, validation_slots)
    cards: list[str] = []
    for row in selected:
        case_id = row["case_id"]
        raw = read_2d(dataset_root / "imagesTr" / f"{case_id}_0000.tif")
        semantic = read_2d(dataset_root / "labelsTr" / f"{case_id}.tif")
        primary = read_2d(dataset_root / "instance_primaryTr" / f"{case_id}.tif")
        secondary = read_2d(dataset_root / "instance_secondaryTr" / f"{case_id}.tif")
        raw_name = f"{case_id}_raw.png"
        semantic_name = f"{case_id}_semantic.png"
        instance_name = f"{case_id}_instances.png"
        Image.fromarray(normalize_raw(raw)).save(preview_root / raw_name)
        Image.fromarray(semantic_preview(raw, semantic)).save(
            preview_root / semantic_name
        )
        Image.fromarray(instance_preview(raw, primary, secondary)).save(
            preview_root / instance_name
        )
        source_text = ", ".join(html.escape(value) for value in row["source_cases"])
        cards.append(f"""
            <article class="scene" data-split="{row['split']}" data-overlap="{int(row['overlap_pixels'] > 0)}">
              <header><strong>{html.escape(case_id)}</strong><span>{row['split']} · {row['cell_count']} Zellen · {row['overlap_pixels']} Überlappungspixel</span></header>
              <div class="panels">
                <figure><img loading="lazy" src="previews/{raw_name}"><figcaption>Rohbild</figcaption></figure>
                <figure><img loading="lazy" src="previews/{semantic_name}"><figcaption>Semantik: Skeleton cyan, Soma pink</figcaption></figure>
                <figure><img loading="lazy" src="previews/{instance_name}"><figcaption>Zell-IDs; Überlappung weiß</figcaption></figure>
              </div>
              <p>{source_text}</p>
            </article>
            """)

    report = dataset_root / "synthetic_multicell_review.html"
    report.write_text(
        """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset143 · synthetische Mehrzellbilder</title>
<style>
:root{color-scheme:dark;font-family:Arial,sans-serif;background:#10151b;color:#edf3f8}body{margin:0;padding:24px}h1{margin:0 0 6px;font-size:24px}.intro{color:#aebac6;margin:0 0 18px}.controls{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;padding:10px 0;background:#10151be8;z-index:2}button{background:#26323e;color:#edf3f8;border:1px solid #425261;border-radius:7px;padding:8px 12px;cursor:pointer}button.active{background:#087f73;border-color:#21baa9}.scene{border-top:1px solid #33414e;padding:18px 0}.scene header{display:flex;justify-content:space-between;gap:12px}.scene header span,.scene p,figcaption{color:#aebac6;font-size:12px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:10px}figure{margin:0}img{display:block;width:100%;height:auto;background:#000;border-radius:6px}figcaption{margin-top:5px}@media(max-width:800px){.panels{grid-template-columns:1fr}.scene header{display:block}}
</style></head><body>
<h1>Dataset143 · synthetische Mehrzellbilder</h1>
<p class="intro">Reproduzierbare Stichprobe. Vollständige Herkunft und Transformationen stehen in synthetic_manifest.json/csv.</p>
<div class="controls"><button class="active" data-filter="all">Alle</button><button data-filter="training">Training</button><button data-filter="validation">Validation</button><button data-filter="overlap">Nur echte Überlappung</button></div>
"""
        + "\n".join(cards)
        + """
<script>const buttons=[...document.querySelectorAll('button[data-filter]')];const scenes=[...document.querySelectorAll('.scene')];buttons.forEach(b=>b.onclick=()=>{buttons.forEach(x=>x.classList.toggle('active',x===b));const f=b.dataset.filter;scenes.forEach(s=>s.hidden=!(f==='all'||s.dataset.split===f||(f==='overlap'&&s.dataset.overlap==='1')))});</script>
</body></html>""",
        encoding="utf-8",
    )
    return report


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_fold(split_path: Path, fold: int) -> tuple[list[str], list[str]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if fold < 0 or fold >= len(payload):
        raise ValueError(f"Fold {fold} not available in {split_path}")
    return list(payload[fold]["train"]), list(payload[fold]["val"])


def scan_donors(
    base_dataset: Path,
    split_path: Path,
    fold: int,
    canvas_size: int,
    edge_margin: int,
    min_soma_pixels: int,
    min_skeleton_pixels: int,
) -> tuple[list[DonorInfo], list[str], list[str]]:
    train_ids, validation_ids = load_fold(split_path, fold)
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    overlap = train_set.intersection(validation_set)
    if overlap:
        raise ValueError(
            f"Train/validation overlap in source split: {sorted(overlap)[:5]}"
        )

    image_paths = sorted((base_dataset / "imagesTr").glob("*_0000.tif"))
    donors: list[DonorInfo] = []
    for index, image_path in enumerate(image_paths, start=1):
        case_id = case_id_from_image(image_path)
        if case_id in train_set:
            split = "training"
        elif case_id in validation_set:
            split = "validation"
        else:
            split = "outside_fold"
        label_path = base_dataset / "labelsTr" / f"{case_id}.tif"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        donors.append(
            analyze_case(
                image_path,
                label_path,
                split,
                canvas_size,
                edge_margin,
                min_soma_pixels,
                min_skeleton_pixels,
            )
        )
        if index % 250 == 0 or index == len(image_paths):
            print(f"Donor audit: {index}/{len(image_paths)}", flush=True)
    return donors, train_ids, validation_ids


def build_dataset(args: argparse.Namespace) -> dict:
    base_dataset = args.base_dataset.resolve()
    source_split = args.base_splits.resolve()
    output_dataset = args.output_dataset.resolve()
    if output_dataset.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_dataset}"
        )
    for required in (
        base_dataset / "imagesTr",
        base_dataset / "labelsTr",
        base_dataset / "dataset.json",
        source_split,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    donors, base_train_ids, base_validation_ids = scan_donors(
        base_dataset,
        source_split,
        args.fold,
        args.canvas_size,
        args.edge_margin,
        args.min_soma_pixels,
        args.min_skeleton_pixels,
    )
    eligible_training = [
        donor for donor in donors if donor.eligible and donor.split == "training"
    ]
    eligible_validation = [
        donor for donor in donors if donor.eligible and donor.split == "validation"
    ]
    if (
        len(eligible_training) < args.min_cells
        or len(eligible_validation) < args.min_cells
    ):
        raise ValueError("Not enough eligible donors in both Fold-0 partitions")

    rng_training = np.random.default_rng(args.seed)
    rng_validation = np.random.default_rng(args.seed + 1)
    training_groups = group_donors(
        eligible_training, rng_training, args.min_cells, args.max_cells
    )
    validation_groups = group_donors(
        eligible_validation, rng_validation, args.min_cells, args.max_cells
    )

    staging = output_dataset.parent / f".{output_dataset.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for directory in (
        "imagesTr",
        "labelsTr",
        "instance_primaryTr",
        "instance_secondaryTr",
        "ambiguousTr",
    ):
        (staging / directory).mkdir()

    try:
        link_methods: Counter[str] = Counter()
        base_images = sorted((base_dataset / "imagesTr").glob("*_0000.tif"))
        for index, source_image in enumerate(base_images, start=1):
            case_id = case_id_from_image(source_image)
            source_label = base_dataset / "labelsTr" / f"{case_id}.tif"
            link_methods[
                hardlink_or_copy(source_image, staging / "imagesTr" / source_image.name)
            ] += 1
            link_methods[
                hardlink_or_copy(source_label, staging / "labelsTr" / source_label.name)
            ] += 1
            if index % 500 == 0 or index == len(base_images):
                print(f"Base cases linked: {index}/{len(base_images)}", flush=True)

        scene_records: list[dict] = []

        def generate_partition(
            groups: Sequence[Sequence[DonorInfo]],
            split_name: str,
            rng: np.random.Generator,
        ) -> list[str]:
            case_ids: list[str] = []
            token = "tr" if split_name == "training" else "va"
            for scene_index, group in enumerate(groups, start=1):
                case_id = f"synmc_{token}_{scene_index:05d}"
                raw, semantic, primary, secondary, metadata = compose_scene(
                    group,
                    args.canvas_size,
                    args.edge_margin,
                    args.context_radius,
                    args.signal_radius,
                    rng,
                )
                ambiguous = np.zeros_like(semantic, dtype=np.uint8)
                write_tiff(staging / "imagesTr" / f"{case_id}_0000.tif", raw)
                write_tiff(staging / "labelsTr" / f"{case_id}.tif", semantic)
                write_tiff(staging / "instance_primaryTr" / f"{case_id}.tif", primary)
                write_tiff(
                    staging / "instance_secondaryTr" / f"{case_id}.tif", secondary
                )
                write_tiff(staging / "ambiguousTr" / f"{case_id}.tif", ambiguous)
                record = {
                    "case_id": case_id,
                    "split": split_name,
                    **metadata,
                }
                scene_records.append(record)
                case_ids.append(case_id)
                if scene_index % 50 == 0 or scene_index == len(groups):
                    print(
                        f"Synthetic {split_name}: {scene_index}/{len(groups)}",
                        flush=True,
                    )
            return case_ids

        synthetic_train_ids = generate_partition(
            training_groups, "training", rng_training
        )
        synthetic_validation_ids = generate_partition(
            validation_groups, "validation", rng_validation
        )

        dataset_json = json.loads(
            (base_dataset / "dataset.json").read_text(encoding="utf-8")
        )
        dataset_json["name"] = output_dataset.name
        dataset_json["numTraining"] = len(base_images) + len(scene_records)
        dataset_json["description"] = (
            "Dataset141 plus deterministic synthetic 512px multi-cell scenes; "
            "semantic labels remain 0/1/2 and synthetic scenes also have instance IDs."
        )
        (staging / "dataset.json").write_text(
            json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        output_split = [
            {
                "train": base_train_ids + synthetic_train_ids,
                "val": base_validation_ids + synthetic_validation_ids,
            }
        ]
        (staging / "splits_final.json").write_text(
            json.dumps(output_split, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "synthetic_manifest.json").write_text(
            json.dumps(scene_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_csv(
            staging / "synthetic_manifest.csv",
            (
                {
                    "case_id": row["case_id"],
                    "split": row["split"],
                    "cell_count": row["cell_count"],
                    "overlap_pixels": row["overlap_pixels"],
                    "source_cases": "|".join(row["source_cases"]),
                }
                for row in scene_records
            ),
            ["case_id", "split", "cell_count", "overlap_pixels", "source_cases"],
        )
        write_csv(
            staging / "donor_audit.csv",
            (asdict(donor) for donor in donors),
            list(asdict(donors[0]).keys()),
        )

        exclusion_counts = Counter(
            donor.exclusion_reason or "eligible" for donor in donors
        )
        summary = {
            "script_version": SCRIPT_VERSION,
            "base_dataset": str(base_dataset),
            "base_split": str(source_split),
            "source_fold": args.fold,
            "output_dataset": str(output_dataset),
            "seed": args.seed,
            "canvas_size": args.canvas_size,
            "signal_radius": args.signal_radius,
            "cell_count_range": [args.min_cells, args.max_cells],
            "base_cases": len(base_images),
            "eligible_training_donors": len(eligible_training),
            "eligible_validation_donors": len(eligible_validation),
            "excluded_donors": len(donors)
            - len(eligible_training)
            - len(eligible_validation),
            "donor_audit_counts": dict(sorted(exclusion_counts.items())),
            "synthetic_training_scenes": len(synthetic_train_ids),
            "synthetic_validation_scenes": len(synthetic_validation_ids),
            "synthetic_scenes": len(scene_records),
            "synthetic_scenes_with_overlap": int(
                sum(row["overlap_pixels"] > 0 for row in scene_records)
            ),
            "total_semantic_cases": len(base_images) + len(scene_records),
            "base_file_materialization": dict(link_methods),
            "instance_labels_available_for": "synthetic scenes only",
        }
        (staging / "build_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report = save_previews(staging, scene_records, args.preview_count)
        (staging / "README.txt").write_text(
            "Dataset143 extends Dataset141 with synthetic multi-cell scenes.\n"
            "imagesTr/ and labelsTr/ are directly compatible with semantic nnU-Net.\n"
            "instance_primaryTr/ and instance_secondaryTr/ exist for synthetic scenes only.\n"
            "Every eligible source cell is used once and never crosses the Fold-0 split.\n",
            encoding="utf-8",
        )
        staging.replace(output_dataset)
        summary["review_html"] = str(output_dataset / report.name)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-splits", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--min-cells", type=int, default=2)
    parser.add_argument("--max-cells", type=int, default=4)
    parser.add_argument("--edge-margin", type=int, default=8)
    parser.add_argument("--context-radius", type=int, default=24)
    parser.add_argument("--signal-radius", type=int, default=8)
    parser.add_argument("--min-soma-pixels", type=int, default=20)
    parser.add_argument("--min-skeleton-pixels", type=int, default=8)
    parser.add_argument("--preview-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=143)
    args = parser.parse_args()
    if args.canvas_size < 128:
        parser.error("--canvas-size must be at least 128")
    if args.preview_count < 1:
        parser.error("--preview-count must be positive")
    if args.signal_radius < 1 or args.signal_radius > args.context_radius:
        parser.error("--signal-radius must be between 1 and --context-radius")
    return args


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
