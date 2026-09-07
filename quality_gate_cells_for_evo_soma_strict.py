from __future__ import annotations

"""Conservative soma and multi-cell QC for Evo crop selection.

This is a selection-only quality gate. It *never* edits the input semantic
segmentation or the skeleton material. Instead it partitions selected instance
IDs into ``safe``, ``review`` and ``excluded`` maps. The crop exporter should
receive only ``01_cells_safe_for_evo.tif``.

The default additional checks are deliberately conservative, while explicit
command-line policies allow a separate relaxed test without altering the
semantic segmentation:

* a selected cell must contain one compact soma with one strong distance-
  transform core;
* small, malformed, multi-core (possibly fused) and poorly contrasted soma
  candidates can be held back from automated export;
* a second soma or skeleton in the raw-image crop is recorded, but is allowed
  by default because the exporter writes only the selected instance's masks;
* the selected instance itself must still represent one coherent soma/skeleton
  cell rather than a fused multi-core object.

The implementation uses the soma mask only to decide whether a crop is
exportable. It does not watershed-split a fused soma automatically.
"""

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import h_maxima


SCRIPT_VERSION = "evo-cell-quality-gate-v3-configurable-soma-qc-2026-08-16"
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


@dataclass
class Candidate:
    identifier: int
    cell: np.ndarray
    skeleton: np.ndarray
    soma: np.ndarray
    soma_components: int
    soma_area: int
    skeleton_area: int
    skeleton_components: int
    core_radius: float
    core_count: int
    core_points: list[tuple[int, int]]
    core_h: float
    solidity: float
    eccentricity: float
    core_disk_ratio: float
    holes: int
    hole_pixels: int
    connected_skeleton_fraction: float
    skeleton_touches_soma: bool
    support_fraction: float
    soma_local_contrast: float
    image_border: bool
    crop_bounds: tuple[int, int, int, int]
    foreign_soma_components: int
    foreign_soma_pixels: int
    foreign_instance_ids: list[int]
    foreign_skeleton_near_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create conservative safe/review/excluded cell maps without "
            "changing the semantic skeleton/soma segmentation."
        )
    )
    parser.add_argument("--semantic", type=Path, required=True, help="Final 0/1/2 semantic TIFF.")
    parser.add_argument("--instances", type=Path, required=True, help="Selected cell-instance TIFF.")
    parser.add_argument(
        "--review-instances",
        type=Path,
        help=(
            "Optional broader assigned-instance map. It is used to detect "
            "competing cells in a proposed crop and to retain non-selected "
            "assignment cases in the review map."
        ),
    )
    parser.add_argument("--original", type=Path, required=True, help="Same-shape raw image TIFF.")
    parser.add_argument("--ridge-evidence", type=Path, help="Optional same-shape neurite evidence TIFF.")
    parser.add_argument("--ridge-weight", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--polarity", choices=("auto", "bright", "dark"), default="bright")

    # Existing conservative topology checks.
    parser.add_argument("--min-soma-area", type=int, default=80)
    parser.add_argument("--max-soma-area", type=int, default=12000)
    parser.add_argument("--min-skeleton-pixels", type=int, default=20)
    parser.add_argument("--contact-radius", type=int, default=4)
    parser.add_argument("--safe-min-support-fraction", type=float, default=0.25)
    parser.add_argument("--review-min-support-fraction", type=float, default=0.10)
    parser.add_argument("--max-skeleton-components-safe", type=int, default=4)
    parser.add_argument("--safe-min-connected-skeleton-fraction", type=float, default=0.90)
    parser.add_argument("--border-margin", type=int, default=4)

    # Soma-core / fused-soma checks. Distances are in input pixels.
    parser.add_argument("--min-soma-core-radius", type=float, default=6.0)
    parser.add_argument(
        "--core-peak-relative-height",
        type=float,
        default=0.55,
        help="A soma DT peak must be at least this fraction of the largest peak (default: 0.55).",
    )
    parser.add_argument(
        "--core-peak-prominence-fraction",
        type=float,
        default=0.18,
        help="H-maxima prominence as a fraction of the largest DT radius (default: 0.18).",
    )
    parser.add_argument(
        "--core-min-separation",
        type=float,
        default=12.0,
        help="Minimum separation between strong soma cores in pixels (default: 12).",
    )
    parser.add_argument(
        "--multi-core-policy",
        choices=("exclude", "review", "allow-if-single-instance"),
        default="exclude",
        help=(
            "How to handle a single connected soma mask with multiple strong "
            "distance-transform cores. 'exclude' preserves the original strict "
            "behaviour; 'review' holds it back; 'allow-if-single-instance' keeps "
            "it exportable only when its selected skeleton is attached and coherent."
        ),
    )
    parser.add_argument(
        "--min-soma-solidity",
        type=float,
        default=0.80,
        help="Reject a single-component soma below this convex-hull solidity (default: 0.80).",
    )
    parser.add_argument(
        "--max-soma-core-disk-ratio",
        type=float,
        default=5.0,
        help=(
            "Reject very elongated/leaking masks when soma area / (pi*r_max^2) "
            "exceeds this value (default: 5.0)."
        ),
    )
    parser.add_argument(
        "--min-soma-local-contrast",
        type=float,
        default=0.015,
        help="Raw normalized median(soma) - median(local annulus); lower values go to review.",
    )
    parser.add_argument("--soma-annulus-inner", type=int, default=3)
    parser.add_argument("--soma-annulus-outer", type=int, default=20)
    parser.add_argument(
        "--soma-hole-policy",
        choices=("ignore", "review", "exclude"),
        default="review",
        help=(
            "How to handle holes inside the selected soma mask. Small holes below "
            "--min-soma-hole-pixels-for-review are recorded as information only."
        ),
    )
    parser.add_argument(
        "--min-soma-hole-pixels-for-review",
        type=int,
        default=1,
        help=(
            "Minimum total enclosed hole area in pixels before --soma-hole-policy "
            "can block or review the crop (default: 1, preserving the strict mode)."
        ),
    )

    # Per-image robust size outlier check. It complements, never relaxes,
    # --min-soma-area / --max-soma-area.
    parser.add_argument("--disable-auto-soma-size-qc", action="store_true")
    parser.add_argument("--auto-soma-low-quantile", type=float, default=0.03)
    parser.add_argument("--auto-soma-high-quantile", type=float, default=0.98)

    # The following values deliberately mirror the eventual exporter defaults.
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--crop-min-size", type=int, default=160)
    parser.add_argument(
        "--rectangular-crops",
        action="store_true",
        help="Use rectangular preview crops. Default mirrors the exporter's --square mode.",
    )
    parser.add_argument(
        "--foreign-soma-min-area",
        type=int,
        default=80,
        help="Minimum area for reporting a second semantic soma in a proposed crop.",
    )
    parser.add_argument(
        "--foreign-soma-policy",
        choices=("allow", "review", "exclude"),
        default="allow",
        help=(
            "How to handle another soma in the raw-image crop. 'allow' keeps "
            "the selected cell exportable because only its instance mask is "
            "written; 'review' holds it back; 'exclude' is the old strict mode."
        ),
    )
    parser.add_argument("--foreign-skeleton-near-distance", type=int, default=18)
    parser.add_argument("--max-foreign-skeleton-near-pixels", type=int, default=30)
    parser.add_argument(
        "--foreign-skeleton-policy",
        choices=("allow", "review"),
        default="allow",
        help=(
            "A foreign skeleton is outside the exported instance mask. Keep it "
            "as information ('allow', default) or route the candidate to review."
        ),
    )
    parser.add_argument("--qc-max-size", type=int, default=2200)
    parser.add_argument("--flagged-qc-max-size", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_2d(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be 2-D after squeeze, got {array.shape}")
    return array


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def robust_unit_scale(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("Image has no finite values.")
    sample = finite[::max(1, finite.size // 1_000_000)]
    low, high = np.percentile(sample, [1.0, 99.5])
    if high <= low:
        low, high = float(np.min(sample)), float(np.max(sample))
    if high <= low:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def build_support(original: np.ndarray, skeleton: np.ndarray, polarity: str) -> tuple[np.ndarray, np.ndarray, str]:
    normalized = robust_unit_scale(original)
    if polarity == "auto":
        selected = "bright"
        if np.any(skeleton) and np.any(~skeleton):
            selected = "bright" if np.median(normalized[skeleton]) >= np.median(normalized[~skeleton]) else "dark"
    else:
        selected = polarity
    direct = normalized if selected == "bright" else 1.0 - normalized
    local = direct - ndi.gaussian_filter(direct, sigma=2.0, mode="reflect")
    support = np.clip(0.70 * direct + 0.30 * np.clip(0.5 + 2.0 * local, 0.0, 1.0), 0.0, 1.0)
    return support.astype(np.float32), direct.astype(np.float32), selected


def touches_border(mask: np.ndarray, margin: int) -> bool:
    if margin <= 0 or not np.any(mask):
        return False
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    return bool(rows.min() < margin or cols.min() < margin or rows.max() >= height - margin or cols.max() >= width - margin)


def axis_bounds(minimum: int, maximum_inclusive: int, limit: int, margin: int, minimum_size: int) -> tuple[int, int]:
    desired = max(maximum_inclusive - minimum + 1 + 2 * margin, minimum_size)
    desired = min(desired, limit)
    center = 0.5 * (minimum + maximum_inclusive)
    start = int(math.floor(center - desired / 2))
    start = max(0, min(start, limit - desired))
    return start, start + desired


def crop_bounds(mask: np.ndarray, margin: int, minimum_size: int, square: bool) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise RuntimeError("Cannot crop an empty cell.")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    height, width = mask.shape
    if square:
        requested = max(int(max(y_max - y_min + 1, x_max - x_min + 1)) + 2 * margin, minimum_size)
        y0, y1 = axis_bounds(int(y_min), int(y_max), height, margin, requested)
        x0, x1 = axis_bounds(int(x_min), int(x_max), width, margin, requested)
    else:
        y0, y1 = axis_bounds(int(y_min), int(y_max), height, margin, minimum_size)
        x0, x1 = axis_bounds(int(x_min), int(x_max), width, margin, minimum_size)
    return y0, y1, x0, x1


def skeleton_attachment(soma: np.ndarray, skeleton: np.ndarray, contact_radius: int) -> tuple[bool, float]:
    if not np.any(soma) or not np.any(skeleton):
        return False, 0.0
    zone = ndi.binary_dilation(soma, iterations=contact_radius) if contact_radius else soma
    touches = bool(np.any(skeleton & zone))
    components, _ = ndi.label(skeleton | soma, structure=STRUCTURE_8)
    soma_component_ids = np.unique(components[soma])
    soma_component_ids = soma_component_ids[soma_component_ids > 0]
    connected_skeleton = skeleton & np.isin(components, soma_component_ids)
    return touches, float(connected_skeleton.sum() / skeleton.sum()) if skeleton.any() else 0.0


def hole_metrics(mask: np.ndarray) -> tuple[int, int]:
    """Return enclosed-hole component count and total area in pixels.

    A one-pixel classification imperfection is biologically very different from
    a large cleft splitting a soma. The original QC only returned a boolean,
    which made these two cases indistinguishable.
    """
    if not np.any(mask):
        return 0, 0
    y, x = np.nonzero(mask)
    y0, y1 = max(0, int(y.min()) - 1), min(mask.shape[0], int(y.max()) + 2)
    x0, x1 = max(0, int(x.min()) - 1), min(mask.shape[1], int(x.max()) + 2)
    sub = mask[y0:y1, x0:x1]
    filled = ndi.binary_fill_holes(sub)
    holes = filled & ~sub
    labels, count = ndi.label(holes, structure=STRUCTURE_8)
    del labels
    return int(count), int(holes.sum())


def soma_shape_metrics(
    soma: np.ndarray,
    min_core_radius: float,
    peak_relative_height: float,
    peak_prominence_fraction: float,
    min_separation: float,
) -> tuple[float, int, list[tuple[int, int]], float, float, float, int, int, int]:
    """Return DT-core and geometry metrics for one connected soma mask."""
    if not soma.any():
        return 0.0, 0, [], 0.0, 0.0, math.inf, 0.0, 0, 0
    labels, component_count = ndi.label(soma, structure=STRUCTURE_8)
    if component_count != 1:
        # Still return metrics for the largest component; caller rejects multi-component soma.
        areas = [(labels == label).sum() for label in range(1, component_count + 1)]
        soma_for_geometry = labels == int(np.argmax(areas) + 1)
    else:
        soma_for_geometry = soma
    distance = ndi.distance_transform_edt(soma_for_geometry)
    radius = float(distance.max())
    if radius <= 0:
        holes, hole_pixels = hole_metrics(soma_for_geometry)
        return 0.0, 0, [], 0.0, 0.0, math.inf, 0.0, holes, hole_pixels

    # h-maxima removes noisy local maxima. A seed also needs a substantial
    # in-mask radius and height relative to the dominant core.
    h = max(1.0, radius * peak_prominence_fraction)
    maxima = h_maxima(distance, h=h)
    maxima_labels, maxima_count = ndi.label(maxima, structure=STRUCTURE_8)
    raw_points: list[tuple[float, int, int]] = []
    threshold = max(float(min_core_radius), radius * peak_relative_height)
    for label in range(1, maxima_count + 1):
        coordinates = np.argwhere(maxima_labels == label)
        if coordinates.size == 0:
            continue
        values = distance[maxima_labels == label]
        y, x = coordinates[int(np.argmax(values))]
        height = float(distance[y, x])
        if height >= threshold:
            raw_points.append((height, int(y), int(x)))
    raw_points.sort(reverse=True)
    points: list[tuple[int, int]] = []
    for _, y, x in raw_points:
        if all(math.hypot(y - py, x - px) >= min_separation for py, px in points):
            points.append((y, x))

    prop = regionprops(soma_for_geometry.astype(np.uint8))[0]
    solidity = float(prop.solidity) if np.isfinite(prop.solidity) else 0.0
    eccentricity = float(prop.eccentricity) if np.isfinite(prop.eccentricity) else 0.0
    # A disc has ratio 1; large values flag very elongated blobs, soma leaks,
    # or merged lobes. This is an outlier feature, not a replacement for cores.
    core_disk_ratio = float(soma_for_geometry.sum() / (math.pi * radius * radius))
    holes, hole_pixels = hole_metrics(soma_for_geometry)
    return radius, len(points), points, h, solidity, eccentricity, core_disk_ratio, holes, hole_pixels


def soma_local_contrast(direct: np.ndarray, soma: np.ndarray, inner: int, outer: int) -> float:
    if not soma.any():
        return float("nan")
    inner_mask = soma if inner <= 0 else ndi.binary_dilation(soma, iterations=inner)
    outer_mask = ndi.binary_dilation(soma, iterations=max(inner + 1, outer))
    annulus = outer_mask & ~inner_mask
    if annulus.sum() < 20:
        return float("nan")
    return float(np.median(direct[soma]) - np.median(direct[annulus]))


def foreign_crop_metrics(
    semantic: np.ndarray,
    broad_instances: np.ndarray | None,
    cell: np.ndarray,
    identifier: int,
    bounds: tuple[int, int, int, int],
    foreign_soma_min_area: int,
    foreign_skeleton_near_distance: int,
) -> tuple[int, int, list[int], int]:
    y0, y1, x0, x1 = bounds
    local_cell = cell[y0:y1, x0:x1]
    local_semantic = semantic[y0:y1, x0:x1]
    foreign_soma = (local_semantic == 2) & ~local_cell
    labels, count = ndi.label(foreign_soma, structure=STRUCTURE_8)
    valid_foreign = np.zeros_like(foreign_soma)
    valid_count = 0
    for label in range(1, count + 1):
        component = labels == label
        if int(component.sum()) >= foreign_soma_min_area:
            valid_foreign |= component
            valid_count += 1
    foreign_ids: list[int] = []
    if broad_instances is not None:
        local_broad = broad_instances[y0:y1, x0:x1]
        foreign_ids = sorted(int(value) for value in np.unique(local_broad) if int(value) > 0 and int(value) != identifier)
    local_foreign_skeleton = (local_semantic == 1) & ~local_cell
    near = (
        local_cell
        if foreign_skeleton_near_distance <= 0
        else ndi.binary_dilation(local_cell, iterations=foreign_skeleton_near_distance)
    )
    near_pixels = int((local_foreign_skeleton & near).sum())
    return valid_count, int(valid_foreign.sum()), foreign_ids, near_pixels


def palette(identifier: int) -> np.ndarray:
    hue = (identifier * 0.61803398875) % 1.0
    anchors = np.asarray([[255, 85, 115], [250, 210, 70], [70, 210, 160], [65, 150, 255], [185, 100, 245]], dtype=np.float32)
    return anchors[int(hue * len(anchors)) % len(anchors)]


def save_qc(path: Path, original: np.ndarray, safe: np.ndarray, review: np.ndarray, excluded: np.ndarray, max_size: int) -> None:
    height, width = original.shape
    stride = max(1, int(np.ceil(max(height, width) / max(max_size, 1))))
    gray = robust_unit_scale(original[::stride, ::stride])
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    safe_small, review_small, excluded_small = safe[::stride, ::stride], review[::stride, ::stride], excluded[::stride, ::stride]
    for identifier in np.unique(safe_small):
        if identifier > 0:
            mask = safe_small == identifier
            rgb[mask] = 0.18 * rgb[mask] + 0.82 * palette(int(identifier))
    rgb[review_small > 0] = 0.18 * rgb[review_small > 0] + 0.82 * np.asarray([255, 190, 0])
    rgb[excluded_small > 0] = 0.28 * rgb[excluded_small > 0] + 0.72 * np.asarray([245, 70, 55])
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def draw_points(rgb: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for y, x in points:
        y0, y1 = max(0, y - 3), min(rgb.shape[0], y + 4)
        x0, x1 = max(0, x - 3), min(rgb.shape[1], x + 4)
        rgb[y0:y1, x0:x1] = color


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80]


def save_flagged_cell_qc(
    path: Path,
    original: np.ndarray,
    semantic: np.ndarray,
    candidate: Candidate,
    max_size: int,
) -> None:
    y0, y1, x0, x1 = candidate.crop_bounds
    raw = robust_unit_scale(original[y0:y1, x0:x1])
    rgb = np.repeat((raw * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    cell = candidate.cell[y0:y1, x0:x1]
    skeleton = candidate.skeleton[y0:y1, x0:x1]
    soma = candidate.soma[y0:y1, x0:x1]
    foreign_soma = (semantic[y0:y1, x0:x1] == 2) & ~cell
    rgb[skeleton] = 0.10 * rgb[skeleton] + 0.90 * np.asarray([0, 220, 255])
    rgb[soma] = 0.18 * rgb[soma] + 0.82 * np.asarray([255, 45, 180])
    rgb[foreign_soma] = 0.15 * rgb[foreign_soma] + 0.85 * np.asarray([245, 60, 55])
    local_points = [(y - y0, x - x0) for y, x in candidate.core_points]
    draw_points(rgb, local_points, (255, 235, 0))
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    scale = min(1.0, max_size / max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.NEAREST)
    image.save(path)


def build_candidate(
    identifier: int,
    semantic: np.ndarray,
    instances: np.ndarray,
    broad_instances: np.ndarray | None,
    support: np.ndarray,
    direct: np.ndarray,
    args: argparse.Namespace,
) -> Candidate:
    cell = instances == identifier
    skeleton = cell & (semantic == 1)
    soma = cell & (semantic == 2)
    soma_components = int(ndi.label(soma, structure=STRUCTURE_8)[1])
    skeleton_components = int(ndi.label(skeleton, structure=STRUCTURE_8)[1])
    (
        core_radius,
        core_count,
        core_points,
        core_h,
        solidity,
        eccentricity,
        core_disk_ratio,
        holes,
        hole_pixels,
    ) = soma_shape_metrics(
        soma,
        args.min_soma_core_radius,
        args.core_peak_relative_height,
        args.core_peak_prominence_fraction,
        args.core_min_separation,
    )
    touches, connected_fraction = skeleton_attachment(soma, skeleton, args.contact_radius)
    bounds = crop_bounds(cell, args.crop_margin, args.crop_min_size, square=not args.rectangular_crops)
    foreign_soma_components, foreign_soma_pixels, foreign_ids, foreign_skeleton_near_pixels = foreign_crop_metrics(
        semantic,
        broad_instances,
        cell,
        identifier,
        bounds,
        args.foreign_soma_min_area,
        args.foreign_skeleton_near_distance,
    )
    return Candidate(
        identifier=identifier,
        cell=cell,
        skeleton=skeleton,
        soma=soma,
        soma_components=soma_components,
        soma_area=int(soma.sum()),
        skeleton_area=int(skeleton.sum()),
        skeleton_components=skeleton_components,
        core_radius=core_radius,
        core_count=core_count,
        core_points=core_points,
        core_h=core_h,
        solidity=solidity,
        eccentricity=eccentricity,
        core_disk_ratio=core_disk_ratio,
        holes=holes,
        hole_pixels=hole_pixels,
        connected_skeleton_fraction=connected_fraction,
        skeleton_touches_soma=touches,
        support_fraction=float(np.mean(support[skeleton] >= 0.5)) if skeleton.any() else 0.0,
        soma_local_contrast=soma_local_contrast(direct, soma, args.soma_annulus_inner, args.soma_annulus_outer),
        image_border=touches_border(cell, args.border_margin),
        crop_bounds=bounds,
        foreign_soma_components=foreign_soma_components,
        foreign_soma_pixels=foreign_soma_pixels,
        foreign_instance_ids=foreign_ids,
        foreign_skeleton_near_pixels=foreign_skeleton_near_pixels,
    )


def effective_size_limits(candidates: list[Candidate], args: argparse.Namespace) -> tuple[float, float, float]:
    areas = np.asarray([candidate.soma_area for candidate in candidates if candidate.soma_components == 1 and candidate.soma_area > 0], dtype=float)
    radii = np.asarray([candidate.core_radius for candidate in candidates if candidate.soma_components == 1 and candidate.core_radius > 0], dtype=float)
    if args.disable_auto_soma_size_qc or areas.size < 12:
        return float(args.min_soma_area), float(args.max_soma_area), float(args.min_soma_core_radius)
    auto_low_area = float(np.quantile(areas, args.auto_soma_low_quantile))
    auto_high_area = float(np.quantile(areas, args.auto_soma_high_quantile))
    auto_low_radius = float(np.quantile(radii, args.auto_soma_low_quantile)) if radii.size else 0.0
    lower_area = max(float(args.min_soma_area), auto_low_area)
    upper_area = min(float(args.max_soma_area), auto_high_area)
    lower_radius = max(float(args.min_soma_core_radius), auto_low_radius)
    if upper_area <= lower_area:
        upper_area = float(args.max_soma_area)
    return lower_area, upper_area, lower_radius


def classify(
    candidate: Candidate,
    args: argparse.Namespace,
    min_area: float,
    max_area: float,
    min_radius: float,
) -> tuple[str, list[str], list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    information: list[str] = []
    if candidate.soma_components != 1:
        hard.append("not_exactly_one_soma_component")
    if candidate.soma_area < min_area:
        hard.append("soma_area_below_image_calibrated_minimum")
    if candidate.soma_area > max_area:
        hard.append("soma_area_above_image_calibrated_maximum")
    if candidate.core_radius < min_radius:
        hard.append("soma_core_radius_too_small")
    if candidate.core_count == 0:
        hard.append("no_strong_soma_core")
    elif candidate.core_count > 1:
        if args.multi_core_policy == "exclude":
            hard.append("multiple_persistent_soma_cores_possible_fusion")
        elif args.multi_core_policy == "review":
            soft.append("multiple_persistent_soma_cores_possible_fusion")
        else:
            # The requested relaxed mode does not assume that two DT maxima are
            # automatically two cells. It still demands one selected, coherent
            # cell instance with a soma-attached skeleton. Other hard geometry
            # checks below remain active.
            coherent_single_instance = (
                candidate.soma_components == 1
                and candidate.skeleton_area >= args.min_skeleton_pixels
                and candidate.skeleton_touches_soma
                and candidate.skeleton_components <= args.max_skeleton_components_safe
                and candidate.connected_skeleton_fraction >= args.safe_min_connected_skeleton_fraction
            )
            if coherent_single_instance:
                information.append("multiple_soma_cores_allowed_for_coherent_single_instance")
            else:
                soft.append("multiple_persistent_soma_cores_needs_manual_review")
    if candidate.solidity < args.min_soma_solidity:
        hard.append("soma_low_solidity_possible_fusion_or_fragment")
    if candidate.core_disk_ratio > args.max_soma_core_disk_ratio:
        hard.append("soma_elongated_or_leaking_relative_to_core")
    if candidate.image_border:
        hard.append("touches_source_image_border")
    if candidate.foreign_soma_components > 0:
        if args.foreign_soma_policy == "exclude":
            hard.append("foreign_soma_in_proposed_export_crop")
        elif args.foreign_soma_policy == "review":
            soft.append("foreign_soma_in_proposed_export_crop")
        else:
            information.append("foreign_soma_in_raw_crop_not_exported")

    if candidate.skeleton_area < args.min_skeleton_pixels:
        hard.append("skeleton_below_min_pixels")
    if not candidate.skeleton_touches_soma:
        soft.append("skeleton_not_touching_soma")
    if candidate.skeleton_components > args.max_skeleton_components_safe:
        soft.append("fragmented_skeleton")
    if candidate.connected_skeleton_fraction < args.safe_min_connected_skeleton_fraction:
        soft.append("skeleton_not_sufficiently_connected_to_soma")
    if candidate.support_fraction < args.review_min_support_fraction:
        soft.append("very_weak_raw_or_ridge_support")
    elif candidate.support_fraction < args.safe_min_support_fraction:
        soft.append("weak_raw_or_ridge_support")
    if np.isfinite(candidate.soma_local_contrast) and candidate.soma_local_contrast < args.min_soma_local_contrast:
        soft.append("low_local_soma_contrast")
    if candidate.hole_pixels:
        if candidate.hole_pixels < args.min_soma_hole_pixels_for_review:
            information.append("soma_tiny_hole_below_review_threshold")
        elif args.soma_hole_policy == "exclude":
            hard.append("soma_contains_large_hole")
        elif args.soma_hole_policy == "review":
            soft.append("soma_contains_large_hole")
        else:
            information.append("soma_hole_not_blocking")
    if candidate.foreign_instance_ids:
        information.append("other_assigned_instance_in_raw_crop_not_exported")
    if candidate.foreign_skeleton_near_pixels > args.max_foreign_skeleton_near_pixels:
        if args.foreign_skeleton_policy == "review":
            soft.append("substantial_foreign_skeleton_close_to_selected_cell")
        else:
            information.append("foreign_skeleton_in_raw_crop_not_exported")

    if hard:
        return "excluded", hard, soft, information
    if soft:
        return "review", hard, soft, information
    return "safe", hard, soft, information


def row_from_candidate(
    candidate: Candidate,
    category: str,
    hard: list[str],
    soft: list[str],
    information: list[str],
) -> dict[str, object]:
    y0, y1, x0, x1 = candidate.crop_bounds
    return {
        "source_instance_id": candidate.identifier,
        "category": category,
        "soma_components": candidate.soma_components,
        "soma_pixels": candidate.soma_area,
        "soma_core_radius_px": round(candidate.core_radius, 4),
        "strong_soma_core_count": candidate.core_count,
        "soma_hmax_prominence_px": round(candidate.core_h, 4),
        "soma_solidity": round(candidate.solidity, 5),
        "soma_eccentricity": round(candidate.eccentricity, 5),
        "soma_area_to_core_disk_ratio": round(candidate.core_disk_ratio, 5),
        "soma_holes": candidate.holes,
        "soma_hole_pixels": candidate.hole_pixels,
        "soma_local_contrast": "" if not np.isfinite(candidate.soma_local_contrast) else round(candidate.soma_local_contrast, 5),
        "skeleton_pixels": candidate.skeleton_area,
        "skeleton_components": candidate.skeleton_components,
        "skeleton_touches_soma": int(candidate.skeleton_touches_soma),
        "connected_skeleton_fraction": round(candidate.connected_skeleton_fraction, 5),
        "support_fraction_ge_0_5": round(candidate.support_fraction, 5),
        "foreign_soma_components_in_crop": candidate.foreign_soma_components,
        "foreign_soma_pixels_in_crop": candidate.foreign_soma_pixels,
        "foreign_instance_ids_in_crop": ";".join(map(str, candidate.foreign_instance_ids)),
        "foreign_skeleton_near_pixels": candidate.foreign_skeleton_near_pixels,
        "crop_x_min": x0,
        "crop_y_min": y0,
        "crop_x_max_exclusive": x1,
        "crop_y_max_exclusive": y1,
        "hard_reasons": ";".join(hard),
        "review_reasons": ";".join(soft),
        "information_notes": ";".join(information),
        "manual_decision": "",
        "manual_notes": "",
    }


def main() -> None:
    args = parse_args()
    if args.min_soma_area < 1 or args.max_soma_area < args.min_soma_area:
        raise ValueError("Soma area limits must be valid.")
    if (
        args.min_skeleton_pixels < 1
        or args.contact_radius < 0
        or args.crop_margin < 0
        or args.crop_min_size < 1
        or args.min_soma_hole_pixels_for_review < 1
    ):
        raise ValueError("Minimum sizes, contact radius and crop parameters must be valid.")
    if not 0 <= args.review_min_support_fraction <= args.safe_min_support_fraction <= 1:
        raise ValueError("Require 0 <= review support <= safe support <= 1.")
    if not 0 <= args.ridge_weight <= 1 or not 0 <= args.safe_min_connected_skeleton_fraction <= 1:
        raise ValueError("Weights and fractions must be between 0 and 1.")
    if not 0 < args.auto_soma_low_quantile < args.auto_soma_high_quantile < 1:
        raise ValueError("Auto soma quantiles must satisfy 0 < low < high < 1.")
    if not 0 < args.min_soma_solidity <= 1 or args.core_min_separation < 0:
        raise ValueError("Invalid soma-core geometry parameters.")

    prepare_output(args.output_dir, args.overwrite)
    semantic = read_2d(args.semantic, "semantic segmentation")
    instances = read_2d(args.instances, "selected instance map").astype(np.uint32)
    original = read_2d(args.original, "original image")
    if not (semantic.shape == instances.shape == original.shape):
        raise RuntimeError("semantic, instances and original must have the same 2-D shape.")
    invalid = set(np.unique(semantic).astype(int).tolist()) - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Unexpected semantic labels: {sorted(invalid)}")
    if np.any((instances > 0) & (semantic == 0)):
        raise RuntimeError("Instance map contains pixels outside semantic skeleton/soma material.")

    broad_instances = None
    if args.review_instances:
        broad_instances = read_2d(args.review_instances, "broader review instance map").astype(np.uint32)
        if broad_instances.shape != semantic.shape:
            raise RuntimeError("Broader review instance map must match semantic shape.")
        if np.any((broad_instances > 0) & (semantic == 0)):
            raise RuntimeError("Broader review instance map contains pixels outside semantic material.")

    skeleton_all = semantic == 1
    raw_support, direct, selected_polarity = build_support(original, skeleton_all, args.polarity)
    support = raw_support
    if args.ridge_evidence:
        ridge = read_2d(args.ridge_evidence, "ridge evidence")
        if ridge.shape != semantic.shape:
            raise RuntimeError("Ridge evidence must match semantic shape.")
        support = (1.0 - args.ridge_weight) * raw_support + args.ridge_weight * robust_unit_scale(ridge)
        support = np.clip(support, 0.0, 1.0).astype(np.float32)

    identifiers = [int(value) for value in np.unique(instances) if int(value) > 0]
    candidates = [build_candidate(identifier, semantic, instances, broad_instances, support, direct, args) for identifier in identifiers]
    effective_min_area, effective_max_area, effective_min_radius = effective_size_limits(candidates, args)
    print(
        "Soma QC calibration: "
        f"area [{effective_min_area:.1f}, {effective_max_area:.1f}], "
        f"core radius >= {effective_min_radius:.2f}px"
    )

    safe = np.zeros_like(instances)
    review = np.zeros_like(instances)
    excluded = np.zeros_like(instances)
    core_markers = np.zeros_like(instances)
    foreign_soma_crop_mask = np.zeros(semantic.shape, dtype=np.uint8)
    rows: list[dict[str, object]] = []
    flagged_dir = args.output_dir / "07_flagged_soma_qc"
    flagged_dir.mkdir(exist_ok=True)

    for candidate in candidates:
        category, hard, soft, information = classify(candidate, args, effective_min_area, effective_max_area, effective_min_radius)
        if category == "safe":
            safe[candidate.cell] = candidate.identifier
        elif category == "review":
            review[candidate.cell] = candidate.identifier
        else:
            excluded[candidate.cell] = candidate.identifier
        for y, x in candidate.core_points:
            y0, y1 = max(0, y - 1), min(core_markers.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(core_markers.shape[1], x + 2)
            core_markers[y0:y1, x0:x1] = candidate.identifier
        if candidate.foreign_soma_components:
            y0, y1, x0, x1 = candidate.crop_bounds
            foreign_soma_crop_mask[y0:y1, x0:x1] |= ((semantic[y0:y1, x0:x1] == 2) & ~candidate.cell[y0:y1, x0:x1]).astype(np.uint8)
        row = row_from_candidate(candidate, category, hard, soft, information)
        rows.append(row)
        if category != "safe":
            reason = row["hard_reasons"] or row["review_reasons"] or "review"
            filename = f"instance_{candidate.identifier:05d}_{category}_{safe_name(str(reason))}.png"
            save_flagged_cell_qc(flagged_dir / filename, original, semantic, candidate, args.flagged_qc_max_size)

    # Broader assigned instances that were intentionally not selected by the
    # overlap-representative stage remain visible as review, never safe.
    if broad_instances is not None:
        selected_ids = set(identifiers)
        for identifier in (int(value) for value in np.unique(broad_instances) if int(value) > 0):
            if identifier in selected_ids:
                continue
            candidate = broad_instances == identifier
            review[candidate] = identifier
            skeleton = candidate & (semantic == 1)
            soma = candidate & (semantic == 2)
            rows.append({
                "source_instance_id": identifier,
                "category": "review",
                "soma_components": int(ndi.label(soma, structure=STRUCTURE_8)[1]),
                "soma_pixels": int(soma.sum()),
                "soma_core_radius_px": "",
                "strong_soma_core_count": "",
                "soma_hmax_prominence_px": "",
                "soma_solidity": "",
                "soma_eccentricity": "",
                "soma_area_to_core_disk_ratio": "",
                "soma_holes": "",
                "soma_hole_pixels": "",
                "soma_local_contrast": "",
                "skeleton_pixels": int(skeleton.sum()),
                "skeleton_components": int(ndi.label(skeleton, structure=STRUCTURE_8)[1]),
                "skeleton_touches_soma": "",
                "connected_skeleton_fraction": "",
                "support_fraction_ge_0_5": "",
                "foreign_soma_components_in_crop": "",
                "foreign_soma_pixels_in_crop": "",
                "foreign_instance_ids_in_crop": "",
                "foreign_skeleton_near_pixels": "",
                "crop_x_min": "",
                "crop_y_min": "",
                "crop_x_max_exclusive": "",
                "crop_y_max_exclusive": "",
                "hard_reasons": "",
                "review_reasons": "not_selected_by_overlap_or_assignment",
                "information_notes": "",
                "manual_decision": "",
                "manual_notes": "",
            })

    status = np.zeros(semantic.shape, dtype=np.uint8)
    status[safe > 0] = 1
    status[review > 0] = 2
    status[excluded > 0] = 3
    max_id = max([*identifiers, 0])
    dtype = np.uint16 if max_id <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(args.output_dir / "01_cells_safe_for_evo.tif", safe.astype(dtype))
    tifffile.imwrite(args.output_dir / "02_cells_for_manual_review.tif", review.astype(dtype))
    tifffile.imwrite(args.output_dir / "03_cells_excluded_from_export.tif", excluded.astype(dtype))
    tifffile.imwrite(args.output_dir / "04_cell_quality_status_map.tif", status)
    tifffile.imwrite(args.output_dir / "05_combined_raw_ridge_support.tif", support.astype(np.float32))
    tifffile.imwrite(args.output_dir / "06_soma_core_markers.tif", core_markers.astype(dtype))
    tifffile.imwrite(args.output_dir / "07_foreign_soma_in_proposed_crops.tif", foreign_soma_crop_mask)
    save_qc(args.output_dir / "08_cell_quality_gate_qc.png", original, safe, review, excluded, args.qc_max_size)
    with (args.output_dir / "cell_quality_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["source_instance_id"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "manual_review_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["source_instance_id"])
        writer.writeheader()
        writer.writerows([row for row in rows if row["category"] != "safe"])

    counts = {category: int(sum(row["category"] == category for row in rows)) for category in ("safe", "review", "excluded")}
    summary = {
        "script_version": SCRIPT_VERSION,
        "segmentation_modified": False,
        "semantic": str(args.semantic.resolve()),
        "instances": str(args.instances.resolve()),
        "original": str(args.original.resolve()),
        "review_instances": str(args.review_instances.resolve()) if args.review_instances else None,
        "ridge_evidence": str(args.ridge_evidence.resolve()) if args.ridge_evidence else None,
        "selected_polarity": selected_polarity,
        "counts": counts,
        "source_instance_count": len(identifiers),
        "effective_soma_limits": {
            "min_area_px": effective_min_area,
            "max_area_px": effective_max_area,
            "min_core_radius_px": effective_min_radius,
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "_script_version.txt").write_text(f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n", encoding="utf-8")
    print("\n" + "=" * 72)
    print("SOMA / SINGLE-INSTANCE QUALITY GATE COMPLETE")
    print("=" * 72)
    print(f"Safe: {counts['safe']}; review: {counts['review']}; excluded: {counts['excluded']}")
    print(
        "Policies: "
        f"multi-core={args.multi_core_policy}; "
        f"soma-hole={args.soma_hole_policy}; "
        f"hole review threshold={args.min_soma_hole_pixels_for_review}px"
    )
    print("Semantic segmentation changed: NO")
    print(f"Safe map for Evo export: {args.output_dir / '01_cells_safe_for_evo.tif'}")
    print(f"Flagged crop QC: {flagged_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
