from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.graph import route_through_array
from skimage.morphology import disk, skeletonize


SCRIPT_VERSION = "net129-no-loss-v4-ridge-evidence-safe-topology-2026-08-12"
BACKGROUND_CLASS = 0
SKELETON_CLASS = 1
SOMA_CLASS = 2


@dataclass
class Candidate:
    pass_index: int
    candidate_id: int
    kind: str
    source_endpoint: int
    target_endpoint: int | None
    source_y: int
    source_x: int
    target_y: int
    target_x: int
    source_component: int
    target_component: int
    target_soma: int
    euclidean_distance: float
    straight_probability_mean: float = 0.0
    straight_image_support_mean: float = 0.5
    direction_cosine: float = 0.0
    path_length: float = 0.0
    path_probability_mean: float = 0.0
    path_probability_q10: float = 0.0
    path_image_support_mean: float = 0.5
    normalized_path_cost: float = math.inf
    detour_ratio: float = math.inf
    score: float = -math.inf
    acceptance_mode: str = "none"
    decision: str = "not_evaluated"
    reason: str = ""
    path: tuple[tuple[int, int], ...] = ()

    def csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row.pop("path")
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "No-loss postprocessing for a net-129 nnU-Net prediction. "
            "Original skeleton pixels are never removed."
        )
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Hard overview.tif with labels 0=background, 1=skeleton, 2=soma.",
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        required=True,
        help="overview.npz written by nnU-Net with --save_probabilities.",
    )
    parser.add_argument(
        "--prior-skeleton-material",
        type=Path,
        default=None,
        help=(
            "Optional binary fullwidth skeleton material from a prior no-loss "
            "pass. It is ORed with class-1 input before routing so material is "
            "carried across staged passes even where class 2 takes semantic precedence."
        ),
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=None,
        help="Optional original 2D image/max projection seen by net 129.",
    )
    parser.add_argument(
        "--ridge-evidence",
        type=Path,
        default=None,
        help=(
            "Optional 2-D ridge/Hessian evidence image. High values must mean "
            "strong neurite-like support. It is used only for route scoring and "
            "never changes existing skeleton material."
        ),
    )
    parser.add_argument(
        "--ridge-weight",
        type=float,
        default=0.0,
        help=(
            "Fraction of image support contributed by --ridge-evidence when a raw "
            "--original is also present (default: 0.0, preserving legacy behavior)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--endpoint-gap",
        type=float,
        default=30.0,
        help="Maximum endpoint-to-endpoint distance in pixels (default: 30).",
    )
    parser.add_argument(
        "--soma-gap",
        type=float,
        default=24.0,
        help="Maximum endpoint-to-soma distance in pixels (default: 24).",
    )
    parser.add_argument(
        "--segment-gap",
        type=float,
        default=22.0,
        help="Maximum endpoint-to-nearby-skeleton-segment distance (default: 22).",
    )
    parser.add_argument(
        "--geometry-rescue-gap",
        type=float,
        default=90.0,
        help=(
            "Longer search radius for strongly aligned fragment chains "
            "(default: 90)."
        ),
    )
    parser.add_argument(
        "--geometry-rescue-cosine",
        type=float,
        default=0.78,
        help=(
            "Minimum directional alignment for a geometry-rescued connection "
            "(default: 0.78)."
        ),
    )
    parser.add_argument(
        "--geometry-rescue-min-score",
        type=float,
        default=0.23,
        help="Minimum combined score for geometry rescue (default: 0.23).",
    )
    parser.add_argument(
        "--geometry-rescue-min-support",
        type=float,
        default=0.30,
        help=(
            "Minimum original-image support for geometry rescue when network "
            "probability is weak (default: 0.30)."
        ),
    )
    parser.add_argument(
        "--geometry-rescue-max-detour",
        type=float,
        default=1.55,
        help="Maximum detour ratio for geometry rescue (default: 1.55).",
    )
    parser.add_argument(
        "--disable-geometry-rescue",
        action="store_true",
        help="Disable long, strongly aligned chain completion.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Iterative connection passes. Two is a useful aggressive default.",
    )
    parser.add_argument(
        "--max-candidates-per-type",
        type=int,
        default=2,
        help="Nearest candidates retained per endpoint and connection type.",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=5000,
        help="Safety limit for expensive probability-guided path searches per pass.",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.34,
        help="Minimum combined acceptance score (default: 0.34).",
    )
    parser.add_argument(
        "--min-path-probability",
        type=float,
        default=0.10,
        help="Minimum mean skeleton probability unless image evidence rescues the path.",
    )
    parser.add_argument(
        "--min-image-support",
        type=float,
        default=0.52,
        help="Minimum original-image support that can rescue a low-probability path.",
    )
    parser.add_argument(
        "--min-direction-cosine",
        type=float,
        default=-0.25,
        help=(
            "Minimum endpoint direction cosine. -0.25 is permissive; "
            "larger values require straighter continuation."
        ),
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.045,
        help="Top candidates within this score difference are marked ambiguous.",
    )
    parser.add_argument(
        "--max-detour",
        type=float,
        default=1.85,
        help="Maximum routed-path length divided by straight distance.",
    )
    parser.add_argument(
        "--route-margin",
        type=int,
        default=8,
        help="Extra pixels around the local route-search rectangle.",
    )
    parser.add_argument(
        "--orientation-steps",
        type=int,
        default=8,
        help="Number of inward skeleton steps used to estimate endpoint direction.",
    )
    parser.add_argument(
        "--contact-radius",
        type=int,
        default=3,
        help="Radius used to decide whether skeleton touches a soma.",
    )
    parser.add_argument(
        "--connection-radius",
        type=int,
        default=0,
        help=(
            "Optional dilation radius for newly added paths. "
            "0 keeps connections one pixel wide."
        ),
    )

    parser.add_argument(
        "--probability-weight",
        type=float,
        default=0.72,
        help="Weight of P(skeleton) in the routing cost.",
    )
    parser.add_argument(
        "--image-weight",
        type=float,
        default=0.28,
        help="Weight of original-image evidence in the routing cost.",
    )
    parser.add_argument(
        "--probability-gamma",
        type=float,
        default=1.4,
        help="Exponent applied to 1-P(skeleton) in the routing cost.",
    )
    parser.add_argument(
        "--original-polarity",
        choices=("auto", "bright", "dark"),
        default="auto",
        help="Whether skeleton signal is bright or dark in the original image.",
    )
    parser.add_argument(
        "--qc-max-size",
        type=int,
        default=2400,
        help="Maximum height or width used for QC PNG rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional candidate statistics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory deliberately.",
    )
    return parser.parse_args()


def read_2d(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    array = np.asarray(tifffile.imread(path))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be 2D after squeeze, got {array.shape}: {path}")
    return array


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {path}\n"
            "Use a new directory or pass --overwrite deliberately."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_probabilities(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Probability file not found: {path}")
    with np.load(path) as archive:
        keys = list(archive.keys())
        if "probabilities" in archive:
            probabilities = np.asarray(archive["probabilities"])
        elif "softmax" in archive:
            probabilities = np.asarray(archive["softmax"])
        elif len(keys) == 1:
            probabilities = np.asarray(archive[keys[0]])
        else:
            raise RuntimeError(
                f"Could not identify probabilities in {path.name}; keys={keys}"
            )

    probabilities = np.squeeze(probabilities)
    if probabilities.ndim != 3:
        raise RuntimeError(
            f"Expected a 3D probability array, got {probabilities.shape}"
        )
    if probabilities.shape[1:] == expected_shape:
        chw = probabilities
    elif probabilities.shape[:2] == expected_shape:
        chw = np.moveaxis(probabilities, -1, 0)
    else:
        raise RuntimeError(
            "Probability/image shape mismatch:\n"
            f"probabilities={probabilities.shape}\n"
            f"prediction={expected_shape}"
        )
    if chw.shape[0] < 3:
        raise RuntimeError(
            f"Expected background/skeleton/soma probabilities, got {chw.shape[0]} channels"
        )
    chw = chw.astype(np.float32, copy=False)
    if not np.all(np.isfinite(chw)):
        raise RuntimeError("Probability array contains NaN or infinity.")
    return np.clip(chw, 0.0, 1.0)


def robust_unit_scale(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        raise RuntimeError("Original image contains no finite values.")
    lo, hi = np.percentile(image[finite], [1.0, 99.0])
    if hi <= lo:
        lo = float(np.min(image[finite]))
        hi = float(np.max(image[finite]))
    if hi <= lo:
        return np.full(image.shape, 0.5, dtype=np.float32)
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    scaled[~finite] = 0.0
    return scaled.astype(np.float32)


def build_image_support(
    original: np.ndarray | None,
    original_skeleton: np.ndarray,
    polarity: str,
) -> tuple[np.ndarray, str]:
    if original is None:
        return np.full(original_skeleton.shape, 0.5, dtype=np.float32), "none"

    normalized = robust_unit_scale(original)
    if polarity == "auto":
        skeleton_values = normalized[original_skeleton]
        background_values = normalized[~original_skeleton]
        if skeleton_values.size and background_values.size:
            background_reference = np.median(
                background_values[
                    :: max(1, background_values.size // 200_000)
                ]
            )
            polarity = (
                "bright"
                if float(np.median(skeleton_values)) >= float(background_reference)
                else "dark"
            )
        else:
            polarity = "bright"

    direct = normalized if polarity == "bright" else 1.0 - normalized
    smooth = ndi.gaussian_filter(normalized, sigma=2.0)
    local_contrast = (
        normalized - smooth if polarity == "bright" else smooth - normalized
    )
    contrast_support = np.clip(0.5 + 2.0 * local_contrast, 0.0, 1.0)
    support = np.clip(0.70 * direct + 0.30 * contrast_support, 0.0, 1.0)
    return support.astype(np.float32), polarity


def build_cost_image(
    skeleton_probability: np.ndarray,
    image_support: np.ndarray,
    skeleton_material: np.ndarray,
    soma_mask: np.ndarray,
    probability_weight: float,
    image_weight: float,
    probability_gamma: float,
    original_available: bool,
) -> np.ndarray:
    pw = max(0.0, probability_weight)
    iw = max(0.0, image_weight) if original_available else 0.0
    total = pw + iw
    if total <= 0:
        pw, iw, total = 1.0, 0.0, 1.0
    pw /= total
    iw /= total

    cost = (
        0.025
        + pw * np.power(1.0 - skeleton_probability, probability_gamma)
        + iw * (1.0 - image_support)
    ).astype(np.float32)
    cost[skeleton_material] = np.minimum(cost[skeleton_material], 0.015)
    cost[soma_mask] = np.minimum(cost[soma_mask], 0.02)
    return np.maximum(cost, 1e-4)


def combine_route_support(
    raw_support: np.ndarray,
    ridge_evidence: np.ndarray | None,
    ridge_weight: float,
    raw_available: bool,
) -> np.ndarray:
    """Combine raw-image and ridge evidence without changing either input.

    The ridge map is intentionally treated as positive-only support: high values
    make a path cheaper, but the segmentation itself is never modified here.
    With the default ridge_weight=0 this function is bit-for-bit equivalent to
    the former raw-only workflow.
    """
    if ridge_evidence is None:
        return raw_support.astype(np.float32, copy=False)

    ridge_support = robust_unit_scale(ridge_evidence)
    if not raw_available:
        return ridge_support

    weight = float(np.clip(ridge_weight, 0.0, 1.0))
    return np.clip(
        (1.0 - weight) * raw_support + weight * ridge_support,
        0.0,
        1.0,
    ).astype(np.float32)


def neighborhood_count(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(binary.astype(np.uint8), kernel, mode="constant", cval=0)


def analysis_graph(binary_material: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thin = skeletonize(binary_material)
    counts = neighborhood_count(thin)
    endpoints = np.argwhere(thin & (counts == 1))
    component_labels, _ = ndi.label(thin, structure=np.ones((3, 3), dtype=np.uint8))
    return thin, endpoints.astype(np.int32), component_labels.astype(np.int32)


def trace_endpoint_direction(
    thin: np.ndarray,
    start: tuple[int, int],
    steps: int,
) -> np.ndarray:
    current = start
    previous: tuple[int, int] | None = None
    visited = [start]
    height, width = thin.shape

    for _ in range(max(1, steps)):
        choices: list[tuple[int, int]] = []
        y, x = current
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                point = (ny, nx)
                if thin[ny, nx] and point != previous:
                    choices.append(point)
        if not choices:
            break
        if len(choices) > 1:
            unseen = [point for point in choices if point not in visited]
            if unseen:
                choices = unseen
        next_point = max(
            choices,
            key=lambda p: (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2,
        )
        previous, current = current, next_point
        visited.append(current)

    vector = np.asarray(start, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return vector / norm


def direction_cosine(
    source: tuple[int, int],
    target: tuple[int, int],
    source_direction: np.ndarray,
    target_direction: np.ndarray | None = None,
) -> float:
    delta = np.asarray(target, dtype=np.float32) - np.asarray(source, dtype=np.float32)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return 1.0
    unit = delta / norm
    source_cos = float(np.dot(source_direction, unit))
    if target_direction is None:
        return source_cos
    target_cos = float(np.dot(target_direction, -unit))
    return 0.5 * (source_cos + target_cos)


def sample_line(
    start: tuple[int, int],
    end: tuple[int, int],
    array: np.ndarray,
) -> np.ndarray:
    rr, cc = line(start[0], start[1], end[0], end[1])
    rr = np.clip(rr, 0, array.shape[0] - 1)
    cc = np.clip(cc, 0, array.shape[1] - 1)
    return array[rr, cc]


def component_soma_contacts(
    component_labels: np.ndarray,
    soma_instances: np.ndarray,
    radius: int,
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    component_coords = np.argwhere(component_labels > 0).astype(np.int32)
    boundary_coords, boundary_labels = soma_boundaries(soma_instances)
    if component_coords.size == 0 or boundary_coords.size == 0:
        return result
    tree = cKDTree(component_coords)
    neighbor_lists = tree.query_ball_point(boundary_coords, max(0, radius))
    for soma_id, neighbors in zip(boundary_labels, neighbor_lists):
        if not neighbors:
            continue
        component_ids = np.unique(
            component_labels[
                component_coords[neighbors, 0],
                component_coords[neighbors, 1],
            ]
        )
        for component_id in component_ids:
            if int(component_id) > 0:
                result.setdefault(int(component_id), set()).add(int(soma_id))
    return result


def soma_boundaries(soma_instances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = soma_instances > 0
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary = mask & ~eroded
    coords = np.argwhere(boundary).astype(np.int32)
    labels = soma_instances[boundary].astype(np.int32)
    return coords, labels


def collect_candidates(
    pass_index: int,
    thin: np.ndarray,
    endpoints: np.ndarray,
    component_labels: np.ndarray,
    soma_instances: np.ndarray,
    endpoint_gap: float,
    soma_gap: float,
    segment_gap: float,
    per_type: int,
    orientation_steps: int,
) -> list[Candidate]:
    if endpoints.size == 0:
        return []

    endpoint_points = [tuple(map(int, point)) for point in endpoints]
    directions = [
        trace_endpoint_direction(thin, point, orientation_steps)
        for point in endpoint_points
    ]
    endpoint_components = [
        int(component_labels[point]) for point in endpoint_points
    ]

    def nearest_plus_aligned(
        options: list[tuple],
        key_index: int,
        distance_index: int,
        cosine_index: int,
    ) -> list[tuple]:
        """Keep the nearest option plus the best directional continuations."""
        if not options:
            return []
        nearest = sorted(options, key=lambda item: item[distance_index])
        aligned = sorted(
            options,
            key=lambda item: (-item[cosine_index], item[distance_index]),
        )
        selected: list[tuple] = []
        used_keys: set[object] = set()
        for option in nearest[:1] + aligned:
            option_key = option[key_index]
            if option_key in used_keys:
                continue
            selected.append(option)
            used_keys.add(option_key)
            if len(selected) >= per_type:
                break
        return selected

    candidates: list[Candidate] = []
    candidate_id = 0

    endpoint_tree = cKDTree(endpoints)
    endpoint_neighbors = endpoint_tree.query_ball_point(endpoints, endpoint_gap)
    seen_pairs: set[tuple[int, int]] = set()
    for source_index, neighbor_indices in enumerate(endpoint_neighbors):
        source = endpoint_points[source_index]
        possible = []
        for target_index in neighbor_indices:
            target_index = int(target_index)
            if target_index == source_index:
                continue
            pair = tuple(sorted((source_index, target_index)))
            if pair in seen_pairs:
                continue
            target = endpoint_points[target_index]
            distance = float(np.linalg.norm(endpoints[source_index] - endpoints[target_index]))
            source_component = endpoint_components[source_index]
            target_component = endpoint_components[target_index]
            if source_component == target_component:
                continue
            cosine = direction_cosine(
                source,
                target,
                directions[source_index],
                directions[target_index],
            )
            possible.append((distance, target_index, pair, cosine))
        selected_pairs = nearest_plus_aligned(
            possible,
            key_index=2,
            distance_index=0,
            cosine_index=3,
        )
        for distance, target_index, pair, cosine in selected_pairs:
            seen_pairs.add(pair)
            target = endpoint_points[target_index]
            candidate_id += 1
            candidates.append(
                Candidate(
                    pass_index=pass_index,
                    candidate_id=candidate_id,
                    kind="endpoint_to_endpoint",
                    source_endpoint=source_index,
                    target_endpoint=target_index,
                    source_y=source[0],
                    source_x=source[1],
                    target_y=target[0],
                    target_x=target[1],
                    source_component=endpoint_components[source_index],
                    target_component=endpoint_components[target_index],
                    target_soma=0,
                    euclidean_distance=distance,
                    direction_cosine=cosine,
                )
            )

    boundary_coords, boundary_labels = soma_boundaries(soma_instances)
    if boundary_coords.size:
        soma_tree = cKDTree(boundary_coords)
        neighbor_lists = soma_tree.query_ball_point(endpoints, soma_gap)
        for source_index, source in enumerate(endpoint_points):
            by_soma: dict[int, tuple[float, int]] = {}
            for boundary_index in neighbor_lists[source_index]:
                distance = float(
                    np.linalg.norm(
                        endpoints[source_index] - boundary_coords[int(boundary_index)]
                    )
                )
                soma_id = int(boundary_labels[int(boundary_index)])
                old = by_soma.get(soma_id)
                if old is None or distance < old[0]:
                    by_soma[soma_id] = (distance, int(boundary_index))
            soma_options = []
            for soma_id, (distance, boundary_index) in by_soma.items():
                target = tuple(map(int, boundary_coords[boundary_index]))
                cosine = direction_cosine(
                    source,
                    target,
                    directions[source_index],
                )
                soma_options.append(
                    (soma_id, distance, boundary_index, cosine)
                )
            selected_somata = nearest_plus_aligned(
                soma_options,
                key_index=0,
                distance_index=1,
                cosine_index=3,
            )
            for soma_id, distance, boundary_index, cosine in selected_somata:
                target = tuple(map(int, boundary_coords[boundary_index]))
                if distance < 1.0:
                    continue
                candidate_id += 1
                candidates.append(
                    Candidate(
                        pass_index=pass_index,
                        candidate_id=candidate_id,
                        kind="endpoint_to_soma",
                        source_endpoint=source_index,
                        target_endpoint=None,
                        source_y=source[0],
                        source_x=source[1],
                        target_y=target[0],
                        target_x=target[1],
                        source_component=endpoint_components[source_index],
                        target_component=0,
                        target_soma=soma_id,
                        euclidean_distance=distance,
                        direction_cosine=cosine,
                    )
                )

    skeleton_coords = np.argwhere(thin).astype(np.int32)
    if skeleton_coords.size:
        skeleton_components = component_labels[thin].astype(np.int32)
        segment_tree = cKDTree(skeleton_coords)
        neighbor_lists = segment_tree.query_ball_point(endpoints, segment_gap)
        for source_index, source in enumerate(endpoint_points):
            source_component = endpoint_components[source_index]
            by_component: dict[int, tuple[float, int]] = {}
            for skeleton_index in neighbor_lists[source_index]:
                distance = float(
                    np.linalg.norm(
                        endpoints[source_index] - skeleton_coords[int(skeleton_index)]
                    )
                )
                target_component = int(skeleton_components[int(skeleton_index)])
                if target_component == source_component or target_component == 0:
                    continue
                old = by_component.get(target_component)
                if old is None or distance < old[0]:
                    by_component[target_component] = (
                        distance,
                        int(skeleton_index),
                    )
            segment_options = []
            for target_component, (distance, skeleton_index) in by_component.items():
                target = tuple(map(int, skeleton_coords[skeleton_index]))
                cosine = direction_cosine(
                    source,
                    target,
                    directions[source_index],
                )
                segment_options.append(
                    (target_component, distance, skeleton_index, cosine)
                )
            selected_segments = nearest_plus_aligned(
                segment_options,
                key_index=0,
                distance_index=1,
                cosine_index=3,
            )
            for (
                target_component,
                distance,
                skeleton_index,
                cosine,
            ) in selected_segments:
                target = tuple(map(int, skeleton_coords[skeleton_index]))
                if distance < 1.0:
                    continue
                candidate_id += 1
                candidates.append(
                    Candidate(
                        pass_index=pass_index,
                        candidate_id=candidate_id,
                        kind="endpoint_to_segment",
                        source_endpoint=source_index,
                        target_endpoint=None,
                        source_y=source[0],
                        source_x=source[1],
                        target_y=target[0],
                        target_x=target[1],
                        source_component=source_component,
                        target_component=target_component,
                        target_soma=0,
                        euclidean_distance=distance,
                        direction_cosine=cosine,
                    )
                )

    return candidates


def route_candidate(
    candidate: Candidate,
    cost_image: np.ndarray,
    skeleton_probability: np.ndarray,
    image_support: np.ndarray,
    route_margin: int,
) -> None:
    start = (candidate.source_y, candidate.source_x)
    end = (candidate.target_y, candidate.target_x)
    height, width = cost_image.shape
    y0 = max(0, min(start[0], end[0]) - route_margin)
    y1 = min(height, max(start[0], end[0]) + route_margin + 1)
    x0 = max(0, min(start[1], end[1]) - route_margin)
    x1 = min(width, max(start[1], end[1]) + route_margin + 1)
    local_start = (start[0] - y0, start[1] - x0)
    local_end = (end[0] - y0, end[1] - x0)

    local_path, total_cost = route_through_array(
        cost_image[y0:y1, x0:x1],
        local_start,
        local_end,
        fully_connected=True,
        geometric=True,
    )
    path = tuple((int(y + y0), int(x + x0)) for y, x in local_path)
    if not path:
        candidate.decision = "rejected"
        candidate.reason = "empty_route"
        return

    rr = np.fromiter((point[0] for point in path), dtype=np.int32)
    cc = np.fromiter((point[1] for point in path), dtype=np.int32)
    probabilities = skeleton_probability[rr, cc]
    supports = image_support[rr, cc]

    step_lengths = np.hypot(np.diff(rr), np.diff(cc))
    path_length = float(step_lengths.sum()) if len(path) > 1 else 0.0
    denominator = max(candidate.euclidean_distance, 1.0)
    candidate.path = path
    candidate.path_length = path_length
    candidate.path_probability_mean = float(np.mean(probabilities))
    candidate.path_probability_q10 = float(np.quantile(probabilities, 0.10))
    candidate.path_image_support_mean = float(np.mean(supports))
    candidate.normalized_path_cost = float(total_cost / max(path_length, 1.0))
    candidate.detour_ratio = path_length / denominator

    cost_score = math.exp(-1.25 * candidate.normalized_path_cost)
    length_scale = {
        "endpoint_to_endpoint": 30.0,
        "endpoint_to_soma": 24.0,
        "endpoint_to_segment": 22.0,
    }[candidate.kind]
    length_score = math.exp(-candidate.euclidean_distance / max(length_scale, 1.0))
    direction_score = float(np.clip((candidate.direction_cosine + 1.0) / 2.0, 0.0, 1.0))
    candidate.score = (
        0.40 * candidate.path_probability_mean
        + 0.10 * candidate.path_probability_q10
        + 0.20 * candidate.path_image_support_mean
        + 0.12 * direction_score
        + 0.10 * cost_score
        + 0.08 * length_score
    )


def evaluate_candidates(
    candidates: list[Candidate],
    cost_image: np.ndarray,
    skeleton_probability: np.ndarray,
    image_support: np.ndarray,
    original_available: bool,
    min_score: float,
    min_path_probability: float,
    min_image_support: float,
    min_direction_cosine: float,
    max_detour: float,
    route_margin: int,
    max_routes: int,
    endpoint_gap: float,
    soma_gap: float,
    segment_gap: float,
    geometry_rescue_enabled: bool,
    geometry_rescue_gap: float,
    geometry_rescue_cosine: float,
    geometry_rescue_min_score: float,
    geometry_rescue_min_support: float,
    geometry_rescue_max_detour: float,
) -> None:
    normal_gaps = {
        "endpoint_to_endpoint": endpoint_gap,
        "endpoint_to_soma": soma_gap,
        "endpoint_to_segment": segment_gap,
    }
    ranked: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        start = (candidate.source_y, candidate.source_x)
        end = (candidate.target_y, candidate.target_x)
        line_probability = sample_line(start, end, skeleton_probability)
        line_support = sample_line(start, end, image_support)
        candidate.straight_probability_mean = float(np.mean(line_probability))
        candidate.straight_image_support_mean = float(np.mean(line_support))
        direction_score = float(
            np.clip((candidate.direction_cosine + 1.0) / 2.0, 0.0, 1.0)
        )
        cheap_score = (
            0.58 * candidate.straight_probability_mean
            + 0.25 * candidate.straight_image_support_mean
            + 0.17 * direction_score
        )
        ranked.append((cheap_score, candidate))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    for route_index, (cheap_score, candidate) in enumerate(ranked):
        if route_index >= max_routes:
            candidate.decision = "rejected"
            candidate.reason = "max_routes_limit"
            continue
        if candidate.direction_cosine < min_direction_cosine:
            candidate.decision = "rejected"
            candidate.reason = "direction"
            continue
        possible_geometry_rescue = (
            geometry_rescue_enabled
            and candidate.euclidean_distance <= geometry_rescue_gap
            and candidate.direction_cosine >= geometry_rescue_cosine
        )
        if cheap_score < 0.16 and not possible_geometry_rescue:
            candidate.decision = "rejected"
            candidate.reason = "weak_straight_evidence"
            continue

        route_candidate(
            candidate,
            cost_image,
            skeleton_probability,
            image_support,
            route_margin,
        )
        if candidate.decision == "rejected":
            continue
        evidence_ok = candidate.path_probability_mean >= min_path_probability
        if original_available:
            evidence_ok = evidence_ok or (
                candidate.path_image_support_mean >= min_image_support
                and candidate.path_probability_q10 >= 0.015
            )
        geometry_evidence_ok = (
            candidate.path_probability_mean >= 0.02
            or (
                original_available
                and candidate.path_image_support_mean
                >= geometry_rescue_min_support
            )
        )
        geometry_rescue_ok = (
            possible_geometry_rescue
            and geometry_evidence_ok
            and candidate.detour_ratio <= geometry_rescue_max_detour
            and candidate.score >= geometry_rescue_min_score
        )
        normal_distance_ok = (
            candidate.euclidean_distance <= normal_gaps[candidate.kind]
        )

        if geometry_rescue_ok:
            candidate.acceptance_mode = "geometry_rescue"
            candidate.decision = "viable"
            candidate.reason = "strong_directional_chain"
        elif not normal_distance_ok:
            candidate.decision = "rejected"
            candidate.reason = "beyond_normal_gap"
        elif not evidence_ok:
            candidate.decision = "rejected"
            candidate.reason = "insufficient_path_evidence"
        elif candidate.detour_ratio > max_detour:
            candidate.decision = "rejected"
            candidate.reason = "detour"
        elif candidate.score < min_score:
            candidate.decision = "rejected"
            candidate.reason = "score"
        else:
            candidate.acceptance_mode = "normal"
            candidate.decision = "viable"
            candidate.reason = "passes_thresholds"


def proposal_soma_union(
    candidate: Candidate,
    component_contacts: dict[int, set[int]],
    soma_instances: np.ndarray,
    contact_radius: int,
) -> set[int]:
    soma_ids = set(component_contacts.get(candidate.source_component, set()))
    if candidate.target_component > 0:
        soma_ids.update(component_contacts.get(candidate.target_component, set()))
    if candidate.target_soma > 0:
        soma_ids.add(candidate.target_soma)
    if candidate.path:
        path_mask = np.zeros(soma_instances.shape, dtype=bool)
        rr = np.fromiter((point[0] for point in candidate.path), dtype=np.int32)
        cc = np.fromiter((point[1] for point in candidate.path), dtype=np.int32)
        path_mask[rr, cc] = True
        if contact_radius > 0:
            path_mask = ndi.binary_dilation(
                path_mask,
                structure=disk(contact_radius),
            )
        soma_ids.update(
            int(value)
            for value in np.unique(soma_instances[path_mask])
            if int(value) > 0
        )
    return soma_ids


def choose_connections(
    candidates: list[Candidate],
    component_contacts: dict[int, set[int]],
    soma_instances: np.ndarray,
    contact_radius: int,
    ambiguity_margin: float,
) -> tuple[list[Candidate], list[Candidate], set[int]]:
    viable = [candidate for candidate in candidates if candidate.decision == "viable"]
    forced_ambiguous_components: set[int] = set()

    for candidate in viable:
        soma_union = proposal_soma_union(
            candidate,
            component_contacts,
            soma_instances,
            contact_radius,
        )
        if len(soma_union) > 1:
            candidate.decision = "ambiguous"
            candidate.reason = "would_join_multiple_somata"
            forced_ambiguous_components.add(candidate.source_component)
            if candidate.target_component > 0:
                forced_ambiguous_components.add(candidate.target_component)

    viable = [candidate for candidate in viable if candidate.decision == "viable"]
    def relative_target_key(
        candidate: Candidate,
        endpoint_id: int,
    ) -> tuple[str, int]:
        """Return the biological target seen from one endpoint.

        Endpoint-to-endpoint and endpoint-to-segment proposals that reach the
        same skeleton component are equivalent alternatives, not competitors.
        """
        if endpoint_id == candidate.source_endpoint:
            if candidate.target_soma > 0:
                return ("soma", candidate.target_soma)
            if candidate.target_component > 0:
                return ("component", candidate.target_component)
        if (
            candidate.target_endpoint is not None
            and endpoint_id == candidate.target_endpoint
        ):
            return ("component", candidate.source_component)
        return ("candidate", candidate.candidate_id)

    by_endpoint: dict[int, list[Candidate]] = {}
    for candidate in viable:
        by_endpoint.setdefault(candidate.source_endpoint, []).append(candidate)
        if candidate.target_endpoint is not None:
            by_endpoint.setdefault(candidate.target_endpoint, []).append(candidate)

    winning_target: dict[int, tuple[str, int]] = {}
    ambiguous_ids: set[int] = set()
    for endpoint_id, endpoint_candidates in by_endpoint.items():
        grouped: dict[tuple[str, int], list[Candidate]] = {}
        for candidate in endpoint_candidates:
            grouped.setdefault(
                relative_target_key(candidate, endpoint_id),
                [],
            ).append(candidate)
        ranked_groups = sorted(
            (
                (max(candidate.score for candidate in group), key, group)
                for key, group in grouped.items()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        top_score, top_key, _ = ranked_groups[0]
        competing_groups = [
            group
            for score, _, group in ranked_groups
            if top_score - score <= ambiguity_margin
        ]
        if len(competing_groups) > 1:
            for candidate in [
                item for group in competing_groups for item in group
            ]:
                ambiguous_ids.add(id(candidate))
            continue
        winning_target[endpoint_id] = top_key

    accepted: list[Candidate] = []
    used_endpoints: set[int] = set()
    for candidate in sorted(viable, key=lambda c: c.score, reverse=True):
        if id(candidate) in ambiguous_ids:
            candidate.decision = "ambiguous"
            candidate.reason = "competing_targets"
            continue
        if candidate.source_endpoint in used_endpoints:
            candidate.decision = "rejected"
            candidate.reason = "source_endpoint_already_used"
            continue
        if winning_target.get(candidate.source_endpoint) != relative_target_key(
            candidate,
            candidate.source_endpoint,
        ):
            candidate.decision = "rejected"
            candidate.reason = "not_best_for_source"
            continue
        if candidate.target_endpoint is not None:
            if candidate.target_endpoint in used_endpoints:
                candidate.decision = "rejected"
                candidate.reason = "target_endpoint_already_used"
                continue
            if winning_target.get(candidate.target_endpoint) != relative_target_key(
                candidate,
                candidate.target_endpoint,
            ):
                candidate.decision = "ambiguous"
                candidate.reason = "not_mutual_endpoint_choice"
                ambiguous_ids.add(id(candidate))
                continue
        candidate.decision = "accepted"
        candidate.reason = "best_noncompeting_candidate"
        accepted.append(candidate)
        used_endpoints.add(candidate.source_endpoint)
        if candidate.target_endpoint is not None:
            used_endpoints.add(candidate.target_endpoint)

    ambiguous = [
        candidate for candidate in candidates if candidate.decision == "ambiguous"
    ]
    return accepted, ambiguous, forced_ambiguous_components


def paths_to_mask(
    shape: tuple[int, int],
    candidates: Iterable[Candidate],
    radius: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        if not candidate.path:
            continue
        rr = np.fromiter((point[0] for point in candidate.path), dtype=np.int32)
        cc = np.fromiter((point[1] for point in candidate.path), dtype=np.int32)
        mask[rr, cc] = True
    if radius > 0 and mask.any():
        mask = ndi.binary_dilation(mask, structure=disk(radius))
    return mask


def analyze_assignments(
    completed_material: np.ndarray,
    soma_instances: np.ndarray,
    contact_radius: int,
    ambiguity_seed: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    material_labels, number_components = ndi.label(
        completed_material,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    contacts = component_soma_contacts(material_labels, soma_instances, contact_radius)

    confident = np.zeros(completed_material.shape, dtype=np.uint32)
    ambiguous = np.zeros(completed_material.shape, dtype=np.uint32)
    unassigned = np.zeros(completed_material.shape, dtype=np.uint32)
    status = np.zeros(completed_material.shape, dtype=np.uint8)
    report: list[dict[str, object]] = []
    ambiguous_counter = 0
    unassigned_counter = 0

    forced_labels = set(
        int(value)
        for value in np.unique(material_labels[ambiguity_seed])
        if int(value) > 0
    )

    component_soma_ids: dict[int, list[int]] = {
        component_id: sorted(contacts.get(component_id, set()))
        for component_id in range(1, number_components + 1)
    }
    ambiguous_soma_ids: set[int] = set()
    for component_id, soma_ids in component_soma_ids.items():
        if len(soma_ids) > 1 or component_id in forced_labels:
            ambiguous_soma_ids.update(soma_ids)

    for component_id in range(1, number_components + 1):
        component_mask = material_labels == component_id
        soma_ids = component_soma_ids[component_id]
        forced = component_id in forced_labels
        shares_ambiguous_soma = bool(set(soma_ids) & ambiguous_soma_ids)

        if len(soma_ids) == 1 and not forced and not shares_ambiguous_soma:
            soma_id = soma_ids[0]
            confident[component_mask] = soma_id
            confident[soma_instances == soma_id] = soma_id
            status[component_mask] = 1
            status[soma_instances == soma_id] = 1
            assignment = "confident"
            output_id = soma_id
        elif len(soma_ids) > 1 or forced or shares_ambiguous_soma:
            ambiguous_counter += 1
            ambiguous[component_mask] = ambiguous_counter
            for soma_id in soma_ids:
                ambiguous[soma_instances == soma_id] = ambiguous_counter
            status[component_mask] = 2
            for soma_id in soma_ids:
                status[soma_instances == soma_id] = 2
            assignment = "ambiguous"
            output_id = ambiguous_counter
        else:
            unassigned_counter += 1
            unassigned[component_mask] = unassigned_counter
            status[component_mask] = 3
            assignment = "unassigned"
            output_id = unassigned_counter

        report.append(
            {
                "skeleton_component": component_id,
                "assignment": assignment,
                "output_instance_id": output_id,
                "touching_soma_ids": ";".join(str(value) for value in soma_ids),
                "number_touching_somata": len(soma_ids),
                "forced_ambiguous_by_competing_connection": int(forced),
                "shares_soma_with_ambiguous_component": int(shares_ambiguous_soma),
                "material_pixels": int(component_mask.sum()),
            }
        )

    return (
        material_labels.astype(np.uint32),
        confident,
        ambiguous,
        unassigned,
        status,
        report,
    )


def save_tiff(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(path, array, photometric="minisblack")


def downsample_for_qc(arrays: Sequence[np.ndarray], max_size: int) -> list[np.ndarray]:
    height, width = arrays[0].shape[:2]
    step = max(1, int(math.ceil(max(height, width) / max_size)))
    return [array[::step, ::step, ...] for array in arrays]


def base_rgb(original: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if original is None:
        gray = np.zeros(shape, dtype=np.float32)
    else:
        gray = robust_unit_scale(original)
    return np.repeat(gray[..., None], 3, axis=2)


def blend(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    if not mask.any():
        return
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.asarray(color)


def save_qc_overlay(
    path: Path,
    original: np.ndarray | None,
    original_skeleton: np.ndarray,
    accepted: np.ndarray,
    ambiguous_proposals: np.ndarray,
    soma_mask: np.ndarray,
    status: np.ndarray,
    max_size: int,
) -> None:
    arrays = [
        base_rgb(original, original_skeleton.shape),
        original_skeleton,
        accepted,
        ambiguous_proposals,
        soma_mask,
        status,
    ]
    rgb_base, raw, added, proposals, soma, status_small = downsample_for_qc(
        arrays, max_size
    )

    material_rgb = rgb_base.copy()
    blend(material_rgb, raw.astype(bool), (0.95, 0.95, 0.95), 0.82)
    blend(material_rgb, soma.astype(bool), (1.0, 0.0, 1.0), 0.65)
    blend(material_rgb, proposals.astype(bool), (0.20, 0.45, 1.0), 0.85)
    blend(material_rgb, added.astype(bool), (1.0, 0.95, 0.0), 0.95)

    assignment_rgb = rgb_base.copy()
    blend(assignment_rgb, status_small == 1, (0.05, 1.0, 0.15), 0.82)
    blend(assignment_rgb, status_small == 2, (1.0, 0.48, 0.0), 0.90)
    blend(assignment_rgb, status_small == 3, (0.0, 0.9, 1.0), 0.90)
    blend(assignment_rgb, soma.astype(bool), (1.0, 0.0, 1.0), 0.48)
    blend(assignment_rgb, added.astype(bool), (1.0, 0.95, 0.0), 0.92)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    axes[0].imshow(np.clip(material_rgb, 0.0, 1.0))
    axes[0].set_title(
        "Material: original=white, added=yellow, competing=blue, soma=magenta"
    )
    axes[1].imshow(np.clip(assignment_rgb, 0.0, 1.0))
    axes[1].set_title(
        "Assignment: confident=green, ambiguous=orange, unassigned=cyan"
    )
    for axis in axes:
        axis.axis("off")
    fig.savefig(path, dpi=180, facecolor="black")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "endpoint_gap",
        "soma_gap",
        "segment_gap",
        "geometry_rescue_gap",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.passes < 1:
        raise ValueError("--passes must be >= 1")
    if args.max_candidates_per_type < 1:
        raise ValueError("--max-candidates-per-type must be >= 1")
    if args.max_routes < 1:
        raise ValueError("--max-routes must be >= 1")
    if not 0.0 <= args.ambiguity_margin <= 1.0:
        raise ValueError("--ambiguity-margin must be between 0 and 1")
    if not 0.0 <= args.ridge_weight <= 1.0:
        raise ValueError("--ridge-weight must be between 0 and 1")
    if not -1.0 <= args.geometry_rescue_cosine <= 1.0:
        raise ValueError("--geometry-rescue-cosine must be between -1 and 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    print(f"POSTPROCESSING VERSION: {SCRIPT_VERSION}")
    print(f"RUNNING FILE: {Path(__file__).resolve()}")

    prepare_output(args.output_dir, args.overwrite)
    prediction = read_2d(args.prediction, "prediction")
    values = set(np.unique(prediction).astype(int).tolist())
    invalid = values - {BACKGROUND_CLASS, SKELETON_CLASS, SOMA_CLASS}
    if invalid:
        raise RuntimeError(
            f"Prediction contains unexpected values {sorted(invalid)}; expected 0/1/2."
        )

    semantic_input_skeleton = prediction == SKELETON_CLASS
    prior_skeleton_material = np.zeros(prediction.shape, dtype=bool)
    if args.prior_skeleton_material is not None:
        prior_array = read_2d(
            args.prior_skeleton_material,
            "prior skeleton material",
        )
        if prior_array.shape != prediction.shape:
            raise RuntimeError(
                "Prior-material/prediction shape mismatch: "
                f"{prior_array.shape} vs {prediction.shape}"
            )
        prior_skeleton_material = prior_array > 0
    # The binary material map is deliberately carried independently from the
    # semantic 0/1/2 map. A skeleton pixel can lie under a soma in the binary
    # topology map, while the final semantic image correctly shows class 2.
    original_skeleton = semantic_input_skeleton | prior_skeleton_material
    soma_mask = prediction == SOMA_CLASS
    probabilities = load_probabilities(args.probabilities, prediction.shape)
    skeleton_probability = probabilities[SKELETON_CLASS]

    original = None
    if args.original is not None:
        original = read_2d(args.original, "original image")
        if original.shape != prediction.shape:
            raise RuntimeError(
                f"Original/prediction shape mismatch: {original.shape} vs {prediction.shape}"
            )

    ridge_evidence = None
    if args.ridge_evidence is not None:
        ridge_evidence = read_2d(args.ridge_evidence, "ridge evidence")
        if ridge_evidence.shape != prediction.shape:
            raise RuntimeError(
                "Ridge-evidence/prediction shape mismatch: "
                f"{ridge_evidence.shape} vs {prediction.shape}"
            )

    raw_image_support, selected_polarity = build_image_support(
        original,
        original_skeleton,
        args.original_polarity,
    )
    image_support = combine_route_support(
        raw_image_support,
        ridge_evidence,
        args.ridge_weight,
        original is not None,
    )
    image_support_available = original is not None or ridge_evidence is not None
    soma_instances, soma_count = ndi.label(
        soma_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    soma_instances = soma_instances.astype(np.uint32)

    print(f"Image shape: {prediction.shape}")
    print(f"Semantic input skeleton pixels: {int(semantic_input_skeleton.sum())}")
    print(f"Preserved input material pixels: {int(original_skeleton.sum())}")
    print(f"Soma instances: {soma_count}")
    print(f"Original-image polarity: {selected_polarity}")
    if ridge_evidence is not None:
        print(f"Ridge evidence: enabled (weight={args.ridge_weight:.2f})")

    completed_material = original_skeleton.copy()
    added_connections = np.zeros(prediction.shape, dtype=bool)
    ambiguous_proposals = np.zeros(prediction.shape, dtype=bool)
    ambiguity_seed = np.zeros(prediction.shape, dtype=bool)
    all_candidates: list[Candidate] = []

    for pass_index in range(1, args.passes + 1):
        thin, endpoints, component_labels = analysis_graph(completed_material)
        contacts = component_soma_contacts(
            component_labels,
            soma_instances,
            args.contact_radius,
        )
        cost_image = build_cost_image(
            skeleton_probability,
            image_support,
            completed_material,
            soma_mask,
            args.probability_weight,
            args.image_weight,
            args.probability_gamma,
            image_support_available,
        )
        rescue_search_gap = (
            0.0 if args.disable_geometry_rescue else args.geometry_rescue_gap
        )
        candidates = collect_candidates(
            pass_index,
            thin,
            endpoints,
            component_labels,
            soma_instances,
            max(args.endpoint_gap, rescue_search_gap),
            max(args.soma_gap, rescue_search_gap),
            max(args.segment_gap, rescue_search_gap),
            args.max_candidates_per_type,
            args.orientation_steps,
        )
        evaluate_candidates(
            candidates,
            cost_image,
            skeleton_probability,
            image_support,
            image_support_available,
            args.min_score,
            args.min_path_probability,
            args.min_image_support,
            args.min_direction_cosine,
            args.max_detour,
            args.route_margin,
            args.max_routes,
            args.endpoint_gap,
            args.soma_gap,
            args.segment_gap,
            not args.disable_geometry_rescue,
            args.geometry_rescue_gap,
            args.geometry_rescue_cosine,
            args.geometry_rescue_min_score,
            args.geometry_rescue_min_support,
            args.geometry_rescue_max_detour,
        )
        accepted, ambiguous, forced_components = choose_connections(
            candidates,
            contacts,
            soma_instances,
            args.contact_radius,
            args.ambiguity_margin,
        )
        accepted_mask = paths_to_mask(
            prediction.shape,
            accepted,
            args.connection_radius,
        )
        ambiguous_mask = paths_to_mask(prediction.shape, ambiguous, 0)

        if forced_components:
            ambiguity_seed |= np.isin(component_labels, list(forced_components))
        # Unused competing proposals remain visible in the QC output but do
        # not demote an otherwise single-soma cell. Only a real potential
        # multi-soma identity conflict enters ambiguity_seed.
        ambiguous_proposals |= ambiguous_mask
        new_pixels = accepted_mask & ~completed_material
        added_connections |= new_pixels
        completed_material |= accepted_mask
        all_candidates.extend(candidates)

        print(
            f"Pass {pass_index}: endpoints={len(endpoints)}, "
            f"candidates={len(candidates)}, accepted={len(accepted)}, "
            f"ambiguous={len(ambiguous)}, new_pixels={int(new_pixels.sum())}"
        )
        if args.verbose:
            decisions: dict[str, int] = {}
            for candidate in candidates:
                decisions[candidate.decision] = decisions.get(candidate.decision, 0) + 1
            print(f"  Decisions: {decisions}")
        if not new_pixels.any():
            print("  No new accepted pixels; stopping iterative search.")
            break

    # This invariant is the central no-loss guarantee.
    completed_material = original_skeleton | added_connections
    if not np.all(completed_material[original_skeleton]):
        raise AssertionError("Internal error: an original skeleton pixel was removed.")

    final_analysis_skeleton = skeletonize(completed_material)
    geometry_rescue_connections = paths_to_mask(
        prediction.shape,
        (
            candidate
            for candidate in all_candidates
            if candidate.decision == "accepted"
            and candidate.acceptance_mode == "geometry_rescue"
        ),
        args.connection_radius,
    )
    geometry_rescue_connections &= ~original_skeleton
    (
        material_component_labels,
        confident_instances,
        ambiguous_instances,
        unassigned_instances,
        status_map,
        assignment_report,
    ) = analyze_assignments(
        completed_material,
        soma_instances,
        args.contact_radius,
        ambiguity_seed,
    )

    save_tiff(
        args.output_dir / "01_original_skeleton_fullwidth.tif",
        original_skeleton.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "02_added_connections_accepted.tif",
        added_connections.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "02b_geometry_rescue_connections.tif",
        geometry_rescue_connections.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "03_competing_connections_ambiguous.tif",
        ambiguous_proposals.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "04_completed_skeleton_fullwidth_NO_LOSS.tif",
        completed_material.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "05_analysis_skeleton_1px_NOT_FOR_MATERIAL.tif",
        final_analysis_skeleton.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "06_soma_mask_original.tif",
        soma_mask.astype(np.uint8),
    )
    save_tiff(args.output_dir / "07_soma_instances.tif", soma_instances)
    save_tiff(
        args.output_dir / "08_skeleton_component_instances_fullwidth.tif",
        material_component_labels,
    )
    save_tiff(
        args.output_dir / "09_confident_cell_instances.tif",
        confident_instances,
    )
    save_tiff(
        args.output_dir / "10_ambiguous_cell_instances.tif",
        ambiguous_instances,
    )
    save_tiff(
        args.output_dir / "11_unassigned_skeleton_instances.tif",
        unassigned_instances,
    )
    save_tiff(args.output_dir / "12_cell_status_map.tif", status_map)
    save_tiff(
        args.output_dir / "13_ambiguity_seed_map.tif",
        ambiguity_seed.astype(np.uint8),
    )
    completed_skeleton_and_soma = np.zeros(prediction.shape, dtype=np.uint8)
    completed_skeleton_and_soma[completed_material] = SKELETON_CLASS
    # Soma takes precedence if a newly routed connection reaches or crosses it.
    completed_skeleton_and_soma[soma_mask] = SOMA_CLASS
    save_tiff(
        args.output_dir / "15_completed_skeleton_and_soma_0-1-2.tif",
        completed_skeleton_and_soma,
    )

    save_qc_overlay(
        args.output_dir / "14_qc_overlay.png",
        original,
        original_skeleton,
        added_connections,
        ambiguous_proposals,
        soma_mask,
        status_map,
        args.qc_max_size,
    )
    write_csv(
        args.output_dir / "connection_candidates.csv",
        [candidate.csv_row() for candidate in all_candidates],
    )
    write_csv(
        args.output_dir / "cell_assignment_report.csv",
        assignment_report,
    )

    accepted_count = sum(
        candidate.decision == "accepted" for candidate in all_candidates
    )
    ambiguous_count = sum(
        candidate.decision == "ambiguous" for candidate in all_candidates
    )
    geometry_rescue_count = sum(
        candidate.decision == "accepted"
        and candidate.acceptance_mode == "geometry_rescue"
        for candidate in all_candidates
    )
    summary = {
        "script_version": SCRIPT_VERSION,
        "prediction": str(args.prediction.resolve()),
        "probabilities": str(args.probabilities.resolve()),
        "prior_skeleton_material": (
            str(args.prior_skeleton_material.resolve())
            if args.prior_skeleton_material else None
        ),
        "original": str(args.original.resolve()) if args.original else None,
        "ridge_evidence": (
            str(args.ridge_evidence.resolve()) if args.ridge_evidence else None
        ),
        "ridge_weight": float(args.ridge_weight),
        "shape": list(prediction.shape),
        "semantic_input_skeleton_pixels": int(semantic_input_skeleton.sum()),
        "prior_skeleton_material_pixels": int(prior_skeleton_material.sum()),
        "original_skeleton_pixels": int(original_skeleton.sum()),
        "added_connection_pixels": int(added_connections.sum()),
        "completed_skeleton_pixels": int(completed_material.sum()),
        "removed_original_skeleton_pixels": int(
            np.sum(original_skeleton & ~completed_material)
        ),
        "accepted_connections": int(accepted_count),
        "geometry_rescued_connections": int(geometry_rescue_count),
        "geometry_rescued_pixels": int(geometry_rescue_connections.sum()),
        "ambiguous_connection_candidates": int(ambiguous_count),
        "soma_instances": int(soma_count),
        "confident_skeleton_pixels": int(np.sum(status_map == 1)),
        "ambiguous_skeleton_pixels": int(np.sum(status_map == 2)),
        "unassigned_skeleton_pixels": int(np.sum(status_map == 3)),
        "selected_original_polarity": selected_polarity,
        "parameters": vars(args) | {
            "prediction": str(args.prediction),
            "probabilities": str(args.probabilities),
            "prior_skeleton_material": (
                str(args.prior_skeleton_material)
                if args.prior_skeleton_material else None
            ),
            "original": str(args.original) if args.original else None,
            "ridge_evidence": (
                str(args.ridge_evidence) if args.ridge_evidence else None
            ),
            "output_dir": str(args.output_dir),
        },
    }
    with (args.output_dir / "run_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    (args.output_dir / "_script_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("POSTPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Preserved input material pixels: {summary['original_skeleton_pixels']}")
    print(f"Added connection pixels:   {summary['added_connection_pixels']}")
    print(f"Completed skeleton pixels: {summary['completed_skeleton_pixels']}")
    print(f"REMOVED original pixels:   {summary['removed_original_skeleton_pixels']}")
    print(f"Accepted connections:      {accepted_count}")
    print(f"Geometry-rescued:          {geometry_rescue_count}")
    print(f"Ambiguous candidates:      {ambiguous_count}")
    print()
    print("Final material mask:")
    print("  04_completed_skeleton_fullwidth_NO_LOSS.tif")
    print("Combined semantic mask:")
    print("  15_completed_skeleton_and_soma_0-1-2.tif")
    print("QC:")
    print("  14_qc_overlay.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
