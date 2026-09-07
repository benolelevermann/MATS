from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.measure import block_reduce
from skimage.morphology import skeletonize


VERSION = "stage5-image-guided-endpoint-paths-v1-2026-07-30"
Point = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "No-loss Stage 5: sucht bildgefuehrte Minimum-Cost-Pfade "
            "ausschliesslich zwischen Skeleton-Endpunkten."
        )
    )
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--original", type=Path)

    parser.add_argument("--min-gap", type=float, default=20.0)
    parser.add_argument("--max-gap", type=float, default=75.0)
    parser.add_argument("--roi-margin", type=int, default=24)
    parser.add_argument("--max-neighbors-per-endpoint", type=int, default=6)
    parser.add_argument("--orientation-steps", type=int, default=8)
    parser.add_argument("--min-chord-direction-cos", type=float, default=-0.65)
    parser.add_argument("--min-path-direction-cos", type=float, default=0.05)
    parser.add_argument("--path-direction-steps", type=int, default=8)
    parser.add_argument("--max-path-ratio", type=float, default=1.80)

    parser.add_argument("--image-weight", type=float, default=0.55)
    parser.add_argument("--probability-weight", type=float, default=0.45)
    parser.add_argument(
        "--image-polarity",
        choices=("auto", "bright", "dark"),
        default="auto",
    )
    parser.add_argument("--image-sigma", type=float, default=1.0)
    parser.add_argument("--background-sigma", type=float, default=5.0)
    parser.add_argument("--cost-power", type=float, default=2.0)

    parser.add_argument("--min-path-score", type=float, default=0.44)
    parser.add_argument("--min-path-mean", type=float, default=0.30)
    parser.add_argument("--min-path-q20", type=float, default=0.10)
    parser.add_argument("--weak-support-threshold", type=float, default=0.12)
    parser.add_argument("--max-weak-run", type=int, default=8)
    parser.add_argument("--ambiguity-margin", type=float, default=0.12)

    parser.add_argument("--soma-contact-radius", type=float, default=3.0)
    parser.add_argument("--skeleton-clearance", type=int, default=1)
    parser.add_argument("--endpoint-unblock-radius", type=int, default=3)
    parser.add_argument("--soma-clearance", type=int, default=1)
    parser.add_argument(
        "--allow-same-component",
        action="store_true",
        help="Erlaubt Verbindungen zwischen zwei Endpunkten derselben Komponente.",
    )
    parser.add_argument(
        "--allow-non-mutual-best",
        action="store_true",
        help="Akzeptiert auch Paare, die nicht gegenseitig beste Partner sind.",
    )
    parser.add_argument(
        "--disable-soma-guard",
        action="store_true",
        help="Deaktiviert den Schutz vor Verbindungen verschiedener Soma-IDs.",
    )
    parser.add_argument(
        "--allow-ambiguous-components",
        action="store_true",
        help="Erlaubt Komponenten, die bereits mehrere Somata beruehren.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Uebernimmt akzeptierte Pfade. Ohne --apply wird nur eine "
            "sichere Kandidatenvorschau erzeugt."
        ),
    )
    parser.add_argument("--connection-radius", type=int, default=0)
    parser.add_argument("--preview-max-size", type=int, default=2600)
    return parser.parse_args()


def load_2d_tiff(path: Path) -> np.ndarray:
    array = np.squeeze(tifffile.imread(path))
    if array.ndim != 2:
        raise RuntimeError(
            f"{path.name}: Erwartet wurde ein 2D-TIFF, gefunden {array.shape}"
        )
    return array


def load_skeleton_probability(
    path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    with np.load(path) as archive:
        keys = list(archive.keys())
        if "probabilities" in archive:
            probabilities = archive["probabilities"]
        elif "softmax" in archive:
            probabilities = archive["softmax"]
        elif len(keys) == 1:
            probabilities = archive[keys[0]]
        else:
            raise RuntimeError(
                f"{path.name}: Wahrscheinlichkeiten nicht eindeutig; Keys={keys}"
            )

        probabilities = np.squeeze(probabilities)
        if probabilities.ndim != 3:
            raise RuntimeError(
                f"{path.name}: Erwartet wurden Klassenwahrscheinlichkeiten, "
                f"gefunden {probabilities.shape}"
            )

        if probabilities.shape[1:] == expected_shape:
            result = np.asarray(probabilities[1], dtype=np.float32).copy()
        elif probabilities.shape[:2] == expected_shape:
            result = np.asarray(probabilities[..., 1], dtype=np.float32).copy()
        else:
            raise RuntimeError(
                f"{path.name}: Probability-Shape {probabilities.shape} passt "
                f"nicht zu {expected_shape}"
            )

    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{path.name}: P(Skeleton) enthaelt NaN oder Inf")
    return np.clip(result, 0.0, 1.0)


def neighbor_count(thin: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(
        thin.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )


def endpoint_coordinates(thin: np.ndarray) -> np.ndarray:
    counts = neighbor_count(thin)
    return np.argwhere(thin & (counts == 1)).astype(np.int32)


def skeleton_neighbors(point: Point, thin: np.ndarray) -> list[Point]:
    row, col = point
    result: list[Point] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = row + dr, col + dc
            if (
                0 <= rr < thin.shape[0]
                and 0 <= cc < thin.shape[1]
                and thin[rr, cc]
            ):
                result.append((rr, cc))
    return result


def endpoint_outward_tangent(
    endpoint: Point,
    thin: np.ndarray,
    steps: int,
) -> np.ndarray | None:
    path = [endpoint]
    previous: Point | None = None
    current = endpoint

    for _ in range(max(1, steps)):
        candidates = [
            point
            for point in skeleton_neighbors(current, thin)
            if point != previous
        ]
        if len(candidates) != 1:
            break
        previous, current = current, candidates[0]
        path.append(current)

    if len(path) < 2:
        return None

    endpoint_vector = np.asarray(path[0], dtype=np.float64)
    inward_reference = np.mean(np.asarray(path[1:], dtype=np.float64), axis=0)
    tangent = endpoint_vector - inward_reference
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-8:
        return None
    return tangent / norm


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return -1.0
    return float(np.dot(first, second) / denominator)


def component_soma_assignments(
    thin: np.ndarray,
    soma: np.ndarray,
    contact_radius: float,
) -> tuple[np.ndarray, dict[int, set[int]], int, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    component_labels, component_count = ndi.label(thin, structure=structure)
    soma_labels, soma_count = ndi.label(soma, structure=structure)
    assignments: dict[int, set[int]] = defaultdict(set)

    skeleton_coordinates = np.argwhere(thin).astype(np.float32)
    soma_coordinates = np.argwhere(soma).astype(np.float32)
    if skeleton_coordinates.size == 0 or soma_coordinates.size == 0:
        return component_labels, assignments, int(component_count), int(soma_count)

    soma_tree = cKDTree(soma_coordinates)
    distances, nearest_indices = soma_tree.query(
        skeleton_coordinates,
        k=1,
        distance_upper_bound=contact_radius,
    )
    valid = np.isfinite(distances)
    if np.any(valid):
        valid_skeleton = skeleton_coordinates[valid].astype(np.int32)
        valid_soma = soma_coordinates[nearest_indices[valid]].astype(np.int32)
        component_ids = component_labels[
            valid_skeleton[:, 0],
            valid_skeleton[:, 1],
        ]
        soma_ids = soma_labels[valid_soma[:, 0], valid_soma[:, 1]]
        pairs = np.unique(np.column_stack((component_ids, soma_ids)), axis=0)
        for component_id, soma_id in pairs:
            if component_id > 0 and soma_id > 0:
                assignments[int(component_id)].add(int(soma_id))

    return component_labels, assignments, int(component_count), int(soma_count)


def robust_scaling(image: np.ndarray) -> tuple[float, float]:
    step = max(1, int(math.ceil(max(image.shape) / 2500)))
    sample = image[::step, ::step].astype(np.float32, copy=False)
    low, high = np.percentile(sample, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def determine_polarity(
    image: np.ndarray,
    thin: np.ndarray,
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    skeleton_values = image[thin]
    if skeleton_values.size == 0:
        return "bright"

    step = max(1, int(math.ceil(max(image.shape) / 2500)))
    background_reference = float(np.median(image[::step, ::step]))
    skeleton_reference = float(np.median(skeleton_values))
    return "bright" if skeleton_reference >= background_reference else "dark"


def normalized_original_crop(
    original: np.ndarray,
    bounds: tuple[int, int, int, int],
    scaling: tuple[float, float],
    polarity: str,
) -> np.ndarray:
    row0, row1, col0, col1 = bounds
    low, high = scaling
    crop = original[row0:row1, col0:col1].astype(np.float32)
    crop = np.clip((crop - low) / (high - low), 0.0, 1.0)
    if polarity == "dark":
        crop = 1.0 - crop
    return crop


def image_support_from_crop(
    normalized_crop: np.ndarray,
    image_sigma: float,
    background_sigma: float,
) -> np.ndarray:
    smooth = ndi.gaussian_filter(
        normalized_crop,
        sigma=max(0.0, image_sigma),
    )
    background = ndi.gaussian_filter(
        normalized_crop,
        sigma=max(image_sigma + 0.1, background_sigma),
    )
    residual = smooth - background
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    noise_scale = max(1.4826 * mad, 1e-4)
    contrast_support = np.clip(
        (residual - 0.35 * noise_scale) / (3.0 * noise_scale),
        0.0,
        1.0,
    )
    return np.clip(
        0.85 * contrast_support + 0.15 * smooth,
        0.0,
        1.0,
    ).astype(np.float32)


def crop_bounds(
    start: Point,
    end: Point,
    shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    row0 = max(0, min(start[0], end[0]) - margin)
    row1 = min(shape[0], max(start[0], end[0]) + margin + 1)
    col0 = max(0, min(start[1], end[1]) - margin)
    col1 = min(shape[1], max(start[1], end[1]) + margin + 1)
    return row0, row1, col0, col1


def disk_mask(
    shape: tuple[int, int],
    center: Point,
    radius: int,
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    row, col = center
    row0 = max(0, row - radius)
    row1 = min(shape[0], row + radius + 1)
    col0 = max(0, col - radius)
    col1 = min(shape[1], col + radius + 1)
    rr, cc = np.ogrid[row0:row1, col0:col1]
    result[row0:row1, col0:col1] = (
        (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    )
    return result


def maximum_true_run(values: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def path_length(path: list[Point]) -> float:
    if len(path) < 2:
        return 0.0
    array = np.asarray(path, dtype=np.float64)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def path_endpoint_cosines(
    path: list[Point],
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    steps: int,
) -> tuple[float, float]:
    if len(path) < 2:
        return -1.0, -1.0
    offset = min(max(1, steps), len(path) - 1)
    start_direction = (
        np.asarray(path[offset], dtype=np.float64)
        - np.asarray(path[0], dtype=np.float64)
    )
    end_direction = (
        np.asarray(path[-offset - 1], dtype=np.float64)
        - np.asarray(path[-1], dtype=np.float64)
    )
    return (
        cosine(start_tangent, start_direction),
        cosine(end_tangent, end_direction),
    )


def nearest_pair_ids(
    endpoints: np.ndarray,
    max_gap: float,
    max_neighbors: int,
) -> list[tuple[int, int]]:
    if len(endpoints) < 2:
        return []

    tree = cKDTree(endpoints.astype(np.float32))
    unique_pairs: set[tuple[int, int]] = set()
    for endpoint_id, endpoint in enumerate(endpoints):
        neighbor_ids = tree.query_ball_point(endpoint, r=max_gap)
        ranked = sorted(
            (
                (
                    float(
                        np.linalg.norm(
                            endpoints[neighbor_id].astype(np.float64)
                            - endpoint.astype(np.float64)
                        )
                    ),
                    int(neighbor_id),
                )
                for neighbor_id in neighbor_ids
                if int(neighbor_id) != endpoint_id
            ),
            key=lambda item: item[0],
        )
        for _, neighbor_id in ranked[:max_neighbors]:
            unique_pairs.add(
                (
                    min(endpoint_id, neighbor_id),
                    max(endpoint_id, neighbor_id),
                )
            )
    return sorted(unique_pairs)


def route_candidate(
    start: Point,
    end: Point,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    thin: np.ndarray,
    soma: np.ndarray,
    probability: np.ndarray | None,
    original: np.ndarray | None,
    original_scaling: tuple[float, float] | None,
    polarity: str | None,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    bounds = crop_bounds(
        start,
        end,
        thin.shape,
        margin=args.roi_margin,
    )
    row0, row1, col0, col1 = bounds
    local_start = (start[0] - row0, start[1] - col0)
    local_end = (end[0] - row0, end[1] - col0)

    evidence_parts: list[tuple[float, np.ndarray]] = []
    image_support: np.ndarray | None = None
    if (
        original is not None
        and original_scaling is not None
        and polarity is not None
        and args.image_weight > 0
    ):
        normalized_crop = normalized_original_crop(
            original,
            bounds,
            original_scaling,
            polarity,
        )
        image_support = image_support_from_crop(
            normalized_crop,
            image_sigma=args.image_sigma,
            background_sigma=args.background_sigma,
        )
        evidence_parts.append((args.image_weight, image_support))

    probability_support: np.ndarray | None = None
    if probability is not None and args.probability_weight > 0:
        probability_support = probability[row0:row1, col0:col1]
        evidence_parts.append((args.probability_weight, probability_support))

    if not evidence_parts:
        raise RuntimeError(
            "Stage 5 benoetigt --original und/oder --probabilities als Kostenbild."
        )

    total_weight = sum(weight for weight, _ in evidence_parts)
    evidence = sum(
        weight * values for weight, values in evidence_parts
    ) / total_weight
    evidence = np.clip(evidence, 0.0, 1.0).astype(np.float32)
    cost = (
        0.03
        + np.power(1.0 - evidence, args.cost_power)
    ).astype(np.float32)

    thin_crop = thin[row0:row1, col0:col1]
    soma_crop = soma[row0:row1, col0:col1]
    blocked = np.zeros_like(thin_crop, dtype=bool)
    if args.skeleton_clearance > 0:
        blocked |= ndi.binary_dilation(
            thin_crop,
            iterations=args.skeleton_clearance,
        )
    else:
        blocked |= thin_crop
    if args.soma_clearance > 0:
        blocked |= ndi.binary_dilation(
            soma_crop,
            iterations=args.soma_clearance,
        )
    else:
        blocked |= soma_crop

    unblock = disk_mask(
        thin_crop.shape,
        local_start,
        args.endpoint_unblock_radius,
    )
    unblock |= disk_mask(
        thin_crop.shape,
        local_end,
        args.endpoint_unblock_radius,
    )
    blocked &= ~unblock
    cost[blocked] = 1e6

    try:
        route, route_cost = route_through_array(
            cost,
            local_start,
            local_end,
            fully_connected=True,
            geometric=True,
        )
    except (ValueError, RuntimeError):
        return None

    path = [
        (int(row + row0), int(col + col0))
        for row, col in route
    ]
    if len(path) < 2:
        return None

    local_rows = np.fromiter(
        (point[0] - row0 for point in path),
        dtype=np.int32,
    )
    local_cols = np.fromiter(
        (point[1] - col0 for point in path),
        dtype=np.int32,
    )
    if np.any(blocked[local_rows[1:-1], local_cols[1:-1]]):
        return None

    direct_distance = float(
        np.linalg.norm(
            np.asarray(end, dtype=np.float64)
            - np.asarray(start, dtype=np.float64)
        )
    )
    length = path_length(path)
    path_ratio = length / max(direct_distance, 1e-6)
    path_start_cos, path_end_cos = path_endpoint_cosines(
        path,
        start_tangent,
        end_tangent,
        args.path_direction_steps,
    )

    path_evidence = evidence[local_rows, local_cols]
    path_mean = float(np.mean(path_evidence))
    path_q20 = float(np.quantile(path_evidence, 0.20))
    weak_run = maximum_true_run(
        path_evidence < args.weak_support_threshold
    )
    image_mean = (
        float(np.mean(image_support[local_rows, local_cols]))
        if image_support is not None
        else math.nan
    )
    probability_mean = (
        float(np.mean(probability_support[local_rows, local_cols]))
        if probability_support is not None
        else math.nan
    )

    direction_score = float(
        np.clip((path_start_cos + path_end_cos) / 2.0, 0.0, 1.0)
    )
    ratio_score = float(
        np.clip(
            1.0
            - (path_ratio - 1.0) / max(args.max_path_ratio - 1.0, 1e-6),
            0.0,
            1.0,
        )
    )
    weak_run_score = float(
        np.clip(
            1.0 - weak_run / max(args.max_weak_run + 1, 1),
            0.0,
            1.0,
        )
    )
    evidence_score = (
        0.55 * path_mean
        + 0.20 * path_q20
        + 0.25 * weak_run_score
    )
    score = (
        0.65 * evidence_score
        + 0.20 * direction_score
        + 0.15 * ratio_score
    )

    return {
        "path": path,
        "route_cost": float(route_cost),
        "path_length": length,
        "path_ratio": path_ratio,
        "path_start_direction_cos": path_start_cos,
        "path_end_direction_cos": path_end_cos,
        "path_mean": path_mean,
        "path_q20": path_q20,
        "max_weak_run": weak_run,
        "image_mean": image_mean,
        "probability_mean": probability_mean,
        "score": float(score),
    }


def build_candidates(
    thin: np.ndarray,
    soma: np.ndarray,
    endpoints: np.ndarray,
    component_labels: np.ndarray,
    component_soma_ids: dict[int, set[int]],
    probability: np.ndarray | None,
    original: np.ndarray | None,
    original_scaling: tuple[float, float] | None,
    polarity: str | None,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    endpoint_points = [tuple(map(int, point)) for point in endpoints]
    tangents = [
        endpoint_outward_tangent(point, thin, args.orientation_steps)
        for point in endpoint_points
    ]
    pairs = nearest_pair_ids(
        endpoints,
        max_gap=args.max_gap,
        max_neighbors=args.max_neighbors_per_endpoint,
    )
    candidates: list[dict[str, object]] = []

    for pair_index, (start_id, end_id) in enumerate(pairs, start=1):
        start = endpoint_points[start_id]
        end = endpoint_points[end_id]
        start_component = int(component_labels[start])
        end_component = int(component_labels[end])
        start_soma_ids = component_soma_ids.get(start_component, set())
        end_soma_ids = component_soma_ids.get(end_component, set())

        vector = (
            np.asarray(end, dtype=np.float64)
            - np.asarray(start, dtype=np.float64)
        )
        distance = float(np.linalg.norm(vector))
        if distance < args.min_gap or distance > args.max_gap:
            continue

        candidate: dict[str, object] = {
            "candidate_id": len(candidates),
            "start_endpoint": start_id,
            "end_endpoint": end_id,
            "start_component": start_component,
            "end_component": end_component,
            "start_soma_ids": ";".join(map(str, sorted(start_soma_ids))),
            "end_soma_ids": ";".join(map(str, sorted(end_soma_ids))),
            "start_row": start[0],
            "start_col": start[1],
            "end_row": end[0],
            "end_col": end[1],
            "distance": distance,
            "chord_start_cos": math.nan,
            "chord_end_cos": math.nan,
            "route_cost": math.nan,
            "path_length": math.nan,
            "path_ratio": math.nan,
            "path_start_direction_cos": math.nan,
            "path_end_direction_cos": math.nan,
            "path_mean": math.nan,
            "path_q20": math.nan,
            "max_weak_run": -1,
            "image_mean": math.nan,
            "probability_mean": math.nan,
            "score": -math.inf,
            "path": [],
            "decision": "candidate",
        }

        if start_component == end_component and not args.allow_same_component:
            candidate["decision"] = "rejected_same_component"
            candidates.append(candidate)
            continue

        if (
            not args.allow_ambiguous_components
            and (
                len(start_soma_ids) > 1
                or len(end_soma_ids) > 1
            )
        ):
            candidate["decision"] = "rejected_ambiguous_component"
            candidates.append(candidate)
            continue

        soma_conflict = bool(
            not args.disable_soma_guard
            and start_soma_ids
            and end_soma_ids
            and len(start_soma_ids | end_soma_ids) > 1
        )
        if soma_conflict:
            candidate["decision"] = "rejected_different_somata"
            candidates.append(candidate)
            continue

        start_tangent = tangents[start_id]
        end_tangent = tangents[end_id]
        if start_tangent is None or end_tangent is None:
            candidate["decision"] = "rejected_missing_tangent"
            candidates.append(candidate)
            continue

        chord_start_cos = cosine(start_tangent, vector)
        chord_end_cos = cosine(end_tangent, -vector)
        candidate["chord_start_cos"] = chord_start_cos
        candidate["chord_end_cos"] = chord_end_cos
        if min(chord_start_cos, chord_end_cos) < args.min_chord_direction_cos:
            candidate["decision"] = "rejected_chord_direction"
            candidates.append(candidate)
            continue

        routed = route_candidate(
            start=start,
            end=end,
            start_tangent=start_tangent,
            end_tangent=end_tangent,
            thin=thin,
            soma=soma,
            probability=probability,
            original=original,
            original_scaling=original_scaling,
            polarity=polarity,
            args=args,
        )
        if routed is None:
            candidate["decision"] = "rejected_no_valid_route"
            candidates.append(candidate)
            continue
        candidate.update(routed)

        if float(candidate["path_ratio"]) > args.max_path_ratio:
            candidate["decision"] = "rejected_path_ratio"
        elif min(
            float(candidate["path_start_direction_cos"]),
            float(candidate["path_end_direction_cos"]),
        ) < args.min_path_direction_cos:
            candidate["decision"] = "rejected_path_direction"
        elif float(candidate["path_mean"]) < args.min_path_mean:
            candidate["decision"] = "rejected_path_mean"
        elif float(candidate["path_q20"]) < args.min_path_q20:
            candidate["decision"] = "rejected_path_q20"
        elif int(candidate["max_weak_run"]) > args.max_weak_run:
            candidate["decision"] = "rejected_weak_run"
        elif float(candidate["score"]) < args.min_path_score:
            candidate["decision"] = "rejected_score"

        candidates.append(candidate)
        if pair_index % 250 == 0:
            print(f"  Gepruefte Endpoint-Paare: {pair_index}/{len(pairs)}")

    return candidates


def select_candidates(
    candidates: list[dict[str, object]],
    ambiguity_margin: float,
    require_mutual_best: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["decision"] == "candidate"
    ]
    by_endpoint: dict[int, list[dict[str, object]]] = defaultdict(list)
    for candidate in eligible:
        by_endpoint[int(candidate["start_endpoint"])].append(candidate)
        by_endpoint[int(candidate["end_endpoint"])].append(candidate)

    best_candidate: dict[int, int] = {}
    ambiguous_endpoints: set[int] = set()
    for endpoint_id, endpoint_candidates in by_endpoint.items():
        ranked = sorted(
            endpoint_candidates,
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        best_candidate[endpoint_id] = int(ranked[0]["candidate_id"])
        if (
            len(ranked) >= 2
            and float(ranked[0]["score"]) - float(ranked[1]["score"])
            < ambiguity_margin
        ):
            ambiguous_endpoints.add(endpoint_id)

    accepted: list[dict[str, object]] = []
    ambiguous: list[dict[str, object]] = []
    used_endpoints: set[int] = set()
    for candidate in sorted(
        eligible,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        candidate_id = int(candidate["candidate_id"])
        start_id = int(candidate["start_endpoint"])
        end_id = int(candidate["end_endpoint"])

        if start_id in ambiguous_endpoints or end_id in ambiguous_endpoints:
            candidate["decision"] = "ambiguous_competing_candidate"
            ambiguous.append(candidate)
            continue

        if require_mutual_best and (
            best_candidate.get(start_id) != candidate_id
            or best_candidate.get(end_id) != candidate_id
        ):
            candidate["decision"] = "rejected_not_mutual_best"
            continue

        if start_id in used_endpoints or end_id in used_endpoints:
            candidate["decision"] = "rejected_endpoint_already_used"
            continue

        candidate["decision"] = "accepted_preview"
        accepted.append(candidate)
        used_endpoints.update((start_id, end_id))

    return accepted, ambiguous


def mask_from_paths(
    shape: tuple[int, int],
    candidates: Iterable[dict[str, object]],
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        for row, col in candidate["path"]:  # type: ignore[index]
            result[int(row), int(col)] = True
    return result


def save_tiff(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(path, array, photometric="minisblack")


def reduced_mask(mask: np.ndarray, step: int) -> np.ndarray:
    return block_reduce(mask, block_size=(step, step), func=np.max).astype(bool)


def save_qc(
    path: Path,
    original: np.ndarray | None,
    raw_skeleton: np.ndarray,
    soma: np.ndarray,
    endpoints: np.ndarray,
    all_valid_paths: np.ndarray,
    ambiguous_paths: np.ndarray,
    proposed_paths: np.ndarray,
    applied_paths: np.ndarray,
    preview_max_size: int,
    apply_mode: bool,
) -> None:
    step = max(
        1,
        int(math.ceil(max(raw_skeleton.shape) / preview_max_size)),
    )
    height = math.ceil(raw_skeleton.shape[0] / step)
    width = math.ceil(raw_skeleton.shape[1] / step)

    if original is not None:
        background = original[::step, ::step].astype(np.float32)
        background = background[:height, :width]
        low, high = np.percentile(background, [1.0, 99.5])
        background = np.clip(
            (background - low) / max(high - low, 1e-6),
            0.0,
            1.0,
        )
    else:
        background = np.zeros((height, width), dtype=np.float32)

    overlay = np.repeat(background[..., None], 3, axis=2) * 0.72
    raw_small = reduced_mask(raw_skeleton, step)[:height, :width]
    soma_small = reduced_mask(soma, step)[:height, :width]
    all_valid_small = reduced_mask(all_valid_paths, step)[:height, :width]
    ambiguous_small = reduced_mask(ambiguous_paths, step)[:height, :width]
    proposed_small = reduced_mask(proposed_paths, step)[:height, :width]
    applied_small = reduced_mask(applied_paths, step)[:height, :width]

    overlay[raw_small] = (0.15, 0.85, 0.25)
    overlay[all_valid_small] = (0.20, 0.55, 1.00)
    overlay[ambiguous_small] = (1.00, 0.40, 0.05)
    overlay[proposed_small] = (1.00, 0.95, 0.10)
    if apply_mode:
        overlay[applied_small] = (0.70, 0.25, 1.00)
    overlay[soma_small] = (1.00, 0.00, 0.85)

    endpoint_mask = np.zeros_like(raw_skeleton, dtype=bool)
    if len(endpoints):
        endpoint_mask[endpoints[:, 0], endpoints[:, 1]] = True
    endpoint_mask = ndi.binary_dilation(
        endpoint_mask,
        iterations=max(1, step),
    )
    endpoint_small = endpoint_mask[::step, ::step][:height, :width]
    overlay[endpoint_small] = (0.00, 1.00, 1.00)

    plt.figure(figsize=(16, 10), dpi=160)
    plt.imshow(overlay)
    plt.axis("off")
    mode = "APPLY" if apply_mode else "PREVIEW"
    plt.title(
        f"Stage 5 {mode}: gruen=Input-Skeleton, gelb=Vorschlag, "
        "violett=angewendet, orange=mehrdeutig, blau=gueltige Alternative, "
        "magenta=Soma, cyan=Endpunkte"
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def write_candidates(
    path: Path,
    candidates: list[dict[str, object]],
) -> None:
    fields = [
        "candidate_id",
        "start_endpoint",
        "end_endpoint",
        "start_component",
        "end_component",
        "start_soma_ids",
        "end_soma_ids",
        "start_row",
        "start_col",
        "end_row",
        "end_col",
        "distance",
        "chord_start_cos",
        "chord_end_cos",
        "route_cost",
        "path_length",
        "path_ratio",
        "path_start_direction_cos",
        "path_end_direction_cos",
        "path_mean",
        "path_q20",
        "max_weak_run",
        "image_mean",
        "probability_mean",
        "score",
        "decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    field: candidate[field]
                    for field in fields
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_weight < 0 or args.probability_weight < 0:
        raise ValueError("Gewichte duerfen nicht negativ sein.")
    if args.image_weight + args.probability_weight <= 0:
        raise ValueError("Mindestens ein Evidenzgewicht muss groesser als 0 sein.")
    if args.max_gap <= args.min_gap:
        raise ValueError("--max-gap muss groesser als --min-gap sein.")

    print(f"ITERATIVES POSTPROCESSING: {VERSION}")
    print(f"RUNNING FILE: {Path(__file__).resolve()}")
    print("MODUS: " + ("APPLY" if args.apply else "PREVIEW (keine Pfade angewendet)"))

    prediction = load_2d_tiff(args.prediction)
    values = set(np.unique(prediction).astype(int).tolist())
    if not values.issubset({0, 1, 2}):
        raise RuntimeError(
            f"{args.prediction.name}: Erwartet 0/1/2, gefunden {sorted(values)}"
        )

    raw_skeleton = prediction == 1
    soma = prediction == 2
    thin = skeletonize(raw_skeleton)
    endpoints_before = endpoint_coordinates(thin)
    (
        component_labels,
        component_soma_ids,
        component_count,
        soma_count,
    ) = component_soma_assignments(
        thin,
        soma,
        contact_radius=args.soma_contact_radius,
    )

    probability = None
    if args.probabilities is not None:
        probability = load_skeleton_probability(
            args.probabilities,
            prediction.shape,
        )

    original = None
    original_scaling = None
    polarity = None
    if args.original is not None:
        original = load_2d_tiff(args.original)
        if original.shape != prediction.shape:
            raise RuntimeError(
                f"Original {original.shape} und Prediction {prediction.shape} "
                "haben unterschiedliche Groessen."
            )
        original_scaling = robust_scaling(original)
        polarity = determine_polarity(
            original,
            thin,
            requested=args.image_polarity,
        )

    if probability is None and original is None:
        raise RuntimeError(
            "Bitte --original und/oder --probabilities angeben."
        )

    print(f"Bildgroesse:             {prediction.shape}")
    print(f"Skeletonpixel Input:     {int(raw_skeleton.sum())}")
    print(f"Endpunkte Input:         {len(endpoints_before)}")
    print(f"Skeletonkomponenten:     {component_count}")
    print(f"Soma-Instanzen:          {soma_count}")
    print(f"Suchbereich:             {args.min_gap:.1f}-{args.max_gap:.1f} px")
    print(
        "Gegenseitig bester Partner: "
        + ("nein" if args.allow_non_mutual_best else "ja")
    )
    if polarity is not None:
        print(f"Originalbild-Polaritaet: {polarity}")

    candidates = build_candidates(
        thin=thin,
        soma=soma,
        endpoints=endpoints_before,
        component_labels=component_labels,
        component_soma_ids=component_soma_ids,
        probability=probability,
        original=original,
        original_scaling=original_scaling,
        polarity=polarity,
        args=args,
    )
    proposed, ambiguous = select_candidates(
        candidates,
        ambiguity_margin=args.ambiguity_margin,
        require_mutual_best=not args.allow_non_mutual_best,
    )

    alternative_candidates = [
        candidate
        for candidate in candidates
        if candidate["decision"]
        in {
            "rejected_not_mutual_best",
            "rejected_endpoint_already_used",
        }
    ]
    proposed_mask = mask_from_paths(prediction.shape, proposed)
    ambiguous_mask = mask_from_paths(prediction.shape, ambiguous)
    alternative_mask = mask_from_paths(
        prediction.shape,
        alternative_candidates,
    )
    alternative_mask &= ~raw_skeleton

    if args.connection_radius > 0:
        proposed_mask = ndi.binary_dilation(
            proposed_mask,
            iterations=args.connection_radius,
        )

    proposed_mask &= ~raw_skeleton
    applied_mask = (
        proposed_mask.copy()
        if args.apply
        else np.zeros_like(proposed_mask)
    )
    completed = raw_skeleton | applied_mask

    # Unverhandelbare No-loss-Garantie.
    if not np.all(completed[raw_skeleton]):
        raise AssertionError("No-loss-Garantie verletzt.")

    completed_thin = skeletonize(completed)
    endpoints_after = endpoint_coordinates(completed_thin)
    semantic = np.zeros(prediction.shape, dtype=np.uint8)
    semantic[completed] = 1
    semantic[soma] = 2

    endpoint_before_mask = np.zeros(prediction.shape, dtype=np.uint8)
    if len(endpoints_before):
        endpoint_before_mask[
            endpoints_before[:, 0],
            endpoints_before[:, 1],
        ] = 1
    endpoint_after_mask = np.zeros(prediction.shape, dtype=np.uint8)
    if len(endpoints_after):
        endpoint_after_mask[
            endpoints_after[:, 0],
            endpoints_after[:, 1],
        ] = 1

    save_tiff(
        args.output_dir / "01_input_skeleton_fullwidth_NO_LOSS.tif",
        raw_skeleton.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "02_analysis_skeleton_1px.tif",
        thin.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "03_endpoints_before.tif",
        endpoint_before_mask,
    )
    save_tiff(
        args.output_dir / "04_valid_alternatives_NOT_SELECTED.tif",
        alternative_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "05_ambiguous_image_guided_paths.tif",
        ambiguous_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "06_proposed_image_guided_connections.tif",
        proposed_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "07_added_connections_APPLIED.tif",
        applied_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "08_completed_skeleton_NO_LOSS.tif",
        completed.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "09_completed_skeleton_and_soma_0-1-2.tif",
        semantic,
    )
    save_tiff(
        args.output_dir / "10_endpoints_after.tif",
        endpoint_after_mask,
    )
    write_candidates(
        args.output_dir / "stage5_connection_candidates.csv",
        candidates,
    )
    save_qc(
        args.output_dir / "stage5_qc_overlay.png",
        original=original,
        raw_skeleton=raw_skeleton,
        soma=soma,
        endpoints=endpoints_before,
        all_valid_paths=alternative_mask,
        ambiguous_paths=ambiguous_mask,
        proposed_paths=proposed_mask,
        applied_paths=applied_mask,
        preview_max_size=args.preview_max_size,
        apply_mode=args.apply,
    )

    decision_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        decision_counts[str(candidate["decision"])] += 1

    summary = {
        "version": VERSION,
        "mode": "apply" if args.apply else "preview",
        "prediction": str(args.prediction),
        "probabilities": (
            str(args.probabilities)
            if args.probabilities
            else None
        ),
        "original": str(args.original) if args.original else None,
        "parameters": {
            "min_gap": args.min_gap,
            "max_gap": args.max_gap,
            "roi_margin": args.roi_margin,
            "max_neighbors_per_endpoint": args.max_neighbors_per_endpoint,
            "orientation_steps": args.orientation_steps,
            "min_chord_direction_cos": args.min_chord_direction_cos,
            "min_path_direction_cos": args.min_path_direction_cos,
            "max_path_ratio": args.max_path_ratio,
            "image_weight": args.image_weight,
            "probability_weight": args.probability_weight,
            "image_polarity": polarity,
            "min_path_score": args.min_path_score,
            "min_path_mean": args.min_path_mean,
            "min_path_q20": args.min_path_q20,
            "weak_support_threshold": args.weak_support_threshold,
            "max_weak_run": args.max_weak_run,
            "ambiguity_margin": args.ambiguity_margin,
            "mutual_best_required": not args.allow_non_mutual_best,
            "soma_guard_enabled": not args.disable_soma_guard,
        },
        "input_skeleton_pixels": int(raw_skeleton.sum()),
        "proposed_connections": len(proposed),
        "ambiguous_connections": len(ambiguous),
        "valid_alternatives_not_selected": len(alternative_candidates),
        "proposed_pixels": int(proposed_mask.sum()),
        "applied_pixels": int(applied_mask.sum()),
        "completed_skeleton_pixels": int(completed.sum()),
        "removed_original_pixels": int(np.sum(raw_skeleton & ~completed)),
        "endpoints_before": int(len(endpoints_before)),
        "endpoints_after": int(len(endpoints_after)),
        "decision_counts": dict(sorted(decision_counts.items())),
    }
    with (args.output_dir / "stage5_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 72)
    print("STAGE 5 ABGESCHLOSSEN")
    print("=" * 72)
    print(f"Modus:                    {summary['mode']}")
    print(f"Vorgeschlagene Pfade:     {len(proposed)}")
    print(f"Mehrdeutige Pfade:        {len(ambiguous)}")
    print(f"Vorgeschlagene Pixel:     {int(proposed_mask.sum())}")
    print(f"Angewendete Pixel:        {int(applied_mask.sum())}")
    print(
        "Endpunkte vorher/nachher: "
        f"{len(endpoints_before)} / {len(endpoints_after)}"
    )
    print(f"ENTFERNTE Originalpixel:  {int(np.sum(raw_skeleton & ~completed))}")
    print(f"QC:                       {args.output_dir / 'stage5_qc_overlay.png'}")
    if not args.apply:
        print()
        print("PREVIEW: Es wurde noch keine Verbindung in den finalen Output uebernommen.")
        print("Pruefe 06_proposed_image_guided_connections.tif und stage5_qc_overlay.png.")
        print("Wenn die Vorschlaege passen, denselben Aufruf mit --apply wiederholen.")


if __name__ == "__main__":
    main()
