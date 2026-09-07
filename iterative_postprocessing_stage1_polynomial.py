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
from skimage.draw import line
from skimage.measure import block_reduce
from skimage.morphology import skeletonize


VERSION = "stage1-short-polynomial-gaps-v3-straight-rescue-2026-07-30"
Point = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stufe 1 eines iterativen No-loss-Postprocessings: kurze Lücken "
            "zwischen Skeleton-Endpunkten werden mit kubischen Polynomen geschlossen."
        )
    )
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--max-gap", type=float, default=12.0)
    parser.add_argument("--min-gap", type=float, default=2.0)
    parser.add_argument("--orientation-steps", type=int, default=6)
    parser.add_argument("--tangent-scale", type=float, default=0.40)
    parser.add_argument("--min-direction-cos", type=float, default=0.00)
    parser.add_argument("--max-curve-ratio", type=float, default=1.45)
    parser.add_argument("--min-score", type=float, default=0.42)
    parser.add_argument("--ambiguity-margin", type=float, default=0.045)
    parser.add_argument("--max-candidates-per-endpoint", type=int, default=3)
    parser.add_argument("--connection-radius", type=int, default=0)
    parser.add_argument("--soma-contact-radius", type=float, default=3.0)
    parser.add_argument("--straight-rescue-min-cos", type=float, default=0.85)
    parser.add_argument(
        "--straight-rescue-max-curve-ratio",
        type=float,
        default=1.20,
    )
    parser.add_argument("--straight-rescue-min-score", type=float, default=0.43)
    parser.add_argument("--straight-rescue-min-image-mean", type=float, default=0.35)
    parser.add_argument("--straight-rescue-min-image-q25", type=float, default=0.30)
    parser.add_argument(
        "--disable-straight-rescue",
        action="store_true",
        help="Deaktiviert die Sonderregel für nahezu gerade, bildgestützte Lücken.",
    )
    parser.add_argument(
        "--disable-soma-guard",
        action="store_true",
        help=(
            "Deaktiviert die Schutzregel, die Verbindungen zwischen Komponenten "
            "mit unterschiedlichen Soma-IDs blockiert."
        ),
    )
    parser.add_argument("--preview-max-size", type=int, default=2600)
    return parser.parse_args()


def load_2d_tiff(path: Path) -> np.ndarray:
    array = np.squeeze(tifffile.imread(path))
    if array.ndim != 2:
        raise RuntimeError(f"{path.name}: Erwartet wurde ein 2D-TIFF, gefunden {array.shape}")
    return array


def load_skeleton_probability(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
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
                f"{path.name}: Wahrscheinlichkeiten nicht eindeutig gefunden; Keys={keys}"
            )

        probabilities = np.squeeze(probabilities)
        if probabilities.ndim != 3:
            raise RuntimeError(
                f"{path.name}: Erwartet wurden Klassenwahrscheinlichkeiten, "
                f"gefunden {probabilities.shape}"
            )

        if probabilities.shape[1:] == expected_shape:
            skeleton_probability = np.asarray(probabilities[1], dtype=np.float32).copy()
        elif probabilities.shape[:2] == expected_shape:
            skeleton_probability = np.asarray(
                probabilities[..., 1], dtype=np.float32
            ).copy()
        else:
            raise RuntimeError(
                f"{path.name}: Wahrscheinlichkeits-Shape {probabilities.shape} "
                f"passt nicht zum Bild {expected_shape}"
            )

    if not np.all(np.isfinite(skeleton_probability)):
        raise RuntimeError(f"{path.name}: P(Skeleton) enthält NaN oder Inf")
    return np.clip(skeleton_probability, 0.0, 1.0)


def neighbor_count(thin: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(thin.astype(np.uint8), kernel, mode="constant", cval=0)


def endpoint_coordinates(thin: np.ndarray) -> np.ndarray:
    counts = neighbor_count(thin)
    return np.argwhere(thin & (counts == 1)).astype(np.int32)


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
            valid_skeleton[:, 0], valid_skeleton[:, 1]
        ]
        soma_ids = soma_labels[valid_soma[:, 0], valid_soma[:, 1]]
        pairs = np.unique(np.column_stack((component_ids, soma_ids)), axis=0)
        for component_id, soma_id in pairs:
            if component_id > 0 and soma_id > 0:
                assignments[int(component_id)].add(int(soma_id))

    return component_labels, assignments, int(component_count), int(soma_count)


def neighbors(point: Point, thin: np.ndarray) -> list[Point]:
    row, col = point
    result: list[Point] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = row + dr, col + dc
            if 0 <= rr < thin.shape[0] and 0 <= cc < thin.shape[1] and thin[rr, cc]:
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
        candidates = [point for point in neighbors(current, thin) if point != previous]
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


def cubic_hermite_path(
    start: Point,
    end: Point,
    start_outward: np.ndarray,
    end_outward: np.ndarray,
    tangent_scale: float,
    shape: tuple[int, int],
) -> list[Point]:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    distance = float(np.linalg.norm(p1 - p0))

    # C(t) is a cubic polynomial. At t=0 it follows the first endpoint
    # outward; at t=1 it approaches the second endpoint against its
    # outward direction.
    m0 = start_outward * distance * tangent_scale
    m1 = -end_outward * distance * tangent_scale

    samples = max(16, int(math.ceil(distance * 5.0)) + 1)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    curve = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    curve[:, 0] = np.clip(curve[:, 0], 0, shape[0] - 1)
    curve[:, 1] = np.clip(curve[:, 1], 0, shape[1] - 1)
    rounded = np.rint(curve).astype(np.int32)

    rasterized: list[Point] = []
    for first, second in zip(rounded[:-1], rounded[1:]):
        rr, cc = line(
            int(first[0]),
            int(first[1]),
            int(second[0]),
            int(second[1]),
        )
        rasterized.extend(zip(rr.tolist(), cc.tolist()))
    rasterized.append(end)

    unique: list[Point] = []
    seen: set[Point] = set()
    for point in rasterized:
        if point not in seen:
            unique.append(point)
            seen.add(point)
    return unique


def polyline_length(points: Iterable[Point]) -> float:
    array = np.asarray(list(points), dtype=np.float64)
    if array.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def image_scaling(image: np.ndarray) -> tuple[float, float]:
    step = max(1, int(math.ceil(max(image.shape) / 2500)))
    sample = image[::step, ::step].astype(np.float32, copy=False)
    low, high = np.percentile(sample, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def normalized_image_values(
    image: np.ndarray | None,
    points: list[Point],
    scaling: tuple[float, float] | None,
) -> np.ndarray | None:
    if image is None or scaling is None:
        return None
    rows = np.fromiter((point[0] for point in points), dtype=np.int32)
    cols = np.fromiter((point[1] for point in points), dtype=np.int32)
    low, high = scaling
    values = (image[rows, cols].astype(np.float32) - low) / (high - low)
    return np.clip(values, 0.0, 1.0)


def build_candidates(
    thin: np.ndarray,
    soma: np.ndarray,
    endpoints: np.ndarray,
    component_labels: np.ndarray,
    component_soma_ids: dict[int, set[int]],
    soma_guard_enabled: bool,
    probability: np.ndarray | None,
    original: np.ndarray | None,
    original_scaling: tuple[float, float] | None,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    endpoint_points = [tuple(map(int, point)) for point in endpoints]
    if len(endpoint_points) < 2:
        return []

    tangents = [
        endpoint_outward_tangent(point, thin, args.orientation_steps)
        for point in endpoint_points
    ]
    tree = cKDTree(endpoints.astype(np.float32))
    pairs = tree.query_pairs(r=args.max_gap)
    candidates: list[dict[str, object]] = []

    for start_id, end_id in sorted(pairs):
        start = endpoint_points[start_id]
        end = endpoint_points[end_id]
        start_component = int(component_labels[start])
        end_component = int(component_labels[end])
        start_soma_ids = component_soma_ids.get(start_component, set())
        end_soma_ids = component_soma_ids.get(end_component, set())
        soma_union = start_soma_ids | end_soma_ids
        soma_guard_conflict = bool(
            soma_guard_enabled
            and start_soma_ids
            and end_soma_ids
            and len(soma_union) > 1
        )
        vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        distance = float(np.linalg.norm(vector))
        if distance < args.min_gap or distance > args.max_gap:
            continue

        start_tangent = tangents[start_id]
        end_tangent = tangents[end_id]
        if start_tangent is None or end_tangent is None:
            continue

        start_cos = cosine(start_tangent, vector)
        end_cos = cosine(end_tangent, -vector)
        if min(start_cos, end_cos) < args.min_direction_cos:
            continue

        path = cubic_hermite_path(
            start,
            end,
            start_tangent,
            end_tangent,
            tangent_scale=args.tangent_scale,
            shape=thin.shape,
        )
        rows = np.fromiter((point[0] for point in path), dtype=np.int32)
        cols = np.fromiter((point[1] for point in path), dtype=np.int32)

        if np.any(soma[rows[1:-1], cols[1:-1]]):
            continue

        curve_length = polyline_length(path)
        curve_ratio = curve_length / max(distance, 1e-6)
        if curve_ratio > args.max_curve_ratio:
            continue

        distance_score = np.clip(
            1.0 - (distance - args.min_gap) / max(args.max_gap - args.min_gap, 1e-6),
            0.0,
            1.0,
        )
        direction_score = np.clip((start_cos + end_cos) / 2.0, 0.0, 1.0)
        curve_score = np.clip(
            1.0 - (curve_ratio - 1.0) / max(args.max_curve_ratio - 1.0, 1e-6),
            0.0,
            1.0,
        )
        geometry_score = (
            0.36 * float(distance_score)
            + 0.46 * float(direction_score)
            + 0.18 * float(curve_score)
        )

        probability_mean = math.nan
        probability_q25 = math.nan
        evidence_parts: list[tuple[float, float]] = []
        if probability is not None:
            probability_values = probability[rows, cols]
            probability_mean = float(np.mean(probability_values))
            probability_q25 = float(np.quantile(probability_values, 0.25))
            evidence_parts.append(
                (0.70, 0.65 * probability_mean + 0.35 * probability_q25)
            )

        image_mean = math.nan
        image_q25 = math.nan
        image_values = normalized_image_values(original, path, original_scaling)
        if image_values is not None:
            image_mean = float(np.mean(image_values))
            image_q25 = float(np.quantile(image_values, 0.25))
            evidence_parts.append((0.30 if probability is not None else 1.0, image_mean))

        if evidence_parts:
            total_weight = sum(weight for weight, _ in evidence_parts)
            evidence_score = sum(
                weight * value for weight, value in evidence_parts
            ) / total_weight
            score = 0.68 * geometry_score + 0.32 * evidence_score
        else:
            evidence_score = math.nan
            score = geometry_score

        straight_rescue_eligible = bool(
            not args.disable_straight_rescue
            and not soma_guard_conflict
            and min(start_cos, end_cos) >= args.straight_rescue_min_cos
            and curve_ratio <= args.straight_rescue_max_curve_ratio
            and score >= args.straight_rescue_min_score
            and math.isfinite(image_mean)
            and math.isfinite(image_q25)
            and image_mean >= args.straight_rescue_min_image_mean
            and image_q25 >= args.straight_rescue_min_image_q25
        )

        decision = (
            "rejected_different_somata"
            if soma_guard_conflict
            else "candidate"
        )
        candidates.append(
            {
                "candidate_id": len(candidates),
                "start_endpoint": start_id,
                "end_endpoint": end_id,
                "start_component": start_component,
                "end_component": end_component,
                "start_soma_ids": ";".join(map(str, sorted(start_soma_ids))),
                "end_soma_ids": ";".join(map(str, sorted(end_soma_ids))),
                "soma_guard_conflict": soma_guard_conflict,
                "start_row": start[0],
                "start_col": start[1],
                "end_row": end[0],
                "end_col": end[1],
                "distance": distance,
                "start_direction_cos": start_cos,
                "end_direction_cos": end_cos,
                "curve_length": curve_length,
                "curve_ratio": curve_ratio,
                "geometry_score": geometry_score,
                "probability_mean": probability_mean,
                "probability_q25": probability_q25,
                "image_mean": image_mean,
                "image_q25": image_q25,
                "evidence_score": evidence_score,
                "score": float(score),
                "straight_rescue_eligible": straight_rescue_eligible,
                "path": path,
                "decision": decision,
            }
        )

    # Retain only the strongest local alternatives. This prevents dense endpoint
    # neighborhoods from generating an unmanageable number of long alternatives.
    by_endpoint: dict[int, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["decision"] != "candidate":
            continue
        by_endpoint[int(candidate["start_endpoint"])].append(candidate)
        by_endpoint[int(candidate["end_endpoint"])].append(candidate)

    retained_ids: set[int] = set()
    for endpoint_candidates in by_endpoint.values():
        endpoint_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        retained_ids.update(
            int(item["candidate_id"])
            for item in endpoint_candidates[: args.max_candidates_per_endpoint]
        )

    return [
        candidate
        for candidate in candidates
        if candidate["decision"] != "candidate"
        or int(candidate["candidate_id"]) in retained_ids
    ]


def select_candidates(
    candidates: list[dict[str, object]],
    min_score: float,
    ambiguity_margin: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["decision"] == "candidate"
        and (
            float(candidate["score"]) >= min_score
            or bool(candidate["straight_rescue_eligible"])
        )
    ]
    by_endpoint: dict[int, list[dict[str, object]]] = defaultdict(list)
    for candidate in eligible:
        by_endpoint[int(candidate["start_endpoint"])].append(candidate)
        by_endpoint[int(candidate["end_endpoint"])].append(candidate)

    ambiguous_endpoints: set[int] = set()
    for endpoint_id, endpoint_candidates in by_endpoint.items():
        endpoint_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        if (
            len(endpoint_candidates) >= 2
            and float(endpoint_candidates[0]["score"])
            - float(endpoint_candidates[1]["score"])
            < ambiguity_margin
        ):
            ambiguous_endpoints.add(endpoint_id)

    ambiguous: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    used_endpoints: set[int] = set()

    for candidate in sorted(eligible, key=lambda item: float(item["score"]), reverse=True):
        start_id = int(candidate["start_endpoint"])
        end_id = int(candidate["end_endpoint"])
        if start_id in ambiguous_endpoints or end_id in ambiguous_endpoints:
            candidate["decision"] = "ambiguous_competing_candidate"
            ambiguous.append(candidate)
            continue
        if start_id in used_endpoints or end_id in used_endpoints:
            candidate["decision"] = "rejected_endpoint_already_used"
            continue
        candidate["decision"] = (
            "accepted_straight_rescue"
            if bool(candidate["straight_rescue_eligible"])
            and float(candidate["score"]) < min_score
            else "accepted"
        )
        accepted.append(candidate)
        used_endpoints.update((start_id, end_id))

    for candidate in candidates:
        if (
            candidate["decision"] == "candidate"
            and float(candidate["score"]) < min_score
            and not bool(candidate["straight_rescue_eligible"])
        ):
            candidate["decision"] = "rejected_score"

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
    candidate_mask: np.ndarray,
    cross_soma_rejected_mask: np.ndarray,
    ambiguous_mask: np.ndarray,
    added_mask: np.ndarray,
    straight_rescue_mask: np.ndarray,
    preview_max_size: int,
) -> None:
    step = max(1, int(math.ceil(max(raw_skeleton.shape) / preview_max_size)))
    height = math.ceil(raw_skeleton.shape[0] / step)
    width = math.ceil(raw_skeleton.shape[1] / step)

    if original is not None:
        background = original[::step, ::step].astype(np.float32)
        background = background[:height, :width]
        low, high = np.percentile(background, [1.0, 99.5])
        background = np.clip((background - low) / max(high - low, 1e-6), 0.0, 1.0)
    else:
        background = np.zeros((height, width), dtype=np.float32)

    overlay = np.repeat(background[..., None], 3, axis=2) * 0.72
    raw_small = reduced_mask(raw_skeleton, step)[:height, :width]
    soma_small = reduced_mask(soma, step)[:height, :width]
    candidate_small = reduced_mask(candidate_mask, step)[:height, :width]
    rejected_small = reduced_mask(cross_soma_rejected_mask, step)[:height, :width]
    ambiguous_small = reduced_mask(ambiguous_mask, step)[:height, :width]
    added_small = reduced_mask(added_mask, step)[:height, :width]
    straight_small = reduced_mask(straight_rescue_mask, step)[:height, :width]

    overlay[raw_small] = (0.15, 0.85, 0.25)
    overlay[candidate_small] = (0.20, 0.55, 1.00)
    overlay[rejected_small] = (1.00, 0.05, 0.05)
    overlay[ambiguous_small] = (1.00, 0.45, 0.05)
    overlay[added_small] = (1.00, 0.95, 0.10)
    overlay[straight_small] = (0.70, 0.25, 1.00)
    overlay[soma_small] = (1.00, 0.00, 0.85)

    endpoint_preview = np.zeros_like(raw_skeleton, dtype=bool)
    endpoint_preview[endpoints[:, 0], endpoints[:, 1]] = True
    endpoint_preview = ndi.binary_dilation(endpoint_preview, iterations=max(1, step))
    endpoint_small = endpoint_preview[::step, ::step][:height, :width]
    overlay[endpoint_small] = (0.00, 1.00, 1.00)

    plt.figure(figsize=(16, 10), dpi=160)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(
        "Stufe 1: grün=Original-Skeleton, gelb=akzeptiert, "
        "violett=Straight Rescue, "
        "rot=Cross-Soma blockiert, orange=mehrdeutig, "
        "blau=Kandidaten, magenta=Soma, cyan=Endpunkte"
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def write_candidates(path: Path, candidates: list[dict[str, object]]) -> None:
    fields = [
        "candidate_id",
        "start_endpoint",
        "end_endpoint",
        "start_component",
        "end_component",
        "start_soma_ids",
        "end_soma_ids",
        "soma_guard_conflict",
        "start_row",
        "start_col",
        "end_row",
        "end_col",
        "distance",
        "start_direction_cos",
        "end_direction_cos",
        "curve_length",
        "curve_ratio",
        "geometry_score",
        "probability_mean",
        "probability_q25",
        "image_mean",
        "image_q25",
        "evidence_score",
        "score",
        "straight_rescue_eligible",
        "decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({field: candidate[field] for field in fields})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ITERATIVES POSTPROCESSING: {VERSION}")
    print(f"RUNNING FILE: {Path(__file__).resolve()}")

    prediction = load_2d_tiff(args.prediction)
    values = set(np.unique(prediction).astype(int).tolist())
    if not values.issubset({0, 1, 2}):
        raise RuntimeError(
            f"{args.prediction.name}: Erwartet wurden 0/1/2, gefunden {sorted(values)}"
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
        probability = load_skeleton_probability(args.probabilities, prediction.shape)

    original = None
    original_scaling = None
    if args.original is not None:
        original = load_2d_tiff(args.original)
        if original.shape != prediction.shape:
            raise RuntimeError(
                f"Original {original.shape} und Prediction {prediction.shape} "
                "haben unterschiedliche Größen"
            )
        original_scaling = image_scaling(original)

    print(f"Bildgröße:              {prediction.shape}")
    print(f"Original-Skeletonpixel: {int(raw_skeleton.sum())}")
    print(f"Endpunkte vorher:       {len(endpoints_before)}")
    print(f"Skeletonkomponenten:    {component_count}")
    print(f"Soma-Instanzen:         {soma_count}")
    print(
        "Cross-Soma-Schutz:     "
        + ("aus" if args.disable_soma_guard else "aktiv")
    )
    print(f"Kurze Lücken bis:       {args.max_gap:.1f} px")

    candidates = build_candidates(
        thin=thin,
        soma=soma,
        endpoints=endpoints_before,
        component_labels=component_labels,
        component_soma_ids=component_soma_ids,
        soma_guard_enabled=not args.disable_soma_guard,
        probability=probability,
        original=original,
        original_scaling=original_scaling,
        args=args,
    )
    accepted, ambiguous = select_candidates(
        candidates,
        min_score=args.min_score,
        ambiguity_margin=args.ambiguity_margin,
    )
    cross_soma_rejected = [
        candidate
        for candidate in candidates
        if candidate["decision"] == "rejected_different_somata"
    ]
    straight_rescued = [
        candidate
        for candidate in accepted
        if candidate["decision"] == "accepted_straight_rescue"
    ]

    candidate_mask = mask_from_paths(prediction.shape, candidates)
    cross_soma_rejected_mask = mask_from_paths(
        prediction.shape,
        cross_soma_rejected,
    )
    ambiguous_mask = mask_from_paths(prediction.shape, ambiguous)
    added_mask = mask_from_paths(prediction.shape, accepted)
    straight_rescue_mask = mask_from_paths(
        prediction.shape,
        straight_rescued,
    )

    if args.connection_radius > 0:
        added_mask = ndi.binary_dilation(
            added_mask,
            iterations=args.connection_radius,
        )
        straight_rescue_mask = ndi.binary_dilation(
            straight_rescue_mask,
            iterations=args.connection_radius,
        )

    # Unverhandelbare No-loss-Regel: finale Materialmaske ist immer OR.
    added_mask &= ~raw_skeleton
    straight_rescue_mask &= ~raw_skeleton
    completed = raw_skeleton | added_mask
    if not np.all(completed[raw_skeleton]):
        raise AssertionError("No-loss-Garantie verletzt")

    completed_thin = skeletonize(completed)
    endpoints_after = endpoint_coordinates(completed_thin)

    semantic = np.zeros(prediction.shape, dtype=np.uint8)
    semantic[completed] = 1
    semantic[soma] = 2

    endpoint_before_mask = np.zeros(prediction.shape, dtype=np.uint8)
    endpoint_before_mask[endpoints_before[:, 0], endpoints_before[:, 1]] = 1
    endpoint_after_mask = np.zeros(prediction.shape, dtype=np.uint8)
    endpoint_after_mask[endpoints_after[:, 0], endpoints_after[:, 1]] = 1

    save_tiff(
        args.output_dir / "01_original_skeleton_fullwidth_NO_LOSS.tif",
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
        args.output_dir / "04_polynomial_candidates.tif",
        candidate_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "05_polynomial_ambiguous.tif",
        ambiguous_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "05a_cross_soma_connections_REJECTED.tif",
        cross_soma_rejected_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "06_added_short_polynomial_connections.tif",
        added_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "06a_straight_rescue_connections.tif",
        straight_rescue_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "07_completed_skeleton_stage1_NO_LOSS.tif",
        completed.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "08_skeleton_and_soma_stage1_0-1-2.tif",
        semantic,
    )
    save_tiff(
        args.output_dir / "09_endpoints_after.tif",
        endpoint_after_mask,
    )

    write_candidates(
        args.output_dir / "stage1_connection_candidates.csv",
        candidates,
    )

    save_qc(
        args.output_dir / "stage1_qc_overlay.png",
        original=original,
        raw_skeleton=raw_skeleton,
        soma=soma,
        endpoints=endpoints_before,
        candidate_mask=candidate_mask,
        cross_soma_rejected_mask=cross_soma_rejected_mask,
        ambiguous_mask=ambiguous_mask,
        added_mask=added_mask,
        straight_rescue_mask=straight_rescue_mask,
        preview_max_size=args.preview_max_size,
    )

    summary = {
        "version": VERSION,
        "prediction": str(args.prediction),
        "probabilities": str(args.probabilities) if args.probabilities else None,
        "original": str(args.original) if args.original else None,
        "parameters": {
            "min_gap": args.min_gap,
            "max_gap": args.max_gap,
            "orientation_steps": args.orientation_steps,
            "tangent_scale": args.tangent_scale,
            "min_direction_cos": args.min_direction_cos,
            "max_curve_ratio": args.max_curve_ratio,
            "min_score": args.min_score,
            "ambiguity_margin": args.ambiguity_margin,
            "max_candidates_per_endpoint": args.max_candidates_per_endpoint,
            "connection_radius": args.connection_radius,
            "soma_contact_radius": args.soma_contact_radius,
            "soma_guard_enabled": not args.disable_soma_guard,
            "straight_rescue_enabled": not args.disable_straight_rescue,
            "straight_rescue_min_cos": args.straight_rescue_min_cos,
            "straight_rescue_max_curve_ratio": args.straight_rescue_max_curve_ratio,
            "straight_rescue_min_score": args.straight_rescue_min_score,
            "straight_rescue_min_image_mean": args.straight_rescue_min_image_mean,
            "straight_rescue_min_image_q25": args.straight_rescue_min_image_q25,
        },
        "original_skeleton_pixels": int(raw_skeleton.sum()),
        "candidate_connections": len(candidates),
        "accepted_connections": len(accepted),
        "ambiguous_connections": len(ambiguous),
        "cross_soma_rejected_connections": len(cross_soma_rejected),
        "straight_rescued_connections": len(straight_rescued),
        "added_pixels": int(added_mask.sum()),
        "completed_skeleton_pixels": int(completed.sum()),
        "removed_original_pixels": int(np.sum(raw_skeleton & ~completed)),
        "endpoints_before": int(len(endpoints_before)),
        "endpoints_after": int(len(endpoints_after)),
    }
    with (args.output_dir / "stage1_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 72)
    print("STUFE 1 ABGESCHLOSSEN")
    print("=" * 72)
    print(f"Kandidaten:               {len(candidates)}")
    print(f"Akzeptierte Verbindungen: {len(accepted)}")
    print(f"Mehrdeutige Verbindungen: {len(ambiguous)}")
    print(f"Cross-Soma blockiert:      {len(cross_soma_rejected)}")
    print(f"Straight Rescue:           {len(straight_rescued)}")
    print(f"Neu ergänzte Pixel:       {int(added_mask.sum())}")
    print(f"Endpunkte vorher/nachher: {len(endpoints_before)} / {len(endpoints_after)}")
    print(f"ENTFERNTE Originalpixel:  {int(np.sum(raw_skeleton & ~completed))}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
