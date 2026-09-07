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
from skimage.measure import block_reduce
from skimage.morphology import skeletonize


VERSION = "stage6-selective-endpoint-caps-v1-2026-07-30"
Point = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "No-loss-Abschlussstufe: schliesst nur sehr kleine Restluecken "
            "durch selektive Verdickung der beteiligten Skeleton-Endpunkte."
        )
    )
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--cap-radius", type=int, default=2)
    parser.add_argument(
        "--max-gap",
        type=float,
        help=(
            "Maximaler Endpunktabstand. Standard: 2 * cap-radius + 1.5."
        ),
    )
    parser.add_argument("--orientation-steps", type=int, default=5)
    parser.add_argument("--min-direction-cos", type=float, default=-0.35)
    parser.add_argument("--ambiguity-margin", type=float, default=0.08)
    parser.add_argument("--soma-contact-radius", type=float, default=3.0)
    parser.add_argument(
        "--disable-soma-caps",
        action="store_true",
        help="Deaktiviert kleine Endpunkt-zu-Soma-Kappen.",
    )
    parser.add_argument(
        "--disable-soma-guard",
        action="store_true",
        help="Deaktiviert den Schutz vor Verbindungen verschiedener Soma-IDs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Uebernimmt die vorgeschlagenen Kappen. Sonst nur Preview.",
    )
    parser.add_argument("--preview-max-size", type=int, default=2600)
    return parser.parse_args()


def load_2d_tiff(path: Path) -> np.ndarray:
    array = np.squeeze(tifffile.imread(path))
    if array.ndim != 2:
        raise RuntimeError(
            f"{path.name}: Erwartet wurde ein 2D-TIFF, gefunden {array.shape}"
        )
    return array


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
    tangent = (
        np.asarray(path[0], dtype=np.float64)
        - np.mean(np.asarray(path[1:], dtype=np.float64), axis=0)
    )
    norm = float(np.linalg.norm(tangent))
    return None if norm <= 1e-8 else tangent / norm


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return -1.0
    return float(np.dot(first, second) / denominator)


def component_soma_assignments(
    thin: np.ndarray,
    soma: np.ndarray,
    contact_radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, set[int]], int, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    component_labels, component_count = ndi.label(thin, structure=structure)
    soma_labels, soma_count = ndi.label(soma, structure=structure)
    assignments: dict[int, set[int]] = defaultdict(set)

    skeleton_coordinates = np.argwhere(thin).astype(np.float32)
    soma_coordinates = np.argwhere(soma).astype(np.float32)
    if skeleton_coordinates.size == 0 or soma_coordinates.size == 0:
        return (
            component_labels,
            soma_labels,
            assignments,
            int(component_count),
            int(soma_count),
        )

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
        soma_ids = soma_labels[
            valid_soma[:, 0],
            valid_soma[:, 1],
        ]
        pairs = np.unique(np.column_stack((component_ids, soma_ids)), axis=0)
        for component_id, soma_id in pairs:
            if component_id > 0 and soma_id > 0:
                assignments[int(component_id)].add(int(soma_id))

    return (
        component_labels,
        soma_labels,
        assignments,
        int(component_count),
        int(soma_count),
    )


def local_cap(
    shape: tuple[int, int],
    points: Iterable[Point],
    radius: int,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    points_list = list(points)
    row0 = max(0, min(point[0] for point in points_list) - radius - 1)
    row1 = min(
        shape[0],
        max(point[0] for point in points_list) + radius + 2,
    )
    col0 = max(0, min(point[1] for point in points_list) - radius - 1)
    col1 = min(
        shape[1],
        max(point[1] for point in points_list) + radius + 2,
    )
    seeds = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    for row, col in points_list:
        seeds[row - row0, col - col0] = True
    cap = ndi.binary_dilation(seeds, iterations=radius)
    return (row0, row1, col0, col1), cap


def cap_global_coordinates(
    bounds: tuple[int, int, int, int],
    cap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row0, _, col0, _ = bounds
    rows, cols = np.nonzero(cap)
    return rows + row0, cols + col0


def pair_cap_is_connected(
    cap: np.ndarray,
    local_start: Point,
    local_end: Point,
) -> bool:
    labels, _ = ndi.label(
        cap,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    start_label = int(labels[local_start])
    end_label = int(labels[local_end])
    return start_label > 0 and start_label == end_label


def build_pair_candidates(
    thin: np.ndarray,
    soma_labels: np.ndarray,
    endpoints: np.ndarray,
    component_labels: np.ndarray,
    component_soma_ids: dict[int, set[int]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    endpoint_points = [tuple(map(int, point)) for point in endpoints]
    tangents = [
        endpoint_outward_tangent(point, thin, args.orientation_steps)
        for point in endpoint_points
    ]
    if len(endpoint_points) < 2:
        return []

    tree = cKDTree(endpoints.astype(np.float32))
    pairs = sorted(tree.query_pairs(r=args.max_gap))
    candidates: list[dict[str, object]] = []

    for start_id, end_id in pairs:
        start = endpoint_points[start_id]
        end = endpoint_points[end_id]
        start_component = int(component_labels[start])
        end_component = int(component_labels[end])
        if start_component == end_component:
            continue

        start_soma_ids = component_soma_ids.get(start_component, set())
        end_soma_ids = component_soma_ids.get(end_component, set())
        soma_conflict = bool(
            not args.disable_soma_guard
            and start_soma_ids
            and end_soma_ids
            and len(start_soma_ids | end_soma_ids) > 1
        )
        if soma_conflict:
            continue

        start_tangent = tangents[start_id]
        end_tangent = tangents[end_id]
        if start_tangent is None or end_tangent is None:
            continue

        vector = (
            np.asarray(end, dtype=np.float64)
            - np.asarray(start, dtype=np.float64)
        )
        distance = float(np.linalg.norm(vector))
        start_cos = cosine(start_tangent, vector)
        end_cos = cosine(end_tangent, -vector)
        if min(start_cos, end_cos) < args.min_direction_cos:
            continue

        bounds, cap = local_cap(
            thin.shape,
            (start, end),
            radius=args.cap_radius,
        )
        row0, _, col0, _ = bounds
        local_start = (start[0] - row0, start[1] - col0)
        local_end = (end[0] - row0, end[1] - col0)
        if not pair_cap_is_connected(cap, local_start, local_end):
            continue

        rows, cols = cap_global_coordinates(bounds, cap)
        touched_components = set(
            component_labels[rows, cols].astype(int).tolist()
        ) - {0}
        expected_components = {start_component, end_component}
        if touched_components - expected_components:
            continue

        touched_somata = set(
            soma_labels[rows, cols].astype(int).tolist()
        ) - {0}
        known_somata = start_soma_ids | end_soma_ids
        if len(touched_somata) > 1:
            continue
        if (
            not args.disable_soma_guard
            and known_somata
            and touched_somata
            and len(known_somata | touched_somata) > 1
        ):
            continue

        distance_score = float(
            np.clip(1.0 - distance / max(args.max_gap, 1e-6), 0.0, 1.0)
        )
        direction_score = float(
            np.clip((start_cos + end_cos) / 2.0, 0.0, 1.0)
        )
        score = 0.65 * distance_score + 0.35 * direction_score
        candidates.append(
            {
                "candidate_id": len(candidates),
                "kind": "endpoint_to_endpoint",
                "start_endpoint": start_id,
                "end_endpoint": end_id,
                "start_row": start[0],
                "start_col": start[1],
                "end_row": end[0],
                "end_col": end[1],
                "distance": distance,
                "start_direction_cos": start_cos,
                "end_direction_cos": end_cos,
                "score": score,
                "bounds": bounds,
                "cap": cap,
                "decision": "candidate",
            }
        )

    return candidates


def select_pair_candidates(
    candidates: list[dict[str, object]],
    ambiguity_margin: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_endpoint: dict[int, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        by_endpoint[int(candidate["start_endpoint"])].append(candidate)
        by_endpoint[int(candidate["end_endpoint"])].append(candidate)

    best_id: dict[int, int] = {}
    ambiguous_endpoints: set[int] = set()
    for endpoint_id, options in by_endpoint.items():
        ranked = sorted(
            options,
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        best_id[endpoint_id] = int(ranked[0]["candidate_id"])
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
        candidates,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        candidate_id = int(candidate["candidate_id"])
        start_id = int(candidate["start_endpoint"])
        end_id = int(candidate["end_endpoint"])
        if start_id in ambiguous_endpoints or end_id in ambiguous_endpoints:
            candidate["decision"] = "ambiguous"
            ambiguous.append(candidate)
            continue
        if (
            best_id.get(start_id) != candidate_id
            or best_id.get(end_id) != candidate_id
        ):
            candidate["decision"] = "rejected_not_mutual"
            continue
        if start_id in used_endpoints or end_id in used_endpoints:
            candidate["decision"] = "rejected_endpoint_used"
            continue
        candidate["decision"] = "accepted"
        accepted.append(candidate)
        used_endpoints.update((start_id, end_id))
    return accepted, ambiguous


def build_soma_caps(
    thin: np.ndarray,
    endpoints: np.ndarray,
    component_labels: np.ndarray,
    soma_labels: np.ndarray,
    component_soma_ids: dict[int, set[int]],
    used_endpoint_ids: set[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    ambiguous: list[dict[str, object]] = []
    if args.disable_soma_caps:
        return accepted, ambiguous

    for endpoint_id, endpoint_array in enumerate(endpoints):
        if endpoint_id in used_endpoint_ids:
            continue
        endpoint = tuple(map(int, endpoint_array))
        component_id = int(component_labels[endpoint])
        known_somata = component_soma_ids.get(component_id, set())
        bounds, cap = local_cap(
            thin.shape,
            (endpoint,),
            radius=args.cap_radius,
        )
        rows, cols = cap_global_coordinates(bounds, cap)
        touched_somata = set(
            soma_labels[rows, cols].astype(int).tolist()
        ) - {0}
        new_somata = touched_somata - known_somata
        if not new_somata:
            continue

        item = {
            "candidate_id": f"soma_{endpoint_id}",
            "kind": "endpoint_to_soma",
            "start_endpoint": endpoint_id,
            "end_endpoint": "",
            "start_row": endpoint[0],
            "start_col": endpoint[1],
            "end_row": "",
            "end_col": "",
            "distance": float(args.cap_radius),
            "start_direction_cos": math.nan,
            "end_direction_cos": math.nan,
            "score": 1.0,
            "bounds": bounds,
            "cap": cap,
            "decision": "candidate",
        }
        if len(new_somata) > 1 or (
            not args.disable_soma_guard
            and known_somata
            and len(known_somata | new_somata) > 1
        ):
            item["decision"] = "ambiguous"
            ambiguous.append(item)
        else:
            item["decision"] = "accepted"
            accepted.append(item)
    return accepted, ambiguous


def mask_from_caps(
    shape: tuple[int, int],
    candidates: Iterable[dict[str, object]],
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        row0, row1, col0, col1 = candidate["bounds"]  # type: ignore[misc]
        result[row0:row1, col0:col1] |= candidate["cap"]  # type: ignore[operator]
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
    proposed: np.ndarray,
    ambiguous: np.ndarray,
    applied: np.ndarray,
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
    proposed_small = reduced_mask(proposed, step)[:height, :width]
    ambiguous_small = reduced_mask(ambiguous, step)[:height, :width]
    applied_small = reduced_mask(applied, step)[:height, :width]
    overlay[raw_small] = (0.15, 0.85, 0.25)
    overlay[ambiguous_small] = (1.00, 0.40, 0.05)
    overlay[proposed_small] = (1.00, 0.95, 0.10)
    if apply_mode:
        overlay[applied_small] = (0.65, 0.25, 1.00)
    overlay[soma_small] = (1.00, 0.00, 0.85)

    endpoint_mask = np.zeros_like(raw_skeleton, dtype=bool)
    if len(endpoints):
        endpoint_mask[endpoints[:, 0], endpoints[:, 1]] = True
    endpoint_small = reduced_mask(endpoint_mask, step)[:height, :width]
    overlay[endpoint_small] = (0.00, 1.00, 1.00)

    plt.figure(figsize=(16, 10), dpi=160)
    plt.imshow(overlay)
    plt.axis("off")
    mode = "APPLY" if apply_mode else "PREVIEW"
    plt.title(
        f"Stage 6 {mode}: gruen=Input, gelb=selektive Endpoint-Kappe, "
        "orange=mehrdeutig, violett=angewendet, magenta=Soma, cyan=Endpunkt"
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
        "kind",
        "start_endpoint",
        "end_endpoint",
        "start_row",
        "start_col",
        "end_row",
        "end_col",
        "distance",
        "start_direction_cos",
        "end_direction_cos",
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
    if args.cap_radius < 1:
        raise ValueError("--cap-radius muss mindestens 1 sein.")
    if args.max_gap is None:
        args.max_gap = 2.0 * args.cap_radius + 1.5
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ITERATIVES POSTPROCESSING: {VERSION}")
    print(f"RUNNING FILE: {Path(__file__).resolve()}")
    print("MODUS: " + ("APPLY" if args.apply else "PREVIEW"))

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
        soma_labels,
        component_soma_ids,
        component_count,
        soma_count,
    ) = component_soma_assignments(
        thin,
        soma,
        contact_radius=args.soma_contact_radius,
    )

    original = None
    if args.original is not None:
        original = load_2d_tiff(args.original)
        if original.shape != prediction.shape:
            raise RuntimeError(
                f"Original {original.shape} und Prediction {prediction.shape} "
                "haben unterschiedliche Groessen."
            )

    pair_candidates = build_pair_candidates(
        thin,
        soma_labels,
        endpoints_before,
        component_labels,
        component_soma_ids,
        args,
    )
    accepted_pairs, ambiguous_pairs = select_pair_candidates(
        pair_candidates,
        ambiguity_margin=args.ambiguity_margin,
    )
    used_endpoint_ids: set[int] = set()
    for candidate in accepted_pairs:
        used_endpoint_ids.add(int(candidate["start_endpoint"]))
        used_endpoint_ids.add(int(candidate["end_endpoint"]))

    soma_caps, ambiguous_soma_caps = build_soma_caps(
        thin,
        endpoints_before,
        component_labels,
        soma_labels,
        component_soma_ids,
        used_endpoint_ids,
        args,
    )
    accepted = accepted_pairs + soma_caps
    ambiguous = ambiguous_pairs + ambiguous_soma_caps

    proposed_mask = mask_from_caps(prediction.shape, accepted)
    ambiguous_mask = mask_from_caps(prediction.shape, ambiguous)
    proposed_mask &= ~raw_skeleton
    ambiguous_mask &= ~raw_skeleton
    applied_mask = (
        proposed_mask.copy()
        if args.apply
        else np.zeros_like(proposed_mask)
    )
    completed = raw_skeleton | applied_mask
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
        args.output_dir / "01_input_skeleton_NO_LOSS.tif",
        raw_skeleton.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "02_endpoints_before.tif",
        endpoint_before_mask,
    )
    save_tiff(
        args.output_dir / "03_proposed_selective_endpoint_caps.tif",
        proposed_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "04_ambiguous_endpoint_caps.tif",
        ambiguous_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "05_added_endpoint_caps_APPLIED.tif",
        applied_mask.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "06_completed_skeleton_NO_LOSS.tif",
        completed.astype(np.uint8),
    )
    save_tiff(
        args.output_dir / "07_completed_skeleton_and_soma_0-1-2.tif",
        semantic,
    )
    save_tiff(
        args.output_dir / "08_endpoints_after.tif",
        endpoint_after_mask,
    )
    write_candidates(
        args.output_dir / "stage6_endpoint_cap_candidates.csv",
        pair_candidates + soma_caps + ambiguous_soma_caps,
    )
    save_qc(
        args.output_dir / "stage6_qc_overlay.png",
        original,
        raw_skeleton,
        soma,
        endpoints_before,
        proposed_mask,
        ambiguous_mask,
        applied_mask,
        args.preview_max_size,
        args.apply,
    )

    summary = {
        "version": VERSION,
        "mode": "apply" if args.apply else "preview",
        "prediction": str(args.prediction),
        "original": str(args.original) if args.original else None,
        "parameters": {
            "cap_radius": args.cap_radius,
            "max_gap": args.max_gap,
            "min_direction_cos": args.min_direction_cos,
            "ambiguity_margin": args.ambiguity_margin,
            "soma_guard_enabled": not args.disable_soma_guard,
            "soma_caps_enabled": not args.disable_soma_caps,
        },
        "component_count": component_count,
        "soma_count": soma_count,
        "accepted_endpoint_pairs": len(accepted_pairs),
        "accepted_endpoint_to_soma_caps": len(soma_caps),
        "ambiguous_caps": len(ambiguous),
        "proposed_pixels": int(proposed_mask.sum()),
        "applied_pixels": int(applied_mask.sum()),
        "input_skeleton_pixels": int(raw_skeleton.sum()),
        "completed_skeleton_pixels": int(completed.sum()),
        "removed_original_pixels": int(np.sum(raw_skeleton & ~completed)),
        "endpoints_before": int(len(endpoints_before)),
        "endpoints_after": int(len(endpoints_after)),
    }
    with (args.output_dir / "stage6_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 72)
    print("STAGE 6 ENDPOINT-KAPPEN ABGESCHLOSSEN")
    print("=" * 72)
    print(f"Radius:                   {args.cap_radius} px")
    print(f"Akzeptierte Endpointpaare:{len(accepted_pairs):8d}")
    print(f"Endpoint-zu-Soma-Kappen: {len(soma_caps):8d}")
    print(f"Mehrdeutige Kappen:      {len(ambiguous):8d}")
    print(f"Vorgeschlagene Pixel:    {int(proposed_mask.sum()):8d}")
    print(f"Angewendete Pixel:       {int(applied_mask.sum()):8d}")
    print(
        "Endpunkte vorher/nachher:"
        f"{len(endpoints_before):8d} / {len(endpoints_after)}"
    )
    print(f"ENTFERNTE Originalpixel: {int(np.sum(raw_skeleton & ~completed)):8d}")
    if not args.apply:
        print()
        print("PREVIEW: Noch keine Kappen in den finalen Output uebernommen.")
        print("Pruefe 03_proposed_selective_endpoint_caps.tif und stage6_qc_overlay.png.")


if __name__ == "__main__":
    main()
