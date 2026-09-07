from __future__ import annotations

"""Create and evaluate staged one-pixel skeleton connectivity candidates.

The script implements the two inference-only experiments in the connectivity
test ladder:

* R2 grows the unchanged R0 skeleton through weak class-1 probabilities and
  thins the result back to one pixel.
* R3 adds conservative, straight, one-pixel connections between mutually
  nearest components when endpoint geometry supports the connection.

It also creates the calibration manifest and writes identical automatic
guard-rail metrics for every candidate. Existing checkpoints and prediction
folders are never deleted.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.morphology import skeletonize


SCRIPT_VERSION = "skeleton-connectivity-experiments-v1-2026-08-27"
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)
NEIGHBOR_KERNEL = CONNECTIVITY_8.copy()
NEIGHBOR_KERNEL[1, 1] = 0
SKELETON = 1
SOMA = 2


@dataclass(frozen=True)
class ConnectionCandidate:
    source_component: int
    target_component: int | None
    target_soma: int | None
    source: tuple[int, int]
    target: tuple[int, int]
    kind: str
    distance: float
    source_direction_cosine: float
    target_direction_cosine: float | None
    smaller_component_size: int

    @property
    def entity_pair(self) -> tuple[str, str]:
        left = f"skeleton:{self.source_component}"
        if self.target_component is not None:
            right = f"skeleton:{self.target_component}"
        else:
            right = f"soma:{self.target_soma}"
        return tuple(sorted((left, right)))


def read_2d(path: Path) -> np.ndarray:
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D TIFF, got {array.shape}: {path}")
    return array


def find_tiff(directory: Path, case_id: str) -> Path:
    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidate = directory / f"{case_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing TIFF for {case_id} in {directory}")


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        key = "probabilities" if "probabilities" in archive else next(iter(archive))
        probabilities = np.squeeze(np.asarray(archive[key]))
    if probabilities.ndim != 3 or probabilities.shape[0] < 3:
        raise ValueError(
            f"Expected validation probabilities shaped (C,H,W), got "
            f"{probabilities.shape}: {path}"
        )
    return probabilities


def case_ids_from_predictions(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {directory}")
    case_ids = sorted(
        {
            path.stem
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        }
    )
    if not case_ids:
        raise ValueError(f"No TIFF predictions found in {directory}")
    return case_ids


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    case_ids = [str(row.get("case_id", "")).strip() for row in rows]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError(f"Manifest is empty or malformed: {path}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Manifest contains duplicate case IDs: {path}")
    return case_ids


def read_validation_ids(split_file: Path, fold: int) -> list[str]:
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or fold < 0 or fold >= len(payload):
        raise ValueError(f"Fold {fold} is not present in {split_file}")
    validation = payload[fold].get("val")
    if not isinstance(validation, list) or not validation:
        raise ValueError(f"Fold {fold} has no validation cases in {split_file}")
    return [str(case_id) for case_id in validation]


def union_components(label: np.ndarray) -> int:
    return int(ndi.label(label > 0, structure=CONNECTIVITY_8)[1])


def select_evenly(items: Sequence[str], count: int) -> list[str]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count + 2)[1:-1]
    return [items[int(round(index))] for index in indices]


def select_calibration_cases(
    baseline_dir: Path,
    split_file: Path,
    formal_manifest: Path,
    output_manifest: Path,
    fold: int = 0,
) -> list[dict[str, object]]:
    """Select six fragmented, three middle and three clean non-formal cases."""

    validation_ids = read_validation_ids(split_file, fold)
    formal = set(read_manifest(formal_manifest))
    available = set(case_ids_from_predictions(baseline_dir))
    candidates = [
        case_id
        for case_id in validation_ids
        if case_id not in formal and case_id in available
    ]
    if len(candidates) < 12:
        raise ValueError(
            f"Need at least 12 validation cases outside the formal panel, found "
            f"{len(candidates)}"
        )

    component_counts = {
        case_id: union_components(read_2d(find_tiff(baseline_dir, case_id)))
        for case_id in candidates
    }
    by_fragmentation = sorted(
        candidates,
        key=lambda case_id: (-component_counts[case_id], case_id),
    )
    fragmented = by_fragmentation[:6]
    used = set(fragmented)

    clean_pool = sorted(
        (case_id for case_id in candidates if case_id not in used),
        key=lambda case_id: (component_counts[case_id], case_id),
    )
    clean = clean_pool[:3]
    used.update(clean)

    middle_pool = sorted(
        (case_id for case_id in candidates if case_id not in used),
        key=lambda case_id: (component_counts[case_id], case_id),
    )
    middle = select_evenly(middle_pool, 3)

    rows: list[dict[str, object]] = []
    for group, selected in (
        ("fragmented", fragmented),
        ("middle", middle),
        ("clean", clean),
    ):
        for case_id in selected:
            rows.append(
                {
                    "order": len(rows) + 1,
                    "case_id": case_id,
                    "fold": fold,
                    "selection_group": group,
                    "baseline_union_components_8": component_counts[case_id],
                }
            )

    if len(rows) != 12 or len({row["case_id"] for row in rows}) != 12:
        raise RuntimeError("Calibration selection did not produce 12 unique cases")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def safe_divide(numerator: float, denominator: float, empty_value: float) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return safe_divide(
        2 * int(np.count_nonzero(prediction & target)), denominator, 1.0
    )


def soma_attached_fraction(label: np.ndarray) -> float:
    skeleton = label == SKELETON
    skeleton_pixels = int(skeleton.sum())
    if not skeleton_pixels:
        return 0.0
    material, _ = ndi.label(label > 0, structure=CONNECTIVITY_8)
    soma_ids = np.unique(material[label == SOMA])
    soma_ids = soma_ids[soma_ids > 0]
    attached = skeleton & np.isin(material, soma_ids)
    return safe_divide(int(attached.sum()), skeleton_pixels, 0.0)


def endpoint_count(skeleton: np.ndarray) -> int:
    neighbors = ndi.convolve(
        skeleton.astype(np.uint8), NEIGHBOR_KERNEL, mode="constant", cval=0
    )
    return int(np.count_nonzero(skeleton & (neighbors == 1)))


def one_pixel_fraction(skeleton: np.ndarray) -> float:
    pixels = int(skeleton.sum())
    if not pixels:
        return 1.0
    thin = skeletonize(skeleton)
    return safe_divide(int(thin.sum()), pixels, 1.0)


def prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    predicted_skeleton = prediction == SKELETON
    target_skeleton = target == SKELETON
    predicted_soma = prediction == SOMA
    target_soma = target == SOMA
    intersection = int(np.count_nonzero(predicted_skeleton & target_skeleton))
    predicted_pixels = int(predicted_skeleton.sum())
    target_pixels = int(target_skeleton.sum())
    return {
        "skeleton_pixels": predicted_pixels,
        "empty_skeleton": int(predicted_pixels == 0),
        "skeleton_precision": safe_divide(intersection, predicted_pixels, target_pixels == 0),
        "skeleton_recall": safe_divide(intersection, target_pixels, predicted_pixels == 0),
        "skeleton_dice": binary_dice(predicted_skeleton, target_skeleton),
        "soma_dice": binary_dice(predicted_soma, target_soma),
        "union_components_8": union_components(prediction),
        "soma_attached_fraction": soma_attached_fraction(prediction),
        "endpoints_8": endpoint_count(predicted_skeleton),
        "one_pixel_fraction": one_pixel_fraction(predicted_skeleton),
    }


def metrics_row(
    case_id: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    baseline_metrics = prediction_metrics(baseline, target)
    candidate_metrics = prediction_metrics(candidate, target)
    row: dict[str, object] = {"case_id": case_id}
    row.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
    row.update({f"candidate_{key}": value for key, value in candidate_metrics.items()})
    return row


def aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric table")
    metric_names = [
        "skeleton_pixels",
        "empty_skeleton",
        "skeleton_precision",
        "skeleton_recall",
        "skeleton_dice",
        "soma_dice",
        "union_components_8",
        "soma_attached_fraction",
        "endpoints_8",
        "one_pixel_fraction",
    ]
    summary: dict[str, object] = {"cases": len(rows)}
    for prefix in ("baseline", "candidate"):
        for metric in metric_names:
            values = np.asarray(
                [float(row[f"{prefix}_{metric}"]) for row in rows],
                dtype=np.float64,
            )
            if metric == "empty_skeleton":
                summary[f"{prefix}_{metric}"] = int(values.sum())
            else:
                summary[f"{prefix}_mean_{metric}"] = float(values.mean())

    baseline_pixels = float(summary["baseline_mean_skeleton_pixels"])
    candidate_pixels = float(summary["candidate_mean_skeleton_pixels"])
    # Keep the summary strict-JSON compatible even for the pathological case in
    # which the baseline is empty but the candidate is not.  Such a candidate
    # must fail the growth gate; a large finite sentinel expresses that without
    # emitting Infinity, which many JSON readers reject.
    skeleton_growth_ratio = safe_divide(
        candidate_pixels,
        baseline_pixels,
        1.0 if candidate_pixels == 0 else 1.0e12,
    )
    soma_dice_drop = float(summary["baseline_mean_soma_dice"]) - float(
        summary["candidate_mean_soma_dice"]
    )
    one_pixel_drop = float(summary["baseline_mean_one_pixel_fraction"]) - float(
        summary["candidate_mean_one_pixel_fraction"]
    )
    additional_empty = int(summary["candidate_empty_skeleton"]) - int(
        summary["baseline_empty_skeleton"]
    )
    gates = {
        "no_additional_empty_skeletons": additional_empty <= 0,
        "soma_dice_drop_at_most_0p005": soma_dice_drop <= 0.005 + 1e-12,
        "skeleton_pixel_growth_at_most_25_percent": skeleton_growth_ratio <= 1.25 + 1e-12,
        "one_pixel_fraction_drop_at_most_0p02": one_pixel_drop <= 0.02 + 1e-12,
    }
    summary.update(
        {
            "additional_empty_skeletons": additional_empty,
            "soma_dice_drop": soma_dice_drop,
            "skeleton_pixel_growth_ratio": skeleton_growth_ratio,
            "one_pixel_fraction_drop": one_pixel_drop,
            "automatic_gates": gates,
            "automatic_gate_pass": all(gates.values()),
        }
    )
    return summary


def threshold_token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def grow_hysteresis_from_baseline(
    baseline: np.ndarray,
    skeleton_probability: np.ndarray,
    threshold_low: float,
) -> np.ndarray:
    if not 0 <= threshold_low < 1:
        raise ValueError(f"T_low must be in [0,1), got {threshold_low}")
    soma = baseline == SOMA
    seed = (baseline == SKELETON) & ~soma
    weak = (skeleton_probability >= threshold_low) & ~soma
    if not seed.any():
        thin = np.zeros(seed.shape, dtype=bool)
    else:
        grown = ndi.binary_propagation(
            seed,
            structure=CONNECTIVITY_8,
            mask=weak | seed,
        )
        thin = skeletonize(grown) & ~soma
    semantic = np.zeros(baseline.shape, dtype=np.uint8)
    semantic[thin] = SKELETON
    semantic[soma] = SOMA
    return semantic


def endpoint_direction(
    skeleton: np.ndarray,
    endpoint: tuple[int, int],
    steps: int = 5,
) -> np.ndarray | None:
    current = endpoint
    previous: tuple[int, int] | None = None
    last = endpoint
    height, width = skeleton.shape
    for _ in range(steps):
        neighbors: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                point = (current[0] + dy, current[1] + dx)
                if not (0 <= point[0] < height and 0 <= point[1] < width):
                    continue
                if skeleton[point] and point != previous:
                    neighbors.append(point)
        if len(neighbors) != 1:
            break
        previous, current = current, neighbors[0]
        last = current
    vector = np.asarray(endpoint, dtype=np.float64) - np.asarray(last, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return None
    return vector / norm


def cosine_to_target(
    direction: np.ndarray | None,
    source: tuple[int, int],
    target: tuple[int, int],
) -> float:
    if direction is None:
        return -1.0
    vector = np.asarray(target, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return -1.0
    return float(np.dot(direction, vector / norm))


def nearest_point(
    tree: cKDTree,
    coordinates: np.ndarray,
    point: tuple[int, int],
    max_gap: float,
) -> tuple[tuple[int, int], float] | None:
    distance, index = tree.query(np.asarray(point), k=1, distance_upper_bound=max_gap)
    if not np.isfinite(distance) or int(index) >= len(coordinates):
        return None
    target = tuple(map(int, coordinates[int(index)]))
    return target, float(distance)


def component_soma_ids(
    component_labels: np.ndarray,
    component_count: int,
    soma_labels: np.ndarray,
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for component in range(1, component_count + 1):
        touching = ndi.binary_dilation(
            component_labels == component, structure=CONNECTIVITY_8
        ) & (soma_labels > 0)
        result[component] = set(map(int, np.unique(soma_labels[touching]))) - {0}
    return result


def build_connection_candidates(
    skeleton: np.ndarray,
    soma: np.ndarray,
    max_gap: float,
    minimum_direction_cosine: float,
) -> tuple[list[ConnectionCandidate], dict[int, set[int]], np.ndarray]:
    component_labels, component_count = ndi.label(skeleton, structure=CONNECTIVITY_8)
    component_sizes = np.bincount(component_labels.ravel(), minlength=component_count + 1)
    soma_labels, soma_count = ndi.label(soma, structure=CONNECTIVITY_8)
    attached_somas = component_soma_ids(
        component_labels, component_count, soma_labels
    )
    neighbor_counts = ndi.convolve(
        skeleton.astype(np.uint8), NEIGHBOR_KERNEL, mode="constant", cval=0
    )
    endpoints = np.argwhere(skeleton & (neighbor_counts == 1))
    directions = {
        tuple(map(int, endpoint)): endpoint_direction(
            skeleton, tuple(map(int, endpoint))
        )
        for endpoint in endpoints
    }

    component_coordinates = {
        component: np.argwhere(component_labels == component)
        for component in range(1, component_count + 1)
    }
    component_trees = {
        component: cKDTree(coordinates)
        for component, coordinates in component_coordinates.items()
        if coordinates.size
    }
    soma_coordinates = {
        soma_id: np.argwhere(soma_labels == soma_id)
        for soma_id in range(1, soma_count + 1)
    }
    soma_trees = {
        soma_id: cKDTree(coordinates)
        for soma_id, coordinates in soma_coordinates.items()
        if coordinates.size
    }

    best_by_pair: dict[tuple[str, str], ConnectionCandidate] = {}
    for endpoint_array in endpoints:
        source = tuple(map(int, endpoint_array))
        source_component = int(component_labels[source])
        source_direction = directions[source]

        for target_component, tree in component_trees.items():
            if target_component == source_component:
                continue
            nearest = nearest_point(
                tree, component_coordinates[target_component], source, max_gap
            )
            if nearest is None:
                continue
            target, distance = nearest
            source_cosine = cosine_to_target(source_direction, source, target)
            if source_cosine < minimum_direction_cosine:
                continue
            target_is_endpoint = bool(neighbor_counts[target] == 1)
            target_cosine: float | None = None
            if target_is_endpoint:
                target_direction = directions.get(target)
                target_cosine = cosine_to_target(target_direction, target, source)
                if target_cosine < minimum_direction_cosine:
                    continue
            smaller_size = int(
                min(component_sizes[source_component], component_sizes[target_component])
            )
            candidate = ConnectionCandidate(
                source_component=source_component,
                target_component=target_component,
                target_soma=None,
                source=source,
                target=target,
                kind=("endpoint_to_endpoint" if target_is_endpoint else "endpoint_to_segment"),
                distance=distance,
                source_direction_cosine=source_cosine,
                target_direction_cosine=target_cosine,
                smaller_component_size=smaller_size,
            )
            pair = candidate.entity_pair
            if pair not in best_by_pair or distance < best_by_pair[pair].distance:
                best_by_pair[pair] = candidate

        if attached_somas[source_component]:
            continue
        for soma_id, tree in soma_trees.items():
            nearest = nearest_point(tree, soma_coordinates[soma_id], source, max_gap)
            if nearest is None:
                continue
            target, distance = nearest
            source_cosine = cosine_to_target(source_direction, source, target)
            if source_cosine < minimum_direction_cosine:
                continue
            candidate = ConnectionCandidate(
                source_component=source_component,
                target_component=None,
                target_soma=soma_id,
                source=source,
                target=target,
                kind="endpoint_to_soma",
                distance=distance,
                source_direction_cosine=source_cosine,
                target_direction_cosine=None,
                smaller_component_size=int(component_sizes[source_component]),
            )
            pair = candidate.entity_pair
            if pair not in best_by_pair or distance < best_by_pair[pair].distance:
                best_by_pair[pair] = candidate

    return list(best_by_pair.values()), attached_somas, component_labels


def reconnect_semantic(
    baseline: np.ndarray,
    max_gap: float,
    minimum_direction_cosine: float = 0.707,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    skeleton = baseline == SKELETON
    soma = baseline == SOMA
    candidates, attached_somas, component_labels = build_connection_candidates(
        skeleton, soma, max_gap, minimum_direction_cosine
    )

    nearest_pair: dict[int, tuple[str, str]] = {}
    skeleton_candidates = [
        candidate for candidate in candidates if candidate.target_component is not None
    ]
    for candidate in sorted(skeleton_candidates, key=lambda item: item.distance):
        nearest_pair.setdefault(candidate.source_component, candidate.entity_pair)
        nearest_pair.setdefault(int(candidate.target_component), candidate.entity_pair)

    used_endpoints: set[tuple[int, int]] = set()
    added = np.zeros(skeleton.shape, dtype=bool)
    records: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: item.distance):
        accepted = True
        reason = "accepted"
        if candidate.distance <= 0:
            accepted, reason = False, "zero_distance"
        elif candidate.distance > min(max_gap, candidate.smaller_component_size):
            accepted, reason = False, "longer_than_smaller_component"
        elif candidate.source in used_endpoints:
            accepted, reason = False, "source_endpoint_already_used"
        elif candidate.target_component is not None:
            target_component = int(candidate.target_component)
            if nearest_pair.get(candidate.source_component) != candidate.entity_pair:
                accepted, reason = False, "not_mutual_nearest_source"
            elif nearest_pair.get(target_component) != candidate.entity_pair:
                accepted, reason = False, "not_mutual_nearest_target"
            elif attached_somas[candidate.source_component] and attached_somas[target_component]:
                accepted, reason = False, "would_merge_two_soma_attached_components"
            elif candidate.kind == "endpoint_to_endpoint" and candidate.target in used_endpoints:
                accepted, reason = False, "target_endpoint_already_used"

        rr, cc = line(
            candidate.source[0],
            candidate.source[1],
            candidate.target[0],
            candidate.target[1],
        )
        intermediate = component_labels[rr[1:-1], cc[1:-1]] if len(rr) > 2 else np.empty(0)
        allowed_components = {
            candidate.source_component,
            int(candidate.target_component or 0),
            0,
        }
        if accepted and any(int(value) not in allowed_components for value in intermediate):
            accepted, reason = False, "crosses_third_skeleton_component"

        if accepted:
            path = np.zeros(skeleton.shape, dtype=bool)
            path[rr, cc] = True
            path &= ~soma
            added |= path & ~skeleton
            used_endpoints.add(candidate.source)
            if candidate.kind == "endpoint_to_endpoint":
                used_endpoints.add(candidate.target)

        records.append(
            {
                "source_component": candidate.source_component,
                "target_component": candidate.target_component or "",
                "target_soma": candidate.target_soma or "",
                "source_y": candidate.source[0],
                "source_x": candidate.source[1],
                "target_y": candidate.target[0],
                "target_x": candidate.target[1],
                "kind": candidate.kind,
                "distance": round(candidate.distance, 4),
                "source_direction_cosine": round(candidate.source_direction_cosine, 4),
                "target_direction_cosine": (
                    ""
                    if candidate.target_direction_cosine is None
                    else round(candidate.target_direction_cosine, 4)
                ),
                "smaller_component_size": candidate.smaller_component_size,
                "accepted": int(accepted),
                "reason": reason,
                "added_pixels": int(np.count_nonzero(added)) if accepted else 0,
            }
        )

    semantic = np.zeros(baseline.shape, dtype=np.uint8)
    semantic[skeleton | added] = SKELETON
    semantic[soma] = SOMA
    return semantic, records


def prepare_variant_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.glob("*.tif")) and not overwrite:
        raise FileExistsError(
            f"Candidate TIFFs already exist in {path}. Use --overwrite or a new output root."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_metric_outputs(
    variant_dir: Path,
    rows: list[dict[str, object]],
    method: str,
    parameters: dict[str, object],
) -> dict[str, object]:
    fields = list(rows[0])
    with (variant_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate_metrics(rows)
    payload = {
        "script_version": SCRIPT_VERSION,
        "method": method,
        "parameters": parameters,
        "prediction_directory": str(variant_dir.resolve()),
        "summary": summary,
    }
    (variant_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def generate_hysteresis_variants(args: argparse.Namespace) -> list[dict[str, object]]:
    case_ids = case_ids_from_predictions(args.baseline_dir)
    variants: list[dict[str, object]] = []
    for threshold in args.t_low:
        variant_dir = args.output_root / f"tlow_{threshold_token(threshold)}"
        prepare_variant_directory(variant_dir, args.overwrite)
        rows: list[dict[str, object]] = []
        for index, case_id in enumerate(case_ids, start=1):
            print(
                f"[R2 T_low={threshold:.3f}] {index}/{len(case_ids)} {case_id}",
                flush=True,
            )
            baseline = read_2d(find_tiff(args.baseline_dir, case_id))
            target = read_2d(find_tiff(args.labels_dir, case_id))
            probability_path = args.probabilities_dir / f"{case_id}.npz"
            if not probability_path.is_file():
                raise FileNotFoundError(
                    f"Missing R0 validation probability archive: {probability_path}"
                )
            probabilities = load_probabilities(probability_path)
            if probabilities.shape[1:] != baseline.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}: probabilities={probabilities.shape}, "
                    f"baseline={baseline.shape}"
                )
            candidate = grow_hysteresis_from_baseline(
                baseline, probabilities[SKELETON], threshold
            )
            tifffile.imwrite(variant_dir / f"{case_id}.tif", candidate)
            rows.append(metrics_row(case_id, baseline, candidate, target))
        summary = write_metric_outputs(
            variant_dir,
            rows,
            method="R2 baseline-seeded 8-connected hysteresis followed by one-pixel thinning",
            parameters={"t_low": threshold, "soma_locked": True},
        )
        variants.append(
            {
                "key": f"tlow_{threshold_token(threshold)}",
                "label": f"R2 T_low={threshold:.2f}",
                "directory": str(variant_dir.resolve()),
                "automatic_gate_pass": summary["automatic_gate_pass"],
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "variants.json").write_text(
        json.dumps(variants, indent=2), encoding="utf-8"
    )
    return variants


def generate_reconnect_variants(args: argparse.Namespace) -> list[dict[str, object]]:
    case_ids = case_ids_from_predictions(args.baseline_dir)
    variants: list[dict[str, object]] = []
    for max_gap in args.max_gap:
        variant_dir = args.output_root / f"gap_{threshold_token(max_gap)}"
        prepare_variant_directory(variant_dir, args.overwrite)
        rows: list[dict[str, object]] = []
        connection_rows: list[dict[str, object]] = []
        for index, case_id in enumerate(case_ids, start=1):
            print(
                f"[R3 gap={max_gap:g}] {index}/{len(case_ids)} {case_id}",
                flush=True,
            )
            baseline = read_2d(find_tiff(args.baseline_dir, case_id))
            target = read_2d(find_tiff(args.labels_dir, case_id))
            candidate, records = reconnect_semantic(
                baseline,
                max_gap=max_gap,
                minimum_direction_cosine=args.minimum_direction_cosine,
            )
            tifffile.imwrite(variant_dir / f"{case_id}.tif", candidate)
            rows.append(metrics_row(case_id, baseline, candidate, target))
            for record in records:
                connection_rows.append({"case_id": case_id, **record})
        summary = write_metric_outputs(
            variant_dir,
            rows,
            method="R3 conservative mutual-nearest straight one-pixel reconnection",
            parameters={
                "max_gap": max_gap,
                "minimum_direction_cosine": args.minimum_direction_cosine,
                "soma_locked": True,
                "remove_baseline_pixels": False,
            },
        )
        if connection_rows:
            with (variant_dir / "connection_candidates.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(connection_rows[0]))
                writer.writeheader()
                writer.writerows(connection_rows)
        variants.append(
            {
                "key": f"gap_{threshold_token(max_gap)}",
                "label": f"R3 max gap={max_gap:g}px",
                "directory": str(variant_dir.resolve()),
                "automatic_gate_pass": summary["automatic_gate_pass"],
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "variants.json").write_text(
        json.dumps(variants, indent=2), encoding="utf-8"
    )
    return variants


def add_common_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select-calibration", help="Create the fixed 12-case calibration manifest."
    )
    select.add_argument("--baseline-dir", type=Path, required=True)
    select.add_argument("--split-file", type=Path, required=True)
    select.add_argument("--formal-manifest", type=Path, required=True)
    select.add_argument("--output-manifest", type=Path, required=True)
    select.add_argument("--fold", type=int, default=0)

    hysteresis = subparsers.add_parser(
        "hysteresis", help="Generate R2 threshold candidates and metrics."
    )
    add_common_generation_arguments(hysteresis)
    hysteresis.add_argument("--probabilities-dir", type=Path, required=True)
    hysteresis.add_argument(
        "--t-low", type=float, nargs="+", default=[0.30, 0.20, 0.10]
    )

    reconnect = subparsers.add_parser(
        "reconnect", help="Generate R3 geometric candidates and metrics."
    )
    add_common_generation_arguments(reconnect)
    reconnect.add_argument(
        "--max-gap", type=float, nargs="+", default=[6.0, 10.0]
    )
    reconnect.add_argument(
        "--minimum-direction-cosine", type=float, default=0.707
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select-calibration":
        rows = select_calibration_cases(
            args.baseline_dir,
            args.split_file,
            args.formal_manifest,
            args.output_manifest,
            args.fold,
        )
        print(f"Calibration manifest: {args.output_manifest.resolve()}")
        for row in rows:
            print(
                f"  {row['order']:>2}  {row['case_id']:<8} "
                f"{row['selection_group']:<10} "
                f"beta0={row['baseline_union_components_8']}"
            )
    elif args.command == "hysteresis":
        variants = generate_hysteresis_variants(args)
        print(json.dumps(variants, indent=2))
    elif args.command == "reconnect":
        variants = generate_reconnect_variants(args)
        print(json.dumps(variants, indent=2))
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
