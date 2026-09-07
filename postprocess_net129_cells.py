from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.morphology import binary_dilation, disk, skeletonize


Point = Tuple[int, int]


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def read_2d_tiff(path: Path, projection_axis: int = 0) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    image = np.squeeze(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        print(
            f"{path.name}: 3D-Shape {image.shape}; "
            f"Max-Projektion entlang Achse {projection_axis}."
        )
        return np.max(image, axis=projection_axis)

    raise RuntimeError(
        f"{path}: 2D- oder 3D-TIFF erwartet, gefunden {image.shape}."
    )


def load_probabilities(
    path: Path,
    expected_shape: Tuple[int, int],
) -> np.ndarray:
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
                f"{path.name}: Wahrscheinlichkeiten nicht eindeutig gefunden. "
                f"Keys: {keys}"
            )

    probabilities = np.squeeze(probabilities)

    if probabilities.ndim != 3:
        raise RuntimeError(
            f"{path.name}: 3D-Array erwartet, gefunden {probabilities.shape}."
        )

    if probabilities.shape[1:] == expected_shape:
        chw = probabilities
    elif probabilities.shape[:2] == expected_shape:
        chw = np.moveaxis(probabilities, -1, 0)
    else:
        raise RuntimeError(
            f"{path.name}: Shape passt nicht.\n"
            f"Probabilities: {probabilities.shape}\n"
            f"Prediction:    {expected_shape}"
        )

    if chw.shape[0] < 3:
        raise RuntimeError(
            f"{path.name}: mindestens drei Klassen erwartet, "
            f"gefunden {chw.shape[0]}."
        )

    if not np.all(np.isfinite(chw)):
        raise RuntimeError(
            f"{path.name}: enthält NaN oder unendliche Werte."
        )

    return chw.astype(np.float32, copy=False)


def save_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        array,
        photometric="minisblack",
    )


# ---------------------------------------------------------------------
# Bild- und Maskenvorbereitung
# ---------------------------------------------------------------------

def robust_normalize(image: np.ndarray, invert: bool = False) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        result = np.zeros_like(image, dtype=np.float32)
    else:
        low, high = np.percentile(finite, [1.0, 99.5])
        if high <= low:
            low = float(finite.min())
            high = float(finite.max())

        if high <= low:
            result = np.zeros_like(image, dtype=np.float32)
        else:
            result = np.clip(
                (image - low) / (high - low),
                0.0,
                1.0,
            )

    if invert:
        result = 1.0 - result

    return result


def make_soma_instances(
    soma_mask: np.ndarray,
    min_area: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    labels, count = ndi.label(
        soma_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    result = np.zeros_like(labels, dtype=np.uint32)
    report: List[Dict[str, object]] = []
    next_id = 1

    for old_id in range(1, count + 1):
        component = labels == old_id
        area = int(component.sum())

        if area == 0:
            continue

        rows, cols = np.nonzero(component)
        keep_for_assignment = area >= min_area

        assigned_id = next_id if keep_for_assignment else 0

        report.append({
            "source_component_id": old_id,
            "soma_id": assigned_id,
            "area_pixels": area,
            "centroid_row": float(rows.mean()),
            "centroid_col": float(cols.mean()),
            "used_for_assignment": int(keep_for_assignment),
        })

        if keep_for_assignment:
            result[component] = next_id
            next_id += 1

    return result, report


# ---------------------------------------------------------------------
# Skeleton-Geometrie
# ---------------------------------------------------------------------

NEIGHBOR_OFFSETS: Tuple[Point, ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def neighbors_on_skeleton(
    skeleton: np.ndarray,
    point: Point,
) -> List[Point]:
    row, col = point
    height, width = skeleton.shape
    result: List[Point] = []

    for dr, dc in NEIGHBOR_OFFSETS:
        rr = row + dr
        cc = col + dc
        if 0 <= rr < height and 0 <= cc < width and skeleton[rr, cc]:
            result.append((rr, cc))

    return result


def endpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    degree = ndi.convolve(
        skeleton.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )
    return np.logical_and(skeleton, degree == 1)


def branchpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    degree = ndi.convolve(
        skeleton.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )
    return np.logical_and(skeleton, degree >= 3)


def trace_endpoint_direction(
    skeleton: np.ndarray,
    endpoint: Point,
    steps: int,
) -> Optional[np.ndarray]:
    path: List[Point] = [endpoint]
    previous: Optional[Point] = None
    current = endpoint

    for _ in range(steps):
        candidates = [
            point
            for point in neighbors_on_skeleton(skeleton, current)
            if point != previous
        ]

        if len(candidates) != 1:
            break

        next_point = candidates[0]
        path.append(next_point)
        previous = current
        current = next_point

    if len(path) < 2:
        return None

    end = np.asarray(path[-1], dtype=np.float32)
    start = np.asarray(path[0], dtype=np.float32)

    # Richtung nach außen: vom inneren Astpunkt zum Endpunkt.
    vector = start - end
    norm = float(np.linalg.norm(vector))

    if norm <= 0:
        return None

    return vector / norm


def angle_degrees(
    vector_a: Optional[np.ndarray],
    vector_b: np.ndarray,
) -> float:
    if vector_a is None:
        return 0.0

    norm_b = float(np.linalg.norm(vector_b))
    if norm_b <= 0:
        return 180.0

    unit_b = vector_b / norm_b
    cosine = float(np.clip(np.dot(vector_a, unit_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def polyline_length(path: Sequence[Point]) -> float:
    if len(path) < 2:
        return 0.0

    coordinates = np.asarray(path, dtype=np.float32)
    differences = np.diff(coordinates, axis=0)
    return float(np.linalg.norm(differences, axis=1).sum())


# ---------------------------------------------------------------------
# Pfadsuche und Bewertung
# ---------------------------------------------------------------------

def build_cost_image(
    skeleton: np.ndarray,
    soma_instances: np.ndarray,
    p_background: np.ndarray,
    p_skeleton: np.ndarray,
    p_soma: np.ndarray,
    original_evidence: Optional[np.ndarray],
    probability_weight: float,
    original_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if original_evidence is None:
        combined_evidence = p_skeleton.copy()
    else:
        total = probability_weight + original_weight
        if total <= 0:
            raise ValueError(
                "probability_weight + original_weight muss > 0 sein."
            )
        combined_evidence = (
            probability_weight * p_skeleton
            + original_weight * original_evidence
        ) / total

    combined_evidence = np.clip(combined_evidence, 0.0, 1.0)

    # Niedrige Kosten auf plausiblen Skeletonpixeln, hohe Kosten im Hintergrund.
    cost = (
        1.0
        + 6.0 * np.square(1.0 - combined_evidence)
        + 2.0 * p_background
        + 1.5 * p_soma
    ).astype(np.float32)

    # Bereits vorhandenes Skeleton soll ohne Umweg weiterverwendet werden können.
    cost[skeleton] = 0.05

    # Durch Somata soll der Pfad nicht hindurchlaufen.
    cost[soma_instances > 0] = 25.0

    return cost, combined_evidence


def crop_bounds(
    shape: Tuple[int, int],
    start: Point,
    end: Point,
    margin: int,
) -> Tuple[int, int, int, int]:
    height, width = shape
    row0 = max(0, min(start[0], end[0]) - margin)
    row1 = min(height, max(start[0], end[0]) + margin + 1)
    col0 = max(0, min(start[1], end[1]) - margin)
    col1 = min(width, max(start[1], end[1]) + margin + 1)
    return row0, row1, col0, col1


def shortest_path_local(
    cost: np.ndarray,
    start: Point,
    end: Point,
    margin: int,
) -> Tuple[List[Point], float]:
    row0, row1, col0, col1 = crop_bounds(
        cost.shape,
        start,
        end,
        margin,
    )

    local_cost = cost[row0:row1, col0:col1].copy()
    local_start = (start[0] - row0, start[1] - col0)
    local_end = (end[0] - row0, end[1] - col0)

    local_cost[local_start] = 0.01
    local_cost[local_end] = 0.01

    path_local, accumulated_cost = route_through_array(
        local_cost,
        local_start,
        local_end,
        fully_connected=True,
        geometric=True,
    )

    path_global = [
        (row + row0, col + col0)
        for row, col in path_local
    ]

    return path_global, float(accumulated_cost)


def evaluate_path(
    path: Sequence[Point],
    start: Point,
    end: Point,
    skeleton: np.ndarray,
    soma_instances: np.ndarray,
    combined_evidence: np.ndarray,
    p_skeleton: np.ndarray,
    original_evidence: Optional[np.ndarray],
    start_direction: Optional[np.ndarray],
    end_direction: Optional[np.ndarray],
    max_gap: float,
    max_angle: float,
    max_detour_ratio: float,
    min_mean_evidence: float,
    min_q20_evidence: float,
    target_soma_id: int = 0,
) -> Dict[str, object]:
    coordinates = np.asarray(path, dtype=np.int32)
    rows = coordinates[:, 0]
    cols = coordinates[:, 1]

    new_pixel_mask = np.logical_not(skeleton[rows, cols])
    if target_soma_id > 0:
        # Das letzte Zielpixel darf auf dem Ziel-Soma liegen.
        target_pixels = soma_instances[rows, cols] == target_soma_id
        new_pixel_mask = np.logical_and(
            new_pixel_mask,
            np.logical_not(target_pixels),
        )

    if np.any(new_pixel_mask):
        evidence_values = combined_evidence[rows[new_pixel_mask], cols[new_pixel_mask]]
        p_skeleton_values = p_skeleton[rows[new_pixel_mask], cols[new_pixel_mask]]
        if original_evidence is None:
            original_values = np.zeros_like(evidence_values)
        else:
            original_values = original_evidence[
                rows[new_pixel_mask],
                cols[new_pixel_mask],
            ]
    else:
        evidence_values = np.asarray([1.0], dtype=np.float32)
        p_skeleton_values = np.asarray([1.0], dtype=np.float32)
        original_values = np.asarray([1.0], dtype=np.float32)

    mean_evidence = float(np.mean(evidence_values))
    q20_evidence = float(np.quantile(evidence_values, 0.20))
    mean_p_skeleton = float(np.mean(p_skeleton_values))
    mean_original = float(np.mean(original_values))

    euclidean_distance = float(
        np.linalg.norm(
            np.asarray(end, dtype=np.float32)
            - np.asarray(start, dtype=np.float32)
        )
    )
    path_length = polyline_length(path)
    detour_ratio = (
        path_length / euclidean_distance
        if euclidean_distance > 0
        else float("inf")
    )

    start_vector = (
        np.asarray(end, dtype=np.float32)
        - np.asarray(start, dtype=np.float32)
    )
    end_vector = -start_vector

    start_angle = angle_degrees(start_direction, start_vector)
    end_angle = (
        angle_degrees(end_direction, end_vector)
        if end_direction is not None
        else 0.0
    )
    worst_angle = max(start_angle, end_angle)

    touched_soma_ids = sorted(
        int(value)
        for value in np.unique(soma_instances[rows, cols])
        if int(value) > 0
    )
    foreign_soma_ids = [
        soma_id
        for soma_id in touched_soma_ids
        if soma_id != target_soma_id
    ]

    distance_score = max(0.0, 1.0 - euclidean_distance / max_gap)
    angle_score = max(0.0, 1.0 - worst_angle / max_angle)
    detour_score = max(
        0.0,
        1.0 - max(0.0, detour_ratio - 1.0) / max(0.01, max_detour_ratio - 1.0),
    )

    score = (
        0.55 * mean_evidence
        + 0.15 * q20_evidence
        + 0.12 * distance_score
        + 0.10 * angle_score
        + 0.08 * detour_score
    )

    valid = (
        mean_evidence >= min_mean_evidence
        and q20_evidence >= min_q20_evidence
        and detour_ratio <= max_detour_ratio
        and worst_angle <= max_angle
        and len(foreign_soma_ids) == 0
    )

    return {
        "score": float(score),
        "valid": bool(valid),
        "euclidean_distance": euclidean_distance,
        "path_length": path_length,
        "detour_ratio": detour_ratio,
        "mean_evidence": mean_evidence,
        "q20_evidence": q20_evidence,
        "mean_p_skeleton": mean_p_skeleton,
        "mean_original": mean_original,
        "start_angle_deg": start_angle,
        "end_angle_deg": end_angle,
        "touched_soma_ids": ",".join(str(value) for value in touched_soma_ids),
        "foreign_soma_ids": ",".join(str(value) for value in foreign_soma_ids),
    }


# ---------------------------------------------------------------------
# Kandidaten erzeugen
# ---------------------------------------------------------------------

def endpoint_pair_candidates(
    skeleton: np.ndarray,
    soma_instances: np.ndarray,
    cost: np.ndarray,
    combined_evidence: np.ndarray,
    p_skeleton: np.ndarray,
    original_evidence: Optional[np.ndarray],
    max_gap: float,
    route_margin: int,
    direction_steps: int,
    max_angle: float,
    max_detour_ratio: float,
    min_mean_evidence: float,
    min_q20_evidence: float,
    round_index: int,
) -> Tuple[List[Dict[str, object]], List[Point]]:
    endpoints = [
        (int(row), int(col))
        for row, col in np.argwhere(endpoint_mask(skeleton))
    ]

    if len(endpoints) < 2:
        return [], endpoints

    endpoint_array = np.asarray(endpoints, dtype=np.float32)
    tree = cKDTree(endpoint_array)
    pairs = sorted(tree.query_pairs(r=max_gap))

    component_labels, _ = ndi.label(
        skeleton,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    directions = {
        endpoint_id: trace_endpoint_direction(
            skeleton,
            point,
            direction_steps,
        )
        for endpoint_id, point in enumerate(endpoints)
    }

    candidates: List[Dict[str, object]] = []

    for first_id, second_id in pairs:
        start = endpoints[first_id]
        end = endpoints[second_id]

        # Bereits in derselben Komponente: keine automatische Schleife erzeugen.
        if component_labels[start] == component_labels[end]:
            continue

        try:
            path, accumulated_cost = shortest_path_local(
                cost,
                start,
                end,
                route_margin,
            )
        except Exception as error:
            print(
                f"Warnung: Pfadsuche {start}->{end} fehlgeschlagen: {error}"
            )
            continue

        metrics = evaluate_path(
            path=path,
            start=start,
            end=end,
            skeleton=skeleton,
            soma_instances=soma_instances,
            combined_evidence=combined_evidence,
            p_skeleton=p_skeleton,
            original_evidence=original_evidence,
            start_direction=directions[first_id],
            end_direction=directions[second_id],
            max_gap=max_gap,
            max_angle=max_angle,
            max_detour_ratio=max_detour_ratio,
            min_mean_evidence=min_mean_evidence,
            min_q20_evidence=min_q20_evidence,
        )

        candidate = {
            "round": round_index,
            "type": "endpoint_to_endpoint",
            "start_endpoint_id": first_id,
            "end_endpoint_id": second_id,
            "start_row": start[0],
            "start_col": start[1],
            "end_row": end[0],
            "end_col": end[1],
            "target_soma_id": 0,
            "accumulated_cost": accumulated_cost,
            "path": path,
            **metrics,
        }
        candidates.append(candidate)

    return candidates, endpoints


def soma_boundary_coordinates(
    soma_instances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(soma_instances > 0):
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.uint32),
        )

    eroded = ndi.binary_erosion(
        soma_instances > 0,
        structure=np.ones((3, 3), dtype=np.uint8),
        border_value=0,
    )
    boundary = np.logical_and(soma_instances > 0, np.logical_not(eroded))
    coordinates = np.argwhere(boundary).astype(np.int32)
    labels = soma_instances[boundary].astype(np.uint32)
    return coordinates, labels


def endpoint_soma_candidates(
    skeleton: np.ndarray,
    soma_instances: np.ndarray,
    cost: np.ndarray,
    combined_evidence: np.ndarray,
    p_skeleton: np.ndarray,
    original_evidence: Optional[np.ndarray],
    max_gap: float,
    route_margin: int,
    direction_steps: int,
    max_angle: float,
    max_detour_ratio: float,
    min_mean_evidence: float,
    min_q20_evidence: float,
    round_index: int,
    endpoints: Sequence[Point],
) -> List[Dict[str, object]]:
    boundary_coordinates, boundary_labels = soma_boundary_coordinates(
        soma_instances
    )

    if len(endpoints) == 0 or boundary_coordinates.shape[0] == 0:
        return []

    tree = cKDTree(boundary_coordinates.astype(np.float32))
    directions = {
        endpoint_id: trace_endpoint_direction(
            skeleton,
            point,
            direction_steps,
        )
        for endpoint_id, point in enumerate(endpoints)
    }

    candidates: List[Dict[str, object]] = []

    for endpoint_id, start in enumerate(endpoints):
        nearby_indices = tree.query_ball_point(
            np.asarray(start, dtype=np.float32),
            r=max_gap,
        )

        if not nearby_indices:
            continue

        by_soma: Dict[int, List[int]] = defaultdict(list)
        for boundary_index in nearby_indices:
            soma_id = int(boundary_labels[boundary_index])
            by_soma[soma_id].append(boundary_index)

        for soma_id, indices in by_soma.items():
            coords = boundary_coordinates[indices]
            distances = np.linalg.norm(
                coords.astype(np.float32)
                - np.asarray(start, dtype=np.float32),
                axis=1,
            )
            best_local = int(np.argmin(distances))
            end_array = coords[best_local]
            end = (int(end_array[0]), int(end_array[1]))

            # Bereits Kontakt zum gleichen Soma: keine neue Verbindung nötig.
            if float(distances[best_local]) <= 1.5:
                continue

            try:
                path, accumulated_cost = shortest_path_local(
                    cost,
                    start,
                    end,
                    route_margin,
                )
            except Exception as error:
                print(
                    f"Warnung: Soma-Pfadsuche {start}->Soma {soma_id} "
                    f"fehlgeschlagen: {error}"
                )
                continue

            metrics = evaluate_path(
                path=path,
                start=start,
                end=end,
                skeleton=skeleton,
                soma_instances=soma_instances,
                combined_evidence=combined_evidence,
                p_skeleton=p_skeleton,
                original_evidence=original_evidence,
                start_direction=directions[endpoint_id],
                end_direction=None,
                max_gap=max_gap,
                max_angle=max_angle,
                max_detour_ratio=max_detour_ratio,
                min_mean_evidence=min_mean_evidence,
                min_q20_evidence=min_q20_evidence,
                target_soma_id=soma_id,
            )

            candidate = {
                "round": round_index,
                "type": "endpoint_to_soma",
                "start_endpoint_id": endpoint_id,
                "end_endpoint_id": -1,
                "start_row": start[0],
                "start_col": start[1],
                "end_row": end[0],
                "end_col": end[1],
                "target_soma_id": soma_id,
                "accumulated_cost": accumulated_cost,
                "path": path,
                **metrics,
            }
            candidates.append(candidate)

    return candidates


# ---------------------------------------------------------------------
# Kandidaten auswählen
# ---------------------------------------------------------------------

def select_connections(
    candidates: List[Dict[str, object]],
    min_score: float,
    ambiguity_margin: float,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    eligible = [
        candidate
        for candidate in candidates
        if bool(candidate["valid"])
        and float(candidate["score"]) >= min_score
    ]

    candidates_by_endpoint: Dict[int, List[Dict[str, object]]] = defaultdict(list)

    for candidate in eligible:
        start_id = int(candidate["start_endpoint_id"])
        candidates_by_endpoint[start_id].append(candidate)

        end_id = int(candidate["end_endpoint_id"])
        if end_id >= 0:
            candidates_by_endpoint[end_id].append(candidate)

    blocked_endpoints = set()
    ambiguous_candidate_ids = set()

    for endpoint_id, endpoint_candidates in candidates_by_endpoint.items():
        endpoint_candidates = sorted(
            endpoint_candidates,
            key=lambda item: float(item["score"]),
            reverse=True,
        )

        if len(endpoint_candidates) < 2:
            continue

        top = endpoint_candidates[0]
        second = endpoint_candidates[1]

        if (
            float(top["score"]) - float(second["score"])
            <= ambiguity_margin
        ):
            blocked_endpoints.add(endpoint_id)
            ambiguous_candidate_ids.add(id(top))
            ambiguous_candidate_ids.add(id(second))

    accepted: List[Dict[str, object]] = []
    ambiguous: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    used_endpoints = set()

    for candidate in sorted(
        candidates,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        start_id = int(candidate["start_endpoint_id"])
        end_id = int(candidate["end_endpoint_id"])

        if id(candidate) in ambiguous_candidate_ids:
            candidate["decision"] = "ambiguous_competition"
            ambiguous.append(candidate)
            continue

        if (
            start_id in blocked_endpoints
            or (end_id >= 0 and end_id in blocked_endpoints)
        ):
            candidate["decision"] = "rejected_blocked_by_ambiguity"
            rejected.append(candidate)
            continue

        if not bool(candidate["valid"]):
            candidate["decision"] = "rejected_geometry_or_evidence"
            rejected.append(candidate)
            continue

        if float(candidate["score"]) < min_score:
            candidate["decision"] = "rejected_low_score"
            rejected.append(candidate)
            continue

        if (
            start_id in used_endpoints
            or (end_id >= 0 and end_id in used_endpoints)
        ):
            candidate["decision"] = "rejected_endpoint_already_used"
            rejected.append(candidate)
            continue

        candidate["decision"] = "accepted"
        accepted.append(candidate)
        used_endpoints.add(start_id)
        if end_id >= 0:
            used_endpoints.add(end_id)

    return accepted, ambiguous, rejected


# ---------------------------------------------------------------------
# Zellzuordnung
# ---------------------------------------------------------------------

def assign_skeleton_components(
    skeleton: np.ndarray,
    soma_instances: np.ndarray,
    contact_radius: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, object]],
]:
    component_labels, component_count = ndi.label(
        skeleton,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_labels = component_labels.astype(np.uint32)

    if np.any(soma_instances > 0):
        distance, nearest_indices = ndi.distance_transform_edt(
            soma_instances == 0,
            return_indices=True,
        )
        nearest_soma = soma_instances[
            nearest_indices[0],
            nearest_indices[1],
        ]
    else:
        distance = np.full(
            skeleton.shape,
            np.inf,
            dtype=np.float32,
        )
        nearest_soma = np.zeros_like(
            soma_instances,
            dtype=np.uint32,
        )

    component_to_somata: Dict[int, List[int]] = {}
    report: List[Dict[str, object]] = []
    ambiguous_soma_ids = set()

    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        contact_pixels = np.logical_and(
            component,
            distance <= contact_radius,
        )
        soma_ids = sorted(
            int(value)
            for value in np.unique(nearest_soma[contact_pixels])
            if int(value) > 0
        )
        component_to_somata[component_id] = soma_ids

        if len(soma_ids) > 1:
            ambiguous_soma_ids.update(soma_ids)

        report.append({
            "skeleton_component_id": component_id,
            "skeleton_pixels": int(component.sum()),
            "estimated_length_pixels": float(component.sum()),
            "number_of_somata": len(soma_ids),
            "soma_ids": ",".join(str(value) for value in soma_ids),
            "status": (
                "confident"
                if len(soma_ids) == 1
                else "ambiguous"
                if len(soma_ids) > 1
                else "unassigned"
            ),
        })

    confident_instances = np.zeros(
        skeleton.shape,
        dtype=np.uint32,
    )
    all_assignable_instances = np.zeros(
        skeleton.shape,
        dtype=np.uint32,
    )
    status_map = np.zeros(
        skeleton.shape,
        dtype=np.uint8,
    )
    ambiguous_mask = np.zeros(
        skeleton.shape,
        dtype=bool,
    )
    unassigned_mask = np.zeros(
        skeleton.shape,
        dtype=bool,
    )

    # Soma selbst eintragen.
    for soma_id in [
        int(value)
        for value in np.unique(soma_instances)
        if int(value) > 0
    ]:
        soma_region = soma_instances == soma_id
        all_assignable_instances[soma_region] = soma_id

        if soma_id in ambiguous_soma_ids:
            status_map[soma_region] = 2
            ambiguous_mask[soma_region] = True
        else:
            confident_instances[soma_region] = soma_id
            status_map[soma_region] = 1

    for component_id, soma_ids in component_to_somata.items():
        component = component_labels == component_id

        if len(soma_ids) == 0:
            status_map[component] = 3
            unassigned_mask[component] = True
        elif len(soma_ids) > 1:
            status_map[component] = 2
            ambiguous_mask[component] = True
        else:
            soma_id = soma_ids[0]
            all_assignable_instances[component] = soma_id

            if soma_id in ambiguous_soma_ids:
                status_map[component] = 2
                ambiguous_mask[component] = True
            else:
                confident_instances[component] = soma_id
                status_map[component] = 1

    return (
        component_labels,
        confident_instances,
        all_assignable_instances,
        status_map,
        ambiguous_mask,
        unassigned_mask,
        report,
    )


# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------

def save_qc_overlay(
    path: Path,
    original: Optional[np.ndarray],
    soma_instances: np.ndarray,
    status_map: np.ndarray,
    added_connections: np.ndarray,
    ambiguous_connection_candidates: np.ndarray,
) -> None:
    if original is None:
        base = np.zeros(
            (*status_map.shape, 3),
            dtype=np.float32,
        )
    else:
        normalized = robust_normalize(original)
        base = np.repeat(
            normalized[..., None],
            3,
            axis=2,
        )
        base *= 0.55

    overlay = base.copy()

    # Zellstatus
    confident = status_map == 1
    ambiguous = status_map == 2
    unassigned = status_map == 3
    soma = soma_instances > 0

    overlay[confident] = (
        0.15 * overlay[confident]
        + 0.85 * np.asarray([0.0, 1.0, 0.0])
    )
    overlay[ambiguous] = (
        0.15 * overlay[ambiguous]
        + 0.85 * np.asarray([1.0, 0.35, 0.0])
    )
    overlay[unassigned] = (
        0.15 * overlay[unassigned]
        + 0.85 * np.asarray([0.0, 0.9, 1.0])
    )
    overlay[soma] = (
        0.10 * overlay[soma]
        + 0.90 * np.asarray([1.0, 0.0, 1.0])
    )

    # Automatisch ergänzte, akzeptierte Verbindungen: gelb.
    overlay[added_connections] = np.asarray([1.0, 1.0, 0.0])

    # Nicht automatisch entschiedene konkurrierende Kandidaten: blau.
    overlay[ambiguous_connection_candidates] = np.asarray([0.1, 0.3, 1.0])

    figure, axis = plt.subplots(figsize=(14, 14))
    axis.imshow(np.clip(overlay, 0.0, 1.0))
    axis.set_title(
        "Postprocessing QC\n"
        "Grün=zugeordnet | Orange=mehrdeutig | Cyan=ohne Soma | "
        "Magenta=Soma | Gelb=ergänzt | Blau=ungeklärter Kandidat"
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Konservatives Postprocessing für Netz 129: "
            "Skeletonlücken schließen, Skeleton-Enden mit Somata verbinden, "
            "Zellen zuordnen und mehrdeutige Fälle markieren."
        )
    )

    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Harte Netz-129-Vorhersage als TIFF: 0=BG, 1=Skeleton, 2=Soma.",
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=None,
        help=(
            "Netz-129-NPZ mit Klassenwahrscheinlichkeiten. "
            "Standard: gleichnamige NPZ neben der Prediction."
        ),
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=None,
        help=(
            "Originalbild oder exakt dieselbe Max-Projektion, die Netz 129 sah. "
            "Stark empfohlen."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--projection-axis",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dark-structures",
        action="store_true",
        help="Setzen, falls Zellen im Original dunkel statt hell sind.",
    )

    # Konservative Standardwerte. Alle Abstände sind Pixel.
    parser.add_argument("--endpoint-gap", type=float, default=18.0)
    parser.add_argument("--soma-gap", type=float, default=14.0)
    parser.add_argument("--contact-radius", type=float, default=3.0)
    parser.add_argument("--route-margin", type=int, default=12)
    parser.add_argument("--direction-steps", type=int, default=7)
    parser.add_argument("--max-angle", type=float, default=75.0)
    parser.add_argument("--max-detour-ratio", type=float, default=1.65)
    parser.add_argument("--min-mean-evidence", type=float, default=0.28)
    parser.add_argument("--min-q20-evidence", type=float, default=0.08)
    parser.add_argument("--min-score", type=float, default=0.36)
    parser.add_argument("--ambiguity-margin", type=float, default=0.055)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-soma-area", type=int, default=1)
    parser.add_argument("--probability-weight", type=float, default=0.70)
    parser.add_argument("--original-weight", type=float, default=0.30)

    args = parser.parse_args()

    prediction_path = args.prediction.resolve()
    probability_path = (
        args.probabilities.resolve()
        if args.probabilities is not None
        else prediction_path.with_suffix(".npz")
    )
    original_path = (
        args.original.resolve()
        if args.original is not None
        else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    if not probability_path.exists():
        raise FileNotFoundError(
            f"Wahrscheinlichkeitsdatei fehlt: {probability_path}"
        )
    if original_path is not None and not original_path.exists():
        raise FileNotFoundError(original_path)

    prediction = read_2d_tiff(
        prediction_path,
        projection_axis=args.projection_axis,
    ).astype(np.uint8)

    unique_values = set(
        np.unique(prediction).astype(int).tolist()
    )
    if not unique_values.issubset({0, 1, 2}):
        raise RuntimeError(
            f"Prediction enthält unerwartete Werte: {sorted(unique_values)}"
        )

    probabilities = load_probabilities(
        probability_path,
        expected_shape=prediction.shape,
    )

    p_background = np.clip(probabilities[0], 0.0, 1.0)
    p_skeleton = np.clip(probabilities[1], 0.0, 1.0)
    p_soma = np.clip(probabilities[2], 0.0, 1.0)

    original: Optional[np.ndarray] = None
    original_evidence: Optional[np.ndarray] = None

    if original_path is not None:
        original = read_2d_tiff(
            original_path,
            projection_axis=args.projection_axis,
        )
        if original.shape != prediction.shape:
            raise RuntimeError(
                f"Originalbild und Prediction haben verschiedene Shapes:\n"
                f"Original:   {original.shape}\n"
                f"Prediction: {prediction.shape}"
            )
        original_evidence = robust_normalize(
            original,
            invert=args.dark_structures,
        )

    raw_skeleton_mask = prediction == 1
    raw_soma_mask = prediction == 2

    soma_instances, soma_report = make_soma_instances(
        raw_soma_mask,
        min_area=args.min_soma_area,
    )

    skeleton = skeletonize(raw_skeleton_mask).astype(bool)

    cost, combined_evidence = build_cost_image(
        skeleton=skeleton,
        soma_instances=soma_instances,
        p_background=p_background,
        p_skeleton=p_skeleton,
        p_soma=p_soma,
        original_evidence=original_evidence,
        probability_weight=args.probability_weight,
        original_weight=args.original_weight,
    )

    added_connections = np.zeros_like(skeleton, dtype=bool)
    ambiguous_connection_candidates = np.zeros_like(skeleton, dtype=bool)
    all_candidate_rows: List[Dict[str, object]] = []

    for round_index in range(1, args.rounds + 1):
        print()
        print("=" * 72)
        print(f"VERBINDUNGSRUNDE {round_index}/{args.rounds}")
        print("=" * 72)

        # Kosten nach jeder Runde aktualisieren.
        cost, combined_evidence = build_cost_image(
            skeleton=skeleton,
            soma_instances=soma_instances,
            p_background=p_background,
            p_skeleton=p_skeleton,
            p_soma=p_soma,
            original_evidence=original_evidence,
            probability_weight=args.probability_weight,
            original_weight=args.original_weight,
        )

        endpoint_candidates, endpoints = endpoint_pair_candidates(
            skeleton=skeleton,
            soma_instances=soma_instances,
            cost=cost,
            combined_evidence=combined_evidence,
            p_skeleton=p_skeleton,
            original_evidence=original_evidence,
            max_gap=args.endpoint_gap,
            route_margin=args.route_margin,
            direction_steps=args.direction_steps,
            max_angle=args.max_angle,
            max_detour_ratio=args.max_detour_ratio,
            min_mean_evidence=args.min_mean_evidence,
            min_q20_evidence=args.min_q20_evidence,
            round_index=round_index,
        )

        soma_candidates = endpoint_soma_candidates(
            skeleton=skeleton,
            soma_instances=soma_instances,
            cost=cost,
            combined_evidence=combined_evidence,
            p_skeleton=p_skeleton,
            original_evidence=original_evidence,
            max_gap=args.soma_gap,
            route_margin=args.route_margin,
            direction_steps=args.direction_steps,
            max_angle=args.max_angle,
            max_detour_ratio=args.max_detour_ratio,
            min_mean_evidence=args.min_mean_evidence,
            min_q20_evidence=args.min_q20_evidence,
            round_index=round_index,
            endpoints=endpoints,
        )

        candidates = endpoint_candidates + soma_candidates

        accepted, ambiguous, rejected = select_connections(
            candidates,
            min_score=args.min_score,
            ambiguity_margin=args.ambiguity_margin,
        )

        for candidate in accepted:
            for row, col in candidate["path"]:
                if soma_instances[row, col] == 0:
                    added_connections[row, col] = True
                    skeleton[row, col] = True

        for candidate in ambiguous:
            for row, col in candidate["path"]:
                if soma_instances[row, col] == 0:
                    ambiguous_connection_candidates[row, col] = True

        for candidate in candidates:
            row = {
                key: value
                for key, value in candidate.items()
                if key != "path"
            }
            row["path_pixel_count"] = len(candidate["path"])
            all_candidate_rows.append(row)

        print(f"Endpunkte:              {len(endpoints)}")
        print(f"Kandidaten gesamt:      {len(candidates)}")
        print(f"Akzeptierte Verbindungen:{len(accepted)}")
        print(f"Mehrdeutige Kandidaten: {len(ambiguous)}")
        print(f"Verworfene Kandidaten:  {len(rejected)}")

        if not accepted:
            print("Keine weitere sichere Verbindung gefunden. Stop.")
            break

        skeleton = skeletonize(skeleton).astype(bool)

    completed_skeleton = skeletonize(skeleton).astype(bool)

    (
        component_labels,
        confident_instances,
        all_assignable_instances,
        status_map,
        ambiguous_assignment_mask,
        unassigned_mask,
        component_report,
    ) = assign_skeleton_components(
        skeleton=completed_skeleton,
        soma_instances=soma_instances,
        contact_radius=args.contact_radius,
    )

    # Ausgaben
    save_tiff(
        output_dir / "01_raw_skeleton_mask.tif",
        raw_skeleton_mask.astype(np.uint8),
    )
    save_tiff(
        output_dir / "02_raw_soma_mask.tif",
        raw_soma_mask.astype(np.uint8),
    )
    save_tiff(
        output_dir / "03_soma_instances.tif",
        soma_instances.astype(np.uint32),
    )
    save_tiff(
        output_dir / "04_skeleton_original_1px.tif",
        skeletonize(raw_skeleton_mask).astype(np.uint8),
    )
    save_tiff(
        output_dir / "05_added_connections.tif",
        added_connections.astype(np.uint8),
    )
    save_tiff(
        output_dir / "06_ambiguous_connection_candidates.tif",
        ambiguous_connection_candidates.astype(np.uint8),
    )
    save_tiff(
        output_dir / "07_skeleton_completed_1px.tif",
        completed_skeleton.astype(np.uint8),
    )
    save_tiff(
        output_dir / "08_skeleton_component_labels.tif",
        component_labels.astype(np.uint32),
    )
    save_tiff(
        output_dir / "09_cell_instances_confident.tif",
        confident_instances.astype(np.uint32),
    )
    save_tiff(
        output_dir / "10_cell_instances_all_assignable.tif",
        all_assignable_instances.astype(np.uint32),
    )
    save_tiff(
        output_dir / "11_cell_status_map.tif",
        status_map.astype(np.uint8),
    )
    save_tiff(
        output_dir / "12_ambiguous_cells_mask.tif",
        ambiguous_assignment_mask.astype(np.uint8),
    )
    save_tiff(
        output_dir / "13_unassigned_skeleton_mask.tif",
        unassigned_mask.astype(np.uint8),
    )
    save_tiff(
        output_dir / "14_endpoint_mask_completed.tif",
        endpoint_mask(completed_skeleton).astype(np.uint8),
    )
    save_tiff(
        output_dir / "15_branchpoint_mask_completed.tif",
        branchpoint_mask(completed_skeleton).astype(np.uint8),
    )

    # CSV-Berichte
    soma_csv = output_dir / "soma_report.csv"
    if soma_report:
        with open(soma_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(soma_report[0].keys()),
            )
            writer.writeheader()
            writer.writerows(soma_report)

    candidate_csv = output_dir / "connection_candidates.csv"
    if all_candidate_rows:
        fieldnames = sorted(
            {
                key
                for row in all_candidate_rows
                for key in row.keys()
            }
        )
        with open(
            candidate_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(all_candidate_rows)

    component_csv = output_dir / "cell_assignment_report.csv"
    if component_report:
        with open(
            component_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(component_report[0].keys()),
            )
            writer.writeheader()
            writer.writerows(component_report)

    save_qc_overlay(
        path=output_dir / "postprocessing_qc_overlay.png",
        original=original,
        soma_instances=soma_instances,
        status_map=status_map,
        added_connections=added_connections,
        ambiguous_connection_candidates=ambiguous_connection_candidates,
    )

    confident_components = sum(
        row["status"] == "confident"
        for row in component_report
    )
    ambiguous_components = sum(
        row["status"] == "ambiguous"
        for row in component_report
    )
    unassigned_components = sum(
        row["status"] == "unassigned"
        for row in component_report
    )

    summary = {
        "prediction": str(prediction_path),
        "probabilities": str(probability_path),
        "original": str(original_path) if original_path is not None else "",
        "raw_skeleton_pixels": int(raw_skeleton_mask.sum()),
        "completed_skeleton_pixels": int(completed_skeleton.sum()),
        "added_connection_pixels": int(added_connections.sum()),
        "soma_instances": int(soma_instances.max()),
        "skeleton_components": len(component_report),
        "confident_components": int(confident_components),
        "ambiguous_components": int(ambiguous_components),
        "unassigned_components": int(unassigned_components),
        "parameters": vars(args),
    }

    with open(
        output_dir / "postprocessing_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    print()
    print("=" * 72)
    print("POSTPROCESSING FERTIG")
    print("=" * 72)
    print(f"Ausgabeordner:              {output_dir}")
    print(f"Soma-Instanzen:             {int(soma_instances.max())}")
    print(f"Ergänzte Verbindungspixel:  {int(added_connections.sum())}")
    print(f"Skeletonkomponenten:        {len(component_report)}")
    print(f"  sicher zugeordnet:        {confident_components}")
    print(f"  mehrdeutig:               {ambiguous_components}")
    print(f"  ohne Soma:                {unassigned_components}")
    print()
    print("Wichtigste Dateien:")
    print("  07_skeleton_completed_1px.tif")
    print("  09_cell_instances_confident.tif")
    print("  11_cell_status_map.tif")
    print("  12_ambiguous_cells_mask.tif")
    print("  postprocessing_qc_overlay.png")
    print("  connection_candidates.csv")
    print("  cell_assignment_report.csv")


if __name__ == "__main__":
    main()
