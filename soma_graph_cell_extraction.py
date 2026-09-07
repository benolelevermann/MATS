from __future__ import annotations

"""Split fused somata and assign a 1-pixel neurite graph to soma instances.

This is an experimental, non-destructive alternative to the connected-component
cell extraction used by the web pipeline. Existing crops are never modified.
"""

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from cell_pipeline_web.pipeline import _export_cell, _load_export_helpers, default_settings


SCRIPT_VERSION = "soma-split-directional-graph-v1-2026-09-04"
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class GraphSplitSettings:
    min_soma_area: int = 20
    large_soma_factor: float = 1.75
    h_maxima_radius_factor: float = 0.16
    max_soma_splits: int = 4
    min_split_part_factor: float = 0.22
    soma_attach_radius: int = 3
    tangent_window: float = 12.0
    direction_weight: float = 18.0
    branch_direction_scale: float = 0.30
    skeleton_probability_weight: float = 2.0
    raw_intensity_weight: float = 0.35
    fragment_margin_threshold: float = 0.12
    min_skeleton_pixels: int = 8
    crop_margin: int = 48
    min_crop_size: int = 128


@dataclass
class FragmentGraph:
    labels: np.ndarray
    count: int
    junction_labels: np.ndarray
    junction_count: int
    fragment_pixels: list[np.ndarray]
    junction_pixels: list[np.ndarray]
    fragment_costs: np.ndarray
    graph: csr_matrix
    junction_fragments: list[list[int]]
    junction_transition_rows: list[dict[str, object]]


def read_2d(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} fehlt: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{label} muss 2-D sein, erhalten: {array.shape}")
    return array


def normalize01(array: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.5) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _component_slices(labels: np.ndarray, count: int) -> list[tuple[slice, slice] | None]:
    objects = ndi.find_objects(labels, max_label=count)
    return [obj if obj is None else (obj[0], obj[1]) for obj in objects]


def _select_marker_components(maxima: np.ndarray, distance: np.ndarray, maximum: int) -> np.ndarray:
    marker_labels, marker_count = ndi.label(maxima, structure=CONNECTIVITY_8)
    if marker_count <= maximum:
        return marker_labels
    scored: list[tuple[float, int]] = []
    for marker_id in range(1, marker_count + 1):
        scored.append((float(distance[marker_labels == marker_id].max()), marker_id))
    keep = {marker_id for _, marker_id in sorted(scored, reverse=True)[:maximum]}
    result = np.zeros(marker_labels.shape, dtype=np.int32)
    for new_id, marker_id in enumerate(sorted(keep), start=1):
        result[marker_labels == marker_id] = new_id
    return result


def _merge_tiny_watershed_parts(
    local_mask: np.ndarray,
    distance: np.ndarray,
    parts: np.ndarray,
    minimum_area: int,
) -> np.ndarray:
    counts = np.bincount(parts.ravel())
    keep = [part_id for part_id in range(1, len(counts)) if int(counts[part_id]) >= minimum_area]
    if len(keep) < 2:
        return np.where(local_mask, 1, 0).astype(np.int32)
    markers = np.zeros(parts.shape, dtype=np.int32)
    for new_id, part_id in enumerate(keep, start=1):
        region = parts == part_id
        maximum_position = np.unravel_index(np.argmax(np.where(region, distance, -1)), distance.shape)
        markers[maximum_position] = new_id
    return watershed(-distance, markers=markers, mask=local_mask, connectivity=CONNECTIVITY_8)


def split_soma_mask(
    soma_mask: np.ndarray,
    settings: GraphSplitSettings,
) -> tuple[np.ndarray, list[dict[str, object]], dict[int, dict[str, object]], float]:
    """Return one instance id per soma, splitting oversized multi-lobed objects."""

    raw_labels, raw_count = ndi.label(soma_mask, structure=CONNECTIVITY_8)
    areas = np.bincount(raw_labels.ravel(), minlength=raw_count + 1)
    valid_areas = np.asarray(
        [int(areas[index]) for index in range(1, raw_count + 1) if int(areas[index]) >= settings.min_soma_area],
        dtype=np.float64,
    )
    if valid_areas.size == 0:
        raise RuntimeError("Keine gueltige Soma-Komponente gefunden.")
    reference_area = float(np.median(valid_areas))
    reference_radius = math.sqrt(reference_area / math.pi)
    large_threshold = settings.large_soma_factor * reference_area
    minimum_part_area = max(settings.min_soma_area, int(round(settings.min_split_part_factor * reference_area)))

    output = np.zeros(soma_mask.shape, dtype=np.uint32)
    report: list[dict[str, object]] = []
    instance_info: dict[int, dict[str, object]] = {}
    output_id = 0
    objects = _component_slices(raw_labels, raw_count)

    for raw_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        area = int(areas[raw_id])
        if area < settings.min_soma_area:
            report.append(
                {
                    "raw_soma_id": raw_id,
                    "area_px": area,
                    "area_over_reference": round(area / reference_area, 4),
                    "candidate_for_split": 0,
                    "detected_peaks": 0,
                    "output_instances": "",
                    "reason": "too_small",
                }
            )
            continue

        local_component = raw_labels[slices] == raw_id
        split_candidate = area >= large_threshold
        local_parts = np.where(local_component, 1, 0).astype(np.int32)
        detected_peaks = 1
        reason = "kept"

        if split_candidate:
            distance = ndi.distance_transform_edt(local_component)
            h_value = max(1.0, settings.h_maxima_radius_factor * reference_radius)
            maxima = h_maxima(distance, h=h_value) & local_component
            markers = _select_marker_components(maxima, distance, settings.max_soma_splits)
            detected_peaks = int(markers.max())
            if detected_peaks >= 2:
                proposed = watershed(
                    -distance,
                    markers=markers,
                    mask=local_component,
                    connectivity=CONNECTIVITY_8,
                )
                proposed = _merge_tiny_watershed_parts(
                    local_component,
                    distance,
                    proposed,
                    minimum_part_area,
                )
                if int(proposed.max()) >= 2:
                    local_parts = proposed
                    reason = "watershed_split"
                else:
                    reason = "oversized_unsplit_after_tiny_merge"
            else:
                reason = "oversized_unsplit_no_multiple_maxima"

        created: list[int] = []
        part_count = int(local_parts.max())
        for local_id in range(1, part_count + 1):
            part = local_parts == local_id
            if not np.any(part):
                continue
            output_id += 1
            output_view = output[slices]
            output_view[part] = output_id
            created.append(output_id)
            instance_info[output_id] = {
                "raw_soma_id": raw_id,
                "raw_soma_area_px": area,
                "soma_area_px": int(part.sum()),
                "split_candidate": bool(split_candidate),
                "split_count": part_count,
                "split_reason": reason,
            }

        report.append(
            {
                "raw_soma_id": raw_id,
                "area_px": area,
                "area_over_reference": round(area / reference_area, 4),
                "candidate_for_split": int(split_candidate),
                "detected_peaks": detected_peaks,
                "output_instances": ";".join(map(str, created)),
                "reason": reason,
            }
        )

    return output, report, instance_info, reference_area


def _outward_tangent(fragment_coords: np.ndarray, junction_coords: np.ndarray, window: float) -> np.ndarray:
    junction_center = junction_coords.mean(axis=0)
    deltas = fragment_coords.astype(np.float64) - junction_center
    distances = np.linalg.norm(deltas, axis=1)
    selection = (distances > 0) & (distances <= window)
    if not np.any(selection):
        selection = distances > 0
    if not np.any(selection):
        return np.asarray([0.0, 0.0])
    selected_deltas = deltas[selection]
    selected_distances = distances[selection]
    far_threshold = np.percentile(selected_distances, 65.0)
    far = selected_deltas[selected_distances >= far_threshold]
    vector = far.mean(axis=0) if far.size else selected_deltas.mean(axis=0)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.asarray([0.0, 0.0])


def _coordinates_by_label(labels: np.ndarray, count: int) -> list[np.ndarray]:
    grouped = [np.empty((0, 2), dtype=np.int64) for _ in range(count + 1)]
    coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return grouped
    ids = labels[coords[:, 0], coords[:, 1]].astype(np.int64)
    order = np.argsort(ids, kind="stable")
    coords = coords[order]
    ids = ids[order]
    boundaries = np.searchsorted(ids, np.arange(1, count + 2))
    for label_id in range(1, count + 1):
        grouped[label_id] = coords[boundaries[label_id - 1] : boundaries[label_id]]
    return grouped


def build_fragment_graph(
    skeleton: np.ndarray,
    skeleton_probability: np.ndarray,
    raw_normalized: np.ndarray,
    settings: GraphSplitSettings,
) -> FragmentGraph:
    neighbor_count = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    neighbor_count = neighbor_count - skeleton.astype(np.uint8)
    junction_mask = skeleton & (neighbor_count >= 3)
    junction_labels, junction_count = ndi.label(junction_mask, structure=CONNECTIVITY_8)
    fragment_labels, fragment_count = ndi.label(skeleton & ~junction_mask, structure=CONNECTIVITY_8)

    fragment_pixels = _coordinates_by_label(fragment_labels, fragment_count)
    fragment_costs = np.zeros(fragment_count, dtype=np.float64)
    for fragment_id in range(1, fragment_count + 1):
        coords = fragment_pixels[fragment_id]
        if coords.size:
            probability = float(np.mean(skeleton_probability[coords[:, 0], coords[:, 1]]))
            intensity = float(np.mean(raw_normalized[coords[:, 0], coords[:, 1]]))
            per_pixel = (
                1.0
                + settings.skeleton_probability_weight * (1.0 - probability)
                + settings.raw_intensity_weight * (1.0 - intensity)
            )
            fragment_costs[fragment_id - 1] = max(1.0, len(coords) * per_pixel)

    junction_pixels = _coordinates_by_label(junction_labels, junction_count)
    junction_objects = _component_slices(junction_labels, junction_count)
    junction_fragments: list[list[int]] = [[]]
    transition_rows: list[dict[str, object]] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    weights: list[float] = []

    for junction_id in range(1, junction_count + 1):
        coords = junction_pixels[junction_id]
        slices = junction_objects[junction_id - 1]
        if slices is None:
            junction_fragments.append([])
            continue
        y0 = max(0, slices[0].start - 1)
        y1 = min(skeleton.shape[0], slices[0].stop + 1)
        x0 = max(0, slices[1].start - 1)
        x1 = min(skeleton.shape[1], slices[1].stop + 1)
        local_junction = junction_labels[y0:y1, x0:x1] == junction_id
        local_dilated = ndi.binary_dilation(local_junction, structure=CONNECTIVITY_8)
        adjacent = np.unique(fragment_labels[y0:y1, x0:x1][local_dilated])
        adjacent_ids = sorted(int(value) for value in adjacent if int(value) > 0)
        junction_fragments.append(adjacent_ids)
        tangents = {
            fragment_id: _outward_tangent(fragment_pixels[fragment_id], coords, settings.tangent_window)
            for fragment_id in adjacent_ids
        }
        degree_scale = 1.0 if len(adjacent_ids) >= 4 else settings.branch_direction_scale
        for left_index, left_id in enumerate(adjacent_ids):
            for right_id in adjacent_ids[left_index + 1 :]:
                dot = float(np.clip(np.dot(tangents[left_id], tangents[right_id]), -1.0, 1.0))
                angle = float(math.acos(dot))
                deviation = abs(math.pi - angle) / math.pi
                direction_penalty = settings.direction_weight * degree_scale * deviation
                base = 0.5 * (
                    fragment_costs[left_id - 1] + fragment_costs[right_id - 1]
                )
                weight = max(1e-3, base + direction_penalty)
                left_node = left_id - 1
                right_node = right_id - 1
                row_indices.extend((left_node, right_node))
                col_indices.extend((right_node, left_node))
                weights.extend((weight, weight))
                transition_rows.append(
                    {
                        "junction_id": junction_id,
                        "junction_degree": len(adjacent_ids),
                        "fragment_a": left_id,
                        "fragment_b": right_id,
                        "angle_degrees": round(math.degrees(angle), 3),
                        "direction_penalty": round(direction_penalty, 4),
                    }
                )

    graph = coo_matrix(
        (weights, (row_indices, col_indices)),
        shape=(fragment_count, fragment_count),
        dtype=np.float64,
    ).tocsr()
    return FragmentGraph(
        labels=fragment_labels,
        count=fragment_count,
        junction_labels=junction_labels,
        junction_count=junction_count,
        fragment_pixels=fragment_pixels,
        junction_pixels=junction_pixels,
        fragment_costs=fragment_costs,
        graph=graph,
        junction_fragments=junction_fragments,
        junction_transition_rows=transition_rows,
    )


def assign_fragments_to_somata(
    fragment_graph: FragmentGraph,
    soma_instances: np.ndarray,
    settings: GraphSplitSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, object]], list[dict[str, object]]]:
    soma_count = int(soma_instances.max())
    fragment_count = fragment_graph.count
    assignment = np.zeros(fragment_graph.labels.shape, dtype=np.uint32)
    ambiguous = np.zeros(fragment_graph.labels.shape, dtype=bool)
    shared_junction = np.zeros(fragment_graph.labels.shape, dtype=bool)
    if fragment_count == 0:
        return assignment, ambiguous, shared_junction, {}, []

    attach_structure = ndi.iterate_structure(CONNECTIVITY_8, settings.soma_attach_radius)
    soma_objects = _component_slices(soma_instances, soma_count)
    seed_fragments: dict[int, list[int]] = {}
    seed_rows: list[dict[str, object]] = []
    for soma_id in range(1, soma_count + 1):
        slices = soma_objects[soma_id - 1]
        if slices is None:
            ids = []
        else:
            radius = settings.soma_attach_radius
            y0 = max(0, slices[0].start - radius)
            y1 = min(soma_instances.shape[0], slices[0].stop + radius)
            x0 = max(0, slices[1].start - radius)
            x1 = min(soma_instances.shape[1], slices[1].stop + radius)
            local_soma = soma_instances[y0:y1, x0:x1] == soma_id
            expanded = ndi.binary_dilation(local_soma, structure=attach_structure)
            ids = sorted(
                int(value)
                for value in np.unique(fragment_graph.labels[y0:y1, x0:x1][expanded])
                if int(value) > 0
            )
        seed_fragments[soma_id] = ids
        seed_rows.append(
            {"soma_id": soma_id, "seed_fragments": ";".join(map(str, ids)), "seed_fragment_count": len(ids)}
        )

    total_nodes = fragment_count + soma_count
    base = fragment_graph.graph.tocoo()
    rows = base.row.tolist()
    cols = base.col.tolist()
    values = base.data.tolist()
    for soma_id, fragments in seed_fragments.items():
        soma_node = fragment_count + soma_id - 1
        for fragment_id in fragments:
            fragment_node = fragment_id - 1
            weight = max(1e-3, 0.5 * fragment_graph.fragment_costs[fragment_node])
            rows.extend((soma_node, fragment_node))
            cols.extend((fragment_node, soma_node))
            values.extend((weight, weight))
    augmented = coo_matrix((values, (rows, cols)), shape=(total_nodes, total_nodes)).tocsr()
    source_nodes = np.arange(fragment_count, total_nodes, dtype=np.int64)
    distances = np.asarray(dijkstra(augmented, directed=False, indices=source_nodes))[:, :fragment_count]

    fragment_owner = np.zeros(fragment_count, dtype=np.int32)
    fragment_margin = np.zeros(fragment_count, dtype=np.float64)
    fragment_report: list[dict[str, object]] = []
    for fragment_index in range(fragment_count):
        column = distances[:, fragment_index]
        finite = np.flatnonzero(np.isfinite(column))
        if finite.size == 0:
            fragment_report.append(
                {
                    "fragment_id": fragment_index + 1,
                    "pixels": len(fragment_graph.fragment_pixels[fragment_index + 1]),
                    "owner_soma_id": 0,
                    "margin": 0.0,
                    "ambiguous": 1,
                    "reason": "unreachable_from_soma",
                }
            )
            coords = fragment_graph.fragment_pixels[fragment_index + 1]
            ambiguous[coords[:, 0], coords[:, 1]] = True
            continue
        ordered = finite[np.argsort(column[finite])]
        best = int(ordered[0])
        best_cost = float(column[best])
        if len(ordered) == 1:
            margin = 1.0
        else:
            second_cost = float(column[int(ordered[1])])
            margin = max(0.0, (second_cost - best_cost) / max(second_cost, 1e-6))
        owner = best + 1
        is_ambiguous = margin < settings.fragment_margin_threshold
        fragment_owner[fragment_index] = owner
        fragment_margin[fragment_index] = margin
        coords = fragment_graph.fragment_pixels[fragment_index + 1]
        if is_ambiguous:
            ambiguous[coords[:, 0], coords[:, 1]] = True
        else:
            assignment[coords[:, 0], coords[:, 1]] = owner
        fragment_report.append(
            {
                "fragment_id": fragment_index + 1,
                "pixels": len(coords),
                "owner_soma_id": owner,
                "margin": round(margin, 6),
                "ambiguous": int(is_ambiguous),
                "reason": "low_cost_margin" if is_ambiguous else "assigned",
            }
        )

    per_soma: dict[int, dict[str, object]] = {
        soma_id: {
            "assigned_fragment_ids": [],
            "assigned_skeleton_pixels": 0,
            "fragment_margins": [],
            "shared_junction_pixels": 0,
        }
        for soma_id in range(1, soma_count + 1)
    }
    for fragment_index, owner in enumerate(fragment_owner):
        if owner <= 0 or fragment_margin[fragment_index] < settings.fragment_margin_threshold:
            continue
        fragment_id = fragment_index + 1
        per_soma[int(owner)]["assigned_fragment_ids"].append(fragment_id)
        pixel_count = len(fragment_graph.fragment_pixels[fragment_id])
        per_soma[int(owner)]["assigned_skeleton_pixels"] += pixel_count
        per_soma[int(owner)]["fragment_margins"].append(float(fragment_margin[fragment_index]))

    for junction_id in range(1, fragment_graph.junction_count + 1):
        incident = fragment_graph.junction_fragments[junction_id]
        owners = sorted(
            {
                int(fragment_owner[fragment_id - 1])
                for fragment_id in incident
                if fragment_owner[fragment_id - 1] > 0
                and fragment_margin[fragment_id - 1] >= settings.fragment_margin_threshold
            }
        )
        coords = fragment_graph.junction_pixels[junction_id]
        if not owners:
            ambiguous[coords[:, 0], coords[:, 1]] = True
            continue
        if len(owners) == 1:
            assignment[coords[:, 0], coords[:, 1]] = owners[0]
        else:
            shared_junction[coords[:, 0], coords[:, 1]] = True
            for owner in owners:
                per_soma[owner]["shared_junction_pixels"] += len(coords)

    for soma_id, details in per_soma.items():
        margins = details.pop("fragment_margins")
        details["minimum_fragment_margin"] = round(float(min(margins)), 6) if margins else 0.0
        details["median_fragment_margin"] = round(float(np.median(margins)), 6) if margins else 0.0
        details["seed_fragment_ids"] = seed_fragments[soma_id]

    return assignment, ambiguous, shared_junction, per_soma, fragment_report


def _color_table(count: int) -> np.ndarray:
    rng = np.random.default_rng(20260904)
    colors = rng.integers(45, 255, size=(count + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors


def _save_overview(
    path: Path,
    raw: np.ndarray,
    soma_instances: np.ndarray,
    skeleton_assignment: np.ndarray,
    ambiguous: np.ndarray,
    shared_junction: np.ndarray,
) -> None:
    gray = np.round(normalize01(raw) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    colors = _color_table(int(soma_instances.max()))
    instance_pixels = soma_instances > 0
    rgb[instance_pixels] = 0.15 * rgb[instance_pixels] + 0.85 * colors[soma_instances[instance_pixels]]
    assigned_pixels = skeleton_assignment > 0
    rgb[assigned_pixels] = 0.15 * rgb[assigned_pixels] + 0.85 * colors[skeleton_assignment[assigned_pixels]]
    rgb[ambiguous] = np.asarray([255, 215, 0], dtype=np.float32)
    rgb[shared_junction] = np.asarray([255, 255, 255], dtype=np.float32)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    image.thumbnail((2500, 2500), Image.Resampling.LANCZOS)
    image.save(path, optimize=True)


def _write_csv(path: Path, rows: list[dict[str, object]], fallback_fields: list[str]) -> None:
    fieldnames = list(rows[0].keys()) if rows else fallback_fields
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _probability_map(path: Path | None, shape: tuple[int, int], channel: int) -> np.ndarray:
    if path is None:
        return np.ones(shape, dtype=np.float32)
    with np.load(path) as archive:
        probabilities = np.squeeze(np.asarray(archive["probabilities"], dtype=np.float32))
    if probabilities.ndim != 3 or probabilities.shape[0] <= channel or tuple(probabilities.shape[1:]) != shape:
        raise RuntimeError(f"Unerwartete Wahrscheinlichkeitsform: {probabilities.shape}, erwartet (*, {shape})")
    return probabilities[channel]


def run_extraction(
    raw_path: Path,
    semantic_path: Path,
    prediction_path: Path,
    output_root: Path,
    project_root: Path,
    probability_path: Path | None = None,
    settings: GraphSplitSettings | None = None,
) -> dict[str, object]:
    config = settings or GraphSplitSettings()
    raw = read_2d(raw_path, "Rohbild")
    semantic = read_2d(semantic_path, "Hysterese").astype(np.uint8)
    prediction = read_2d(prediction_path, "Prediction").astype(np.uint8)
    if raw.shape != semantic.shape or raw.shape != prediction.shape:
        raise RuntimeError(f"Formen stimmen nicht ueberein: raw={raw.shape}, semantic={semantic.shape}, pred={prediction.shape}")
    invalid = set(np.unique(semantic).astype(int).tolist()) - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Ungueltige semantische Labels: {sorted(invalid)}")
    if output_root.exists():
        raise FileExistsError(f"Ausgabe existiert bereits: {output_root}")
    output_root.mkdir(parents=True)

    skeleton = semantic == 1
    soma_instances, soma_report, soma_info, reference_area = split_soma_mask(semantic == 2, config)
    skeleton_probability = _probability_map(probability_path, raw.shape, channel=1)
    graph = build_fragment_graph(skeleton, skeleton_probability, normalize01(raw), config)
    assignment, ambiguous, shared_junction, per_soma, fragment_report = assign_fragments_to_somata(
        graph, soma_instances, config
    )

    tifffile.imwrite(output_root / "soma_split_instances.tif", soma_instances.astype(np.uint16))
    tifffile.imwrite(output_root / "skeleton_instance_assignment.tif", assignment.astype(np.uint16))
    tifffile.imwrite(output_root / "ambiguous_skeleton.tif", ambiguous.astype(np.uint8))
    tifffile.imwrite(output_root / "shared_junctions.tif", shared_junction.astype(np.uint8))
    tifffile.imwrite(output_root / "junction_labels.tif", graph.junction_labels.astype(np.uint16))
    _save_overview(
        output_root / "overview_graph_assignment.png",
        raw,
        soma_instances,
        assignment,
        ambiguous,
        shared_junction,
    )
    _write_csv(output_root / "soma_split_report.csv", soma_report, ["raw_soma_id", "reason"])
    _write_csv(output_root / "fragment_assignment_report.csv", fragment_report, ["fragment_id", "reason"])
    _write_csv(
        output_root / "junction_transition_report.csv",
        graph.junction_transition_rows,
        ["junction_id", "fragment_a", "fragment_b", "angle_degrees", "direction_penalty"],
    )

    cells_root = output_root / "cells"
    cells_root.mkdir()
    helper = _load_export_helpers(project_root)
    export_settings = default_settings(
        project_root,
        min_soma_area=config.min_soma_area,
        min_skeleton_pixels=config.min_skeleton_pixels,
        crop_margin=config.crop_margin,
        min_crop_size=config.min_crop_size,
        finalize_fiji=False,
    )
    manifest: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    output_index = 0
    soma_order: list[tuple[float, float, int]] = []
    for soma_id in range(1, int(soma_instances.max()) + 1):
        coords = np.argwhere(soma_instances == soma_id)
        if coords.size:
            center = coords.mean(axis=0)
            soma_order.append((float(center[0]), float(center[1]), soma_id))

    for _, _, soma_id in sorted(soma_order):
        soma_coords = np.argwhere(soma_instances == soma_id)
        skeleton_coords = np.argwhere(assignment == soma_id)
        # Crossing pixels can legitimately be shared between two independent instance masks.
        details = per_soma.get(soma_id, {})
        if int(details.get("shared_junction_pixels", 0)):
            incident_junction_ids = []
            owned_fragments = set(int(value) for value in details.get("assigned_fragment_ids", []))
            for junction_id in range(1, graph.junction_count + 1):
                if owned_fragments.intersection(graph.junction_fragments[junction_id]):
                    incident_junction_ids.append(junction_id)
            if incident_junction_ids:
                extra = np.argwhere(np.isin(graph.junction_labels, incident_junction_ids))
                skeleton_coords = np.unique(np.vstack((skeleton_coords, extra)), axis=0)

        if len(skeleton_coords) < config.min_skeleton_pixels:
            rejected.append(
                {
                    "soma_id": soma_id,
                    "raw_soma_id": soma_info[soma_id]["raw_soma_id"],
                    "reason": "not_enough_confident_skeleton",
                    "skeleton_pixels": len(skeleton_coords),
                }
            )
            continue
        output_index += 1
        margins = float(details.get("median_fragment_margin", 0.0))
        split_reason = str(soma_info[soma_id]["split_reason"])
        review_status = (
            "review_medium_confidence"
            if split_reason != "kept" or margins < 0.30 or int(details.get("shared_junction_pixels", 0)) > 0
            else "accepted_high_confidence"
        )
        qc = {
            **soma_info[soma_id],
            **details,
            "graph_review_status": review_status,
        }
        try:
            row = _export_cell(
                raw,
                prediction,
                semantic,
                skeleton_coords,
                soma_coords,
                cells_root,
                output_index,
                helper,
                export_settings,
                source="soma_split_directional_graph",
                source_component=int(soma_info[soma_id]["raw_soma_id"]),
                conflict_group=int(soma_info[soma_id]["raw_soma_id"])
                if soma_info[soma_id]["split_count"] > 1
                else None,
                qc=qc,
            )
            cell_dir = cells_root / str(row["folder"])
            metadata_path = cell_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            bounds = metadata["bounds"]
            y0, y1 = int(bounds["y_min"]), int(bounds["y_max_exclusive"])
            x0, x1 = int(bounds["x_min"]), int(bounds["x_max_exclusive"])
            local_target = read_2d(cell_dir / "cell_mask.tif", "Zellmaske").astype(bool)
            global_foreground = semantic[y0:y1, x0:x1] > 0
            foreign_foreground = global_foreground & ~local_target
            foreign_soma = (soma_instances[y0:y1, x0:x1] > 0) & (
                soma_instances[y0:y1, x0:x1] != soma_id
            )
            tifffile.imwrite(cell_dir / "ignore_mask.tif", foreign_foreground.astype(np.uint8))
            metadata["script_version"] = SCRIPT_VERSION
            metadata["status"] = review_status
            metadata["qc"]["foreign_foreground_pixels_in_crop"] = int(foreign_foreground.sum())
            metadata["qc"]["foreign_soma_pixels_in_crop"] = int(foreign_soma.sum())
            metadata["qc"]["foreign_soma_instances_in_crop"] = int(
                len(set(np.unique(soma_instances[y0:y1, x0:x1][foreign_soma]).tolist()) - {0})
            )
            metadata["qc"]["training_safe_without_ignore"] = bool(
                not np.any(foreign_foreground) and review_status == "accepted_high_confidence"
            )
            metadata["review_layers"]["ignore_mask"] = "ignore_mask.tif"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            row["source"] = "soma_split_directional_graph"
            manifest.append(row)
        except Exception as error:
            output_index -= 1
            rejected.append(
                {
                    "soma_id": soma_id,
                    "raw_soma_id": soma_info[soma_id]["raw_soma_id"],
                    "reason": str(error),
                    "skeleton_pixels": len(skeleton_coords),
                }
            )

    _write_csv(cells_root / "manifest.csv", manifest, ["folder", "source"])
    _write_csv(cells_root / "rejected_somas.csv", rejected, ["soma_id", "raw_soma_id", "reason"])
    summary = {
        "script_version": SCRIPT_VERSION,
        "raw": str(raw_path.resolve()),
        "semantic": str(semantic_path.resolve()),
        "prediction": str(prediction_path.resolve()),
        "probabilities": str(probability_path.resolve()) if probability_path else None,
        "shape": list(raw.shape),
        "reference_soma_area_px": round(reference_area, 3),
        "input_soma_components": int(ndi.label(semantic == 2, structure=CONNECTIVITY_8)[1]),
        "output_soma_instances": int(soma_instances.max()),
        "split_candidate_components": int(sum(int(row["candidate_for_split"]) for row in soma_report)),
        "successfully_split_components": int(sum(row["reason"] == "watershed_split" for row in soma_report)),
        "skeleton_pixels": int(skeleton.sum()),
        "graph_fragments": graph.count,
        "graph_junctions": graph.junction_count,
        "assigned_skeleton_pixels": int(np.count_nonzero(assignment)),
        "ambiguous_skeleton_pixels": int(np.count_nonzero(ambiguous)),
        "shared_junction_pixels": int(np.count_nonzero(shared_junction)),
        "exported_cells": len(manifest),
        "rejected_somata": len(rejected),
        "settings": asdict(config),
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--large-soma-factor", type=float, default=1.75)
    parser.add_argument("--fragment-margin-threshold", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = GraphSplitSettings(
        large_soma_factor=args.large_soma_factor,
        fragment_margin_threshold=args.fragment_margin_threshold,
    )
    summary = run_extraction(
        raw_path=args.raw,
        semantic_path=args.semantic,
        prediction_path=args.prediction,
        probability_path=args.probabilities,
        output_root=args.output_root,
        project_root=args.project_root,
        settings=settings,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
