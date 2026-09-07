from __future__ import annotations

"""
Compare three cell-assignment strategies on one completed 0/1/2 segmentation.

Input semantics
---------------
0 = background
1 = completed skeleton material (never changed by this script)
2 = soma

The two paper-derived methods are deliberately named "inspired": this is a
raster/2-D adaptation for direct comparison on nnU-Net output, not the original
authors' MATLAB implementation.
"""

import argparse
import csv
import heapq
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


SCRIPT_VERSION = "paper-cell-assignment-comparison-v1-2026-08-12"

OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
OFFSET_TO_DIRECTION = {offset: index for index, offset in enumerate(OFFSETS)}
STEP_LENGTHS = np.asarray(
    [math.hypot(row, col) for row, col in OFFSETS], dtype=np.float64
)
UNIT_DIRECTIONS = np.asarray(
    [(row / length, col / length) for (row, col), length in zip(OFFSETS, STEP_LENGTHS)],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare topological, Matrix-Forest-inspired and G-Cut-inspired "
            "cell assignment on a completed 0/1/2 segmentation."
        )
    )
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--root-radius",
        type=float,
        default=3.0,
        help="Maximum distance in pixels from a skeleton root to a soma boundary.",
    )
    parser.add_argument("--mft-tau", type=float, default=1.0)
    parser.add_argument("--mft-seed-strength", type=float, default=25.0)
    parser.add_argument("--mft-min-confidence", type=float, default=0.58)
    parser.add_argument("--mft-margin", type=float, default=0.12)
    parser.add_argument("--gcut-turn-weight", type=float, default=1.35)
    parser.add_argument("--gcut-radial-weight", type=float, default=0.22)
    parser.add_argument("--gcut-margin", type=float, default=0.10)
    parser.add_argument(
        "--safe-max-ambiguous-fraction",
        type=float,
        default=0.05,
        help="Maximum ambiguous-context fraction for a soma to be called safe.",
    )
    parser.add_argument("--safe-min-skeleton-pixels", type=int, default=12)
    parser.add_argument("--border-margin", type=int, default=2)
    parser.add_argument("--qc-max-size", type=int, default=2200)
    return parser.parse_args()


def load_2d(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.squeeze(tifffile.imread(path))
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2-D image, found {array.shape} in {path}")
    return array


def output_dtype(maximum: int) -> np.dtype:
    return np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32


def normalize_original(array: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if array is None:
        return np.zeros(shape, dtype=np.uint8)
    values = array.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)


def soma_centers(labels: np.ndarray, count: int) -> dict[int, tuple[float, float]]:
    ids = np.arange(1, count + 1, dtype=np.int32)
    centers = ndi.center_of_mass(np.ones(labels.shape, dtype=np.uint8), labels, ids)
    return {int(label): (float(center[0]), float(center[1])) for label, center in zip(ids, centers)}


def soma_boundary_tree(
    soma_instances: np.ndarray,
) -> tuple[cKDTree | None, np.ndarray, np.ndarray]:
    soma_mask = soma_instances > 0
    eroded = ndi.binary_erosion(soma_mask, structure=np.ones((3, 3), dtype=bool))
    boundary = soma_mask & ~eroded
    coords = np.argwhere(boundary)
    if coords.size == 0:
        return None, coords, np.empty(0, dtype=np.int32)
    labels = soma_instances[boundary].astype(np.int32, copy=False)
    return cKDTree(coords), coords, labels


def component_graph(coords: np.ndarray, width: int) -> tuple[list[list[tuple[int, int]]], csr_matrix]:
    """Return pixel adjacency and an undirected weighted adjacency matrix."""
    count = len(coords)
    lookup = {int(row) * width + int(col): index for index, (row, col) in enumerate(coords)}
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    height_guard = int(coords[:, 0].max()) + 2

    for index, (row_value, col_value) in enumerate(coords):
        row = int(row_value)
        col = int(col_value)
        for direction, (dr, dc) in enumerate(OFFSETS):
            nr = row + dr
            nc = col + dc
            if nr < 0 or nc < 0 or nr >= height_guard or nc >= width:
                continue
            other = lookup.get(nr * width + nc)
            if other is None:
                continue
            adjacency[index].append((other, direction))
            if other > index:
                weight = 1.0 / STEP_LENGTHS[direction]
                rows.extend((index, other))
                cols.extend((other, index))
                weights.extend((weight, weight))

    matrix = csr_matrix((weights, (rows, cols)), shape=(count, count), dtype=np.float64)
    return adjacency, matrix


def roots_for_component(
    coords: np.ndarray,
    tree: cKDTree | None,
    boundary_labels: np.ndarray,
    radius: float,
) -> dict[int, np.ndarray]:
    if tree is None or len(coords) == 0:
        return {}
    distances, indices = tree.query(coords, k=1, distance_upper_bound=radius)
    valid = np.isfinite(distances) & (indices < len(boundary_labels))
    grouped: dict[int, list[int]] = defaultdict(list)
    for node_index in np.flatnonzero(valid):
        soma_id = int(boundary_labels[int(indices[node_index])])
        if soma_id > 0:
            grouped[soma_id].append(int(node_index))
    return {
        soma_id: np.asarray(sorted(set(node_indices)), dtype=np.int32)
        for soma_id, node_indices in grouped.items()
    }


def topology_assignment(
    node_count: int,
    roots: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    labels = np.zeros(node_count, dtype=np.int32)
    status = np.full(node_count, 3, dtype=np.uint8)
    soma_ids = sorted(roots)
    if len(soma_ids) == 1:
        labels.fill(soma_ids[0])
        status.fill(1)
    elif len(soma_ids) > 1:
        status.fill(2)
    return labels, status, {"number_of_somata": float(len(soma_ids))}


def matrix_forest_assignment(
    weighted_adjacency: csr_matrix,
    roots: dict[int, np.ndarray],
    tau: float,
    seed_strength: float,
    min_confidence: float,
    min_margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    node_count = weighted_adjacency.shape[0]
    soma_ids = np.asarray(sorted(roots), dtype=np.int32)
    if len(soma_ids) <= 1:
        return topology_assignment(node_count, roots)

    degree = np.asarray(weighted_adjacency.sum(axis=1)).ravel()
    laplacian = diags(degree) - weighted_adjacency
    seed_diagonal = np.zeros(node_count, dtype=np.float64)
    target = np.zeros((node_count, len(soma_ids)), dtype=np.float64)
    for column, soma_id in enumerate(soma_ids):
        nodes = roots[int(soma_id)]
        seed_diagonal[nodes] = 1.0
        target[nodes, column] = 1.0

    system = (
        eye(node_count, dtype=np.float64, format="csr")
        + float(tau) * laplacian
        + float(seed_strength) * diags(seed_diagonal)
    )
    scores = np.asarray(spsolve(system.tocsc(), seed_strength * target))
    if scores.ndim == 1:
        scores = scores[:, None]
    scores = np.clip(scores, 0.0, None)
    totals = scores.sum(axis=1, keepdims=True)
    probabilities = scores / np.maximum(totals, 1e-12)

    order = np.argsort(probabilities, axis=1)
    winner_columns = order[:, -1]
    best = probabilities[np.arange(node_count), winner_columns]
    second = probabilities[np.arange(node_count), order[:, -2]]
    margin = best - second

    labels = soma_ids[winner_columns].astype(np.int32, copy=False)
    confident = (best >= min_confidence) & (margin >= min_margin)
    status = np.where(confident, 1, 2).astype(np.uint8)

    for soma_id, nodes in roots.items():
        labels[nodes] = int(soma_id)
        status[nodes] = 1

    return labels, status, {
        "number_of_somata": float(len(soma_ids)),
        "mean_best_probability": float(np.mean(best)),
        "mean_probability_margin": float(np.mean(margin)),
    }


def update_two_best(
    best_cost: np.ndarray,
    best_label: np.ndarray,
    second_cost: np.ndarray,
    second_label: np.ndarray,
    state: int,
    label: int,
    cost: float,
) -> bool:
    epsilon = 1e-12
    if best_label[state] == label:
        if cost + epsilon < best_cost[state]:
            best_cost[state] = cost
            return True
        return False
    if second_label[state] == label:
        if cost + epsilon < second_cost[state]:
            second_cost[state] = cost
            if second_cost[state] < best_cost[state]:
                best_cost[state], second_cost[state] = second_cost[state], best_cost[state]
                best_label[state], second_label[state] = second_label[state], best_label[state]
            return True
        return False
    if cost + epsilon < best_cost[state]:
        second_cost[state] = best_cost[state]
        second_label[state] = best_label[state]
        best_cost[state] = cost
        best_label[state] = label
        return True
    if cost + epsilon < second_cost[state]:
        second_cost[state] = cost
        second_label[state] = label
        return True
    return False


def gcut_directional_assignment(
    coords: np.ndarray,
    adjacency: list[list[tuple[int, int]]],
    roots: dict[int, np.ndarray],
    centers: dict[int, tuple[float, float]],
    turn_weight: float,
    radial_weight: float,
    min_margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Direction-aware multi-source geodesic branch assignment.

    This transfers the G-Cut idea (soma-rooted growth orientation and global
    competition) to a 2-D raster graph. It is intentionally not called an exact
    reproduction of the original linear program.
    """
    node_count = len(coords)
    soma_ids = sorted(roots)
    if len(soma_ids) <= 1:
        return topology_assignment(node_count, roots)

    directions_per_node = len(OFFSETS) + 1
    unknown_direction = len(OFFSETS)
    state_count = node_count * directions_per_node
    best_cost = np.full(state_count, np.inf, dtype=np.float64)
    second_cost = np.full(state_count, np.inf, dtype=np.float64)
    best_label = np.zeros(state_count, dtype=np.int32)
    second_label = np.zeros(state_count, dtype=np.int32)
    queue: list[tuple[float, int, int]] = []

    for soma_id, nodes in roots.items():
        for node in nodes:
            state = int(node) * directions_per_node + unknown_direction
            if update_two_best(
                best_cost, best_label, second_cost, second_label, state, int(soma_id), 0.0
            ):
                heapq.heappush(queue, (0.0, state, int(soma_id)))

    while queue:
        cost, state, label = heapq.heappop(queue)
        if not (
            (best_label[state] == label and cost <= best_cost[state] + 1e-12)
            or (second_label[state] == label and cost <= second_cost[state] + 1e-12)
        ):
            continue
        node = state // directions_per_node
        previous_direction = state % directions_per_node
        row, col = coords[node]
        center_row, center_col = centers[label]

        for neighbor, direction in adjacency[node]:
            step = STEP_LENGTHS[direction]
            turn_penalty = 0.0
            if previous_direction != unknown_direction:
                cosine = float(
                    np.dot(UNIT_DIRECTIONS[previous_direction], UNIT_DIRECTIONS[direction])
                )
                turn_penalty = turn_weight * (1.0 - cosine) * 0.5

            radial_row = float(row) - center_row
            radial_col = float(col) - center_col
            radial_norm = math.hypot(radial_row, radial_col)
            radial_penalty = 0.0
            if radial_norm > 1e-6:
                radial_cosine = (
                    radial_row * UNIT_DIRECTIONS[direction, 0]
                    + radial_col * UNIT_DIRECTIONS[direction, 1]
                ) / radial_norm
                radial_penalty = radial_weight * (1.0 - radial_cosine) * 0.5

            new_cost = cost + step * (1.0 + turn_penalty + radial_penalty)
            new_state = int(neighbor) * directions_per_node + direction
            if update_two_best(
                best_cost,
                best_label,
                second_cost,
                second_label,
                new_state,
                label,
                new_cost,
            ):
                heapq.heappush(queue, (new_cost, new_state, label))

    node_best_cost = np.full(node_count, np.inf, dtype=np.float64)
    node_second_cost = np.full(node_count, np.inf, dtype=np.float64)
    node_best_label = np.zeros(node_count, dtype=np.int32)
    node_second_label = np.zeros(node_count, dtype=np.int32)

    for node in range(node_count):
        candidates: dict[int, float] = {}
        start = node * directions_per_node
        end = start + directions_per_node
        for label_array, cost_array in (
            (best_label[start:end], best_cost[start:end]),
            (second_label[start:end], second_cost[start:end]),
        ):
            for label, cost in zip(label_array, cost_array):
                label_value = int(label)
                if label_value <= 0 or not np.isfinite(cost):
                    continue
                candidates[label_value] = min(candidates.get(label_value, np.inf), float(cost))
        ordered = sorted(candidates.items(), key=lambda item: item[1])
        if ordered:
            node_best_label[node], node_best_cost[node] = ordered[0]
        if len(ordered) > 1:
            node_second_label[node], node_second_cost[node] = ordered[1]

    relative_margin = (node_second_cost - node_best_cost) / np.maximum(
        node_second_cost, 1e-9
    )
    no_competitor = ~np.isfinite(node_second_cost)
    confident = no_competitor | (relative_margin >= min_margin)
    status = np.where(confident & (node_best_label > 0), 1, 2).astype(np.uint8)
    status[node_best_label <= 0] = 3

    for soma_id, nodes in roots.items():
        node_best_label[nodes] = int(soma_id)
        status[nodes] = 1

    finite_margin = relative_margin[np.isfinite(relative_margin)]
    return node_best_label, status, {
        "number_of_somata": float(len(soma_ids)),
        "mean_relative_cost_margin": (
            float(np.mean(finite_margin)) if finite_margin.size else 1.0
        ),
    }


def palette_color(identifier: int) -> np.ndarray:
    # Deterministic high-saturation color without storing a huge lookup table.
    hue = (identifier * 0.618033988749895) % 1.0
    sector = int(hue * 6.0)
    fraction = hue * 6.0 - sector
    p, q, t = 0.18, 1.0 - 0.82 * fraction, 0.18 + 0.82 * fraction
    colors = (
        (1.0, t, p),
        (q, 1.0, p),
        (p, 1.0, t),
        (p, q, 1.0),
        (t, p, 1.0),
        (1.0, p, q),
    )
    return np.round(np.asarray(colors[sector % 6]) * 255.0).astype(np.uint8)


def resize_for_qc(array: np.ndarray, max_size: int, nearest: bool) -> np.ndarray:
    height, width = array.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale >= 1.0:
        return array
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array).resize(size, mode))


def save_qc(
    path: Path,
    original_u8: np.ndarray,
    assigned: np.ndarray,
    safe_ids: set[int],
    status: np.ndarray,
    soma_instances: np.ndarray,
    max_size: int,
) -> None:
    background = resize_for_qc(original_u8, max_size, nearest=False)
    assigned_small = resize_for_qc(assigned, max_size, nearest=True)
    status_small = resize_for_qc(status, max_size, nearest=True)
    soma_small = resize_for_qc(soma_instances.astype(np.uint32), max_size, nearest=True)
    rgb = np.repeat(background[..., None], 3, axis=2).astype(np.float32)

    for identifier in np.unique(assigned_small):
        identifier = int(identifier)
        if identifier <= 0:
            continue
        mask = assigned_small == identifier
        color = palette_color(identifier).astype(np.float32)
        alpha = 0.82 if identifier in safe_ids else 0.48
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color

    ambiguous = status_small == 2
    unassigned = status_small == 3
    rgb[ambiguous] = 0.18 * rgb[ambiguous] + 0.82 * np.asarray([255, 145, 0])
    rgb[unassigned] = 0.18 * rgb[unassigned] + 0.82 * np.asarray([0, 225, 255])

    # Make soma pixels visible in the color of their cell ID.
    for identifier in np.unique(soma_small):
        identifier = int(identifier)
        if identifier <= 0:
            continue
        mask = soma_small == identifier
        color = palette_color(identifier).astype(np.float32)
        rgb[mask] = 0.12 * rgb[mask] + 0.88 * color

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def touches_border(mask: np.ndarray, margin: int) -> bool:
    if margin <= 0 or not np.any(mask):
        return False
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    return bool(
        rows.min() < margin
        or cols.min() < margin
        or rows.max() >= height - margin
        or cols.max() >= width - margin
    )


def export_method(
    method_name: str,
    method_dir: Path,
    skeleton_material: np.ndarray,
    soma_instances: np.ndarray,
    material_coords: np.ndarray,
    nearest_thin_indices: np.ndarray,
    thin_labels: np.ndarray,
    thin_status: np.ndarray,
    ambiguous_context_by_soma: dict[int, int],
    original_u8: np.ndarray,
    min_skeleton_pixels: int,
    max_ambiguous_fraction: float,
    border_margin: int,
    qc_max_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    method_dir.mkdir(parents=True, exist_ok=True)
    height, width = skeleton_material.shape
    assigned = np.zeros((height, width), dtype=np.uint32)
    status = np.zeros((height, width), dtype=np.uint8)
    assigned[material_coords[:, 0], material_coords[:, 1]] = thin_labels[nearest_thin_indices]
    status[material_coords[:, 0], material_coords[:, 1]] = thin_status[nearest_thin_indices]
    if np.any(status[skeleton_material] == 0):
        raise RuntimeError(
            f"{method_name}: at least one input skeleton pixel was not classified."
        )
    assigned[status != 1] = 0

    soma_count = int(soma_instances.max())
    assigned_counts = np.bincount(assigned.ravel(), minlength=soma_count + 1)
    cell_rows: list[dict[str, object]] = []
    safe_ids: set[int] = set()

    for soma_id in range(1, soma_count + 1):
        skeleton_pixels = int(assigned_counts[soma_id])
        ambiguous_context = int(ambiguous_context_by_soma.get(soma_id, 0))
        denominator = skeleton_pixels + ambiguous_context
        ambiguous_fraction = ambiguous_context / denominator if denominator else 1.0
        soma_border = touches_border(soma_instances == soma_id, border_margin)
        assigned_border = touches_border(assigned == soma_id, border_margin)
        border = soma_border or assigned_border
        safe = (
            skeleton_pixels >= min_skeleton_pixels
            and ambiguous_fraction <= max_ambiguous_fraction
            and not border
        )
        if safe:
            safe_ids.add(soma_id)
        cell_rows.append(
            {
                "method": method_name,
                "soma_id": soma_id,
                "assigned_skeleton_pixels": skeleton_pixels,
                "ambiguous_context_pixels": ambiguous_context,
                "ambiguous_fraction": round(ambiguous_fraction, 6),
                "touches_image_border": int(border),
                "safe_cell": int(safe),
            }
        )

    assigned_cells = assigned.copy()
    soma_mask = soma_instances > 0
    assigned_cells[soma_mask] = soma_instances[soma_mask]

    safe_cells = np.zeros_like(assigned_cells)
    if safe_ids:
        safe_lookup = np.zeros(soma_count + 1, dtype=bool)
        safe_lookup[list(safe_ids)] = True
        keep = (assigned_cells <= soma_count) & safe_lookup[assigned_cells]
        safe_cells[keep] = assigned_cells[keep]

    unsafe_cells = assigned_cells.copy()
    if safe_ids:
        safe_lookup = np.zeros(soma_count + 1, dtype=bool)
        safe_lookup[list(safe_ids)] = True
        unsafe_cells[safe_lookup[np.minimum(unsafe_cells, soma_count)]] = 0

    ambiguous_mask = status == 2
    unassigned_mask = status == 3
    ambiguous_instances, ambiguous_count = ndi.label(
        ambiguous_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    unassigned_instances, unassigned_count = ndi.label(
        unassigned_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )

    dtype = output_dtype(max(soma_count, int(ambiguous_count), int(unassigned_count)))
    tifffile.imwrite(method_dir / "01_cell_instances_assigned.tif", assigned_cells.astype(dtype))
    tifffile.imwrite(method_dir / "02_cell_instances_safe.tif", safe_cells.astype(dtype))
    tifffile.imwrite(method_dir / "03_cell_instances_unsafe.tif", unsafe_cells.astype(dtype))
    tifffile.imwrite(method_dir / "04_ambiguous_skeleton_mask.tif", ambiguous_mask.astype(np.uint8))
    tifffile.imwrite(
        method_dir / "05_ambiguous_regions_instances.tif", ambiguous_instances.astype(dtype)
    )
    tifffile.imwrite(method_dir / "06_unassigned_skeleton_mask.tif", unassigned_mask.astype(np.uint8))
    tifffile.imwrite(
        method_dir / "07_unassigned_skeleton_instances.tif", unassigned_instances.astype(dtype)
    )
    tifffile.imwrite(method_dir / "08_assignment_status_map.tif", status)
    save_qc(
        method_dir / "09_qc_cell_assignments.png",
        original_u8,
        assigned,
        safe_ids,
        status,
        soma_instances,
        qc_max_size,
    )

    with open(method_dir / "cell_report.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cell_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cell_rows)

    skeleton_pixels = int(np.count_nonzero(skeleton_material))
    assigned_pixels = int(np.count_nonzero(status == 1))
    ambiguous_pixels = int(np.count_nonzero(ambiguous_mask))
    unassigned_pixels = int(np.count_nonzero(unassigned_mask))
    if assigned_pixels + ambiguous_pixels + unassigned_pixels != skeleton_pixels:
        raise RuntimeError(
            f"{method_name}: assignment partition does not preserve all skeleton pixels."
        )
    summary: dict[str, object] = {
        "method": method_name,
        "soma_count": soma_count,
        "safe_cell_count": len(safe_ids),
        "skeleton_pixels": skeleton_pixels,
        "assigned_skeleton_pixels": assigned_pixels,
        "ambiguous_skeleton_pixels": ambiguous_pixels,
        "unassigned_skeleton_pixels": unassigned_pixels,
        "assigned_fraction": round(assigned_pixels / max(skeleton_pixels, 1), 6),
        "ambiguous_fraction": round(ambiguous_pixels / max(skeleton_pixels, 1), 6),
        "unassigned_fraction": round(unassigned_pixels / max(skeleton_pixels, 1), 6),
    }
    with open(method_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary, cell_rows


def write_html(output_dir: Path, summaries: list[dict[str, object]]) -> None:
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(summary[key]))}</td>"
            for key in (
                "method",
                "safe_cell_count",
                "assigned_fraction",
                "ambiguous_fraction",
                "unassigned_fraction",
            )
        )
        + "</tr>"
        for summary in summaries
    )
    cards = "".join(
        f"""
        <section class="card">
          <h2>{html.escape(str(summary['method']))}</h2>
          <img src="{html.escape(str(summary['method']))}/09_qc_cell_assignments.png"
               alt="QC {html.escape(str(summary['method']))}">
          <p><a href="{html.escape(str(summary['method']))}/02_cell_instances_safe.tif">Safe cell instances</a></p>
        </section>
        """
        for summary in summaries
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Cell assignment comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f4f6f8; color: #172033; }}
h1 {{ margin-bottom: 6px; }}
.note {{ max-width: 1100px; background: #fff7d6; padding: 12px 16px; border-left: 5px solid #e0a800; }}
table {{ border-collapse: collapse; background: white; margin: 22px 0; }}
th, td {{ border: 1px solid #ccd4df; padding: 8px 12px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 18px; }}
.card {{ background: white; border: 1px solid #ccd4df; border-radius: 8px; padding: 14px; }}
.card img {{ width: 100%; image-rendering: auto; }}
@media (max-width: 1150px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>Cell assignment comparison</h1>
<p>Input skeleton material is identical for all methods and is never deleted.</p>
<p class="note"><b>Important:</b> Matrix-Forest-inspired and G-Cut-inspired are transparent
2-D raster adaptations for this dataset. They are not claimed to be byte-for-byte
reproductions of the authors' original software.</p>
<table><thead><tr><th>Method</th><th>Safe cells</th><th>Assigned</th><th>Ambiguous</th><th>Unassigned</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="grid">{cards}</div>
</body></html>"""
    (output_dir / "comparison.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CELL ASSIGNMENT COMPARISON: {SCRIPT_VERSION}")
    print(f"RUNNING FILE: {Path(__file__).resolve()}")

    segmentation = load_2d(args.segmentation)
    unique_values = set(np.unique(segmentation).astype(int).tolist())
    invalid = unique_values - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Unexpected segmentation values: {sorted(invalid)}")

    skeleton_material = segmentation == 1
    soma_mask = segmentation == 2
    if not np.any(skeleton_material):
        raise RuntimeError("No skeleton pixels (class 1) found.")
    if not np.any(soma_mask):
        raise RuntimeError("No soma pixels (class 2) found.")

    original = load_2d(args.original) if args.original else None
    if original is not None and original.shape != segmentation.shape:
        raise RuntimeError(
            f"Shape mismatch: segmentation={segmentation.shape}, original={original.shape}"
        )
    original_u8 = normalize_original(original, segmentation.shape)

    thin = skeletonize(skeleton_material)
    component_labels, component_count = ndi.label(
        thin, structure=np.ones((3, 3), dtype=np.uint8)
    )
    soma_instances, soma_count = ndi.label(
        soma_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    soma_instances = soma_instances.astype(np.int32, copy=False)
    centers = soma_centers(soma_instances, int(soma_count))
    boundary_tree, _, boundary_labels = soma_boundary_tree(soma_instances)

    thin_coords = np.argwhere(thin)
    material_coords = np.argwhere(skeleton_material)
    thin_tree = cKDTree(thin_coords)
    _, nearest_thin_indices = thin_tree.query(material_coords, k=1)
    nearest_thin_indices = np.asarray(nearest_thin_indices, dtype=np.int64)
    flat_to_thin = {
        int(row) * segmentation.shape[1] + int(col): index
        for index, (row, col) in enumerate(thin_coords)
    }

    method_names = (
        "topology_baseline",
        "matrix_forest_inspired",
        "gcut_inspired",
    )
    method_labels = {
        name: np.zeros(len(thin_coords), dtype=np.int32) for name in method_names
    }
    method_status = {
        name: np.full(len(thin_coords), 3, dtype=np.uint8) for name in method_names
    }
    ambiguous_context: dict[str, dict[int, int]] = {
        name: defaultdict(int) for name in method_names
    }
    component_rows: list[dict[str, object]] = []
    objects = ndi.find_objects(component_labels)

    print(f"Image shape: {segmentation.shape}")
    print(f"Skeleton material pixels: {int(skeleton_material.sum())}")
    print(f"Analysis skeleton pixels: {int(thin.sum())}")
    print(f"Skeleton components: {component_count}")
    print(f"Soma instances: {soma_count}")

    for component_id, object_slice in enumerate(objects, start=1):
        if object_slice is None:
            continue
        local = component_labels[object_slice] == component_id
        local_coords = np.argwhere(local)
        offsets = np.asarray([object_slice[0].start, object_slice[1].start])
        coords = local_coords + offsets
        adjacency, weighted_adjacency = component_graph(coords, segmentation.shape[1])
        roots = roots_for_component(
            coords, boundary_tree, boundary_labels, args.root_radius
        )
        root_ids = sorted(roots)

        results = {
            "topology_baseline": topology_assignment(len(coords), roots),
            "matrix_forest_inspired": matrix_forest_assignment(
                weighted_adjacency,
                roots,
                args.mft_tau,
                args.mft_seed_strength,
                args.mft_min_confidence,
                args.mft_margin,
            ),
            "gcut_inspired": gcut_directional_assignment(
                coords,
                adjacency,
                roots,
                centers,
                args.gcut_turn_weight,
                args.gcut_radial_weight,
                args.gcut_margin,
            ),
        }

        global_indices = np.asarray(
            [
                flat_to_thin[int(row) * segmentation.shape[1] + int(col)]
                for row, col in coords
            ],
            dtype=np.int64,
        )
        for method_name, (labels, status, diagnostics) in results.items():
            method_labels[method_name][global_indices] = labels
            method_status[method_name][global_indices] = status
            ambiguous_count = int(np.count_nonzero(status == 2))
            if ambiguous_count:
                for soma_id in root_ids:
                    ambiguous_context[method_name][soma_id] += ambiguous_count
            component_rows.append(
                {
                    "method": method_name,
                    "component_id": component_id,
                    "analysis_pixels": len(coords),
                    "touching_soma_ids": ";".join(map(str, root_ids)),
                    "number_of_somata": len(root_ids),
                    "assigned_pixels": int(np.count_nonzero(status == 1)),
                    "ambiguous_pixels": ambiguous_count,
                    "unassigned_pixels": int(np.count_nonzero(status == 3)),
                    "diagnostics": json.dumps(diagnostics, sort_keys=True),
                }
            )
        if component_id % 100 == 0 or component_id == component_count:
            print(f"Processed components: {component_id}/{component_count}")

    tifffile.imwrite(
        args.output_dir / "00_input_completed_skeleton_PRESERVED.tif",
        skeleton_material.astype(np.uint8),
    )
    tifffile.imwrite(
        args.output_dir / "00_soma_instances.tif",
        soma_instances.astype(output_dtype(int(soma_count))),
    )

    summaries: list[dict[str, object]] = []
    all_cell_rows: list[dict[str, object]] = []
    for method_name in method_names:
        summary, cell_rows = export_method(
            method_name=method_name,
            method_dir=args.output_dir / method_name,
            skeleton_material=skeleton_material,
            soma_instances=soma_instances,
            material_coords=material_coords,
            nearest_thin_indices=nearest_thin_indices,
            thin_labels=method_labels[method_name],
            thin_status=method_status[method_name],
            ambiguous_context_by_soma=ambiguous_context[method_name],
            original_u8=original_u8,
            min_skeleton_pixels=args.safe_min_skeleton_pixels,
            max_ambiguous_fraction=args.safe_max_ambiguous_fraction,
            border_margin=args.border_margin,
            qc_max_size=args.qc_max_size,
        )
        summaries.append(summary)
        all_cell_rows.extend(cell_rows)

    with open(args.output_dir / "comparison_summary.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with open(args.output_dir / "all_cells_report.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_cell_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_cell_rows)
    with open(args.output_dir / "component_report.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(component_rows[0].keys()))
        writer.writeheader()
        writer.writerows(component_rows)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "segmentation": str(args.segmentation.resolve()),
        "original": str(args.original.resolve()) if args.original else None,
        "shape": list(segmentation.shape),
        "skeleton_material_pixels": int(skeleton_material.sum()),
        "removed_skeleton_pixels": 0,
        "soma_count": int(soma_count),
        "parameters": vars(args) | {
            "segmentation": str(args.segmentation),
            "original": str(args.original) if args.original else None,
            "output_dir": str(args.output_dir),
        },
    }
    with open(args.output_dir / "run_metadata.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    (args.output_dir / "METHOD_NOTES.txt").write_text(
        "The input skeleton is identical for all methods and is never deleted.\n\n"
        "topology_baseline:\n"
        "  A connected skeleton component is safe only when it contacts exactly one soma.\n\n"
        "matrix_forest_inspired:\n"
        "  Regularized graph-Laplacian/forest influence is propagated from soma roots.\n"
        "  This is a transparent 2-D raster adaptation, not the authors' original code.\n\n"
        "gcut_inspired:\n"
        "  Soma-rooted multi-source paths compete using path length, turn continuity and\n"
        "  outward growth orientation. This is a 2-D adaptation, not the original G-Cut LP.\n",
        encoding="utf-8",
    )
    write_html(args.output_dir, summaries)

    print("\n" + "=" * 72)
    print("CELL ASSIGNMENT COMPARISON COMPLETE")
    print("=" * 72)
    for summary in summaries:
        print(
            f"{summary['method']}: safe={summary['safe_cell_count']}, "
            f"assigned={summary['assigned_fraction']:.3f}, "
            f"ambiguous={summary['ambiguous_fraction']:.3f}, "
            f"unassigned={summary['unassigned_fraction']:.3f}"
        )
    print(f"HTML: {args.output_dir / 'comparison.html'}")
    print("Removed input skeleton pixels: 0")


if __name__ == "__main__":
    main()
