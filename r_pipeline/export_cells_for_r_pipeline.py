from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from canonical_skeleton import canonicalize_skeleton, detached_component_count


SCRIPT_VERSION = "r-pipeline-cell-export-v6-snt-path-roots-2026-09-10"
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify cells from the no-loss 0/1/2 postprocessing result and "
            "export every safe cell into a pipeline-compatible cell folder."
        )
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Original 2D overview/max-projection image.",
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        required=True,
        help="15_completed_skeleton_and_soma_0-1-2.tif.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contact-radius",
        type=int,
        default=4,
        help=(
            "Maximum pixel distance used to associate a skeleton component "
            "with a soma (default: 4)."
        ),
    )
    parser.add_argument(
        "--min-soma-area",
        type=int,
        default=20,
        help="Minimum soma area used as a real cell seed (default: 20).",
    )
    parser.add_argument(
        "--min-skeleton-pixels",
        type=int,
        default=8,
        help=(
            "Minimum assigned full-width skeleton pixels required for a safe "
            "cell export (default: 8)."
        ),
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=32,
        help="Context margin around each exported cell in pixels (default: 32).",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=128,
        help="Minimum crop height and width (default: 128).",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="Export square crops.",
    )
    parser.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="Number of the first cell folder (default: 1).",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Zero-padding width, for example cell0001 (default: 4).",
    )
    parser.add_argument(
        "--allow-soma-only",
        action="store_true",
        help=(
            "Also classify a soma without assigned skeleton as safe. "
            "Not recommended for morphology/PCA."
        ),
    )
    return parser.parse_args()


def read_2d(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be 2D, got {array.shape}: {path}")
    return array


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty:\n{path}\n"
            "Use a new directory so existing cell folders are not overwritten."
        )
    path.mkdir(parents=True, exist_ok=True)


def save_tiff(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(path, array, photometric="minisblack")


def axis_bounds(
    minimum: int,
    maximum_inclusive: int,
    limit: int,
    margin: int,
    minimum_size: int,
) -> tuple[int, int]:
    desired = max(
        maximum_inclusive - minimum + 1 + 2 * margin,
        minimum_size,
    )
    desired = min(desired, limit)
    center = 0.5 * (minimum + maximum_inclusive)
    start = int(math.floor(center - desired / 2))
    start = max(0, min(start, limit - desired))
    return start, start + desired


def crop_bounds(
    mask: np.ndarray,
    margin: int,
    minimum_size: int,
    square: bool,
) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise RuntimeError("Cannot crop an empty cell.")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    height, width = mask.shape
    if square:
        requested = max(
            int(max(y_max - y_min + 1, x_max - x_min + 1)) + 2 * margin,
            minimum_size,
        )
        y0, y1 = axis_bounds(
            int(y_min), int(y_max), height, margin, requested
        )
        x0, x1 = axis_bounds(
            int(x_min), int(x_max), width, margin, requested
        )
    else:
        y0, y1 = axis_bounds(
            int(y_min), int(y_max), height, margin, minimum_size
        )
        x0, x1 = axis_bounds(
            int(x_min), int(x_max), width, margin, minimum_size
        )
    return y0, y1, x0, x1


def remove_small_soma_seeds(
    soma: np.ndarray,
    minimum_area: int,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    raw_labels, raw_count = ndi.label(soma, structure=CONNECTIVITY_8)
    areas = np.bincount(raw_labels.ravel())
    valid_old_ids = [
        component_id
        for component_id in range(1, raw_count + 1)
        if int(areas[component_id]) >= minimum_area
    ]
    seed_mask = np.isin(raw_labels, valid_old_ids)
    labels, count = ndi.label(seed_mask, structure=CONNECTIVITY_8)
    rejected = [
        component_id
        for component_id in range(1, raw_count + 1)
        if component_id not in valid_old_ids
    ]
    return seed_mask, labels.astype(np.uint32), count, rejected


def classify_topology(
    skeleton: np.ndarray,
    soma_labels: np.ndarray,
    soma_count: int,
    contact_radius: int,
    min_skeleton_pixels: int,
    allow_soma_only: bool,
) -> dict[str, object]:
    skeleton_labels, skeleton_count = ndi.label(
        skeleton,
        structure=CONNECTIVITY_8,
    )
    expanded_soma = ndi.grey_dilation(
        soma_labels,
        size=(2 * contact_radius + 1, 2 * contact_radius + 1),
    )

    component_to_somata: dict[int, list[int]] = {}
    uniquely_assigned: dict[int, list[int]] = {
        soma_id: [] for soma_id in range(1, soma_count + 1)
    }
    ambiguous_components: list[int] = []
    unassigned_components: list[int] = []
    soma_has_ambiguous_contact = {
        soma_id: False for soma_id in range(1, soma_count + 1)
    }

    for component_id in range(1, skeleton_count + 1):
        component = skeleton_labels == component_id
        touched = sorted(
            int(value)
            for value in np.unique(expanded_soma[component])
            if int(value) > 0
        )
        component_to_somata[component_id] = touched
        if len(touched) == 1:
            uniquely_assigned[touched[0]].append(component_id)
        elif len(touched) > 1:
            ambiguous_components.append(component_id)
            for soma_id in touched:
                soma_has_ambiguous_contact[soma_id] = True
        else:
            unassigned_components.append(component_id)

    safe_soma_ids: list[int] = []
    ambiguous_soma_ids: list[int] = []
    soma_only_ids: list[int] = []
    soma_skeleton_pixels: dict[int, int] = {}
    for soma_id in range(1, soma_count + 1):
        component_ids = uniquely_assigned[soma_id]
        pixel_count = int(np.isin(skeleton_labels, component_ids).sum())
        soma_skeleton_pixels[soma_id] = pixel_count
        if soma_has_ambiguous_contact[soma_id]:
            ambiguous_soma_ids.append(soma_id)
        elif pixel_count >= min_skeleton_pixels:
            safe_soma_ids.append(soma_id)
        elif allow_soma_only:
            safe_soma_ids.append(soma_id)
            soma_only_ids.append(soma_id)
        else:
            soma_only_ids.append(soma_id)

    safe_instances = np.zeros(skeleton.shape, dtype=np.uint32)
    ambiguous_instances = np.zeros(skeleton.shape, dtype=np.uint32)
    safe_component_ids: set[int] = set()

    next_safe_id = 1
    safe_id_to_soma: dict[int, int] = {}
    for soma_id in safe_soma_ids:
        component_ids = uniquely_assigned[soma_id]
        cell_mask = (soma_labels == soma_id) | np.isin(
            skeleton_labels,
            component_ids,
        )
        safe_instances[cell_mask] = next_safe_id
        safe_component_ids.update(component_ids)
        safe_id_to_soma[next_safe_id] = soma_id
        next_safe_id += 1

    next_ambiguous_id = 1
    for soma_id in ambiguous_soma_ids:
        component_ids = uniquely_assigned[soma_id]
        related_ambiguous = [
            component_id
            for component_id in ambiguous_components
            if soma_id in component_to_somata[component_id]
        ]
        mask = (
            (soma_labels == soma_id)
            | np.isin(skeleton_labels, component_ids)
            | np.isin(skeleton_labels, related_ambiguous)
        )
        ambiguous_instances[mask] = next_ambiguous_id
        next_ambiguous_id += 1

    ambiguous_skeleton = np.isin(
        skeleton_labels,
        ambiguous_components,
    )
    unassigned_skeleton = np.isin(
        skeleton_labels,
        unassigned_components,
    )
    status = np.zeros(skeleton.shape, dtype=np.uint8)
    status[safe_instances > 0] = 1
    status[ambiguous_instances > 0] = 2
    status[ambiguous_skeleton] = 2
    status[unassigned_skeleton] = 3

    return {
        "skeleton_labels": skeleton_labels.astype(np.uint32),
        "skeleton_count": skeleton_count,
        "component_to_somata": component_to_somata,
        "uniquely_assigned": uniquely_assigned,
        "ambiguous_components": ambiguous_components,
        "unassigned_components": unassigned_components,
        "safe_soma_ids": safe_soma_ids,
        "ambiguous_soma_ids": ambiguous_soma_ids,
        "soma_only_ids": soma_only_ids,
        "soma_skeleton_pixels": soma_skeleton_pixels,
        "safe_instances": safe_instances,
        "safe_id_to_soma": safe_id_to_soma,
        "ambiguous_instances": ambiguous_instances,
        "unassigned_skeleton": unassigned_skeleton,
        "status": status,
    }


def write_pixel_csv(
    path: Path,
    skeleton: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> None:
    coordinates = np.argwhere(skeleton)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "global_x", "global_y", "class"])
        for y, x in coordinates:
            writer.writerow(
                [int(x), int(y), int(x + x_offset), int(y + y_offset), 1]
            )


def write_swc(
    path: Path,
    skeleton: np.ndarray,
    soma: np.ndarray,
) -> dict[str, int | float]:
    """Write an SNT-style SWC with a shared primary-neurite root.

    Every primary neurite must explicitly touch the soma mask. Its first node is
    connected directly to one shared root. Manual SNT exports place this zero-
    length ``Shared root`` between the starts of otherwise independent primary
    paths, rather than at the centroid of the complete soma. We reproduce that
    convention by using the arithmetic mean of the soma-adjacent component
    roots. With one primary path, that attachment is the root and SNT adds a
    second, straight two-point path from it to the soma center. With multiple
    paths, each path starts with a zero-length node at the shared root before
    its straight connector to the attachment. All remaining neurite edges
    follow adjacent pixels of the canonical 1-px mask.
    """

    thin, canonical_report = canonicalize_skeleton(
        skeleton,
        soma=soma,
        max_soma_gap_px=3.0,
    )
    soma = np.asarray(soma, dtype=bool)
    if detached_component_count(thin, soma):
        raise ValueError(
            "Skeleton contains a component that is not attached to the soma after "
            "the allowed 3-px gap repair."
        )
    soma_coordinates = np.argwhere(soma)
    if soma_coordinates.size:
        soma_y, soma_x = soma_coordinates.mean(axis=0)
        soma_radius = max(
            1.0,
            math.sqrt(float(soma_coordinates.shape[0]) / math.pi),
        )
    else:
        raise ValueError("Cannot write a cell SWC without a soma mask.")

    labels, count = ndi.label(thin, structure=CONNECTIVITY_8)
    height, width = thin.shape

    def neighbors(point: tuple[int, int]):
        y, x = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    yield ny, nx

    component_roots: dict[int, tuple[int, int]] = {}
    adjacent_to_soma = ndi.binary_dilation(soma, structure=CONNECTIVITY_8)
    for component_id in range(1, count + 1):
        candidates = np.argwhere(
            (labels == component_id) & adjacent_to_soma
        )
        if candidates.size == 0:
            raise ValueError(
                f"Skeleton component {component_id} has no explicit soma attachment."
            )
        distances = (
            (candidates[:, 0] - soma_y) ** 2
            + (candidates[:, 1] - soma_x) ** 2
        )
        component_roots[component_id] = tuple(
            map(int, candidates[int(np.argmin(distances))])
        )

    shared_root_y = float(
        np.mean([point[0] for point in component_roots.values()])
    )
    shared_root_x = float(
        np.mean([point[1] for point in component_roots.values()])
    )
    rows: list[tuple[int, int, float, float, float, float, int]] = [
        (
            1,
            1,
            shared_root_x,
            shared_root_y,
            0.0,
            soma_radius,
            -1,
        )
    ]
    next_node_id = 2

    shared_path_start_nodes = 0
    soma_center_connector_nodes = 0
    for component_id in range(1, count + 1):
        component_root = component_roots[component_id]
        visited: set[tuple[int, int]] = {component_root}
        if count == 1:
            # The SWC root itself represents the first skeleton pixel. Continue
            # the one primary path with its neighboring pixels.
            queue_with_parents = deque(
                (point, 1)
                for point in sorted(neighbors(component_root))
                if labels[point] == component_id
            )
            visited.update(point for point, _parent in queue_with_parents)
        else:
            # SNT stores a zero-length start node for every primary path at the
            # shared root and then draws a straight connector to the real
            # soma-adjacent skeleton pixel.
            path_start_id = next_node_id
            next_node_id += 1
            rows.append(
                (
                    path_start_id,
                    3,
                    shared_root_x,
                    shared_root_y,
                    0.0,
                    1.0,
                    1,
                )
            )
            shared_path_start_nodes += 1
            queue_with_parents = deque([(component_root, path_start_id)])
        while queue_with_parents:
            (y, x), parent_id = queue_with_parents.popleft()
            node_id = next_node_id
            next_node_id += 1
            rows.append((node_id, 3, float(x), float(y), 0.0, 1.0, parent_id))
            unvisited_neighbors: list[tuple[int, int]] = []
            for point in sorted(neighbors((y, x))):
                if labels[point] == component_id and point not in visited:
                    unvisited_neighbors.append(point)
            for point in unvisited_neighbors:
                visited.add(point)
                queue_with_parents.append((point, node_id))

    if count == 1:
        # A single manual SNT primary path has a second path from the neurite
        # attachment into the soma. Keep the duplicate path-start node because
        # it is part of the manual SWC tree convention.
        connector_start_id = next_node_id
        connector_end_id = next_node_id + 1
        rows.append(
            (
                connector_start_id,
                3,
                shared_root_x,
                shared_root_y,
                0.0,
                1.0,
                1,
            )
        )
        rows.append(
            (
                connector_end_id,
                3,
                float(soma_x),
                float(soma_y),
                0.0,
                1.0,
                connector_start_id,
            )
        )
        soma_center_connector_nodes = 2

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Generated by {SCRIPT_VERSION}\n")
        handle.write("# https://github.com/benolelevermann/MATS\n")
        handle.write("#\n")
        handle.write("# All positions and radii in pixels\n")
        handle.write("# Voxel separation (x,y,z): 1.0, 1.0, 1.0\n")
        handle.write("#\n")
        for row in rows:
            handle.write(
                f"{row[0]} {row[1]} {row[2]:.3f} {row[3]:.3f} "
                f"{row[4]:.3f} {row[5]:.3f} {row[6]}\n"
            )
    return {
        "nodes": len(rows),
        "soma_connector_nodes": (
            1 + shared_path_start_nodes + soma_center_connector_nodes
        ),
        "shared_path_start_nodes": shared_path_start_nodes,
        "soma_center_connector_nodes": soma_center_connector_nodes,
        "skeleton_nodes": int(thin.sum()),
        "skeleton_components": int(count),
        "soma_gap_pixels_added": canonical_report.soma_gap_pixels_added,
        "shared_root_x": shared_root_x,
        "shared_root_y": shared_root_y,
    }


def normalize_grayscale(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def instance_color(instance_id: int) -> np.ndarray:
    hue = (instance_id * 0.618033988749895) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.85, 1.0), dtype=np.float32)


def save_qc_overlay(
    path: Path,
    original: np.ndarray,
    safe_instances: np.ndarray,
    ambiguous_instances: np.ndarray,
    ambiguous_skeleton: np.ndarray,
    unassigned_skeleton: np.ndarray,
) -> None:
    gray = normalize_grayscale(original).astype(np.float32) / 255.0
    rgb = np.repeat(gray[..., None], 3, axis=2)
    safe_display = ndi.grey_dilation(safe_instances, size=(3, 3))
    for instance_id in np.unique(safe_display):
        if int(instance_id) <= 0:
            continue
        mask = safe_display == instance_id
        color = instance_color(int(instance_id))
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * color
    ambiguous = (ambiguous_instances > 0) | ambiguous_skeleton
    ambiguous = ndi.binary_dilation(ambiguous, iterations=1)
    unassigned = ndi.binary_dilation(unassigned_skeleton, iterations=1)
    rgb[ambiguous] = 0.2 * rgb[ambiguous] + 0.8 * np.array([1.0, 0.45, 0.0])
    rgb[unassigned] = 0.2 * rgb[unassigned] + 0.8 * np.array([0.0, 0.9, 1.0])
    Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if args.contact_radius < 0:
        raise ValueError("--contact-radius must be >= 0")
    if args.min_soma_area < 1:
        raise ValueError("--min-soma-area must be >= 1")
    if args.min_skeleton_pixels < 0:
        raise ValueError("--min-skeleton-pixels must be >= 0")
    if args.margin < 0 or args.min_crop_size < 1:
        raise ValueError("--margin must be >= 0 and --min-crop-size >= 1")
    if args.start_number < 0 or args.digits < 1:
        raise ValueError("--start-number must be >= 0 and --digits >= 1")

    print(f"R PIPELINE CELL EXPORTER: {SCRIPT_VERSION}")
    print(f"RUNNING FILE: {Path(__file__).resolve()}")
    original = read_2d(args.original, "original image")
    semantic = read_2d(args.semantic, "semantic mask")
    if original.shape != semantic.shape:
        raise RuntimeError(
            f"Shape mismatch: original={original.shape}, semantic={semantic.shape}"
        )
    semantic_values = set(np.unique(semantic).astype(int).tolist())
    invalid = semantic_values - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Unexpected semantic values: {sorted(invalid)}")

    skeleton = semantic == 1
    soma = semantic == 2
    soma_seed_mask, soma_labels, soma_count, rejected_soma_ids = (
        remove_small_soma_seeds(soma, args.min_soma_area)
    )
    classification = classify_topology(
        skeleton=skeleton,
        soma_labels=soma_labels,
        soma_count=soma_count,
        contact_radius=args.contact_radius,
        min_skeleton_pixels=args.min_skeleton_pixels,
        allow_soma_only=args.allow_soma_only,
    )
    safe_instances = classification["safe_instances"]
    ambiguous_instances = classification["ambiguous_instances"]
    unassigned_skeleton = classification["unassigned_skeleton"]
    ambiguous_skeleton = np.isin(
        classification["skeleton_labels"],
        classification["ambiguous_components"],
    )
    safe_ids = [
        int(value) for value in np.unique(safe_instances) if int(value) > 0
    ]
    if not safe_ids:
        raise RuntimeError(
            "No safe cells found. Inspect the classification outputs and try "
            "a slightly larger --contact-radius."
        )

    ensure_empty_output(args.output_dir)
    diagnostics = args.output_dir / "_classification"
    diagnostics.mkdir()
    save_tiff(
        diagnostics / "safe_cell_instances_overview.tif",
        safe_instances.astype(np.uint32),
    )
    save_tiff(
        diagnostics / "ambiguous_cell_instances_overview.tif",
        ambiguous_instances.astype(np.uint32),
    )
    save_tiff(
        diagnostics / "unassigned_skeleton_overview.tif",
        unassigned_skeleton.astype(np.uint8),
    )
    save_tiff(
        diagnostics / "cell_status_overview.tif",
        classification["status"].astype(np.uint8),
    )
    save_tiff(
        diagnostics / "valid_soma_seeds_overview.tif",
        soma_seed_mask.astype(np.uint8),
    )
    save_qc_overlay(
        diagnostics / "cell_classification_overlay.png",
        original,
        safe_instances,
        ambiguous_instances,
        ambiguous_skeleton,
        unassigned_skeleton,
    )

    safe_id_to_soma = classification["safe_id_to_soma"]
    manifest_rows: list[dict[str, object]] = []
    for index, safe_id in enumerate(safe_ids):
        output_number = args.start_number + index
        folder_name = f"cell{output_number:0{args.digits}d}"
        cell_dir = args.output_dir / folder_name
        cell_dir.mkdir()

        global_cell_mask = safe_instances == safe_id
        soma_id = int(safe_id_to_soma[safe_id])
        global_soma_mask = soma_labels == soma_id
        global_skeleton_mask = global_cell_mask & skeleton
        y0, y1, x0, x1 = crop_bounds(
            global_cell_mask,
            args.margin,
            args.min_crop_size,
            args.square,
        )
        raw_crop = original[y0:y1, x0:x1]
        skeleton_crop = global_skeleton_mask[y0:y1, x0:x1]
        soma_crop = global_soma_mask[y0:y1, x0:x1]
        cell_crop = skeleton_crop | soma_crop
        seg_crop = np.zeros(cell_crop.shape, dtype=np.uint8)
        seg_crop[skeleton_crop] = 1
        seg_crop[soma_crop] = 2

        save_tiff(cell_dir / "raw.tif", raw_crop)
        save_tiff(cell_dir / "skeleton.tif", skeleton_crop.astype(np.uint8))
        save_tiff(cell_dir / "soma.tif", soma_crop.astype(np.uint8))
        save_tiff(cell_dir / "seg.tif", seg_crop)
        save_tiff(cell_dir / "cell_mask.tif", cell_crop.astype(np.uint8))
        write_pixel_csv(cell_dir / "seg.csv", skeleton_crop, x0, y0)
        write_swc(cell_dir / "seg-000.swc", skeleton_crop, soma_crop)

        soma_coordinates = np.argwhere(soma_crop)
        local_y, local_x = soma_coordinates.mean(axis=0)
        bounds = {
            "x_min": int(x0),
            "y_min": int(y0),
            "x_max_exclusive": int(x1),
            "y_max_exclusive": int(y1),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
        }
        location = {
            "local_x": float(local_x),
            "local_y": float(local_y),
            "global_x": float(local_x + x0),
            "global_y": float(local_y + y0),
        }
        metadata = {
            "script_version": SCRIPT_VERSION,
            "status": "safe",
            "folder": folder_name,
            "safe_instance_id": safe_id,
            "source_soma_id": soma_id,
            "skeleton_pixels": int(skeleton_crop.sum()),
            "soma_pixels": int(soma_crop.sum()),
            "bounds": bounds,
            "location": location,
        }
        (cell_dir / "bounds.json").write_text(
            json.dumps(bounds, indent=2), encoding="utf-8"
        )
        (cell_dir / "location.json").write_text(
            json.dumps(location, indent=2), encoding="utf-8"
        )
        (cell_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        manifest_rows.append(
            {
                "folder": folder_name,
                "status": "safe",
                "safe_instance_id": safe_id,
                "source_soma_id": soma_id,
                "x_min": x0,
                "y_min": y0,
                "x_max_exclusive": x1,
                "y_max_exclusive": y1,
                "global_centroid_x": location["global_x"],
                "global_centroid_y": location["global_y"],
                "skeleton_pixels": int(skeleton_crop.sum()),
                "soma_pixels": int(soma_crop.sum()),
            }
        )
        print(
            f"[{index + 1}/{len(safe_ids)}] {folder_name}: "
            f"soma={soma_id}, crop={x1 - x0}x{y1 - y0}, "
            f"skeleton={int(skeleton_crop.sum())}, "
            f"soma pixels={int(soma_crop.sum())}"
        )

    with (args.output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "safe_cells_exported": len(safe_ids),
        "ambiguous_somata": len(classification["ambiguous_soma_ids"]),
        "soma_without_enough_skeleton": len(classification["soma_only_ids"]),
        "unassigned_skeleton_components": len(
            classification["unassigned_components"]
        ),
        "ambiguous_skeleton_components": len(
            classification["ambiguous_components"]
        ),
        "small_soma_components_not_used_as_seeds": len(rejected_soma_ids),
        "contact_radius": args.contact_radius,
        "min_soma_area": args.min_soma_area,
        "min_skeleton_pixels": args.min_skeleton_pixels,
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "_export_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n",
        encoding="utf-8",
    )
    print()
    print(f"Exported {len(safe_ids)} safe cells to:")
    print(args.output_dir.resolve())
    print()
    print("NEXT: Run finalize_cells_with_fiji.py to create:")
    print("seg.traces, soma.zip, bounds.zip and location.zip")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
