
from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import csv
import math

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import disk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Ole\20260721_CellClassification_v2")

RAW_PATH = (
    PROJECT_ROOT
    / "overview_inference"
    / "input_2d"
    / "overview_max_0000.tif"
)

PRED_PATH = (
    PROJECT_ROOT
    / "overview_inference"
    / "output_2d"
    / "overview_max.tif"
)

# Created by nnUNet prediction with --save_probabilities.
# The script also runs without this file, but probability-based
# skeleton recovery is the preferred method.
PROBABILITY_PATH = (
    PROJECT_ROOT
    / "overview_inference"
    / "output_2d"
    / "overview_max.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "overview_inference"
    / "postprocessed_v3"
)


# ============================================================
# PARAMETERS
# ============================================================

# Soma components smaller than this are preserved, but they are
# treated as uncertain and are not used as reliable assignment seeds.
MIN_RELIABLE_SOMA_AREA_PX = 20

# Skeleton recovery from probability map (hysteresis-style).
USE_PROBABILITIES = True
SKELETON_LOW_PROBABILITY = 0.12
SOMA_EXCLUSION_PROBABILITY = 0.50

# Weak skeleton probability pixels are only accepted this far away
# from a hard skeleton pixel. Prevents flooding through background.
MAX_PROBABILITY_RECOVERY_DISTANCE_PX = 12

# Conservative geometric gap closing. This is performed separately
# inside each reliable soma's nearest-region, so cells are less likely
# to be joined to one another.
GAP_CLOSE_RADIUS_PX = 2

# Assignment-only dilation. Does not change the saved skeleton itself.
ASSIGNMENT_BRIDGE_RADIUS_PX = 5

# Strong ambiguity rules.
# A skeleton component is ambiguous when it is divided between multiple
# soma IDs or physically reaches multiple soma regions.
TOUCH_SOMA_RADIUS_PX = 3

# A long skeleton component can cross a nearest-soma boundary by just
# one or two pixels. That should not automatically mark both cells.
VORONOI_SECOND_FRACTION_THRESHOLD = 0.25
VORONOI_SECOND_PIXEL_MIN = 3

# Optional softer ambiguity rule based on nearest and second-nearest
# soma centroid distances. Disabled by default in dense FOVs.
USE_SOFT_CENTROID_AMBIGUITY = False
AMBIGUITY_SECOND_NEAREST_RATIO = 1.20
AMBIGUITY_SECOND_NEAREST_MARGIN_PX = 12.0

# Far fragments are preserved and highlighted separately. By default
# they do not mark the complete cell as ambiguous.
FAR_SKELETON_DISTANCE_PX = 120.0
MARK_FAR_CELL_AS_AMBIGUOUS = False

# Visible marker settings. These are deliberately thick because a
# 5000x5000 image is usually shown as a much smaller preview.
CELL_OUTLINE_RADIUS_PX = 8
SOMA_MARKER_MARGIN_PX = 18
SOMA_MARKER_LINE_WIDTH_PX = 8
SHOW_CELL_ID_TEXT = True

MAX_PREVIEW_SIZE = 2500
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(np.asarray(image)).astype(np.float32)
    p1, p99 = np.percentile(image, [1, 99])

    if p99 <= p1:
        p1 = float(image.min())
        p99 = float(image.max())

    if p99 <= p1:
        return np.zeros(image.shape, dtype=np.uint8)

    image = (image - p1) / (p99 - p1)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def save_preview(array: np.ndarray, path: Path, max_size: int) -> None:
    image = Image.fromarray(array)
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    image.save(path)


def output_integer_dtype(maximum_id: int):
    if maximum_id <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def load_probabilities(npz_path, expected_shape):
    """
    Lädt nnU-Net-Klassenwahrscheinlichkeiten und bringt sie
    in die Form (Klassen, Höhe, Breite).

    Unterstützt unter anderem:
        (3, H, W)
        (3, 1, H, W)
        (1, 3, H, W)
        (H, W, 3)
    """

    with np.load(npz_path) as data:
        keys = list(data.keys())
        print("NPZ keys:", keys)

        if "probabilities" in data:
            probabilities = np.asarray(data["probabilities"])
        elif "softmax" in data:
            probabilities = np.asarray(data["softmax"])
        elif len(keys) == 1:
            probabilities = np.asarray(data[keys[0]])
            print(f"Nutze NPZ-Key: {keys[0]}")
        else:
            raise RuntimeError(
                f"Keine probabilities oder softmax in der NPZ gefunden. "
                f"Vorhandene Keys: {keys}"
            )

    print("Probability shape before normalization:", probabilities.shape)

    # Zusätzliche Dimensionen der Größe 1 entfernen.
    # Bei dir: (3, 1, 5000, 5000) -> (3, 5000, 5000)
    probabilities = np.squeeze(probabilities)

    print("Probability shape after squeeze:", probabilities.shape)

    if probabilities.ndim != 3:
        raise RuntimeError(
            f"Nach Entfernen der Singleton-Dimensionen werden 3 Dimensionen "
            f"erwartet, gefunden wurde {probabilities.shape}"
        )

    height, width = expected_shape

    # Standardformat von nnU-Net: Klassen, Höhe, Breite
    if probabilities.shape[1:] == (height, width):
        probabilities_chw = probabilities

    # Alternativformat: Höhe, Breite, Klassen
    elif probabilities.shape[:2] == (height, width):
        probabilities_chw = np.moveaxis(probabilities, -1, 0)

    else:
        raise RuntimeError(
            f"Probability-Shape passt nicht zur Prediction. "
            f"Probabilities: {probabilities.shape}, "
            f"Prediction: {expected_shape}"
        )

    if probabilities_chw.shape[0] < 3:
        raise RuntimeError(
            f"Es werden mindestens 3 Klassen erwartet: "
            f"Background, Skeleton und Soma. "
            f"Gefunden wurden {probabilities_chw.shape[0]} Klassen."
        )

    print("Normalized probability shape:", probabilities_chw.shape)

    return probabilities_chw.astype(np.float32, copy=False)

def relabel_reliable_somata(
    soma_mask: np.ndarray,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    all_instances = cc_label(soma_mask, connectivity=2)
    reliable_instances = np.zeros_like(all_instances, dtype=np.int32)
    uncertain_soma_mask = np.zeros_like(soma_mask, dtype=bool)

    original_to_reliable: dict[int, int] = {}
    next_id = 0

    for region in regionprops(all_instances):
        original_id = int(region.label)

        if int(region.area) >= min_area:
            next_id += 1
            reliable_instances[all_instances == original_id] = next_id
            original_to_reliable[original_id] = next_id
        else:
            uncertain_soma_mask[all_instances == original_id] = True

    return (
        all_instances,
        reliable_instances,
        uncertain_soma_mask,
        original_to_reliable,
    )


def recover_skeleton_from_probabilities(
    hard_skeleton: np.ndarray,
    soma_mask: np.ndarray,
    probabilities: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if probabilities is None or not USE_PROBABILITIES:
        return hard_skeleton.copy(), np.zeros_like(hard_skeleton, dtype=bool)

    skeleton_probability = probabilities[1]
    soma_probability = probabilities[2]

    distance_from_hard = ndi.distance_transform_edt(~hard_skeleton)

    weak_skeleton_region = (
        (skeleton_probability >= SKELETON_LOW_PROBABILITY)
        & (soma_probability < SOMA_EXCLUSION_PROBABILITY)
        & (distance_from_hard <= MAX_PROBABILITY_RECOVERY_DISTANCE_PX)
        & (~soma_mask)
    )

    propagation_mask = weak_skeleton_region | hard_skeleton

    recovered = ndi.binary_propagation(
        hard_skeleton,
        structure=np.ones((3, 3), dtype=bool),
        mask=propagation_mask,
    )

    recovered &= ~soma_mask
    probability_added = recovered & ~hard_skeleton

    return recovered, probability_added


def nearest_seed_maps(
    reliable_soma_instances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if reliable_soma_instances.max() == 0:
        raise RuntimeError(
            "No reliable soma seeds remain. Reduce "
            "MIN_RELIABLE_SOMA_AREA_PX."
        )

    distance, nearest_coordinates = ndi.distance_transform_edt(
        reliable_soma_instances == 0,
        return_indices=True,
    )

    nearest_id = reliable_soma_instances[
        nearest_coordinates[0],
        nearest_coordinates[1],
    ]

    return distance, nearest_id


def assign_original_skeleton_components(
    skeleton_mask: np.ndarray,
    reliable_soma_instances: np.ndarray,
    nearest_soma_id: np.ndarray,
    distance_to_soma: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    set[int],
    list[dict[str, object]],
]:
    skeleton_components = cc_label(skeleton_mask, connectivity=2)
    assigned_map = np.zeros_like(reliable_soma_instances, dtype=np.int32)

    ambiguous_cell_ids: set[int] = set()
    report_rows: list[dict[str, object]] = []

    soma_regions = regionprops(reliable_soma_instances)
    soma_centroids = np.array(
        [region.centroid for region in soma_regions],
        dtype=np.float32,
    )
    soma_ids = np.array(
        [int(region.label) for region in soma_regions],
        dtype=np.int32,
    )

    for region in regionprops(skeleton_components):
        component_id = int(region.label)
        coordinates = region.coords
        yy = coordinates[:, 0]
        xx = coordinates[:, 1]

        nearest_ids = nearest_soma_id[yy, xx]
        nearest_ids = nearest_ids[nearest_ids > 0]

        if nearest_ids.size == 0:
            chosen_id = 0
            id_counts: Counter[int] = Counter()
        else:
            id_counts = Counter(nearest_ids.tolist())
            chosen_id = int(id_counts.most_common(1)[0][0])
            assigned_map[yy, xx] = chosen_id

        flags: list[str] = []

        split_ids = sorted(id_counts.keys())
        if len(id_counts) > 1:
            ordered_counts = id_counts.most_common()
            total_count = sum(id_counts.values())
            second_id_by_pixels, second_count = ordered_counts[1]
            second_fraction = second_count / max(1, total_count)

            if (
                second_count >= VORONOI_SECOND_PIXEL_MIN
                and second_fraction >= VORONOI_SECOND_FRACTION_THRESHOLD
            ):
                flags.append("component_substantially_split_between_somata")
                ambiguous_cell_ids.update(
                    int(item_id) for item_id, _ in ordered_counts
                )
        else:
            second_count = 0
            second_fraction = 0.0

        minimum_distance = float(distance_to_soma[yy, xx].min())
        mean_distance = float(distance_to_soma[yy, xx].mean())
        maximum_distance = float(distance_to_soma[yy, xx].max())

        if maximum_distance > FAR_SKELETON_DISTANCE_PX:
            flags.append("far_from_reliable_soma")
            if MARK_FAR_CELL_AS_AMBIGUOUS and chosen_id > 0:
                ambiguous_cell_ids.add(chosen_id)

        touched_ids: list[int] = []
        if TOUCH_SOMA_RADIUS_PX > 0:
            y0, x0, y1, x1 = region.bbox
            margin = TOUCH_SOMA_RADIUS_PX

            ys = slice(max(0, y0 - margin), min(skeleton_mask.shape[0], y1 + margin))
            xs = slice(max(0, x0 - margin), min(skeleton_mask.shape[1], x1 + margin))

            local_component = skeleton_components[ys, xs] == component_id
            local_dilated = ndi.binary_dilation(
                local_component,
                structure=disk(TOUCH_SOMA_RADIUS_PX),
            )

            local_somata = reliable_soma_instances[ys, xs]
            touched_ids_array = np.unique(local_somata[local_dilated])
            touched_ids = [
                int(value)
                for value in touched_ids_array
                if int(value) > 0
            ]

            if len(touched_ids) > 1:
                flags.append("touches_multiple_somata")
                ambiguous_cell_ids.update(touched_ids)

        # Softer centroid-based ambiguity check.
        centroid = np.array(region.centroid, dtype=np.float32)

        if len(soma_centroids) >= 2:
            squared_distances = np.sum(
                (soma_centroids - centroid[None, :]) ** 2,
                axis=1,
            )
            order = np.argpartition(squared_distances, kth=1)[:2]
            order = order[np.argsort(squared_distances[order])]

            first_distance = float(math.sqrt(squared_distances[order[0]]))
            second_distance = float(math.sqrt(squared_distances[order[1]]))
            first_id = int(soma_ids[order[0]])
            second_id = int(soma_ids[order[1]])

            similar_ratio = (
                first_distance > 0
                and second_distance / first_distance
                <= AMBIGUITY_SECOND_NEAREST_RATIO
            )
            similar_margin = (
                second_distance - first_distance
                <= AMBIGUITY_SECOND_NEAREST_MARGIN_PX
            )

            if (
                USE_SOFT_CENTROID_AMBIGUITY
                and similar_ratio
                and similar_margin
            ):
                flags.append("two_similarly_close_somata")
                ambiguous_cell_ids.update([first_id, second_id])
        else:
            first_id = chosen_id
            second_id = 0
            first_distance = minimum_distance
            second_distance = np.nan

        report_rows.append({
            "skeleton_component_id": component_id,
            "pixels": int(region.area),
            "assigned_soma_id": chosen_id,
            "voronoi_soma_ids": ";".join(map(str, split_ids)),
            "voronoi_second_pixel_count": int(second_count),
            "voronoi_second_fraction": float(second_fraction),
            "touched_soma_ids": ";".join(map(str, touched_ids)),
            "nearest_centroid_soma_id": first_id,
            "second_centroid_soma_id": second_id,
            "nearest_centroid_distance": first_distance,
            "second_centroid_distance": second_distance,
            "min_pixel_distance_to_soma": minimum_distance,
            "mean_pixel_distance_to_soma": mean_distance,
            "max_pixel_distance_to_soma": maximum_distance,
            "flags": ";".join(flags),
        })

    return (
        skeleton_components,
        assigned_map,
        ambiguous_cell_ids,
        report_rows,
    )


def soma_constrained_gap_closing(
    skeleton_mask: np.ndarray,
    assigned_skeleton_map: np.ndarray,
    reliable_soma_instances: np.ndarray,
    nearest_soma_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    completed_skeleton = skeleton_mask.copy()
    closing_added = np.zeros_like(skeleton_mask, dtype=bool)
    completed_assignment = assigned_skeleton_map.copy()

    combined_ids = np.where(
        reliable_soma_instances > 0,
        reliable_soma_instances,
        assigned_skeleton_map,
    )

    object_slices = ndi.find_objects(combined_ids)
    structure = disk(GAP_CLOSE_RADIUS_PX)
    margin = GAP_CLOSE_RADIUS_PX + 3

    for cell_id, object_slice in enumerate(object_slices, start=1):
        if object_slice is None:
            continue

        y0 = max(0, object_slice[0].start - margin)
        y1 = min(skeleton_mask.shape[0], object_slice[0].stop + margin)
        x0 = max(0, object_slice[1].start - margin)
        x1 = min(skeleton_mask.shape[1], object_slice[1].stop + margin)

        local_skeleton = (
            assigned_skeleton_map[y0:y1, x0:x1] == cell_id
        )
        local_soma = (
            reliable_soma_instances[y0:y1, x0:x1] == cell_id
        )

        if not local_skeleton.any():
            continue

        local_domain = local_skeleton | local_soma

        if GAP_CLOSE_RADIUS_PX > 0:
            local_closed = ndi.binary_closing(
                local_domain,
                structure=structure,
            )
        else:
            local_closed = local_domain

        local_candidate = (
            local_closed
            & (~local_domain)
            & (nearest_soma_id[y0:y1, x0:x1] == cell_id)
            & (reliable_soma_instances[y0:y1, x0:x1] == 0)
        )

        closing_added[y0:y1, x0:x1] |= local_candidate
        completed_assignment[y0:y1, x0:x1][local_candidate] = cell_id

    completed_skeleton |= closing_added

    return completed_skeleton, closing_added, completed_assignment


def create_cell_instances(
    reliable_soma_instances: np.ndarray,
    completed_skeleton: np.ndarray,
    completed_assignment: np.ndarray,
    nearest_soma_id: np.ndarray,
) -> np.ndarray:
    maximum_id = int(reliable_soma_instances.max())

    cell_instances = np.zeros(
        reliable_soma_instances.shape,
        dtype=output_integer_dtype(maximum_id),
    )

    soma_mask = reliable_soma_instances > 0
    cell_instances[soma_mask] = reliable_soma_instances[soma_mask]

    assigned = completed_assignment.copy()

    missing_assignment = completed_skeleton & (assigned == 0)
    assigned[missing_assignment] = nearest_soma_id[missing_assignment]

    cell_instances[completed_skeleton] = assigned[completed_skeleton]

    return cell_instances


def make_instance_colors(maximum_id: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    colors = rng.integers(
        45,
        255,
        size=(maximum_id + 1, 3),
        dtype=np.uint8,
    )
    colors[0] = [0, 0, 0]
    return colors


def make_visible_overlay(
    raw_uint8: np.ndarray,
    hard_skeleton: np.ndarray,
    probability_added: np.ndarray,
    closing_added: np.ndarray,
    soma_mask: np.ndarray,
    cell_instances: np.ndarray,
    ambiguous_cell_ids: set[int],
    uncertain_soma_instances: np.ndarray,
) -> np.ndarray:
    overlay = np.stack(
        [raw_uint8, raw_uint8, raw_uint8],
        axis=-1,
    )

    # Original hard skeleton: red.
    overlay[hard_skeleton] = [255, 70, 50]

    # Probability-recovered pixels: green.
    overlay[probability_added] = [40, 255, 80]

    # Geometrically closed pixels: cyan.
    overlay[closing_added] = [0, 255, 255]

    # Reliable and uncertain soma: purple.
    overlay[soma_mask] = [190, 40, 255]
    overlay[uncertain_soma_instances > 0] = [255, 130, 0]

    ambiguous_mask = np.isin(
        cell_instances,
        np.array(sorted(ambiguous_cell_ids), dtype=cell_instances.dtype),
    ) if ambiguous_cell_ids else np.zeros(
        cell_instances.shape,
        dtype=bool,
    )

    if ambiguous_mask.any():
        outer = ndi.binary_dilation(
            ambiguous_mask,
            structure=disk(CELL_OUTLINE_RADIUS_PX),
        )
        inner = ndi.binary_erosion(
            ambiguous_mask,
            structure=disk(max(1, CELL_OUTLINE_RADIUS_PX // 3)),
        )
        outline = outer & ~inner
        overlay[outline] = [255, 230, 0]

    pil_image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_image)

    reliable_regions = {
        int(region.label): region
        for region in regionprops(
            cell_instances * (soma_mask.astype(cell_instances.dtype))
        )
    }

    for cell_id in sorted(ambiguous_cell_ids):
        region = reliable_regions.get(cell_id)
        if region is None:
            continue

        y0, x0, y1, x1 = region.bbox
        margin = SOMA_MARKER_MARGIN_PX

        box = (
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(cell_instances.shape[1] - 1, x1 + margin),
            min(cell_instances.shape[0] - 1, y1 + margin),
        )

        draw.ellipse(
            box,
            outline=(255, 230, 0),
            width=SOMA_MARKER_LINE_WIDTH_PX,
        )

        if SHOW_CELL_ID_TEXT:
            draw.text(
                (box[0], max(0, box[1] - 16)),
                f"? {cell_id}",
                fill=(255, 230, 0),
            )

    # Tiny/uncertain soma candidates get orange boxes.
    for region in regionprops(uncertain_soma_instances):
        y0, x0, y1, x1 = region.bbox
        margin = 8
        draw.rectangle(
            (
                max(0, x0 - margin),
                max(0, y0 - margin),
                min(cell_instances.shape[1] - 1, x1 + margin),
                min(cell_instances.shape[0] - 1, y1 + margin),
            ),
            outline=(255, 130, 0),
            width=4,
        )

    return np.asarray(pil_image)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading raw image:")
    print(RAW_PATH)
    raw = np.squeeze(tifffile.imread(RAW_PATH))

    print("Reading nnU-Net prediction:")
    print(PRED_PATH)
    prediction = np.squeeze(tifffile.imread(PRED_PATH)).astype(np.uint8)

    if raw.shape != prediction.shape:
        raise RuntimeError(
            f"Raw/prediction shape mismatch: "
            f"{raw.shape} versus {prediction.shape}"
        )

    hard_skeleton = prediction == 1
    soma_mask = prediction == 2
    raw_uint8 = normalize_to_uint8(raw)

    probabilities = None
    if USE_PROBABILITIES:
        if PROBABILITY_PATH.exists():
            print("Reading class probabilities:")
            print(PROBABILITY_PATH)
            probabilities = load_probabilities(
                PROBABILITY_PATH,
                prediction.shape,
            )
        else:
            print(
                "Probability file not found. The script will only "
                "perform conservative geometric gap closing:"
            )
            print(PROBABILITY_PATH)

    (
        all_soma_instances,
        reliable_soma_instances,
        uncertain_soma_mask,
        _,
    ) = relabel_reliable_somata(
        soma_mask,
        MIN_RELIABLE_SOMA_AREA_PX,
    )

    uncertain_soma_instances = cc_label(
        uncertain_soma_mask,
        connectivity=2,
    )

    print()
    print(f"All soma components: {int(all_soma_instances.max())}")
    print(
        "Reliable soma seeds: "
        f"{int(reliable_soma_instances.max())}"
    )
    print(
        "Uncertain small soma candidates: "
        f"{int(uncertain_soma_instances.max())}"
    )

    recovered_skeleton, probability_added = (
        recover_skeleton_from_probabilities(
            hard_skeleton,
            soma_mask,
            probabilities,
        )
    )

    distance_to_soma, nearest_soma_id = nearest_seed_maps(
        reliable_soma_instances
    )

    (
        skeleton_components,
        assigned_skeleton_map,
        ambiguous_cell_ids,
        component_report,
    ) = assign_original_skeleton_components(
        recovered_skeleton,
        reliable_soma_instances,
        nearest_soma_id,
        distance_to_soma,
    )

    (
        completed_skeleton,
        closing_added,
        completed_assignment,
    ) = soma_constrained_gap_closing(
        recovered_skeleton,
        assigned_skeleton_map,
        reliable_soma_instances,
        nearest_soma_id,
    )

    cell_instances = create_cell_instances(
        reliable_soma_instances,
        completed_skeleton,
        completed_assignment,
        nearest_soma_id,
    )

    ambiguous_cell_mask = np.isin(
        cell_instances,
        np.array(
            sorted(ambiguous_cell_ids),
            dtype=cell_instances.dtype,
        ),
    ) if ambiguous_cell_ids else np.zeros(
        prediction.shape,
        dtype=bool,
    )

    ambiguous_skeleton_mask = (
        completed_skeleton & ambiguous_cell_mask
    )

    overlay = make_visible_overlay(
        raw_uint8=raw_uint8,
        hard_skeleton=hard_skeleton,
        probability_added=probability_added,
        closing_added=closing_added,
        soma_mask=soma_mask,
        cell_instances=cell_instances,
        ambiguous_cell_ids=ambiguous_cell_ids,
        uncertain_soma_instances=uncertain_soma_instances,
    )

    maximum_cell_id = int(cell_instances.max())
    instance_colors = make_instance_colors(maximum_cell_id)
    instance_preview = instance_colors[cell_instances]

    # Non-destructive outputs.
    tifffile.imwrite(
        OUTPUT_DIR / "original_prediction.tif",
        prediction,
    )
    tifffile.imwrite(
        OUTPUT_DIR / "original_skeleton.tif",
        hard_skeleton.astype(np.uint8),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "probability_recovered_pixels.tif",
        probability_added.astype(np.uint8),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "gap_closing_added_pixels.tif",
        closing_added.astype(np.uint8),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "completed_skeleton.tif",
        completed_skeleton.astype(np.uint8),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "all_soma_instances.tif",
        all_soma_instances.astype(
            output_integer_dtype(int(all_soma_instances.max()))
        ),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "reliable_soma_instances.tif",
        reliable_soma_instances.astype(
            output_integer_dtype(
                int(reliable_soma_instances.max())
            )
        ),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "uncertain_small_soma.tif",
        uncertain_soma_mask.astype(np.uint8),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "cell_instances.tif",
        cell_instances,
    )
    tifffile.imwrite(
        OUTPUT_DIR / "ambiguous_cells_mask.tif",
        ambiguous_cell_mask.astype(np.uint8),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "ambiguous_skeleton_mask.tif",
        ambiguous_skeleton_mask.astype(np.uint8),
    )

    save_preview(
        overlay,
        OUTPUT_DIR / "overview_visible_qc_overlay.png",
        MAX_PREVIEW_SIZE,
    )
    save_preview(
        instance_preview,
        OUTPUT_DIR / "overview_instances.png",
        MAX_PREVIEW_SIZE,
    )

    write_csv(
        OUTPUT_DIR / "skeleton_component_report.csv",
        component_report,
    )

    with (
        OUTPUT_DIR / "ambiguous_cell_ids.txt"
    ).open("w", encoding="utf-8") as handle:
        for cell_id in sorted(ambiguous_cell_ids):
            handle.write(f"{cell_id}\n")

    print()
    print("Finished.")
    print(f"Original skeleton pixels: {int(hard_skeleton.sum())}")
    print(
        "Probability-recovered pixels: "
        f"{int(probability_added.sum())}"
    )
    print(
        "Geometric gap-closing pixels: "
        f"{int(closing_added.sum())}"
    )
    print(
        "Completed skeleton pixels: "
        f"{int(completed_skeleton.sum())}"
    )
    print(f"Ambiguous cell IDs: {len(ambiguous_cell_ids)}")
    print()
    print("Open this first:")
    print(OUTPUT_DIR / "overview_visible_qc_overlay.png")
    print()
    print("Legend:")
    print("  red    = original hard skeleton")
    print("  green  = recovered from skeleton probability")
    print("  cyan   = conservative geometric gap closing")
    print("  purple = soma")
    print("  yellow = ambiguous cell outline / marker")
    print("  orange = tiny uncertain soma candidate")


if __name__ == "__main__":
    main()
