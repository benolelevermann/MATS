from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


SCRIPT_VERSION = "safe-cell-export-v3-2026-07-28"
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)

STATUS_BACKGROUND = 0
STATUS_SAFE = 1
STATUS_AMBIGUOUS = 2
STATUS_UNASSIGNED_SKELETON = 3
STATUS_INSUFFICIENT_SKELETON = 4
STATUS_BORDER_REJECTED = 5
STATUS_SMALL_SOMA = 6


# -----------------------------------------------------------------------------
# Kommandozeile und Ein-/Ausgabe
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Findet eindeutig zuordenbare Soma-Skeleton-Zellen in einer "
            "0/1/2-Segmentierung und exportiert jede sichere Zelle in einen "
            "eigenen pipeline-kompatiblen Ordner."
        )
    )

    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Originales 2D-Übersichtsbild bzw. Max-Projektion.",
    )

    parser.add_argument(
        "--semantic",
        type=Path,
        required=True,
        help="2D-TIFF mit 0=Hintergrund, 1=Skeleton und 2=Soma.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--contact-radius",
        type=int,
        default=4,
        help=(
            "Maximaler Pixelabstand, in dem Skeleton und Soma als verbunden "
            "gelten (Standard: 4)."
        ),
    )

    parser.add_argument(
        "--min-soma-area",
        type=int,
        default=20,
        help="Minimale Soma-Fläche in Pixeln (Standard: 20).",
    )

    parser.add_argument(
        "--min-skeleton-pixels",
        type=int,
        default=8,
        help=(
            "Minimale Zahl eindeutig zugeordneter Skeletonpixel für einen "
            "sicheren Export (Standard: 8)."
        ),
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=48,
        help=(
            "Zusätzlicher Kontext-Rand um die vollständige Zelle "
            "in Pixeln (Standard: 48)."
        ),
    )

    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=128,
        help="Minimale Höhe und Breite eines Crops (Standard: 128).",
    )

    parser.add_argument(
        "--edge-clearance",
        type=int,
        default=2,
        help=(
            "Mindestabstand zwischen Zellmaske und Crop-Rand. "
            "Der Export wird abgelehnt, falls dieser Abstand nicht "
            "eingehalten werden kann (Standard: 2)."
        ),
    )

    parser.add_argument(
        "--square",
        action="store_true",
        help="Quadratische Crops exportieren.",
    )

    parser.add_argument(
        "--allow-soma-only",
        action="store_true",
        help=(
            "Auch Soma ohne genügend Skeleton exportieren. "
            "Für Morphologie/PCA normalerweise nicht sinnvoll."
        ),
    )

    parser.add_argument(
        "--allow-border-touching",
        action="store_true",
        help=(
            "Auch Zellen exportieren, deren Segmentierung den Rand des "
            "Übersichtsbildes berührt. Standardmäßig werden sie ausgelassen, "
            "weil sie bereits im Quellbild abgeschnitten sein können."
        ),
    )

    parser.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="Nummer des ersten Zellordners (Standard: 1).",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Anzahl führender Nullen, z. B. cell0001 (Standard: 4).",
    )

    parser.add_argument(
        "--qc-max-side",
        type=int,
        default=3000,
        help="Maximale Kantenlänge des QC-PNGs (Standard: 3000).",
    )

    parser.add_argument(
        "--save-diagnostic-masks",
        action="store_true",
        help=(
            "Zusätzliche Vollauflösungs-Masken für ambige und "
            "unzugeordnete Bereiche speichern."
        ),
    )

    parser.add_argument(
        "--save-overview-instances",
        action="store_true",
        help=(
            "Eine uint32-Instanzkarte aller exportierten Zellen speichern. "
            "Bei einem 220-Megapixel-Bild ist diese ungefähr 880 MB groß."
        ),
    )

    return parser.parse_args()


def read_2d(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} nicht gefunden: {path}")

    try:
        array = np.asarray(tifffile.memmap(path))
        mode = "memory-mapped"
    except Exception:
        array = np.asarray(tifffile.imread(path))
        mode = "vollständig eingelesen"

    array = np.squeeze(array)

    if array.ndim != 2:
        raise RuntimeError(
            f"{name} muss 2D sein, gefunden {array.shape}: {path}"
        )

    print(f"{name}: {array.shape}, {array.dtype}, {mode}")

    return array


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Ausgabeordner ist nicht leer:\n{path}\n"
            "Bitte einen neuen Ordner verwenden, damit keine Zellordner "
            "überschrieben werden."
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_tiff(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        array,
        photometric="minisblack",
        bigtiff=bool(array.nbytes >= 4_000_000_000),
    )


# -----------------------------------------------------------------------------
# Topologische Klassifikation
# -----------------------------------------------------------------------------


def remove_small_soma_seeds(
    soma: np.ndarray,
    minimum_area: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, list[int]]:
    raw_labels, raw_count = ndi.label(
        soma,
        structure=CONNECTIVITY_8,
    )

    areas = np.bincount(
        raw_labels.ravel(),
        minlength=raw_count + 1,
    )

    valid_old_ids = np.asarray(
        [
            component_id
            for component_id in range(1, raw_count + 1)
            if int(areas[component_id]) >= minimum_area
        ],
        dtype=np.int32,
    )

    valid_lookup = np.zeros(
        raw_count + 1,
        dtype=bool,
    )

    if valid_old_ids.size:
        valid_lookup[valid_old_ids] = True

    seed_mask = valid_lookup[raw_labels]
    rejected_mask = soma & ~seed_mask

    labels, count = ndi.label(
        seed_mask,
        structure=CONNECTIVITY_8,
    )

    rejected_ids = [
        component_id
        for component_id in range(1, raw_count + 1)
        if not valid_lookup[component_id]
    ]

    return (
        seed_mask,
        labels.astype(np.uint32, copy=False),
        count,
        rejected_mask,
        rejected_ids,
    )


def map_skeleton_components_to_somata(
    skeleton_labels: np.ndarray,
    skeleton_count: int,
    soma_labels: np.ndarray,
    soma_count: int,
    contact_radius: int,
) -> dict[int, set[int]]:
    """
    Ordnet jeder Skeletonkomponente alle Soma zu, die innerhalb
    des Kontakt-Radius liegen.

    Jedes Soma wird einzeln und lokal erweitert. Dadurch werden
    überlappende Soma-Kontaktbereiche korrekt als mehrdeutig erkannt.
    """
    component_to_somata: dict[int, set[int]] = {
        component_id: set()
        for component_id in range(1, skeleton_count + 1)
    }

    height, width = soma_labels.shape
    soma_objects = ndi.find_objects(soma_labels)

    for soma_id, object_slice in enumerate(
        soma_objects,
        start=1,
    ):
        if soma_id > soma_count or object_slice is None:
            continue

        y_slice, x_slice = object_slice

        y0 = max(
            0,
            y_slice.start - contact_radius,
        )
        y1 = min(
            height,
            y_slice.stop + contact_radius,
        )

        x0 = max(
            0,
            x_slice.start - contact_radius,
        )
        x1 = min(
            width,
            x_slice.stop + contact_radius,
        )

        local_soma = (
            soma_labels[y0:y1, x0:x1]
            == soma_id
        )

        if contact_radius > 0:
            local_soma = ndi.binary_dilation(
                local_soma,
                structure=CONNECTIVITY_8,
                iterations=contact_radius,
            )

        local_skeleton_labels = skeleton_labels[
            y0:y1,
            x0:x1,
        ]

        touched_components = np.unique(
            local_skeleton_labels[local_soma]
        )

        for component_id in touched_components:
            component_id = int(component_id)

            if component_id > 0:
                component_to_somata[
                    component_id
                ].add(soma_id)

    return component_to_somata


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

    skeleton_labels = skeleton_labels.astype(
        np.uint32,
        copy=False,
    )

    skeleton_sizes = np.bincount(
        skeleton_labels.ravel(),
        minlength=skeleton_count + 1,
    )

    component_to_somata = map_skeleton_components_to_somata(
        skeleton_labels=skeleton_labels,
        skeleton_count=skeleton_count,
        soma_labels=soma_labels,
        soma_count=soma_count,
        contact_radius=contact_radius,
    )

    uniquely_assigned: dict[int, list[int]] = {
        soma_id: []
        for soma_id in range(1, soma_count + 1)
    }

    ambiguous_components: list[int] = []
    unassigned_components: list[int] = []

    soma_has_ambiguous_contact = {
        soma_id: False
        for soma_id in range(1, soma_count + 1)
    }

    for component_id in range(1, skeleton_count + 1):
        touched = sorted(
            component_to_somata[component_id]
        )

        if len(touched) == 1:
            uniquely_assigned[
                touched[0]
            ].append(component_id)

        elif len(touched) > 1:
            ambiguous_components.append(
                component_id
            )

            for soma_id in touched:
                soma_has_ambiguous_contact[
                    soma_id
                ] = True

        else:
            unassigned_components.append(
                component_id
            )

    candidate_safe_soma_ids: list[int] = []
    ambiguous_soma_ids: list[int] = []
    insufficient_soma_ids: list[int] = []
    soma_skeleton_pixels: dict[int, int] = {}

    for soma_id in range(1, soma_count + 1):
        component_ids = uniquely_assigned[
            soma_id
        ]

        pixel_count = int(
            sum(
                int(skeleton_sizes[component_id])
                for component_id in component_ids
            )
        )

        soma_skeleton_pixels[
            soma_id
        ] = pixel_count

        if soma_has_ambiguous_contact[soma_id]:
            ambiguous_soma_ids.append(
                soma_id
            )

        elif pixel_count >= min_skeleton_pixels:
            candidate_safe_soma_ids.append(
                soma_id
            )

        elif allow_soma_only:
            candidate_safe_soma_ids.append(
                soma_id
            )
            insufficient_soma_ids.append(
                soma_id
            )

        else:
            insufficient_soma_ids.append(
                soma_id
            )

    return {
        "skeleton_labels": skeleton_labels,
        "skeleton_count": skeleton_count,
        "skeleton_sizes": skeleton_sizes,
        "component_to_somata": component_to_somata,
        "uniquely_assigned": uniquely_assigned,
        "ambiguous_components": ambiguous_components,
        "unassigned_components": unassigned_components,
        "candidate_safe_soma_ids": candidate_safe_soma_ids,
        "ambiguous_soma_ids": ambiguous_soma_ids,
        "insufficient_soma_ids": insufficient_soma_ids,
        "soma_skeleton_pixels": soma_skeleton_pixels,
    }


# -----------------------------------------------------------------------------
# Vollständige Crops
# -----------------------------------------------------------------------------


def union_slices(
    slices: list[tuple[slice, slice]],
) -> tuple[int, int, int, int]:
    valid = [
        item
        for item in slices
        if item is not None
    ]

    if not valid:
        raise RuntimeError(
            "Keine gültige Bounding Box vorhanden."
        )

    y0 = min(
        item[0].start
        for item in valid
    )

    y1 = max(
        item[0].stop
        for item in valid
    )

    x0 = min(
        item[1].start
        for item in valid
    )

    x1 = max(
        item[1].stop
        for item in valid
    )

    return (
        int(y0),
        int(y1),
        int(x0),
        int(x1),
    )


def axis_bounds(
    minimum: int,
    maximum_inclusive: int,
    limit: int,
    margin: int,
    minimum_size: int,
) -> tuple[int, int]:
    desired = max(
        maximum_inclusive
        - minimum
        + 1
        + 2 * margin,
        minimum_size,
    )

    desired = min(
        desired,
        limit,
    )

    center = 0.5 * (
        minimum
        + maximum_inclusive
    )

    start = int(
        math.floor(
            center
            - desired / 2
        )
    )

    start = max(
        0,
        min(
            start,
            limit - desired,
        ),
    )

    return (
        start,
        start + desired,
    )


def crop_bounds_from_bbox(
    core_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin: int,
    minimum_size: int,
    square: bool,
) -> tuple[int, int, int, int]:
    core_y0, core_y1, core_x0, core_x1 = core_bbox

    y_min = core_y0
    y_max = core_y1 - 1

    x_min = core_x0
    x_max = core_x1 - 1

    height, width = image_shape

    if square:
        requested = max(
            max(
                y_max - y_min + 1,
                x_max - x_min + 1,
            )
            + 2 * margin,
            minimum_size,
        )

        requested = min(
            requested,
            height,
            width,
        )

        y0, y1 = axis_bounds(
            y_min,
            y_max,
            height,
            margin,
            requested,
        )

        x0, x1 = axis_bounds(
            x_min,
            x_max,
            width,
            margin,
            requested,
        )

    else:
        y0, y1 = axis_bounds(
            y_min,
            y_max,
            height,
            margin,
            minimum_size,
        )

        x0, x1 = axis_bounds(
            x_min,
            x_max,
            width,
            margin,
            minimum_size,
        )

    return (
        y0,
        y1,
        x0,
        x1,
    )


def mask_touches_source_border(
    local_mask: np.ndarray,
    global_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> bool:
    y0, y1, x0, x1 = global_bbox
    height, width = image_shape

    return bool(
        (
            y0 == 0
            and np.any(local_mask[0, :])
        )
        or (
            y1 == height
            and np.any(local_mask[-1, :])
        )
        or (
            x0 == 0
            and np.any(local_mask[:, 0])
        )
        or (
            x1 == width
            and np.any(local_mask[:, -1])
        )
    )


def mask_has_edge_clearance(
    mask: np.ndarray,
    clearance: int,
) -> bool:
    if clearance <= 0:
        return True

    if (
        mask.shape[0] <= 2 * clearance
        or mask.shape[1] <= 2 * clearance
    ):
        return False

    return not bool(
        np.any(mask[:clearance, :])
        or np.any(mask[-clearance:, :])
        or np.any(mask[:, :clearance])
        or np.any(mask[:, -clearance:])
    )


def prepare_cell_candidate(
    soma_id: int,
    component_ids: list[int],
    soma_labels: np.ndarray,
    skeleton_labels: np.ndarray,
    soma_objects: list[
        Optional[tuple[slice, slice]]
    ],
    skeleton_objects: list[
        Optional[tuple[slice, slice]]
    ],
    image_shape: tuple[int, int],
) -> dict[str, object]:
    soma_slice = soma_objects[
        soma_id - 1
    ]

    if soma_slice is None:
        raise RuntimeError(
            f"Soma {soma_id} besitzt keine Bounding Box."
        )

    component_slices: list[
        tuple[slice, slice]
    ] = []

    for component_id in component_ids:
        component_slice = skeleton_objects[
            component_id - 1
        ]

        if component_slice is not None:
            component_slices.append(
                component_slice
            )

    core_bbox = union_slices(
        [
            soma_slice,
            *component_slices,
        ]
    )

    y0, y1, x0, x1 = core_bbox

    local_soma = (
        soma_labels[y0:y1, x0:x1]
        == soma_id
    )

    if component_ids:
        local_skeleton = np.isin(
            skeleton_labels[
                y0:y1,
                x0:x1,
            ],
            np.asarray(
                component_ids,
                dtype=np.uint32,
            ),
        )
    else:
        local_skeleton = np.zeros(
            local_soma.shape,
            dtype=bool,
        )

    local_cell = (
        local_soma
        | local_skeleton
    )

    touches_border = mask_touches_source_border(
        local_mask=local_cell,
        global_bbox=core_bbox,
        image_shape=image_shape,
    )

    return {
        "soma_id": soma_id,
        "component_ids": component_ids,
        "core_bbox": core_bbox,
        "touches_source_border": touches_border,
        "expected_soma_pixels": int(
            local_soma.sum()
        ),
        "expected_skeleton_pixels": int(
            local_skeleton.sum()
        ),
    }


# -----------------------------------------------------------------------------
# Pipeline-Dateien pro Zelle
# -----------------------------------------------------------------------------


def write_pixel_csv(
    path: Path,
    skeleton: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> None:
    coordinates = np.argwhere(
        skeleton
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "x",
                "y",
                "global_x",
                "global_y",
                "class",
            ]
        )

        for y, x in coordinates:
            writer.writerow(
                [
                    int(x),
                    int(y),
                    int(x + x_offset),
                    int(y + y_offset),
                    1,
                ]
            )


def write_swc(
    path: Path,
    skeleton: np.ndarray,
    soma: np.ndarray,
) -> None:
    """
    Lokalen SWC-Spanning-Forest schreiben,
    am Soma verwurzelt.
    """
    thin = skeletonize(
        skeleton
    )

    soma_coordinates = np.argwhere(
        soma
    )

    if soma_coordinates.size:
        soma_y, soma_x = (
            soma_coordinates.mean(axis=0)
        )

        soma_radius = max(
            1.0,
            math.sqrt(
                float(
                    soma_coordinates.shape[0]
                )
                / math.pi
            ),
        )

    else:
        skeleton_coordinates = np.argwhere(
            thin
        )

        if skeleton_coordinates.size:
            soma_y, soma_x = (
                skeleton_coordinates.mean(axis=0)
            )

        else:
            soma_y = soma_x = 0.0

        soma_radius = 1.0

    labels, count = ndi.label(
        thin,
        structure=CONNECTIVITY_8,
    )

    rows: list[
        tuple[
            int,
            int,
            float,
            float,
            float,
            float,
            int,
        ]
    ] = [
        (
            1,
            1,
            float(soma_x),
            float(soma_y),
            0.0,
            soma_radius,
            -1,
        )
    ]

    next_node_id = 2
    height, width = thin.shape

    for component_id in range(
        1,
        count + 1,
    ):
        coordinates = np.argwhere(
            labels == component_id
        )

        if coordinates.size == 0:
            continue

        distances = (
            (
                coordinates[:, 0]
                - soma_y
            )
            ** 2
            + (
                coordinates[:, 1]
                - soma_x
            )
            ** 2
        )

        root = tuple(
            map(
                int,
                coordinates[
                    int(
                        np.argmin(
                            distances
                        )
                    )
                ],
            )
        )

        queue: deque[
            tuple[
                tuple[int, int],
                int,
            ]
        ] = deque(
            [
                (
                    root,
                    1,
                )
            ]
        )

        visited: set[
            tuple[int, int]
        ] = {
            root
        }

        while queue:
            (
                y,
                x,
            ), parent_id = queue.popleft()

            node_id = next_node_id
            next_node_id += 1

            rows.append(
                (
                    node_id,
                    3,
                    float(x),
                    float(y),
                    0.0,
                    1.0,
                    parent_id,
                )
            )

            neighbors: list[
                tuple[int, int]
            ] = []

            for dy in (
                -1,
                0,
                1,
            ):
                for dx in (
                    -1,
                    0,
                    1,
                ):
                    if (
                        dy == 0
                        and dx == 0
                    ):
                        continue

                    ny = y + dy
                    nx = x + dx
                    point = (
                        ny,
                        nx,
                    )

                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and labels[
                            ny,
                            nx,
                        ]
                        == component_id
                        and point
                        not in visited
                    ):
                        neighbors.append(
                            point
                        )

            neighbors.sort()

            for point in neighbors:
                visited.add(
                    point
                )

                queue.append(
                    (
                        point,
                        node_id,
                    )
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            f"# Generated by {SCRIPT_VERSION}\n"
        )

        handle.write(
            "# Local crop coordinates; "
            "id type x y z radius parent\n"
        )

        for row in rows:
            handle.write(
                f"{row[0]} "
                f"{row[1]} "
                f"{row[2]:.3f} "
                f"{row[3]:.3f} "
                f"{row[4]:.3f} "
                f"{row[5]:.3f} "
                f"{row[6]}\n"
            )


# -----------------------------------------------------------------------------
# Diagnostik
# -----------------------------------------------------------------------------


def normalize_grayscale(
    array: np.ndarray,
) -> np.ndarray:
    values = array.astype(
        np.float32,
        copy=False,
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    low, high = np.percentile(
        finite,
        [
            1.0,
            99.5,
        ],
    )

    if high <= low:
        high = low + 1.0

    scaled = np.clip(
        (
            values
            - low
        )
        / (
            high
            - low
        ),
        0.0,
        1.0,
    )

    return np.round(
        scaled
        * 255.0
    ).astype(
        np.uint8
    )


def instance_color(
    instance_id: int,
) -> np.ndarray:
    hue = (
        instance_id
        * 0.618033988749895
    ) % 1.0

    return np.asarray(
        colorsys.hsv_to_rgb(
            hue,
            0.85,
            1.0,
        ),
        dtype=np.float32,
    )


def build_status_map(
    semantic: np.ndarray,
    skeleton_labels: np.ndarray,
    soma_labels: np.ndarray,
    component_status_lut: np.ndarray,
    soma_status_lut: np.ndarray,
    rejected_small_soma: np.ndarray,
    row_chunk: int = 1024,
) -> np.ndarray:
    status = np.zeros(
        semantic.shape,
        dtype=np.uint8,
    )

    for y0 in range(
        0,
        semantic.shape[0],
        row_chunk,
    ):
        y1 = min(
            semantic.shape[0],
            y0 + row_chunk,
        )

        local_skeleton_labels = (
            skeleton_labels[y0:y1]
        )

        local_soma_labels = (
            soma_labels[y0:y1]
        )

        local_status = (
            component_status_lut[
                local_skeleton_labels
            ]
        )

        soma_status = (
            soma_status_lut[
                local_soma_labels
            ]
        )

        soma_mask = (
            local_soma_labels > 0
        )

        local_status[
            soma_mask
        ] = soma_status[
            soma_mask
        ]

        local_status[
            rejected_small_soma[
                y0:y1
            ]
        ] = STATUS_SMALL_SOMA

        status[
            y0:y1
        ] = local_status

    return status


def save_qc_overlay(
    path: Path,
    original: np.ndarray,
    skeleton_labels: np.ndarray,
    soma_labels: np.ndarray,
    component_status_lut: np.ndarray,
    soma_status_lut: np.ndarray,
    component_instance_lut: np.ndarray,
    soma_instance_lut: np.ndarray,
    rejected_small_soma: np.ndarray,
    max_side: int,
) -> None:
    step = max(
        1,
        math.ceil(
            max(
                original.shape
            )
            / max_side
        ),
    )

    gray = normalize_grayscale(
        original[
            ::step,
            ::step,
        ]
    ).astype(
        np.float32
    ) / 255.0

    rgb = np.repeat(
        gray[..., None],
        3,
        axis=2,
    )

    skeleton_small = skeleton_labels[
        ::step,
        ::step,
    ]

    soma_small = soma_labels[
        ::step,
        ::step,
    ]

    rejected_small = rejected_small_soma[
        ::step,
        ::step,
    ]

    component_instances = (
        component_instance_lut[
            skeleton_small
        ]
    )

    soma_instances = (
        soma_instance_lut[
            soma_small
        ]
    )

    safe_instances = np.maximum(
        component_instances,
        soma_instances,
    )

    for instance_id in np.unique(
        safe_instances
    ):
        instance_id = int(
            instance_id
        )

        if instance_id <= 0:
            continue

        mask = (
            safe_instances
            == instance_id
        )

        color = instance_color(
            instance_id
        )

        rgb[mask] = (
            0.2
            * rgb[mask]
            + 0.8
            * color
        )

    component_status = (
        component_status_lut[
            skeleton_small
        ]
    )

    soma_status = (
        soma_status_lut[
            soma_small
        ]
    )

    combined_status = (
        component_status.copy()
    )

    combined_status[
        soma_small > 0
    ] = soma_status[
        soma_small > 0
    ]

    combined_status[
        rejected_small
    ] = STATUS_SMALL_SOMA

    color_map = {
        STATUS_AMBIGUOUS: np.asarray(
            [
                1.0,
                0.45,
                0.0,
            ]
        ),
        STATUS_UNASSIGNED_SKELETON: np.asarray(
            [
                0.0,
                0.9,
                1.0,
            ]
        ),
        STATUS_INSUFFICIENT_SKELETON: np.asarray(
            [
                1.0,
                1.0,
                0.0,
            ]
        ),
        STATUS_BORDER_REJECTED: np.asarray(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
        STATUS_SMALL_SOMA: np.asarray(
            [
                1.0,
                0.0,
                1.0,
            ]
        ),
    }

    for (
        status_value,
        color,
    ) in color_map.items():
        mask = (
            combined_status
            == status_value
        )

        if np.any(mask):
            rgb[mask] = (
                0.15
                * rgb[mask]
                + 0.85
                * color
            )

    image = np.round(
        np.clip(
            rgb,
            0,
            1,
        )
        * 255
    ).astype(
        np.uint8
    )

    Image.fromarray(
        image
    ).save(
        path
    )


def save_optional_overview_instances(
    path: Path,
    skeleton_labels: np.ndarray,
    soma_labels: np.ndarray,
    component_instance_lut: np.ndarray,
    soma_instance_lut: np.ndarray,
    row_chunk: int = 512,
) -> None:
    temporary = path.with_suffix(
        ".tmp.dat"
    )

    output = np.memmap(
        temporary,
        dtype=np.uint32,
        mode="w+",
        shape=skeleton_labels.shape,
    )

    for y0 in range(
        0,
        skeleton_labels.shape[0],
        row_chunk,
    ):
        y1 = min(
            skeleton_labels.shape[0],
            y0 + row_chunk,
        )

        skeleton_instances = (
            component_instance_lut[
                skeleton_labels[
                    y0:y1
                ]
            ]
        )

        soma_instances = (
            soma_instance_lut[
                soma_labels[
                    y0:y1
                ]
            ]
        )

        output[
            y0:y1
        ] = np.maximum(
            skeleton_instances,
            soma_instances,
        )

    output.flush()

    tifffile.imwrite(
        path,
        output,
        photometric="minisblack",
        bigtiff=True,
    )

    del output

    temporary.unlink(
        missing_ok=True
    )


# -----------------------------------------------------------------------------
# Hauptprogramm
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.contact_radius < 0:
        raise ValueError(
            "--contact-radius muss >= 0 sein"
        )

    if args.min_soma_area < 1:
        raise ValueError(
            "--min-soma-area muss >= 1 sein"
        )

    if args.min_skeleton_pixels < 0:
        raise ValueError(
            "--min-skeleton-pixels muss >= 0 sein"
        )

    if (
        args.margin < 0
        or args.min_crop_size < 1
    ):
        raise ValueError(
            "--margin muss >= 0 und "
            "--min-crop-size >= 1 sein"
        )

    if args.edge_clearance < 0:
        raise ValueError(
            "--edge-clearance muss >= 0 sein"
        )

    if (
        args.start_number < 0
        or args.digits < 1
    ):
        raise ValueError(
            "--start-number muss >= 0 "
            "und --digits >= 1 sein"
        )

    if args.qc_max_side < 256:
        raise ValueError(
            "--qc-max-side muss >= 256 sein"
        )

    print(
        f"SAFE CELL EXPORTER: "
        f"{SCRIPT_VERSION}"
    )

    print(
        f"AUSGEFÜHRTE DATEI: "
        f"{Path(__file__).resolve()}"
    )

    original = read_2d(
        args.original.resolve(),
        "Originalbild",
    )

    semantic = read_2d(
        args.semantic.resolve(),
        "Semantische Maske",
    )

    if original.shape != semantic.shape:
        raise RuntimeError(
            "Shape stimmt nicht überein: "
            f"original={original.shape}, "
            f"semantic={semantic.shape}"
        )

    semantic_values = set(
        np.unique(
            semantic
        ).astype(
            int
        ).tolist()
    )

    invalid = (
        semantic_values
        - {
            0,
            1,
            2,
        }
    )

    if invalid:
        raise RuntimeError(
            "Unerwartete Werte in der "
            f"Segmentierung: {sorted(invalid)}"
        )

    ensure_empty_output(
        args.output_dir
    )

    diagnostics = (
        args.output_dir
        / "_classification"
    )

    diagnostics.mkdir()

    skeleton = (
        semantic == 1
    )

    soma = (
        semantic == 2
    )

    (
        soma_seed_mask,
        soma_labels,
        soma_count,
        rejected_small_soma,
        rejected_soma_ids,
    ) = remove_small_soma_seeds(
        soma,
        args.min_soma_area,
    )

    classification = classify_topology(
        skeleton=skeleton,
        soma_labels=soma_labels,
        soma_count=soma_count,
        contact_radius=args.contact_radius,
        min_skeleton_pixels=(
            args.min_skeleton_pixels
        ),
        allow_soma_only=(
            args.allow_soma_only
        ),
    )

    skeleton_labels = classification[
        "skeleton_labels"
    ]

    skeleton_count = int(
        classification[
            "skeleton_count"
        ]
    )

    uniquely_assigned = classification[
        "uniquely_assigned"
    ]

    candidate_safe_soma_ids = classification[
        "candidate_safe_soma_ids"
    ]

    ambiguous_soma_ids = set(
        classification[
            "ambiguous_soma_ids"
        ]
    )

    insufficient_soma_ids = set(
        classification[
            "insufficient_soma_ids"
        ]
    )

    ambiguous_components = set(
        classification[
            "ambiguous_components"
        ]
    )

    unassigned_components = set(
        classification[
            "unassigned_components"
        ]
    )

    soma_objects = ndi.find_objects(
        soma_labels
    )

    skeleton_objects = ndi.find_objects(
        skeleton_labels
    )

    accepted_candidates: list[
        dict[str, object]
    ] = []

    rejected_border_soma_ids: set[
        int
    ] = set()

    candidate_rows: list[
        dict[str, object]
    ] = []

    for soma_id in sorted(
        candidate_safe_soma_ids
    ):
        component_ids = list(
            uniquely_assigned[
                soma_id
            ]
        )

        candidate = prepare_cell_candidate(
            soma_id=soma_id,
            component_ids=component_ids,
            soma_labels=soma_labels,
            skeleton_labels=skeleton_labels,
            soma_objects=soma_objects,
            skeleton_objects=skeleton_objects,
            image_shape=semantic.shape,
        )

        if (
            candidate[
                "touches_source_border"
            ]
            and not args.allow_border_touching
        ):
            rejected_border_soma_ids.add(
                soma_id
            )

            candidate_rows.append(
                {
                    "source_soma_id": soma_id,
                    "status": "border_rejected",
                    "reason": (
                        "cell mask touches "
                        "source overview border"
                    ),
                    "skeleton_pixels": candidate[
                        "expected_skeleton_pixels"
                    ],
                    "soma_pixels": candidate[
                        "expected_soma_pixels"
                    ],
                    "component_ids": ",".join(
                        map(
                            str,
                            component_ids,
                        )
                    ),
                }
            )

            continue

        accepted_candidates.append(
            candidate
        )

        candidate_rows.append(
            {
                "source_soma_id": soma_id,
                "status": "safe",
                "reason": (
                    "unique soma-skeleton "
                    "assignment"
                ),
                "skeleton_pixels": candidate[
                    "expected_skeleton_pixels"
                ],
                "soma_pixels": candidate[
                    "expected_soma_pixels"
                ],
                "component_ids": ",".join(
                    map(
                        str,
                        component_ids,
                    )
                ),
            }
        )

    component_status_lut = np.zeros(
        skeleton_count + 1,
        dtype=np.uint8,
    )

    soma_status_lut = np.zeros(
        soma_count + 1,
        dtype=np.uint8,
    )

    if ambiguous_components:
        component_status_lut[
            np.asarray(
                sorted(
                    ambiguous_components
                )
            )
        ] = STATUS_AMBIGUOUS

    if unassigned_components:
        component_status_lut[
            np.asarray(
                sorted(
                    unassigned_components
                )
            )
        ] = STATUS_UNASSIGNED_SKELETON

    for soma_id in ambiguous_soma_ids:
        soma_status_lut[
            soma_id
        ] = STATUS_AMBIGUOUS

    for soma_id in insufficient_soma_ids:
        if (
            soma_status_lut[
                soma_id
            ]
            == STATUS_BACKGROUND
        ):
            soma_status_lut[
                soma_id
            ] = STATUS_INSUFFICIENT_SKELETON

    for soma_id in rejected_border_soma_ids:
        soma_status_lut[
            soma_id
        ] = STATUS_BORDER_REJECTED

    component_instance_lut = np.zeros(
        skeleton_count + 1,
        dtype=np.uint32,
    )

    soma_instance_lut = np.zeros(
        soma_count + 1,
        dtype=np.uint32,
    )

    manifest_rows: list[
        dict[str, object]
    ] = []

    exported_records: list[
        dict[str, object]
    ] = []

    for (
        export_index,
        candidate,
    ) in enumerate(
        accepted_candidates
    ):
        output_number = (
            args.start_number
            + export_index
        )

        instance_id = (
            export_index
            + 1
        )

        folder_name = (
            f"cell"
            f"{output_number:0{args.digits}d}"
        )

        cell_dir = (
            args.output_dir
            / folder_name
        )

        soma_id = int(
            candidate[
                "soma_id"
            ]
        )

        component_ids = [
            int(value)
            for value in candidate[
                "component_ids"
            ]
        ]

        core_bbox = candidate[
            "core_bbox"
        ]

        y0, y1, x0, x1 = crop_bounds_from_bbox(
            core_bbox=core_bbox,
            image_shape=semantic.shape,
            margin=args.margin,
            minimum_size=args.min_crop_size,
            square=args.square,
        )

        raw_crop = np.asarray(
            original[
                y0:y1,
                x0:x1,
            ]
        )

        local_skeleton_labels = (
            skeleton_labels[
                y0:y1,
                x0:x1,
            ]
        )

        local_soma_labels = (
            soma_labels[
                y0:y1,
                x0:x1,
            ]
        )

        if component_ids:
            skeleton_crop = np.isin(
                local_skeleton_labels,
                np.asarray(
                    component_ids,
                    dtype=np.uint32,
                ),
            )

        else:
            skeleton_crop = np.zeros(
                local_soma_labels.shape,
                dtype=bool,
            )

        soma_crop = (
            local_soma_labels
            == soma_id
        )

        cell_crop = (
            skeleton_crop
            | soma_crop
        )

        actual_skeleton_pixels = int(
            skeleton_crop.sum()
        )

        actual_soma_pixels = int(
            soma_crop.sum()
        )

        expected_skeleton_pixels = int(
            candidate[
                "expected_skeleton_pixels"
            ]
        )

        expected_soma_pixels = int(
            candidate[
                "expected_soma_pixels"
            ]
        )

        if (
            actual_skeleton_pixels
            != expected_skeleton_pixels
        ):
            raise RuntimeError(
                f"{folder_name}: "
                "Skeleton wurde beim Croppen "
                "abgeschnitten: "
                f"erwartet "
                f"{expected_skeleton_pixels}, "
                f"gefunden "
                f"{actual_skeleton_pixels}."
            )

        if (
            actual_soma_pixels
            != expected_soma_pixels
        ):
            raise RuntimeError(
                f"{folder_name}: "
                "Soma wurde beim Croppen "
                "abgeschnitten: "
                f"erwartet "
                f"{expected_soma_pixels}, "
                f"gefunden "
                f"{actual_soma_pixels}."
            )

        if not mask_has_edge_clearance(
            cell_crop,
            args.edge_clearance,
        ):
            raise RuntimeError(
                f"{folder_name}: "
                "Zellmaske liegt näher als "
                f"{args.edge_clearance} Pixel "
                "am Crop-Rand. "
                "Erhöhe --margin oder "
                "--min-crop-size."
            )

        cell_dir.mkdir()

        seg_crop = np.zeros(
            cell_crop.shape,
            dtype=np.uint8,
        )

        seg_crop[
            skeleton_crop
        ] = 1

        seg_crop[
            soma_crop
        ] = 2

        save_tiff(
            cell_dir
            / "raw.tif",
            raw_crop,
        )

        save_tiff(
            cell_dir
            / "skeleton.tif",
            skeleton_crop.astype(
                np.uint8
            ),
        )

        save_tiff(
            cell_dir
            / "soma.tif",
            soma_crop.astype(
                np.uint8
            ),
        )

        save_tiff(
            cell_dir
            / "seg.tif",
            seg_crop,
        )

        save_tiff(
            cell_dir
            / "cell_mask.tif",
            cell_crop.astype(
                np.uint8
            ),
        )

        write_pixel_csv(
            cell_dir
            / "seg.csv",
            skeleton_crop,
            x0,
            y0,
        )

        write_swc(
            cell_dir
            / "seg-000.swc",
            skeleton_crop,
            soma_crop,
        )

        soma_coordinates = np.argwhere(
            soma_crop
        )

        local_y, local_x = (
            soma_coordinates.mean(
                axis=0
            )
        )

        bounds = {
            "x_min": int(x0),
            "y_min": int(y0),
            "x_max_exclusive": int(x1),
            "y_max_exclusive": int(y1),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
            "core_x_min": int(
                core_bbox[2]
            ),
            "core_y_min": int(
                core_bbox[0]
            ),
            "core_x_max_exclusive": int(
                core_bbox[3]
            ),
            "core_y_max_exclusive": int(
                core_bbox[1]
            ),
        }

        location = {
            "local_x": float(local_x),
            "local_y": float(local_y),
            "global_x": float(
                local_x
                + x0
            ),
            "global_y": float(
                local_y
                + y0
            ),
        }

        metadata = {
            "script_version": SCRIPT_VERSION,
            "status": "safe",
            "folder": folder_name,
            "safe_instance_id": instance_id,
            "source_soma_id": soma_id,
            "source_skeleton_component_ids": (
                component_ids
            ),
            "skeleton_pixels": (
                actual_skeleton_pixels
            ),
            "soma_pixels": (
                actual_soma_pixels
            ),
            "touches_source_border": bool(
                candidate[
                    "touches_source_border"
                ]
            ),
            "crop_completeness_verified": True,
            "edge_clearance_pixels": (
                args.edge_clearance
            ),
            "bounds": bounds,
            "location": location,
        }

        (
            cell_dir
            / "bounds.json"
        ).write_text(
            json.dumps(
                bounds,
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            cell_dir
            / "location.json"
        ).write_text(
            json.dumps(
                location,
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            cell_dir
            / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        soma_instance_lut[
            soma_id
        ] = instance_id

        soma_status_lut[
            soma_id
        ] = STATUS_SAFE

        for component_id in component_ids:
            component_instance_lut[
                component_id
            ] = instance_id

            component_status_lut[
                component_id
            ] = STATUS_SAFE

        manifest_row = {
            "folder": folder_name,
            "status": "safe",
            "safe_instance_id": instance_id,
            "source_soma_id": soma_id,
            "source_skeleton_component_ids": (
                ",".join(
                    map(
                        str,
                        component_ids,
                    )
                )
            ),
            "x_min": x0,
            "y_min": y0,
            "x_max_exclusive": x1,
            "y_max_exclusive": y1,
            "global_centroid_x": location[
                "global_x"
            ],
            "global_centroid_y": location[
                "global_y"
            ],
            "skeleton_pixels": (
                actual_skeleton_pixels
            ),
            "soma_pixels": (
                actual_soma_pixels
            ),
            "crop_width": x1 - x0,
            "crop_height": y1 - y0,
            "crop_completeness_verified": True,
        }

        manifest_rows.append(
            manifest_row
        )

        exported_records.append(
            metadata
        )

        print(
            f"[{export_index + 1}/"
            f"{len(accepted_candidates)}] "
            f"{folder_name}: "
            f"soma={soma_id}, "
            f"crop={x1 - x0}x{y1 - y0}, "
            f"skeleton="
            f"{actual_skeleton_pixels}, "
            f"soma="
            f"{actual_soma_pixels}"
        )

    exported_soma_ids = {
        int(
            record[
                "source_soma_id"
            ]
        )
        for record in exported_records
    }

    for soma_id in range(
        1,
        soma_count + 1,
    ):
        if soma_id in exported_soma_ids:
            continue

        for component_id in uniquely_assigned[
            soma_id
        ]:
            if (
                component_status_lut[
                    component_id
                ]
                == STATUS_BACKGROUND
            ):
                if (
                    soma_id
                    in rejected_border_soma_ids
                ):
                    component_status_lut[
                        component_id
                    ] = STATUS_BORDER_REJECTED

                else:
                    component_status_lut[
                        component_id
                    ] = (
                        STATUS_INSUFFICIENT_SKELETON
                    )

    manifest_fields = [
        "folder",
        "status",
        "safe_instance_id",
        "source_soma_id",
        "source_skeleton_component_ids",
        "x_min",
        "y_min",
        "x_max_exclusive",
        "y_max_exclusive",
        "global_centroid_x",
        "global_centroid_y",
        "skeleton_pixels",
        "soma_pixels",
        "crop_width",
        "crop_height",
        "crop_completeness_verified",
    ]

    with (
        args.output_dir
        / "manifest.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_fields,
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    candidate_fields = [
        "source_soma_id",
        "status",
        "reason",
        "skeleton_pixels",
        "soma_pixels",
        "component_ids",
    ]

    with (
        diagnostics
        / "soma_classification.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=candidate_fields,
        )

        writer.writeheader()
        writer.writerows(
            candidate_rows
        )

    with (
        diagnostics
        / "skeleton_component_assignment.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "component_id",
                "pixel_count",
                "touched_soma_ids",
                "status",
            ]
        )

        skeleton_sizes = classification[
            "skeleton_sizes"
        ]

        component_to_somata = classification[
            "component_to_somata"
        ]

        for component_id in range(
            1,
            skeleton_count + 1,
        ):
            touched = sorted(
                component_to_somata[
                    component_id
                ]
            )

            if (
                component_id
                in ambiguous_components
            ):
                status_name = "ambiguous"

            elif (
                component_id
                in unassigned_components
            ):
                status_name = "unassigned"

            elif (
                component_status_lut[
                    component_id
                ]
                == STATUS_SAFE
            ):
                status_name = "safe"

            elif (
                component_status_lut[
                    component_id
                ]
                == STATUS_BORDER_REJECTED
            ):
                status_name = (
                    "border_rejected"
                )

            else:
                status_name = (
                    "insufficient_or_"
                    "not_exported"
                )

            writer.writerow(
                [
                    component_id,
                    int(
                        skeleton_sizes[
                            component_id
                        ]
                    ),
                    ",".join(
                        map(
                            str,
                            touched,
                        )
                    ),
                    status_name,
                ]
            )

    status_map = build_status_map(
        semantic=semantic,
        skeleton_labels=skeleton_labels,
        soma_labels=soma_labels,
        component_status_lut=(
            component_status_lut
        ),
        soma_status_lut=(
            soma_status_lut
        ),
        rejected_small_soma=(
            rejected_small_soma
        ),
    )

    save_tiff(
        diagnostics
        / "cell_status_overview.tif",
        status_map,
    )

    save_qc_overlay(
        path=(
            diagnostics
            / "cell_classification_overlay.png"
        ),
        original=original,
        skeleton_labels=skeleton_labels,
        soma_labels=soma_labels,
        component_status_lut=(
            component_status_lut
        ),
        soma_status_lut=(
            soma_status_lut
        ),
        component_instance_lut=(
            component_instance_lut
        ),
        soma_instance_lut=(
            soma_instance_lut
        ),
        rejected_small_soma=(
            rejected_small_soma
        ),
        max_side=args.qc_max_side,
    )

    if args.save_diagnostic_masks:
        ambiguous_component_lut = np.zeros(
            skeleton_count + 1,
            dtype=bool,
        )

        unassigned_component_lut = np.zeros(
            skeleton_count + 1,
            dtype=bool,
        )

        if ambiguous_components:
            ambiguous_component_lut[
                np.asarray(
                    sorted(
                        ambiguous_components
                    )
                )
            ] = True

        if unassigned_components:
            unassigned_component_lut[
                np.asarray(
                    sorted(
                        unassigned_components
                    )
                )
            ] = True

        save_tiff(
            diagnostics
            / "ambiguous_skeleton_overview.tif",
            ambiguous_component_lut[
                skeleton_labels
            ].astype(
                np.uint8
            ),
        )

        save_tiff(
            diagnostics
            / "unassigned_skeleton_overview.tif",
            unassigned_component_lut[
                skeleton_labels
            ].astype(
                np.uint8
            ),
        )

        save_tiff(
            diagnostics
            / "valid_soma_seeds_overview.tif",
            soma_seed_mask.astype(
                np.uint8
            ),
        )

        save_tiff(
            diagnostics
            / "rejected_small_soma_overview.tif",
            rejected_small_soma.astype(
                np.uint8
            ),
        )

    if args.save_overview_instances:
        save_optional_overview_instances(
            path=(
                diagnostics
                / "safe_cell_instances_overview.tif"
            ),
            skeleton_labels=skeleton_labels,
            soma_labels=soma_labels,
            component_instance_lut=(
                component_instance_lut
            ),
            soma_instance_lut=(
                soma_instance_lut
            ),
        )

    summary = {
        "script_version": SCRIPT_VERSION,
        "safe_cells_exported": len(
            manifest_rows
        ),
        "candidate_safe_somata_before_border_check": len(
            candidate_safe_soma_ids
        ),
        "border_touching_candidates_rejected": len(
            rejected_border_soma_ids
        ),
        "ambiguous_somata": len(
            ambiguous_soma_ids
        ),
        "soma_without_enough_skeleton": len(
            insufficient_soma_ids
        ),
        "unassigned_skeleton_components": len(
            unassigned_components
        ),
        "ambiguous_skeleton_components": len(
            ambiguous_components
        ),
        "small_soma_components_not_used_as_seeds": len(
            rejected_soma_ids
        ),
        "contact_radius": args.contact_radius,
        "min_soma_area": args.min_soma_area,
        "min_skeleton_pixels": (
            args.min_skeleton_pixels
        ),
        "margin": args.margin,
        "min_crop_size": (
            args.min_crop_size
        ),
        "edge_clearance": (
            args.edge_clearance
        ),
        "square_crops": bool(
            args.square
        ),
        "allow_border_touching": bool(
            args.allow_border_touching
        ),
        "crop_completeness_verified_for_every_export": True,
        "status_codes": {
            "0": "background",
            "1": "safe exported cell",
            "2": "ambiguous",
            "3": "unassigned skeleton",
            "4": (
                "insufficient skeleton "
                "/ not exported"
            ),
            "5": (
                "border-touching "
                "candidate rejected"
            ),
            "6": "small soma rejected",
        },
    }

    (
        args.output_dir
        / "export_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        args.output_dir
        / "_export_version.txt"
    ).write_text(
        f"{SCRIPT_VERSION}\n"
        f"{Path(__file__).resolve()}\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Exportiert: "
        f"{len(manifest_rows)} "
        "sichere Zellen"
    )

    print(
        f"Ausgabe:    "
        f"{args.output_dir.resolve()}"
    )

    print()
    print(
        "Jeder exportierte Crop wurde "
        "pixelgenau auf Vollständigkeit geprüft."
    )

    print(
        "Zellen am Rand des Quellbilds "
        "wurden standardmäßig nicht exportiert."
    )

    print()
    print("NÄCHSTER SCHRITT:")

    print(
        "finalize_cells_with_fiji.py "
        "ausführen, um seg.traces, "
        "soma.zip, bounds.zip und "
        "location.zip zu erzeugen."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            f"\nFEHLER: {error}",
            file=sys.stderr,
        )

        raise
