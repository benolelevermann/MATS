from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

NEIGHBORHOOD = np.ones((3, 3), dtype=np.uint8)


# ---------------------------------------------------------------------
# Einlesen und allgemeine Hilfsfunktionen
# ---------------------------------------------------------------------


def read_2d(path: Path) -> np.ndarray:
    image = np.squeeze(np.asarray(tifffile.imread(path)))

    if image.ndim != 2:
        raise RuntimeError(f"2D-TIFF erwartet, gefunden {image.shape}: {path}")

    return image


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalisierung nur für das QC-PNG.
    Die TIFF-Ausgaben werden dadurch nicht verändert.
    """
    image = image.astype(
        np.float32,
        copy=False,
    )

    valid = image[np.isfinite(image)]

    if valid.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    low, high = np.percentile(
        valid,
        [1.0, 99.5],
    )

    if high <= low:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    return np.clip(
        (image - low) / (high - low),
        0.0,
        1.0,
    )


def line_pixels(
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Speichersparende Bresenham-Linie.

    Es wird keine Vollbildmaske pro Verbindung erzeugt.
    """
    y0, x0 = start
    y1, x1 = end

    ys: list[int] = []
    xs: list[int] = []

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1

    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1

    error = dx + dy

    while True:
        ys.append(y0)
        xs.append(x0)

        if x0 == x1 and y0 == y1:
            break

        error2 = 2 * error

        if error2 >= dy:
            error += dy
            x0 += sx

        if error2 <= dx:
            error += dx
            y0 += sy

    return (
        np.asarray(
            ys,
            dtype=np.int32,
        ),
        np.asarray(
            xs,
            dtype=np.int32,
        ),
    )


# ---------------------------------------------------------------------
# Skeleton-Endpunkte und Richtungen
# ---------------------------------------------------------------------


def find_endpoints(
    thin_skeleton: np.ndarray,
) -> np.ndarray:
    """
    Endpunkt:
    Skeletonpixel mit genau einem Nachbarn in der 3x3-Umgebung.
    """
    kernel = NEIGHBORHOOD.copy()
    kernel[1, 1] = 0

    neighbor_count = ndi.convolve(
        thin_skeleton.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    endpoint_mask = thin_skeleton & (neighbor_count == 1)

    return np.argwhere(endpoint_mask)


def outward_direction(
    thin_skeleton: np.ndarray,
    component_labels: np.ndarray,
    endpoint: tuple[int, int],
    steps: int,
) -> Optional[np.ndarray]:
    """
    Verfolgt das Skeleton vom Endpunkt einige Pixel nach innen.

    Anschließend wird daraus ein Vektor berechnet, der vom Skeleton
    nach außen zeigt. Nur in ungefähr dieser Richtung darf verbunden
    werden.
    """
    component_id = int(component_labels[endpoint])

    current = endpoint

    previous: Optional[tuple[int, int]] = None

    path = [endpoint]

    height, width = thin_skeleton.shape

    for _ in range(steps):
        y, x = current

        candidates: list[tuple[int, int]] = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                point = (
                    y + dy,
                    x + dx,
                )

                if point == previous:
                    continue

                if not (0 <= point[0] < height and 0 <= point[1] < width):
                    continue

                if (
                    thin_skeleton[point]
                    and int(component_labels[point]) == component_id
                ):
                    candidates.append(point)

        # Bei einer Verzweigung ist die Richtung nicht eindeutig.
        if len(candidates) != 1:
            break

        previous = current
        current = candidates[0]
        path.append(current)

    if len(path) < 2:
        return None

    vector = np.asarray(
        endpoint,
        dtype=float,
    ) - np.asarray(
        path[-1],
        dtype=float,
    )

    length = float(np.linalg.norm(vector))

    if length < 1e-6:
        return None

    return vector / length


def angle_degrees(
    direction: Optional[np.ndarray],
    start: tuple[int, int],
    target: tuple[int, int],
) -> float:
    """
    Winkel zwischen der Skeleton-Endpunktrichtung und einer
    möglichen Verbindung.
    """
    if direction is None:
        return 180.0

    connection = np.asarray(
        target,
        dtype=float,
    ) - np.asarray(
        start,
        dtype=float,
    )

    length = float(np.linalg.norm(connection))

    if length < 1e-6:
        return 180.0

    connection /= length

    cosine = float(
        np.clip(
            np.dot(
                direction,
                connection,
            ),
            -1.0,
            1.0,
        )
    )

    return math.degrees(math.acos(cosine))


# ---------------------------------------------------------------------
# Zuordnung bestehender Skeletonkomponenten zu Soma
# ---------------------------------------------------------------------


def soma_contacts(
    skeleton_labels: np.ndarray,
    soma_labels: np.ndarray,
    radius: int,
) -> dict[int, set[int]]:
    """
    Liefert für jede Skeletonkomponente die Soma-IDs,
    mit denen sie bereits Kontakt hat.

    Die Berechnung erfolgt somaweise in kleinen Ausschnitten.
    Dadurch wird keine große Distanztransformations-Matrix benötigt.
    """
    result: dict[
        int,
        set[int],
    ] = {}

    height, width = soma_labels.shape

    soma_objects = ndi.find_objects(soma_labels)

    for soma_id, object_slice in enumerate(
        soma_objects,
        start=1,
    ):
        if object_slice is None:
            continue

        y_slice, x_slice = object_slice

        y0 = max(
            0,
            y_slice.start - radius,
        )
        y1 = min(
            height,
            y_slice.stop + radius,
        )

        x0 = max(
            0,
            x_slice.start - radius,
        )
        x1 = min(
            width,
            x_slice.stop + radius,
        )

        local_soma = (
            soma_labels[
                y0:y1,
                x0:x1,
            ]
            == soma_id
        )

        if radius > 0:
            local_soma = ndi.binary_dilation(
                local_soma,
                iterations=radius,
            )

        local_components = skeleton_labels[
            y0:y1,
            x0:x1,
        ]

        component_ids = np.unique(local_components[local_soma])

        for component_id in component_ids:
            if component_id == 0:
                continue

            result.setdefault(
                int(component_id),
                set(),
            ).add(soma_id)

    return result


class Components:
    """
    Verwaltet bereits verbundene Skeletonkomponenten.

    Entscheidend:
    Eine Verbindung wird abgelehnt, wenn dadurch zwei verschiedene
    Soma miteinander verbunden würden.
    """

    def __init__(
        self,
        count: int,
        contacts: dict[int, set[int]],
    ) -> None:
        self.parent = np.arange(
            count + 1,
            dtype=np.int32,
        )

        self.somas = {
            component_id: set(
                contacts.get(
                    component_id,
                    set(),
                )
            )
            for component_id in range(
                1,
                count + 1,
            )
        }

    def root(
        self,
        component_id: int,
    ) -> int:
        parent = int(self.parent[component_id])

        if parent != component_id:
            self.parent[component_id] = self.root(parent)

        return int(self.parent[component_id])

    def soma_ids(
        self,
        component_id: int,
    ) -> set[int]:
        return self.somas[self.root(component_id)]

    def merge_allowed(
        self,
        first: int,
        second: int,
    ) -> bool:
        root_first = self.root(first)
        root_second = self.root(second)

        if root_first == root_second:
            return False

        combined_somas = self.somas[root_first] | self.somas[root_second]

        # Zwei verschiedene Soma dürfen niemals verbunden werden.
        return len(combined_somas) <= 1

    def merge(
        self,
        first: int,
        second: int,
    ) -> None:
        root_first = self.root(first)
        root_second = self.root(second)

        self.parent[root_second] = root_first

        self.somas[root_first] |= self.somas[root_second]

    def assign_soma(
        self,
        component_id: int,
        soma_id: int,
    ) -> bool:
        root = self.root(component_id)

        existing_somas = self.somas[root]

        if existing_somas and soma_id not in existing_somas:
            return False

        existing_somas.add(soma_id)

        return True


# ---------------------------------------------------------------------
# Verbindungen einzeichnen
# ---------------------------------------------------------------------


def draw_connection(
    added_connections: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
    width: int,
) -> None:
    rr, cc = line_pixels(
        start,
        target,
    )

    added_connections[
        rr,
        cc,
    ] = True

    if width <= 1:
        return

    radius = max(
        1,
        width // 2,
    )

    y0 = max(
        0,
        int(rr.min()) - radius,
    )
    y1 = min(
        added_connections.shape[0],
        int(rr.max()) + radius + 1,
    )

    x0 = max(
        0,
        int(cc.min()) - radius,
    )
    x1 = min(
        added_connections.shape[1],
        int(cc.max()) + radius + 1,
    )

    local = added_connections[
        y0:y1,
        x0:x1,
    ]

    added_connections[
        y0:y1,
        x0:x1,
    ] = ndi.binary_dilation(
        local,
        iterations=radius,
    )


# ---------------------------------------------------------------------
# Neue konservative Verbindungssuche
# ---------------------------------------------------------------------


def connect_gaps(
    raw_skeleton: np.ndarray,
    soma_mask: np.ndarray,
    max_skeleton_gap: float,
    max_soma_gap: float,
    max_angle: float,
    tangent_steps: int,
    attach_radius: int,
    rounds: int,
    connection_width: int,
) -> tuple[
    np.ndarray,
    list[dict[str, object]],
]:
    """
    Neue Reihenfolge:

    1. Skeleton-Endpunkt zu Skeleton-Endpunkt
    2. Erst danach Skeleton-Endpunkt zu Soma

    Sicherheitsregeln:

    - Richtung wird an beiden Skeleton-Endpunkten geprüft.
    - Verschiedene Soma dürfen niemals verbunden werden.
    - Jeder Endpunkt darf nur einmal verwendet werden.
    - Pro Skeletonkomponente höchstens eine neue Soma-Verbindung.
    - Eine Soma-Verbindung darf nicht durch ein anderes Soma laufen.
    """
    added_connections = np.zeros_like(
        raw_skeleton,
        dtype=bool,
    )

    report: list[dict[str, object]] = []

    soma_labels, _ = ndi.label(
        soma_mask,
        structure=NEIGHBORHOOD,
    )

    # Nur die Soma-Randpixel für die räumliche Suche benutzen.
    soma_boundary = soma_mask & ~ndi.binary_erosion(soma_mask)

    soma_boundary_points = np.argwhere(soma_boundary)

    if len(soma_boundary_points) > 0:
        soma_tree: Optional[cKDTree] = cKDTree(soma_boundary_points)
    else:
        soma_tree = None

    for round_number in range(
        1,
        rounds + 1,
    ):
        completed_skeleton = raw_skeleton | added_connections

        # Skeletonisierung nur für Endpunkt- und Richtungsanalyse.
        thin_skeleton = skeletonize(completed_skeleton)

        component_labels, component_count = ndi.label(
            thin_skeleton,
            structure=NEIGHBORHOOD,
        )

        endpoint_array = find_endpoints(thin_skeleton)

        print(
            f"Runde {round_number}: "
            f"{component_count} Komponenten, "
            f"{len(endpoint_array)} Endpunkte"
        )

        if len(endpoint_array) == 0:
            break

        points = [
            (
                int(point[0]),
                int(point[1]),
            )
            for point in endpoint_array
        ]

        component_ids = [int(component_labels[point]) for point in points]

        directions = [
            outward_direction(
                thin_skeleton,
                component_labels,
                point,
                tangent_steps,
            )
            for point in points
        ]

        existing_contacts = soma_contacts(
            component_labels,
            soma_labels,
            attach_radius,
        )

        groups = Components(
            component_count,
            existing_contacts,
        )

        used_endpoints: set[int] = set()

        # =============================================================
        # 1. Skeleton-Skeleton-Lücken zuerst schließen
        # =============================================================

        skeleton_candidates: list[dict[str, object]] = []

        endpoint_tree = cKDTree(endpoint_array)

        endpoint_pairs = endpoint_tree.query_pairs(max_skeleton_gap)

        for (
            first_index,
            second_index,
        ) in endpoint_pairs:
            first_component = component_ids[first_index]

            second_component = component_ids[second_index]

            if first_component == second_component:
                continue

            first_point = points[first_index]

            second_point = points[second_index]

            first_angle = angle_degrees(
                directions[first_index],
                first_point,
                second_point,
            )

            second_angle = angle_degrees(
                directions[second_index],
                second_point,
                first_point,
            )

            # Beide Enden müssen ungefähr aufeinander zeigen.
            if first_angle > max_angle or second_angle > max_angle:
                continue

            rr, cc = line_pixels(
                first_point,
                second_point,
            )

            # Nicht durch ein Soma verbinden.
            if np.any(
                soma_mask[
                    rr,
                    cc,
                ]
            ):
                continue

            # Die Linie darf keine dritte Skeletonkomponente schneiden.
            crossed_components = np.unique(
                component_labels[
                    rr,
                    cc,
                ]
            )

            crossed_components = crossed_components[crossed_components != 0]

            allowed_components = {
                first_component,
                second_component,
            }

            crosses_third_component = any(
                int(value) not in allowed_components for value in crossed_components
            )

            if crosses_third_component:
                continue

            distance = float(
                np.linalg.norm(np.asarray(first_point) - np.asarray(second_point))
            )

            # Kurze und gut ausgerichtete Kandidaten zuerst.
            score = distance + 0.25 * first_angle + 0.25 * second_angle

            skeleton_candidates.append(
                {
                    "first_index": first_index,
                    "second_index": second_index,
                    "first_component": first_component,
                    "second_component": second_component,
                    "start": first_point,
                    "target": second_point,
                    "distance": distance,
                    "angle_start": first_angle,
                    "angle_target": second_angle,
                    "score": score,
                }
            )

        skeleton_candidates.sort(key=lambda candidate: float(candidate["score"]))

        accepted_skeleton_connections = 0

        for candidate in skeleton_candidates:
            first_index = int(candidate["first_index"])

            second_index = int(candidate["second_index"])

            if first_index in used_endpoints or second_index in used_endpoints:
                continue

            first_component = int(candidate["first_component"])

            second_component = int(candidate["second_component"])

            # Kernregel:
            # Zwei verschiedene Soma-Zellen nicht künstlich verbinden.
            if not groups.merge_allowed(
                first_component,
                second_component,
            ):
                continue

            start = candidate["start"]

            target = candidate["target"]

            assert isinstance(
                start,
                tuple,
            )

            assert isinstance(
                target,
                tuple,
            )

            draw_connection(
                added_connections,
                start,
                target,
                connection_width,
            )

            groups.merge(
                first_component,
                second_component,
            )

            used_endpoints.add(first_index)

            used_endpoints.add(second_index)

            accepted_skeleton_connections += 1

            report.append(
                {
                    "round": round_number,
                    "type": "skeleton_to_skeleton",
                    "start_y": start[0],
                    "start_x": start[1],
                    "target_y": target[0],
                    "target_x": target[1],
                    "distance": candidate["distance"],
                    "angle_start": candidate["angle_start"],
                    "angle_target": candidate["angle_target"],
                }
            )

        # =============================================================
        # 2. Erst anschließend Skeleton-Soma-Verbindungen
        # =============================================================

        soma_candidates: list[dict[str, object]] = []

        if soma_tree is not None:
            for (
                endpoint_index,
                endpoint,
            ) in enumerate(points):
                if endpoint_index in used_endpoints:
                    continue

                component_id = component_ids[endpoint_index]

                # Bereits mit einem Soma verbundene Komponenten
                # nicht nochmals sternförmig anbinden.
                if groups.soma_ids(component_id):
                    continue

                nearby_boundary_indices = soma_tree.query_ball_point(
                    endpoint,
                    max_soma_gap,
                )

                # Für jedes Soma nur den besten Randpunkt behalten.
                best_per_soma: dict[
                    int,
                    dict[str, object],
                ] = {}

                for boundary_index in nearby_boundary_indices:
                    target_array = soma_boundary_points[boundary_index]

                    target = (
                        int(target_array[0]),
                        int(target_array[1]),
                    )

                    soma_id = int(soma_labels[target])

                    if soma_id == 0:
                        continue

                    current_angle = angle_degrees(
                        directions[endpoint_index],
                        endpoint,
                        target,
                    )

                    # Soma muss vor dem Endpunkt liegen,
                    # nicht seitlich oder dahinter.
                    if current_angle > max_angle:
                        continue

                    rr, cc = line_pixels(
                        endpoint,
                        target,
                    )

                    crossed_soma = np.unique(
                        soma_labels[
                            rr,
                            cc,
                        ]
                    )

                    crossed_soma = crossed_soma[crossed_soma != 0]

                    # Verbindung darf nur das Ziel-Soma berühren.
                    crosses_foreign_soma = any(
                        int(value) != soma_id for value in crossed_soma
                    )

                    if crosses_foreign_soma:
                        continue

                    distance = float(
                        np.linalg.norm(np.asarray(endpoint) - np.asarray(target))
                    )

                    score = distance + 0.35 * current_angle

                    old_candidate = best_per_soma.get(soma_id)

                    if old_candidate is None or score < float(old_candidate["score"]):
                        best_per_soma[soma_id] = {
                            "endpoint_index": endpoint_index,
                            "component_id": component_id,
                            "soma_id": soma_id,
                            "start": endpoint,
                            "target": target,
                            "distance": distance,
                            "angle_start": current_angle,
                            "score": score,
                        }

                soma_candidates.extend(best_per_soma.values())

        soma_candidates.sort(key=lambda candidate: float(candidate["score"]))

        connected_roots: set[int] = set()

        accepted_soma_connections = 0

        for candidate in soma_candidates:
            endpoint_index = int(candidate["endpoint_index"])

            component_id = int(candidate["component_id"])

            soma_id = int(candidate["soma_id"])

            root = groups.root(component_id)

            if endpoint_index in used_endpoints:
                continue

            # Pro zusammenhängender Skeletonkomponente
            # höchstens eine neue Soma-Anbindung.
            if root in connected_roots:
                continue

            if groups.soma_ids(component_id):
                continue

            if not groups.assign_soma(
                component_id,
                soma_id,
            ):
                continue

            start = candidate["start"]

            target = candidate["target"]

            assert isinstance(
                start,
                tuple,
            )

            assert isinstance(
                target,
                tuple,
            )

            draw_connection(
                added_connections,
                start,
                target,
                connection_width,
            )

            connected_roots.add(groups.root(component_id))

            used_endpoints.add(endpoint_index)

            accepted_soma_connections += 1

            report.append(
                {
                    "round": round_number,
                    "type": "skeleton_to_soma",
                    "start_y": start[0],
                    "start_x": start[1],
                    "target_y": target[0],
                    "target_x": target[1],
                    "distance": candidate["distance"],
                    "angle_start": candidate["angle_start"],
                    "angle_target": "",
                }
            )

        print(
            f"  akzeptiert: "
            f"{accepted_skeleton_connections} Skeleton-Lücken, "
            f"{accepted_soma_connections} Soma-Anbindungen"
        )

        if accepted_skeleton_connections + accepted_soma_connections == 0:
            break

    return (
        added_connections,
        report,
    )


def rescue_unassigned_components(
    completed_skeleton: np.ndarray,
    soma_mask: np.ndarray,
    max_rescue_gap: float,
    max_rescue_angle: float,
    ambiguity_margin: float,
    tangent_steps: int,
    attach_radius: int,
    rescue_rounds: int,
    connection_width: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """
    Verbindet Skeletonfragmente ohne Somakontakt schrittweise mit
    Skeletonkomponenten, die bereits genau einem Soma zugeordnet sind.

    Anders als die erste Verbindungsstufe wird hier nicht nur
    Endpunkt -> Endpunkt verwendet, sondern:

        freier Endpunkt -> beliebiger naher Punkt eines
                           soma-verankerten Skeletons

    Dadurch können auch Lücken geschlossen werden, bei denen das Ziel
    in der Mitte eines Skeletonsegments liegt.

    Sicherheitsregeln:

    - Ein Ziel-Skeleton muss genau einem Soma zugeordnet sein.
    - Die Verbindung darf nicht durch ein anderes Soma führen.
    - Liegen zwei verschiedene Soma ähnlich plausibel, wird nicht verbunden.
    - Pro freier Komponente wird pro Runde höchstens eine Verbindung ergänzt.
    - Nach jeder Runde werden die Komponenten neu bestimmt.
    """

    rescue_connections = np.zeros_like(
        completed_skeleton,
        dtype=bool,
    )

    report: list[dict[str, object]] = []

    soma_labels, _ = ndi.label(
        soma_mask,
        structure=NEIGHBORHOOD,
    )

    for round_number in range(
        1,
        rescue_rounds + 1,
    ):
        current_skeleton = completed_skeleton | rescue_connections

        # Nur für Komponenten- und Endpunktanalyse skeletonisieren.
        thin_skeleton = skeletonize(current_skeleton)

        component_labels, component_count = ndi.label(
            thin_skeleton,
            structure=NEIGHBORHOOD,
        )

        if component_count == 0:
            break

        # Welche Skeletonkomponente berührt welches Soma?
        contacts = soma_contacts(
            component_labels,
            soma_labels,
            attach_radius,
        )

        component_to_soma: dict[int, int] = {}

        anchored_component_ids: list[int] = []
        free_component_ids: list[int] = []
        ambiguous_component_ids: list[int] = []

        for component_id in range(
            1,
            component_count + 1,
        ):
            contacted_soma = contacts.get(
                component_id,
                set(),
            )

            if len(contacted_soma) == 1:
                soma_id = next(iter(contacted_soma))

                component_to_soma[component_id] = int(soma_id)

                anchored_component_ids.append(component_id)

            elif len(contacted_soma) == 0:
                free_component_ids.append(component_id)

            else:
                # Komponente berührt bereits mehrere Soma.
                # Daran wird nichts weiter angeschlossen.
                ambiguous_component_ids.append(component_id)

        print(
            f"Rescue-Runde {round_number}: "
            f"{len(anchored_component_ids)} verankert, "
            f"{len(free_component_ids)} frei, "
            f"{len(ambiguous_component_ids)} mehrdeutig"
        )

        if len(anchored_component_ids) == 0 or len(free_component_ids) == 0:
            break

        # Lookup-Tabelle ist speichersparender als viele np.isin-Aufrufe.
        anchored_lookup = np.zeros(
            component_count + 1,
            dtype=bool,
        )

        anchored_lookup[
            np.asarray(
                anchored_component_ids,
                dtype=np.int32,
            )
        ] = True

        anchored_mask = anchored_lookup[component_labels]

        # Alle Pixel der bereits soma-verankerten Skeletons sind mögliche Ziele.
        target_points = np.argwhere(anchored_mask)

        if len(target_points) == 0:
            break

        target_component_ids = component_labels[
            target_points[:, 0],
            target_points[:, 1],
        ]

        target_soma_ids = np.asarray(
            [
                component_to_soma[int(component_id)]
                for component_id in target_component_ids
            ],
            dtype=np.int32,
        )

        target_tree = cKDTree(target_points.astype(np.float64))

        endpoint_array = find_endpoints(thin_skeleton)

        if len(endpoint_array) == 0:
            break

        endpoint_component_ids = component_labels[
            endpoint_array[:, 0],
            endpoint_array[:, 1],
        ]

        endpoints_by_component: dict[
            int,
            list[int],
        ] = {}

        for endpoint_index, component_id in enumerate(endpoint_component_ids):
            endpoints_by_component.setdefault(
                int(component_id),
                [],
            ).append(endpoint_index)

        component_sizes = np.bincount(
            component_labels.ravel(),
            minlength=component_count + 1,
        )

        candidates: list[dict[str, object]] = []

        number_of_neighbors = min(
            64,
            len(target_points),
        )

        for free_component_id in free_component_ids:
            endpoint_indices = endpoints_by_component.get(
                free_component_id,
                [],
            )

            if not endpoint_indices:
                continue

            # Für jedes mögliche Soma nur den besten Kandidaten behalten.
            best_candidate_per_soma: dict[
                int,
                dict[str, object],
            ] = {}

            for endpoint_index in endpoint_indices:
                endpoint = (
                    int(
                        endpoint_array[
                            endpoint_index,
                            0,
                        ]
                    ),
                    int(
                        endpoint_array[
                            endpoint_index,
                            1,
                        ]
                    ),
                )

                direction = outward_direction(
                    thin_skeleton,
                    component_labels,
                    endpoint,
                    tangent_steps,
                )

                distances, target_indices = target_tree.query(
                    np.asarray(
                        endpoint,
                        dtype=np.float64,
                    ),
                    k=number_of_neighbors,
                    distance_upper_bound=max_rescue_gap,
                )

                distances = np.atleast_1d(distances)

                target_indices = np.atleast_1d(target_indices)

                for distance, target_index in zip(
                    distances,
                    target_indices,
                ):
                    if not np.isfinite(distance):
                        continue

                    target_index = int(target_index)

                    if target_index >= len(target_points):
                        continue

                    target = (
                        int(
                            target_points[
                                target_index,
                                0,
                            ]
                        ),
                        int(
                            target_points[
                                target_index,
                                1,
                            ]
                        ),
                    )

                    target_component_id = int(target_component_ids[target_index])

                    target_soma_id = int(target_soma_ids[target_index])

                    if target_component_id == free_component_id:
                        continue

                    # Bei sehr kleinen Fragmenten ist die Richtung oft
                    # nicht zuverlässig bestimmbar.
                    if direction is None:
                        current_angle = 0.65 * max_rescue_angle
                    else:
                        current_angle = angle_degrees(
                            direction,
                            endpoint,
                            target,
                        )

                    if current_angle > max_rescue_angle:
                        continue

                    rr, cc = line_pixels(
                        endpoint,
                        target,
                    )

                    # Die Verbindung darf kein fremdes Soma schneiden.
                    crossed_soma_ids = np.unique(
                        soma_labels[
                            rr,
                            cc,
                        ]
                    )

                    crossed_soma_ids = crossed_soma_ids[crossed_soma_ids != 0]

                    crosses_foreign_soma = any(
                        int(soma_id) != target_soma_id for soma_id in crossed_soma_ids
                    )

                    if crosses_foreign_soma:
                        continue

                    # Ebenso darf die Linie keine bereits zu einem
                    # anderen Soma gehörende Skeletonkomponente kreuzen.
                    crossed_component_ids = np.unique(
                        component_labels[
                            rr,
                            cc,
                        ]
                    )

                    invalid_crossing = False

                    for crossed_component_id in crossed_component_ids:
                        crossed_component_id = int(crossed_component_id)

                        if crossed_component_id == 0:
                            continue

                        crossed_soma_id = component_to_soma.get(crossed_component_id)

                        if (
                            crossed_soma_id is not None
                            and crossed_soma_id != target_soma_id
                        ):
                            invalid_crossing = True
                            break

                    if invalid_crossing:
                        continue

                    score = float(distance) + 0.25 * float(current_angle)

                    # Sehr kleine Fragmente bekommen eine leichte Strafe,
                    # weil sie häufiger Rauschen darstellen.
                    free_size = int(component_sizes[free_component_id])

                    if free_size < 5:
                        score += 5.0

                    candidate = {
                        "round": round_number,
                        "type": "rescue_to_anchored_skeleton",
                        "free_component": free_component_id,
                        "target_component": target_component_id,
                        "target_soma": target_soma_id,
                        "start": endpoint,
                        "target": target,
                        "distance": float(distance),
                        "angle_start": float(current_angle),
                        "score": float(score),
                    }

                    previous = best_candidate_per_soma.get(target_soma_id)

                    if previous is None or score < float(previous["score"]):
                        best_candidate_per_soma[target_soma_id] = candidate

            if not best_candidate_per_soma:
                continue

            sorted_candidates = sorted(
                best_candidate_per_soma.values(),
                key=lambda candidate: float(candidate["score"]),
            )

            best_candidate = sorted_candidates[0]

            # Falls zwei verschiedene Soma fast gleich plausibel sind,
            # wird keine künstliche Entscheidung getroffen.
            if len(sorted_candidates) >= 2:
                best_score = float(sorted_candidates[0]["score"])

                second_score = float(sorted_candidates[1]["score"])

                if second_score - best_score < ambiguity_margin:
                    continue

            candidates.append(best_candidate)

        candidates.sort(key=lambda candidate: float(candidate["score"]))

        used_free_components: set[int] = set()

        accepted_this_round = 0

        for candidate in candidates:
            free_component_id = int(candidate["free_component"])

            if free_component_id in used_free_components:
                continue

            start = candidate["start"]
            target = candidate["target"]

            assert isinstance(
                start,
                tuple,
            )

            assert isinstance(
                target,
                tuple,
            )

            rr, cc = line_pixels(
                start,
                target,
            )

            new_pixel_mask = np.logical_not(
                current_skeleton[
                    rr,
                    cc,
                ]
            )

            if not np.any(new_pixel_mask):
                continue

            draw_connection(
                rescue_connections,
                start,
                target,
                connection_width,
            )

            used_free_components.add(free_component_id)

            accepted_this_round += 1

            report.append(
                {
                    "round": round_number,
                    "type": "rescue_to_anchored_skeleton",
                    "start_y": start[0],
                    "start_x": start[1],
                    "target_y": target[0],
                    "target_x": target[1],
                    "distance": candidate["distance"],
                    "angle_start": candidate["angle_start"],
                    "angle_target": "",
                    "target_soma": candidate["target_soma"],
                }
            )

        print(f"  Rescue-Verbindungen akzeptiert: " f"{accepted_this_round}")

        if accepted_this_round == 0:
            break

    return (
        rescue_connections,
        report,
    )


# ---------------------------------------------------------------------
# Bereinigung
# ---------------------------------------------------------------------


def remove_small(
    mask: np.ndarray,
    minimum_size: int,
) -> np.ndarray:
    """
    Optionaler Größenfilter.

    Wert 0:
    nichts aufgrund der Größe entfernen.
    """
    if minimum_size <= 1:
        return mask.copy()

    labels, count = ndi.label(
        mask,
        structure=NEIGHBORHOOD,
    )

    if count == 0:
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    sizes = np.bincount(labels.ravel())

    keep_ids = np.flatnonzero(sizes >= minimum_size)

    keep_ids = keep_ids[keep_ids != 0]

    return np.isin(
        labels,
        keep_ids,
    )


def keep_connected_pairs(
    skeleton: np.ndarray,
    soma: np.ndarray,
    attach_radius: int,
    min_soma_area: int,
    min_skeleton_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nach dem Verbinden werden entfernt:

    - Soma ohne Skeletonkontakt
    - Skeletonkomponenten ohne Somakontakt
    """
    skeleton = remove_small(
        skeleton,
        min_skeleton_size,
    )

    soma = remove_small(
        soma,
        min_soma_area,
    )

    skeleton_labels, _ = ndi.label(
        skeleton,
        structure=NEIGHBORHOOD,
    )

    soma_labels, _ = ndi.label(
        soma,
        structure=NEIGHBORHOOD,
    )

    contacts = soma_contacts(
        skeleton_labels,
        soma_labels,
        attach_radius,
    )

    skeleton_ids = np.asarray(
        sorted(component_id for component_id, soma_ids in contacts.items() if soma_ids),
        dtype=np.int32,
    )

    if skeleton_ids.size == 0:
        return (
            np.zeros_like(
                skeleton,
                dtype=bool,
            ),
            np.zeros_like(
                soma,
                dtype=bool,
            ),
        )

    kept_skeleton = np.isin(
        skeleton_labels,
        skeleton_ids,
    )

    soma_ids_to_keep: set[int] = set()

    for skeleton_id in skeleton_ids:
        soma_ids_to_keep |= contacts[int(skeleton_id)]

    kept_soma = np.isin(
        soma_labels,
        np.asarray(
            sorted(soma_ids_to_keep),
            dtype=np.int32,
        ),
    )

    return (
        kept_skeleton,
        kept_soma,
    )


# ---------------------------------------------------------------------
# QC-PNG
# ---------------------------------------------------------------------


def save_qc(
    path: Path,
    original: Optional[np.ndarray],
    raw_segmentation: np.ndarray,
    final_segmentation: np.ndarray,
    added_connections: np.ndarray,
    removed_skeleton: np.ndarray,
    removed_soma: np.ndarray,
    max_side: int,
) -> None:
    """
    Das QC-Bild wird bei sehr großen Bildern verkleinert erzeugt.
    Die TIFF-Ausgaben bleiben in voller Auflösung.
    """
    step = max(
        1,
        math.ceil(max(raw_segmentation.shape) / max_side),
    )

    raw_small = raw_segmentation[
        ::step,
        ::step,
    ]

    final_small = final_segmentation[
        ::step,
        ::step,
    ]

    added_small = added_connections[
        ::step,
        ::step,
    ]

    removed_skeleton_small = removed_skeleton[
        ::step,
        ::step,
    ]

    removed_soma_small = removed_soma[
        ::step,
        ::step,
    ]

    if original is None:
        base = np.zeros(
            raw_small.shape,
            dtype=np.float32,
        )
    else:
        base = normalize(
            original[
                ::step,
                ::step,
            ]
        )

    raw_rgb = (
        np.repeat(
            base[..., None],
            3,
            axis=2,
        )
        * 0.4
    )

    raw_rgb[raw_small == 1] = [0.0, 1.0, 1.0]

    raw_rgb[raw_small == 2] = [1.0, 1.0, 1.0]

    final_rgb = (
        np.repeat(
            base[..., None],
            3,
            axis=2,
        )
        * 0.4
    )

    final_rgb[final_small == 1] = [0.0, 1.0, 0.0]

    final_rgb[final_small == 2] = [1.0, 0.0, 1.0]

    changes_rgb = (
        np.repeat(
            base[..., None],
            3,
            axis=2,
        )
        * 0.3
    )

    changes_rgb[added_small] = [0.0, 1.0, 0.0]

    changes_rgb[removed_skeleton_small] = [1.0, 0.0, 0.0]

    changes_rgb[removed_soma_small] = [1.0, 0.65, 0.0]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7),
    )

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Input")

    axes[1].imshow(final_rgb)
    axes[1].set_title("Final")

    axes[2].imshow(changes_rgb)
    axes[2].set_title(
        "Grün = verbunden | " "Rot = Skeleton entfernt | " "Orange = Soma entfernt"
    )

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Konservatives Postprocessing für " "0=Hintergrund, 1=Skeleton, 2=Soma."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--original",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-skeleton-gap",
        type=float,
        default=22,
    )

    parser.add_argument(
        "--max-soma-gap",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--max-angle",
        type=float,
        default=30,
    )

    parser.add_argument(
        "--tangent-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--attach-radius",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--connection-width",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min-soma-area",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--min-skeleton-size",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-rescue-gap",
        type=float,
        default=60,
        help=(
            "Maximale Distanz, über die ein freies Skeletonfragment "
            "mit einem bereits Soma-zugeordneten Skeleton verbunden wird."
        ),
    )

    parser.add_argument(
        "--max-rescue-angle",
        type=float,
        default=70,
        help=(
            "Maximaler Winkel für die Rettungsverbindung. "
            "Diese Stufe ist weniger streng als die erste Lückenschließung."
        ),
    )

    parser.add_argument(
        "--rescue-rounds",
        type=int,
        default=5,
        help=("Iterative Runden zur schrittweisen Anbindung " "freier Fragmente."),
    )

    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=8,
        help=(
            "Sind zwei unterschiedliche Soma im Score näher als "
            "dieser Wert, wird das Fragment nicht automatisch verbunden."
        ),
    )

    parser.add_argument(
        "--qc-max-side",
        type=int,
        default=2500,
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Eingaben prüfen
    # -------------------------------------------------------------

    input_path = args.input.resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    segmentation = read_2d(input_path).astype(np.uint8)

    values = set(int(value) for value in np.unique(segmentation))

    if not values.issubset({0, 1, 2}):
        raise RuntimeError(
            "Nur Labelwerte 0, 1 und 2 erlaubt. " f"Gefunden: {sorted(values)}"
        )

    original: Optional[np.ndarray] = None

    if args.original is not None:
        original_path = args.original.resolve()

        if not original_path.exists():
            raise FileNotFoundError(original_path)

        original = read_2d(original_path)

        if original.shape != segmentation.shape:
            raise RuntimeError(
                "Original und Segmentierung haben unterschiedliche Shapes: "
                f"{original.shape} und {segmentation.shape}"
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_skeleton = segmentation == 1
    raw_soma = segmentation == 2

    # -------------------------------------------------------------
    # Stufe 1:
    # Sichere kurze Skeleton-Skeleton- und Skeleton-Soma-Verbindungen
    # -------------------------------------------------------------

    strict_connections, report = connect_gaps(
        raw_skeleton=raw_skeleton,
        soma_mask=raw_soma,
        max_skeleton_gap=args.max_skeleton_gap,
        max_soma_gap=args.max_soma_gap,
        max_angle=args.max_angle,
        tangent_steps=args.tangent_steps,
        attach_radius=args.attach_radius,
        rounds=args.rounds,
        connection_width=args.connection_width,
    )

    skeleton_after_strict_connections = raw_skeleton | strict_connections

    # -------------------------------------------------------------
    # Stufe 2:
    # Freie Fragmente an bereits soma-verankerte Skeletons anbinden
    # -------------------------------------------------------------

    rescue_connections, rescue_report = rescue_unassigned_components(
        completed_skeleton=skeleton_after_strict_connections,
        soma_mask=raw_soma,
        max_rescue_gap=args.max_rescue_gap,
        max_rescue_angle=args.max_rescue_angle,
        ambiguity_margin=args.ambiguity_margin,
        tangent_steps=args.tangent_steps,
        attach_radius=args.attach_radius,
        rescue_rounds=args.rescue_rounds,
        connection_width=args.connection_width,
    )

    report.extend(rescue_report)

    added_connections = strict_connections | rescue_connections

    completed_skeleton = raw_skeleton | added_connections

    # -------------------------------------------------------------
    # Bereinigung:
    # Nur Soma-Skeleton-Paare behalten
    # -------------------------------------------------------------

    final_skeleton, final_soma = keep_connected_pairs(
        skeleton=completed_skeleton,
        soma=raw_soma,
        attach_radius=args.attach_radius,
        min_soma_area=args.min_soma_area,
        min_skeleton_size=args.min_skeleton_size,
    )

    removed_skeleton = completed_skeleton & ~final_skeleton

    removed_soma = raw_soma & ~final_soma

    final_segmentation = np.zeros_like(
        segmentation,
        dtype=np.uint8,
    )

    final_segmentation[final_skeleton] = 1

    # Soma überschreibt eine Verbindungslinie innerhalb des Somas.
    final_segmentation[final_soma] = 2

    # -------------------------------------------------------------
    # TIFF-Ausgaben
    # -------------------------------------------------------------

    tifffile.imwrite(
        output_dir / "01a_strict_connections.tif",
        strict_connections.astype(np.uint8),
        photometric="minisblack",
    )

    tifffile.imwrite(
        output_dir / "01b_rescue_connections.tif",
        rescue_connections.astype(np.uint8),
        photometric="minisblack",
    )

    tifffile.imwrite(
        output_dir / "01_added_connections.tif",
        added_connections.astype(np.uint8),
        photometric="minisblack",
    )

    tifffile.imwrite(
        output_dir / "02_completed_skeleton_before_cleanup.tif",
        completed_skeleton.astype(np.uint8),
        photometric="minisblack",
    )

    tifffile.imwrite(
        output_dir / "03_removed_skeleton.tif",
        removed_skeleton.astype(np.uint8),
        photometric="minisblack",
    )

    tifffile.imwrite(
        output_dir / "04_removed_soma.tif",
        removed_soma.astype(np.uint8),
        photometric="minisblack",
    )

    tifffile.imwrite(
        output_dir / "05_final_segmentation.tif",
        final_segmentation,
        photometric="minisblack",
    )

    # -------------------------------------------------------------
    # Verbindungsbericht
    # -------------------------------------------------------------

    report_fields = [
        "round",
        "type",
        "start_y",
        "start_x",
        "target_y",
        "target_x",
        "distance",
        "angle_start",
        "angle_target",
        "target_soma",
    ]

    with open(
        output_dir / "accepted_connections.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=report_fields,
        )

        writer.writeheader()
        writer.writerows(report)

    # -------------------------------------------------------------
    # QC-PNG
    # -------------------------------------------------------------

    save_qc(
        path=output_dir / "postprocessing_qc.png",
        original=original,
        raw_segmentation=segmentation,
        final_segmentation=final_segmentation,
        added_connections=added_connections,
        removed_skeleton=removed_skeleton,
        removed_soma=removed_soma,
        max_side=args.qc_max_side,
    )

    # -------------------------------------------------------------
    # Zusammenfassung
    # -------------------------------------------------------------

    print()
    print("Fertig")
    print(f"Akzeptierte Verbindungen: {len(report)}")
    print(f"Skeletonpixel roh:        {int(raw_skeleton.sum())}")
    print(f"Neu verbundene Pixel:     {int(added_connections.sum())}")
    print(f"Skeletonpixel final:      {int(final_skeleton.sum())}")
    print(f"Entfernte Skeletonpixel:  {int(removed_skeleton.sum())}")
    print(f"Somapixel roh:            {int(raw_soma.sum())}")
    print(f"Somapixel final:          {int(final_soma.sum())}")
    print(f"Entfernte Somapixel:      {int(removed_soma.sum())}")
    print(f"Ausgabeordner:            {output_dir}")


if __name__ == "__main__":
    main()
