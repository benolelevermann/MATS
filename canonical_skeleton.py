from __future__ import annotations

"""Canonical one-pixel centerlines for the MATS soma/skeleton pipeline."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.morphology import skeletonize


CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class CanonicalizationReport:
    input_pixels: int
    output_pixels: int
    pixels_removed_by_thinning: int
    soma_gap_pixels_added: int
    components_before: int
    components_after: int
    detached_components_before: int
    detached_components_after: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _component_count(mask: np.ndarray) -> int:
    return int(ndi.label(mask, structure=CONNECTIVITY_8)[1])


def detached_component_count(skeleton: np.ndarray, soma: np.ndarray) -> int:
    skeleton = np.asarray(skeleton, dtype=bool)
    soma = np.asarray(soma, dtype=bool)
    labels, count = ndi.label(skeleton, structure=CONNECTIVITY_8)
    if count == 0:
        return 0
    adjacent_to_soma = ndi.binary_dilation(soma, structure=CONNECTIVITY_8)
    return sum(
        not np.any((labels == component_id) & adjacent_to_soma)
        for component_id in range(1, count + 1)
    )


def _bridge_short_soma_gaps(
    skeleton: np.ndarray,
    soma: np.ndarray,
    max_soma_gap_px: float,
) -> tuple[np.ndarray, int]:
    """Make short soma attachments explicit in the mask instead of hiding them in SWC edges."""

    result = np.asarray(skeleton, dtype=bool).copy()
    soma = np.asarray(soma, dtype=bool)
    if not result.any() or not soma.any() or max_soma_gap_px <= 0:
        return result, 0

    labels, count = ndi.label(result, structure=CONNECTIVITY_8)
    adjacent_to_soma = ndi.binary_dilation(soma, structure=CONNECTIVITY_8)
    distance, nearest_soma = ndi.distance_transform_edt(
        ~soma,
        return_indices=True,
    )
    added = np.zeros(result.shape, dtype=bool)
    for component_id in range(1, count + 1):
        component = labels == component_id
        if np.any(component & adjacent_to_soma):
            continue
        coordinates = np.argwhere(component)
        component_distances = distance[component]
        nearest_index = int(np.argmin(component_distances))
        gap_distance = float(component_distances[nearest_index])
        if gap_distance > max_soma_gap_px:
            continue
        y, x = map(int, coordinates[nearest_index])
        soma_y = int(nearest_soma[0, y, x])
        soma_x = int(nearest_soma[1, y, x])
        rows, columns = line(y, x, soma_y, soma_x)
        added[rows, columns] = True

    added &= ~soma
    added &= ~result
    result |= added
    return result, int(added.sum())


def canonicalize_skeleton(
    skeleton: np.ndarray,
    soma: np.ndarray | None = None,
    max_soma_gap_px: float = 0.0,
) -> tuple[np.ndarray, CanonicalizationReport]:
    """Return one canonical 1-px mask and an auditable transformation report."""

    source = np.asarray(skeleton, dtype=bool)
    soma_mask = (
        np.zeros(source.shape, dtype=bool)
        if soma is None
        else np.asarray(soma, dtype=bool)
    )
    if source.shape != soma_mask.shape:
        raise ValueError("Skeleton and soma masks must have the same shape.")

    source = source & ~soma_mask
    components_before = _component_count(source)
    detached_before = detached_component_count(source, soma_mask)
    thin = skeletonize(source)
    pixels_removed_by_thinning = max(0, int(source.sum()) - int(thin.sum()))
    thin, gap_pixels = _bridge_short_soma_gaps(
        thin,
        soma_mask,
        max_soma_gap_px=max_soma_gap_px,
    )
    thin = skeletonize(thin) & ~soma_mask
    report = CanonicalizationReport(
        input_pixels=int(source.sum()),
        output_pixels=int(thin.sum()),
        pixels_removed_by_thinning=pixels_removed_by_thinning,
        soma_gap_pixels_added=gap_pixels,
        components_before=components_before,
        components_after=_component_count(thin),
        detached_components_before=detached_before,
        detached_components_after=detached_component_count(thin, soma_mask),
    )
    return thin, report


def canonicalize_semantic(
    semantic: np.ndarray,
    max_soma_gap_px: float = 0.0,
) -> tuple[np.ndarray, CanonicalizationReport]:
    semantic = np.asarray(semantic)
    if semantic.ndim != 2:
        raise ValueError(f"Expected a 2-D semantic map, got {semantic.shape}.")
    values = set(np.unique(semantic).astype(int).tolist())
    if not values <= {0, 1, 2}:
        raise ValueError(f"Semantic map contains unsupported labels: {sorted(values)}")
    soma = semantic == 2
    skeleton, report = canonicalize_skeleton(
        semantic == 1,
        soma=soma,
        max_soma_gap_px=max_soma_gap_px,
    )
    canonical = np.zeros(semantic.shape, dtype=np.uint8)
    canonical[skeleton] = 1
    canonical[soma] = 2
    return canonical, report


def is_canonical_one_pixel(skeleton: np.ndarray) -> bool:
    mask = np.asarray(skeleton, dtype=bool)
    return bool(np.array_equal(mask, skeletonize(mask)))
