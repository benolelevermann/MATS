from __future__ import annotations

"""Keep one crop candidate per ambiguous multi-cell overlap group.

This script is deliberately a *selection layer*, not a segmentation editor.
It never changes the input 0/1/2 segmentation or assigns ambiguous pixels to a
cell. Instead it keeps ordinary safe cells and, for every overlap group, keeps
one deterministic, usable representative cell for crop export. Competing cells
remain visible in the QC outputs and report.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


SCRIPT_VERSION = "overlap-representative-selection-v1-2026-08-12"
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one exportable cell per ambiguous overlap group without "
            "modifying the input skeleton/soma segmentation."
        )
    )
    parser.add_argument("--semantic", type=Path, required=True,
                        help="Completed 2-D semantic TIFF: 0 background, 1 skeleton, 2 soma.")
    parser.add_argument("--assigned-instances", type=Path, required=True,
                        help="01_cell_instances_assigned.tif from one assignment method.")
    parser.add_argument("--safe-instances", type=Path, required=True,
                        help="02_cell_instances_safe.tif from the same assignment method.")
    parser.add_argument("--ambiguous-regions", type=Path, required=True,
                        help="05_ambiguous_regions_instances.tif from the same method.")
    parser.add_argument("--status-map", type=Path, required=True,
                        help="08_assignment_status_map.tif from the same method.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path,
                        help="Optional matching raw overview used only as a QC background.")
    parser.add_argument("--cell-report", type=Path,
                        help="Optional cell_report.csv; ambiguity fraction improves ranking.")
    parser.add_argument("--contact-radius", type=int, default=8,
                        help="Pixels used to find cells bordering one ambiguous region (default: 8).")
    parser.add_argument("--min-skeleton-pixels", type=int, default=12,
                        help="Minimum assigned skeleton pixels for an overlap-group winner (default: 12).")
    parser.add_argument("--border-margin", type=int, default=2,
                        help="Exclude winner candidates touching this overview border (default: 2).")
    parser.add_argument("--qc-max-size", type=int, default=2200,
                        help="Maximum width or height of the QC PNG (default: 2200).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an existing output directory.")
    return parser.parse_args()


def read_2d(path: Path, description: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{description} must be 2-D; got {array.shape}: {path}")
    return array


def validate_shape(reference: np.ndarray, other: np.ndarray, description: str) -> None:
    if reference.shape != other.shape:
        raise RuntimeError(
            f"Shape mismatch for {description}: expected {reference.shape}, got {other.shape}"
        )


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {path}\n"
            "Use a new directory or pass --overwrite deliberately."
        )
    path.mkdir(parents=True, exist_ok=True)


def label_dtype(maximum: int) -> np.dtype:
    return np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32


def touches_border(mask: np.ndarray, margin: int) -> bool:
    if not np.any(mask):
        return False
    height, width = mask.shape
    rows, cols = np.where(mask)
    return bool(
        rows.min() < margin or cols.min() < margin
        or rows.max() >= height - margin or cols.max() >= width - margin
    )


def read_ambiguity_fraction(path: Path | None) -> dict[int, float]:
    if path is None or not path.is_file():
        return {}
    result: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                soma_id = int(float(row.get("soma_id", "")))
                fraction = float(row.get("ambiguous_fraction", "0"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(fraction):
                result[soma_id] = max(0.0, min(1.0, fraction))
    return result


def merge_candidate_sets(records: list[tuple[int, set[int]]]) -> list[dict[str, set[int]]]:
    """Merge ambiguous regions that touch at least one common candidate cell."""
    groups: list[dict[str, set[int]]] = []
    for region_id, candidates in records:
        matches = [i for i, group in enumerate(groups) if group["candidates"] & candidates]
        if not matches:
            groups.append({"regions": {region_id}, "candidates": set(candidates)})
            continue
        merged_regions = {region_id}
        merged_candidates = set(candidates)
        for index in reversed(matches):
            group = groups.pop(index)
            merged_regions |= group["regions"]
            merged_candidates |= group["candidates"]
        groups.append({"regions": merged_regions, "candidates": merged_candidates})
    return groups


def normalize_u8(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99.5])
    high = max(high, low + 1e-6)
    return np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def save_qc(
    path: Path,
    original: np.ndarray | None,
    semantic: np.ndarray,
    representatives: np.ndarray,
    selected_winners: np.ndarray,
    excluded_competitors: np.ndarray,
    unresolved_ambiguity: np.ndarray,
    max_size: int,
) -> None:
    step = max(1, int(np.ceil(max(semantic.shape) / max(max_size, 1))))
    semantic = semantic[::step, ::step]
    representatives = representatives[::step, ::step]
    selected_winners = selected_winners[::step, ::step]
    excluded_competitors = excluded_competitors[::step, ::step]
    unresolved_ambiguity = unresolved_ambiguity[::step, ::step]
    if original is None:
        gray = np.zeros(semantic.shape, dtype=np.uint8)
    else:
        gray = normalize_u8(original[::step, ::step])

    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    skeleton = semantic == 1
    soma = semantic == 2
    rgb[skeleton] = 0.55 * rgb[skeleton] + 0.45 * np.asarray([175, 175, 175])
    rgb[soma] = 0.45 * rgb[soma] + 0.55 * np.asarray([220, 85, 190])

    maximum_id = int(representatives.max())
    rng = np.random.default_rng(20260812)
    colors = np.zeros((maximum_id + 1, 3), dtype=np.uint8)
    if maximum_id:
        colors[1:] = rng.integers(70, 255, size=(maximum_id, 3), dtype=np.uint8)
    selected = representatives > 0
    rgb[selected] = 0.24 * rgb[selected] + 0.76 * colors[representatives[selected]]
    rgb[selected_winners > 0] = 0.12 * rgb[selected_winners > 0] + 0.88 * np.asarray([40, 155, 255])
    rgb[excluded_competitors > 0] = 0.15 * rgb[excluded_competitors > 0] + 0.85 * np.asarray([255, 140, 0])
    rgb[unresolved_ambiguity] = 0.15 * rgb[unresolved_ambiguity] + 0.85 * np.asarray([0, 235, 235])
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if args.contact_radius < 1:
        raise ValueError("--contact-radius must be >= 1")
    if args.min_skeleton_pixels < 1:
        raise ValueError("--min-skeleton-pixels must be >= 1")
    if args.border_margin < 0:
        raise ValueError("--border-margin must be >= 0")

    semantic = read_2d(args.semantic, "Semantic segmentation")
    valid = set(np.unique(semantic).astype(int).tolist()) <= {0, 1, 2}
    if not valid:
        raise RuntimeError("Semantic segmentation must contain only classes 0, 1, and 2.")
    assigned = read_2d(args.assigned_instances, "Assigned instances")
    safe = read_2d(args.safe_instances, "Safe instances")
    ambiguous_regions = read_2d(args.ambiguous_regions, "Ambiguous regions")
    status = read_2d(args.status_map, "Assignment status")
    for array, name in ((assigned, "assigned instances"), (safe, "safe instances"),
                        (ambiguous_regions, "ambiguous regions"), (status, "status map")):
        validate_shape(semantic, array, name)
    original = read_2d(args.original, "Original image") if args.original else None
    if original is not None:
        validate_shape(semantic, original, "original image")
    prepare_output(args.output_dir, args.overwrite)

    material = semantic > 0
    if np.any((assigned > 0) & ~material):
        raise RuntimeError("Assigned instance pixels exist outside the semantic skeleton/soma material.")
    if np.any((safe > 0) & (assigned == 0)):
        raise RuntimeError("Safe instances must be a subset of assigned instances.")

    if not np.any(ambiguous_regions):
        ambiguous_regions, _ = ndi.label(status == 2, structure=CONNECTIVITY_8)
    region_ids = np.unique(ambiguous_regions)
    region_ids = region_ids[region_ids > 0]
    contact_structure = ndi.generate_binary_structure(2, 2)
    region_candidate_records: list[tuple[int, set[int]]] = []
    unresolved_region_ids: set[int] = set()
    for region_id in region_ids.astype(int):
        region = ambiguous_regions == region_id
        neighbourhood = ndi.binary_dilation(
            region, structure=contact_structure, iterations=args.contact_radius
        )
        candidates = set(np.unique(assigned[neighbourhood]).astype(int).tolist()) - {0}
        if len(candidates) >= 2:
            region_candidate_records.append((region_id, candidates))
        else:
            unresolved_region_ids.add(region_id)

    groups = merge_candidate_sets(region_candidate_records)
    report_quality = read_ambiguity_fraction(args.cell_report)
    candidate_cache: dict[int, dict[str, object]] = {}
    selected_winner_ids: set[int] = set()
    excluded_ids: set[int] = set()
    group_labels = np.zeros(semantic.shape, dtype=np.uint32)
    report_rows: list[dict[str, object]] = []

    for group_number, group in enumerate(groups, start=1):
        candidate_ids = sorted(group["candidates"])
        candidate_records: list[dict[str, object]] = []
        for cell_id in candidate_ids:
            if cell_id not in candidate_cache:
                mask = assigned == cell_id
                skeleton_pixels = int(np.count_nonzero(mask & (semantic == 1)))
                soma_pixels = int(np.count_nonzero(mask & (semantic == 2)))
                candidate_cache[cell_id] = {
                    "cell_id": cell_id,
                    "mask": mask,
                    "skeleton_pixels": skeleton_pixels,
                    "soma_pixels": soma_pixels,
                    "touches_border": touches_border(mask, args.border_margin),
                    "ambiguity_fraction": report_quality.get(cell_id, 0.0),
                }
            candidate_records.append(candidate_cache[cell_id])

        eligible = [
            record for record in candidate_records
            if int(record["skeleton_pixels"]) >= args.min_skeleton_pixels
            and int(record["soma_pixels"]) > 0
            and not bool(record["touches_border"])
        ]
        winner_id: int | None = None
        if eligible:
            def rank(record: dict[str, object]) -> tuple[float, int, int, int]:
                quality = 1.0 - float(record["ambiguity_fraction"])
                score = int(record["skeleton_pixels"]) * (0.2 + 0.8 * quality) + 0.05 * int(record["soma_pixels"])
                return (score, int(record["skeleton_pixels"]), int(record["soma_pixels"]), -int(record["cell_id"]))
            winner = max(eligible, key=rank)
            winner_id = int(winner["cell_id"])
            selected_winner_ids.add(winner_id)
            excluded_ids.update(set(candidate_ids) - {winner_id})
        else:
            unresolved_region_ids.update(group["regions"])
            excluded_ids.update(candidate_ids)

        for region_id in group["regions"]:
            group_labels[ambiguous_regions == region_id] = group_number
        report_rows.append(
            {
                "overlap_group": group_number,
                "ambiguous_region_ids": ";".join(map(str, sorted(group["regions"]))),
                "candidate_cell_ids": ";".join(map(str, candidate_ids)),
                "selected_cell_id": winner_id if winner_id is not None else "",
                "selection_reason": "largest_eligible_assigned_cell" if winner_id is not None else "no_eligible_candidate",
                "candidate_count": len(candidate_ids),
            }
        )

    safe_ids = set(np.unique(safe).astype(int).tolist()) - {0}
    group_candidate_ids = set().union(*(group["candidates"] for group in groups)) if groups else set()
    ordinary_safe_ids = safe_ids - group_candidate_ids
    final_ids = ordinary_safe_ids | selected_winner_ids

    representatives = np.where(np.isin(assigned, list(final_ids)), assigned, 0)
    winners = np.where(np.isin(assigned, list(selected_winner_ids)), assigned, 0)
    excluded = np.where(np.isin(assigned, list(excluded_ids)), assigned, 0)
    unresolved = np.isin(ambiguous_regions, list(unresolved_region_ids))
    max_id = int(max(assigned.max(), safe.max(), 1))
    dtype = label_dtype(max_id)
    representatives = representatives.astype(dtype, copy=False)
    winners = winners.astype(dtype, copy=False)
    excluded = excluded.astype(dtype, copy=False)

    if np.any((representatives > 0) & (assigned == 0)):
        raise AssertionError("Representative selection created material outside the assigned instance map.")
    if np.any((representatives > 0) & ~material):
        raise AssertionError("Representative selection created material outside semantic material.")

    selection_status = np.zeros(semantic.shape, dtype=np.uint8)
    selection_status[representatives > 0] = 1
    selection_status[winners > 0] = 2
    selection_status[excluded > 0] = 3
    selection_status[unresolved] = 4

    tifffile.imwrite(args.output_dir / "01_representative_cells_safe.tif", representatives)
    tifffile.imwrite(args.output_dir / "02_overlap_group_labels.tif", group_labels)
    tifffile.imwrite(args.output_dir / "03_selected_overlap_winners.tif", winners)
    tifffile.imwrite(args.output_dir / "04_excluded_overlap_competitors.tif", excluded)
    tifffile.imwrite(args.output_dir / "05_unresolved_ambiguous_regions.tif", unresolved.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "06_crop_selection_status_map.tif", selection_status)
    save_qc(
        args.output_dir / "07_qc_overlap_representatives.png",
        original, semantic, representatives, winners, excluded, unresolved, args.qc_max_size,
    )

    report_columns = [
        "overlap_group", "ambiguous_region_ids", "candidate_cell_ids", "selected_cell_id",
        "selection_reason", "candidate_count",
    ]
    with (args.output_dir / "overlap_group_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=report_columns)
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "semantic": str(args.semantic.resolve()),
        "assigned_instances": str(args.assigned_instances.resolve()),
        "safe_instances": str(args.safe_instances.resolve()),
        "ambiguous_regions": str(args.ambiguous_regions.resolve()),
        "input_skeleton_pixels": int(np.count_nonzero(semantic == 1)),
        "selected_cells": int(len(final_ids)),
        "ordinary_safe_cells": int(len(ordinary_safe_ids)),
        "overlap_groups": int(len(groups)),
        "overlap_group_winners": int(len(selected_winner_ids)),
        "excluded_competing_cells": int(len(excluded_ids)),
        "unresolved_ambiguous_regions": int(len(unresolved_region_ids)),
        "note": (
            "This is a crop-selection map. It does not alter the input semantic "
            "segmentation and deliberately does not assign ambiguous pixels to a cell."
        ),
        "parameters": {
            "contact_radius": args.contact_radius,
            "min_skeleton_pixels": args.min_skeleton_pixels,
            "border_margin": args.border_margin,
        },
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "_script_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("OVERLAP REPRESENTATIVE SELECTION COMPLETE")
    print("=" * 72)
    print(f"Input skeleton pixels (unchanged upstream): {int(np.count_nonzero(semantic == 1))}")
    print(f"Ordinary safe cells retained:              {len(ordinary_safe_ids)}")
    print(f"Overlap groups:                            {len(groups)}")
    print(f"One representative selected per group:     {len(selected_winner_ids)}")
    print(f"Unresolved ambiguous regions:              {len(unresolved_region_ids)}")
    print(f"Crop selection: {args.output_dir / '01_representative_cells_safe.tif'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
