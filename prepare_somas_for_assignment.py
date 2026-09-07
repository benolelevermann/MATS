from __future__ import annotations

"""Prepare soma seeds for cell assignment without deleting skeleton material.

This is a conservative, raster implementation of the soma-reconstruction
principle used in Zehtabian et al. (2022): fill enclosed holes in each predicted
soma and reject tiny soma components as invalid cell seeds.

It deliberately does not alter class-1 skeleton pixels. Small / unassigned
skeleton fragments are reported as debris candidates, but remain present in the
material-preserving semantic segmentation for downstream QC.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


SCRIPT_VERSION = "soma-fill-filter-no-skeleton-loss-v1-2026-08-12"
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill enclosed soma holes and exclude tiny soma components from "
            "cell assignment without deleting any skeleton material."
        )
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Hard 2-D 0/1/2 segmentation from the skeleton-recall model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory.",
    )
    parser.add_argument(
        "--original",
        type=Path,
        help="Optional same-size raw overview for a QC overlay.",
    )
    parser.add_argument(
        "--min-soma-area",
        type=int,
        default=80,
        help="Minimum retained soma area in pixels (default: 80).",
    )
    parser.add_argument(
        "--max-fill-hole-area",
        type=int,
        default=2500,
        help=(
            "Maximum enclosed hole area filled in one soma component in pixels "
            "(default: 2500)."
        ),
    )
    parser.add_argument(
        "--max-fill-ratio",
        type=float,
        default=1.5,
        help=(
            "Maximum filled-hole area divided by original soma area "
            "(default: 1.5)."
        ),
    )
    parser.add_argument(
        "--debris-max-skeleton-pixels",
        type=int,
        default=8,
        help=(
            "Skeleton components up to this size are marked as debris candidates "
            "for QC only, never deleted (default: 8)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def read_2d(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be 2-D; got {array.shape}: {path}")
    return array


def require_valid_semantic(segmentation: np.ndarray) -> None:
    invalid = set(np.unique(segmentation).astype(int).tolist()) - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Expected only 0/1/2 labels, found: {sorted(invalid)}")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {path}\n"
            "Use a new directory or pass --overwrite deliberately."
        )
    path.mkdir(parents=True, exist_ok=True)


def normalize_u8(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    return np.round(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def save_qc(path: Path, original: np.ndarray | None, skeleton: np.ndarray,
            kept_soma: np.ndarray, rejected_soma: np.ndarray, added_soma: np.ndarray) -> None:
    if original is None:
        gray = np.zeros(skeleton.shape, dtype=np.uint8)
    else:
        gray = normalize_u8(original)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    # Preserve skeleton in cyan only for visualization.
    rgb[skeleton] = 0.30 * rgb[skeleton] + 0.70 * np.asarray([0, 220, 235])
    rgb[kept_soma] = 0.10 * rgb[kept_soma] + 0.90 * np.asarray([245, 60, 180])
    rgb[added_soma] = 0.15 * rgb[added_soma] + 0.85 * np.asarray([100, 245, 110])
    rgb[rejected_soma] = 0.15 * rgb[rejected_soma] + 0.85 * np.asarray([255, 145, 0])
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if args.min_soma_area < 1:
        raise ValueError("--min-soma-area must be >= 1")
    if args.max_fill_hole_area < 0:
        raise ValueError("--max-fill-hole-area must be >= 0")
    if args.max_fill_ratio < 0:
        raise ValueError("--max-fill-ratio must be >= 0")
    if args.debris_max_skeleton_pixels < 0:
        raise ValueError("--debris-max-skeleton-pixels must be >= 0")

    segmentation = read_2d(args.prediction, "Prediction")
    require_valid_semantic(segmentation)
    original = read_2d(args.original, "Original image") if args.original else None
    if original is not None and original.shape != segmentation.shape:
        raise RuntimeError(
            f"Shape mismatch: prediction={segmentation.shape}, original={original.shape}"
        )
    prepare_output(args.output_dir, args.overwrite)

    skeleton = segmentation == 1
    raw_soma = segmentation == 2
    raw_labels, raw_count = ndi.label(raw_soma, structure=CONNECTIVITY_8)
    raw_sizes = np.bincount(raw_labels.ravel(), minlength=raw_count + 1)

    kept_soma = np.zeros(raw_soma.shape, dtype=bool)
    rejected_soma = np.zeros(raw_soma.shape, dtype=bool)
    added_soma = np.zeros(raw_soma.shape, dtype=bool)
    component_rows: list[dict[str, object]] = []

    for component_id in range(1, raw_count + 1):
        component = raw_labels == component_id
        original_area = int(raw_sizes[component_id])
        if original_area < args.min_soma_area:
            rejected_soma |= component
            component_rows.append(
                {
                    "raw_soma_id": component_id,
                    "original_area_px": original_area,
                    "filled_area_px": original_area,
                    "added_hole_pixels": 0,
                    "kept_as_soma_seed": 0,
                    "reason": "below_min_soma_area",
                }
            )
            continue

        filled = ndi.binary_fill_holes(component)
        holes = filled & ~component
        hole_area = int(holes.sum())
        fill_allowed = (
            hole_area <= args.max_fill_hole_area
            and hole_area <= args.max_fill_ratio * max(original_area, 1)
        )
        # A predicted soma and skeleton are mutually exclusive classes. A hole
        # inside a soma can nevertheless contain class-1 material (for example
        # when a neurite crosses the predicted soma). Keep that material as
        # skeleton rather than overwriting it with the soma fill.
        final_component = (filled if fill_allowed else component) & ~skeleton
        kept_soma |= final_component
        if fill_allowed:
            added_soma |= holes
        component_rows.append(
            {
                "raw_soma_id": component_id,
                "original_area_px": original_area,
                "filled_area_px": int(final_component.sum()),
                "added_hole_pixels": hole_area if fill_allowed else 0,
                "kept_as_soma_seed": 1,
                "reason": "filled_holes" if hole_area and fill_allowed else (
                    "hole_too_large_not_filled" if hole_area else "kept"
                ),
            }
        )

    # Skeleton remains bit-identical to the input segmentation. Small isolated
    # components are merely marked to make cleanup visible in QC.
    skel_labels, skel_count = ndi.label(skeleton, structure=CONNECTIVITY_8)
    skel_sizes = np.bincount(skel_labels.ravel(), minlength=skel_count + 1)
    debris_ids = [
        i for i in range(1, skel_count + 1)
        if int(skel_sizes[i]) <= args.debris_max_skeleton_pixels
    ]
    debris_candidates = np.isin(skel_labels, debris_ids)

    semantic = np.zeros(segmentation.shape, dtype=np.uint8)
    semantic[skeleton] = 1
    semantic[kept_soma] = 2
    if not np.array_equal(semantic == 1, skeleton):
        raise AssertionError("Skeleton material was unexpectedly changed.")

    kept_instances, kept_count = ndi.label(kept_soma, structure=CONNECTIVITY_8)
    tifffile.imwrite(args.output_dir / "01_input_prediction_0-1-2.tif", segmentation.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "02_soma_filled_kept_mask.tif", kept_soma.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "03_rejected_small_soma_mask.tif", rejected_soma.astype(np.uint8))
    tifffile.imwrite(
        args.output_dir / "04_semantic_soma_filled_and_filtered_0-1-2.tif", semantic
    )
    tifffile.imwrite(
        args.output_dir / "05_skeleton_debris_candidates_NOT_REMOVED.tif",
        debris_candidates.astype(np.uint8),
    )
    tifffile.imwrite(
        args.output_dir / "06_kept_soma_instances.tif",
        kept_instances.astype(np.uint16 if kept_count <= 65535 else np.uint32),
    )
    save_qc(
        args.output_dir / "07_soma_preprocessing_qc.png",
        original, skeleton, kept_soma, rejected_soma, added_soma,
    )

    with (args.output_dir / "soma_component_report.csv").open("w", newline="", encoding="utf-8") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=list(component_rows[0].keys()) if component_rows else [
            "raw_soma_id", "original_area_px", "filled_area_px", "added_hole_pixels", "kept_as_soma_seed", "reason"
        ])
        writer.writeheader()
        writer.writerows(component_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "prediction": str(args.prediction.resolve()),
        "original": str(args.original.resolve()) if args.original else None,
        "shape": list(segmentation.shape),
        "input_skeleton_pixels": int(skeleton.sum()),
        "output_skeleton_pixels": int((semantic == 1).sum()),
        "removed_skeleton_pixels": 0,
        "input_soma_components": int(raw_count),
        "kept_soma_components": int(kept_count),
        "rejected_small_soma_components": int(sum(row["kept_as_soma_seed"] == 0 for row in component_rows)),
        "filled_soma_pixels_added": int(added_soma.sum()),
        "skeleton_debris_candidates_NOT_REMOVED_pixels": int(debris_candidates.sum()),
        "parameters": vars(args) | {"prediction": str(args.prediction), "original": str(args.original) if args.original else None, "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "_script_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("SOMA PREPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Input skeleton pixels:        {int(skeleton.sum())}")
    print(f"Output skeleton pixels:       {int((semantic == 1).sum())}")
    print("REMOVED skeleton pixels:      0")
    print(f"Kept soma instances:          {int(kept_count)}")
    print(f"Rejected small soma instances:{int(sum(row['kept_as_soma_seed'] == 0 for row in component_rows))}")
    print(f"Added filled soma pixels:     {int(added_soma.sum())}")
    print(f"Output semantic: {args.output_dir / '04_semantic_soma_filled_and_filtered_0-1-2.tif'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
