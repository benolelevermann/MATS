from __future__ import annotations

"""Prepare soma seeds without ever removing skeleton material.

This script implements a conservative soma-preparation step inspired by the
local soma reconstruction idea in Zehtabian et al.: a predicted soma can fill
enclosed holes and recover *locally connected*, high-intensity pixels. It does
not globally threshold the image, does not grow through skeleton pixels, and
does not silently split large/merged soma candidates. Those are written to a
separate review mask instead.
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


SCRIPT_VERSION = "paper-guided-soma-preparation-v1-2026-08-12"
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare robust soma seeds while preserving every original class-1 "
            "skeleton pixel."
        )
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--polarity", choices=("auto", "bright", "dark"), default="bright")
    parser.add_argument("--min-soma-area", type=int, default=80)
    parser.add_argument("--max-fill-hole-area", type=int, default=2500)
    parser.add_argument("--max-fill-ratio", type=float, default=1.5)
    parser.add_argument(
        "--recovery-radius", type=float, default=5.0,
        help="Maximum local recovery distance around one predicted soma (default: 5 px).",
    )
    parser.add_argument(
        "--recovery-min-support", type=float, default=0.62,
        help="Minimum raw-image support for new soma pixels (default: 0.62).",
    )
    parser.add_argument(
        "--recovery-quantile", type=float, default=0.10,
        help="Per-soma lower support quantile allowed for recovery (default: 0.10).",
    )
    parser.add_argument(
        "--max-recovery-ratio", type=float, default=0.35,
        help="New local soma pixels may be at most this fraction of the original area (default: 0.35).",
    )
    parser.add_argument(
        "--review-large-soma-area", type=int, default=12000,
        help="Large retained soma components are flagged for review, not removed (default: 12000).",
    )
    parser.add_argument("--qc-max-size", type=int, default=2200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_2d(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be 2-D after squeeze; got {array.shape}")
    return array


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {path}\n"
            "Use a new test directory or pass --overwrite deliberately."
        )
    path.mkdir(parents=True, exist_ok=True)


def robust_unit_scale(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("Original image contains no finite values.")
    sample = finite[::max(1, finite.size // 1_000_000)]
    low, high = np.percentile(sample, [1.0, 99.5])
    if high <= low:
        low, high = float(np.min(sample)), float(np.max(sample))
    if high <= low:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def choose_polarity(normalized: np.ndarray, skeleton: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested
    if np.any(skeleton):
        foreground = normalized[skeleton]
        background = normalized[~skeleton]
        if foreground.size and background.size:
            return "bright" if float(np.median(foreground)) >= float(np.median(background)) else "dark"
    return "bright"


def image_support(original: np.ndarray, skeleton: np.ndarray, polarity: str) -> tuple[np.ndarray, str]:
    normalized = robust_unit_scale(original)
    selected = choose_polarity(normalized, skeleton, polarity)
    direct = normalized if selected == "bright" else 1.0 - normalized
    local = direct - ndi.gaussian_filter(direct, sigma=2.0, mode="reflect")
    support = np.clip(0.70 * direct + 0.30 * np.clip(0.5 + 2.0 * local, 0.0, 1.0), 0.0, 1.0)
    return support.astype(np.float32), selected


def one_component_connected_to(core: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(allowed, structure=STRUCTURE_8)
    if count == 0 or not np.any(core):
        return core.copy()
    identifiers = np.unique(labels[core])
    identifiers = identifiers[identifiers > 0]
    if identifiers.size == 0:
        return core.copy()
    return np.isin(labels, identifiers)


def output_dtype(max_identifier: int) -> np.dtype:
    return np.uint16 if max_identifier <= np.iinfo(np.uint16).max else np.uint32


def save_qc(
    path: Path,
    original: np.ndarray,
    skeleton: np.ndarray,
    kept: np.ndarray,
    filled: np.ndarray,
    recovered: np.ndarray,
    rejected: np.ndarray,
    review_large: np.ndarray,
    max_size: int,
) -> None:
    height, width = original.shape
    stride = max(1, int(np.ceil(max(height, width) / max(max_size, 1))))
    gray = robust_unit_scale(original[::stride, ::stride])
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    layers = [
        (skeleton[::stride, ::stride], np.asarray([0, 220, 235]), 0.75),
        (kept[::stride, ::stride], np.asarray([245, 60, 180]), 0.75),
        (filled[::stride, ::stride], np.asarray([100, 245, 100]), 0.85),
        (recovered[::stride, ::stride], np.asarray([255, 225, 0]), 0.90),
        (rejected[::stride, ::stride], np.asarray([255, 125, 0]), 0.82),
        (review_large[::stride, ::stride], np.asarray([175, 110, 255]), 0.60),
    ]
    for mask, color, alpha in layers:
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if args.min_soma_area < 1 or args.max_fill_hole_area < 0:
        raise ValueError("Invalid soma area setting.")
    if args.recovery_radius < 0 or args.max_recovery_ratio < 0:
        raise ValueError("Recovery radius and maximum recovery ratio must be non-negative.")
    if not 0 <= args.recovery_min_support <= 1 or not 0 <= args.recovery_quantile <= 1:
        raise ValueError("Recovery support values must be between 0 and 1.")

    prepare_output(args.output_dir, args.overwrite)
    segmentation = read_2d(args.prediction, "prediction")
    original = read_2d(args.original, "original")
    if segmentation.shape != original.shape:
        raise RuntimeError(f"Shape mismatch: prediction={segmentation.shape}, original={original.shape}")
    invalid = set(np.unique(segmentation).astype(int).tolist()) - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Expected semantic labels 0/1/2; found {sorted(invalid)}")

    skeleton = segmentation == 1
    raw_soma = segmentation == 2
    support, selected_polarity = image_support(original, skeleton, args.polarity)
    raw_instances, count = ndi.label(raw_soma, structure=STRUCTURE_8)
    sizes = np.bincount(raw_instances.ravel(), minlength=count + 1)

    kept = np.zeros(segmentation.shape, dtype=bool)
    rejected = np.zeros(segmentation.shape, dtype=bool)
    fill_added = np.zeros(segmentation.shape, dtype=bool)
    recovery_added = np.zeros(segmentation.shape, dtype=bool)
    review_large = np.zeros(segmentation.shape, dtype=bool)
    rows: list[dict[str, object]] = []

    for identifier in range(1, count + 1):
        component = raw_instances == identifier
        original_area = int(sizes[identifier])
        if original_area < args.min_soma_area:
            rejected |= component
            rows.append({
                "raw_soma_id": identifier, "original_area_px": original_area,
                "final_area_px": 0, "filled_hole_pixels": 0,
                "intensity_recovered_pixels": 0, "kept_as_seed": 0,
                "flagged_large_for_review": 0, "reason": "below_min_soma_area",
            })
            continue

        other_somas = raw_soma & ~component
        holes_filled = ndi.binary_fill_holes(component)
        holes = holes_filled & ~component & ~skeleton
        hole_area = int(holes.sum())
        fill_ok = hole_area <= args.max_fill_hole_area and hole_area <= args.max_fill_ratio * original_area
        core = (holes_filled if fill_ok else component) & ~skeleton & ~other_somas
        if fill_ok:
            fill_added |= core & ~component

        if args.recovery_radius > 0:
            threshold = max(
                args.recovery_min_support,
                float(np.quantile(support[component], args.recovery_quantile)),
            )
            nearby = ndi.distance_transform_edt(~core) <= args.recovery_radius
            candidates = nearby & (support >= threshold) & ~skeleton & ~other_somas
            # Do not let two nearby predicted soma seeds grow into each other.
            # A potential merged soma is kept and flagged separately rather than
            # being created by the recovery step.
            if np.any(other_somas):
                candidates &= ndi.distance_transform_edt(~other_somas) > args.recovery_radius
            candidates &= ~kept
            recovered_component = one_component_connected_to(core, candidates | core)
            proposed_recovery = recovered_component & ~core
            recovery_pixels = int(proposed_recovery.sum())
            recovery_ok = recovery_pixels <= args.max_recovery_ratio * original_area
            final_component = recovered_component if recovery_ok else core
            if recovery_ok:
                recovery_added |= proposed_recovery
        else:
            threshold = float("nan")
            recovery_pixels = 0
            recovery_ok = False
            final_component = core

        # Preserve the original class-1 material even when it is enclosed by a soma.
        final_component &= ~skeleton
        kept |= final_component
        final_area = int(final_component.sum())
        large = final_area >= args.review_large_soma_area
        if large:
            review_large |= final_component
        rows.append({
            "raw_soma_id": identifier, "original_area_px": original_area,
            "final_area_px": final_area,
            "filled_hole_pixels": int((core & ~component).sum()) if fill_ok else 0,
            "intensity_recovered_pixels": int((final_component & ~core).sum()) if recovery_ok else 0,
            "recovery_threshold": round(threshold, 5) if np.isfinite(threshold) else "",
            "kept_as_seed": 1, "flagged_large_for_review": int(large),
            "reason": "kept_with_local_recovery" if recovery_ok and recovery_pixels else (
                "kept_filled_holes" if fill_ok and hole_area else "kept"
            ),
        })

    semantic = np.zeros(segmentation.shape, dtype=np.uint8)
    semantic[skeleton] = 1
    semantic[kept] = 2
    if not np.array_equal(semantic == 1, skeleton):
        raise AssertionError("Internal error: an original skeleton pixel was changed.")
    instances, kept_count = ndi.label(kept, structure=STRUCTURE_8)

    tifffile.imwrite(args.output_dir / "01_input_prediction_0-1-2.tif", segmentation.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "02_soma_hole_filled_pixels.tif", fill_added.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "03_soma_local_intensity_recovery_pixels.tif", recovery_added.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "04_rejected_small_soma_mask.tif", rejected.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "05_large_soma_review_mask_NOT_REMOVED.tif", review_large.astype(np.uint8))
    tifffile.imwrite(args.output_dir / "06_semantic_soma_prepared_0-1-2.tif", semantic)
    tifffile.imwrite(args.output_dir / "07_kept_soma_instances.tif", instances.astype(output_dtype(int(kept_count))))
    tifffile.imwrite(args.output_dir / "08_raw_image_support.tif", support.astype(np.float32))
    save_qc(
        args.output_dir / "09_soma_preparation_qc.png", original, skeleton, kept,
        fill_added, recovery_added, rejected, review_large, args.qc_max_size,
    )
    with (args.output_dir / "soma_component_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["raw_soma_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "prediction": str(args.prediction.resolve()), "original": str(args.original.resolve()),
        "shape": list(segmentation.shape), "selected_polarity": selected_polarity,
        "input_skeleton_pixels": int(skeleton.sum()), "output_skeleton_pixels": int((semantic == 1).sum()),
        "removed_original_skeleton_pixels": 0, "input_soma_instances": int(count),
        "kept_soma_instances": int(kept_count), "rejected_small_soma_instances": int(np.sum([r["kept_as_seed"] == 0 for r in rows])),
        "hole_fill_pixels": int(fill_added.sum()), "local_recovery_pixels": int(recovery_added.sum()),
        "large_soma_review_instances": int(np.sum([r["flagged_large_for_review"] == 1 for r in rows])),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "_script_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n", encoding="utf-8"
    )

    print("\nSOMA PREPARATION COMPLETE")
    print(f"Input skeleton pixels: {int(skeleton.sum())}")
    print(f"Output skeleton pixels: {int((semantic == 1).sum())}")
    print("REMOVED original skeleton pixels: 0")
    print(f"Kept soma seeds: {int(kept_count)}; rejected tiny somas: {int(rejected.sum())} pixels")
    print(f"Prepared semantic: {args.output_dir / '06_semantic_soma_prepared_0-1-2.tif'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
