from __future__ import annotations

"""Separate export-ready cells from review cases without altering segmentation.

The input instance map can be the overlap-representative output. Each candidate
is checked for one soma, sufficient skeleton, soma contact, image/ridge support,
and crop-border safety. The source semantic image is never edited: this script
only writes safe/review/excluded *selection maps* plus a review manifest.
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


SCRIPT_VERSION = "evo-cell-quality-gate-v1-2026-08-12"
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create safe/review cell maps for Evo crop export.")
    parser.add_argument("--semantic", type=Path, required=True, help="Final 0/1/2 semantic TIFF.")
    parser.add_argument("--instances", type=Path, required=True, help="Selected cell-instance TIFF.")
    parser.add_argument(
        "--review-instances",
        type=Path,
        help=(
            "Optional broader assigned-instance map. IDs that are not present "
            "in --instances are included in the manual-review map rather than "
            "silently disappearing after overlap selection."
        ),
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--ridge-evidence", type=Path)
    parser.add_argument("--ridge-weight", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--polarity", choices=("auto", "bright", "dark"), default="bright")
    parser.add_argument("--min-soma-area", type=int, default=80)
    parser.add_argument("--max-soma-area", type=int, default=12000)
    parser.add_argument("--min-skeleton-pixels", type=int, default=20)
    parser.add_argument("--contact-radius", type=int, default=4)
    parser.add_argument("--safe-min-support-fraction", type=float, default=0.25)
    parser.add_argument("--review-min-support-fraction", type=float, default=0.10)
    parser.add_argument("--max-skeleton-components-safe", type=int, default=4)
    parser.add_argument(
        "--safe-min-connected-skeleton-fraction", type=float, default=0.90,
        help="Minimum skeleton fraction connected to the cell's soma for safe status (default: 0.90).",
    )
    parser.add_argument("--border-margin", type=int, default=4)
    parser.add_argument("--qc-max-size", type=int, default=2200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_2d(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be 2-D after squeeze, got {array.shape}")
    return array


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def robust_unit_scale(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("Image has no finite values.")
    sample = finite[::max(1, finite.size // 1_000_000)]
    low, high = np.percentile(sample, [1.0, 99.5])
    if high <= low:
        low, high = float(np.min(sample)), float(np.max(sample))
    if high <= low:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def build_support(original: np.ndarray, skeleton: np.ndarray, polarity: str) -> tuple[np.ndarray, str]:
    normalized = robust_unit_scale(original)
    if polarity == "auto":
        selected = "bright"
        if np.any(skeleton) and np.any(~skeleton):
            selected = "bright" if np.median(normalized[skeleton]) >= np.median(normalized[~skeleton]) else "dark"
    else:
        selected = polarity
    direct = normalized if selected == "bright" else 1.0 - normalized
    local = direct - ndi.gaussian_filter(direct, sigma=2.0, mode="reflect")
    return np.clip(0.70 * direct + 0.30 * np.clip(0.5 + 2.0 * local, 0.0, 1.0), 0.0, 1.0).astype(np.float32), selected


def touches_border(mask: np.ndarray, margin: int) -> bool:
    if margin <= 0 or not np.any(mask):
        return False
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    return bool(rows.min() < margin or cols.min() < margin or rows.max() >= height - margin or cols.max() >= width - margin)


def palette(identifier: int) -> np.ndarray:
    # Golden-ratio hue sequence; reproducible and visually distinct enough for QC.
    hue = (identifier * 0.61803398875) % 1.0
    anchors = np.asarray([[255, 85, 115], [250, 210, 70], [70, 210, 160], [65, 150, 255], [185, 100, 245]], dtype=np.float32)
    index = int(hue * len(anchors)) % len(anchors)
    return anchors[index]


def save_qc(path: Path, original: np.ndarray, safe: np.ndarray, review: np.ndarray, excluded: np.ndarray, max_size: int) -> None:
    height, width = original.shape
    stride = max(1, int(np.ceil(max(height, width) / max(max_size, 1))))
    gray = robust_unit_scale(original[::stride, ::stride])
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    safe_small, review_small, excluded_small = safe[::stride, ::stride], review[::stride, ::stride], excluded[::stride, ::stride]
    for identifier in np.unique(safe_small):
        if identifier <= 0:
            continue
        mask = safe_small == identifier
        rgb[mask] = 0.18 * rgb[mask] + 0.82 * palette(int(identifier))
    rgb[review_small > 0] = 0.18 * rgb[review_small > 0] + 0.82 * np.asarray([255, 190, 0])
    rgb[excluded_small > 0] = 0.28 * rgb[excluded_small > 0] + 0.72 * np.asarray([245, 70, 55])
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if args.min_soma_area < 1 or args.min_skeleton_pixels < 1 or args.contact_radius < 0:
        raise ValueError("Minimum areas and contact radius must be valid.")
    if not 0 <= args.review_min_support_fraction <= args.safe_min_support_fraction <= 1:
        raise ValueError("Require 0 <= review support <= safe support <= 1.")
    if not 0 <= args.ridge_weight <= 1:
        raise ValueError("--ridge-weight must be between 0 and 1.")
    if not 0 <= args.safe_min_connected_skeleton_fraction <= 1:
        raise ValueError("--safe-min-connected-skeleton-fraction must be between 0 and 1.")

    prepare_output(args.output_dir, args.overwrite)
    semantic = read_2d(args.semantic, "semantic segmentation")
    instances = read_2d(args.instances, "selected instance map").astype(np.uint32)
    original = read_2d(args.original, "original image")
    if not (semantic.shape == instances.shape == original.shape):
        raise RuntimeError("semantic, instances and original must have the same 2-D shape.")
    invalid = set(np.unique(semantic).astype(int).tolist()) - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Unexpected semantic labels: {sorted(invalid)}")
    if np.any((instances > 0) & (semantic == 0)):
        raise RuntimeError("Instance map contains pixels outside semantic skeleton/soma material.")
    review_instances = None
    if args.review_instances:
        review_instances = read_2d(args.review_instances, "broader review instance map").astype(np.uint32)
        if review_instances.shape != semantic.shape:
            raise RuntimeError("Broader review instance map must match semantic shape.")
        if np.any((review_instances > 0) & (semantic == 0)):
            raise RuntimeError("Broader review instance map contains pixels outside semantic material.")

    skeleton_all = semantic == 1
    raw_support, selected_polarity = build_support(original, skeleton_all, args.polarity)
    support = raw_support
    if args.ridge_evidence:
        ridge = read_2d(args.ridge_evidence, "ridge evidence")
        if ridge.shape != semantic.shape:
            raise RuntimeError("Ridge evidence must match semantic shape.")
        support = (1.0 - args.ridge_weight) * raw_support + args.ridge_weight * robust_unit_scale(ridge)
        support = np.clip(support, 0.0, 1.0).astype(np.float32)

    safe = np.zeros_like(instances)
    review = np.zeros_like(instances)
    excluded = np.zeros_like(instances)
    rows: list[dict[str, object]] = []

    identifiers = [int(value) for value in np.unique(instances) if int(value) > 0]
    for identifier in identifiers:
        cell = instances == identifier
        skeleton = cell & (semantic == 1)
        soma = cell & (semantic == 2)
        soma_labels, soma_count = ndi.label(soma, structure=STRUCTURE_8)
        soma_area = int(soma.sum())
        skeleton_area = int(skeleton.sum())
        skeleton_components = int(ndi.label(skeleton, structure=STRUCTURE_8)[1])
        if np.any(soma):
            soma_contact_zone = (
                ndi.binary_dilation(soma, iterations=args.contact_radius)
                if args.contact_radius > 0 else soma
            )
            touches = bool(np.any(skeleton & soma_contact_zone))
        else:
            touches = False
        cell_components, _ = ndi.label(skeleton | soma, structure=STRUCTURE_8)
        soma_component_ids = np.unique(cell_components[soma])
        soma_component_ids = soma_component_ids[soma_component_ids > 0]
        connected_skeleton = skeleton & np.isin(cell_components, soma_component_ids)
        connected_skeleton_fraction = (
            float(connected_skeleton.sum() / skeleton_area) if skeleton_area else 0.0
        )
        support_fraction = float(np.mean(support[skeleton] >= 0.5)) if skeleton_area else 0.0
        border = touches_border(cell, args.border_margin)

        reasons: list[str] = []
        if soma_count != 1:
            reasons.append("not_exactly_one_soma")
        if soma_area < args.min_soma_area:
            reasons.append("soma_below_min_area")
        if skeleton_area < args.min_skeleton_pixels:
            reasons.append("skeleton_below_min_pixels")
        if border:
            reasons.append("touches_image_border")
        hard_reject = bool(reasons)
        soft_reasons: list[str] = []
        if soma_area > args.max_soma_area:
            soft_reasons.append("large_soma_review")
        if not touches:
            soft_reasons.append("skeleton_not_touching_soma")
        if skeleton_components > args.max_skeleton_components_safe:
            soft_reasons.append("fragmented_skeleton")
        if connected_skeleton_fraction < args.safe_min_connected_skeleton_fraction:
            soft_reasons.append("skeleton_not_sufficiently_connected_to_soma")
        if support_fraction < args.review_min_support_fraction:
            soft_reasons.append("very_weak_raw_or_ridge_support")
        elif support_fraction < args.safe_min_support_fraction:
            soft_reasons.append("weak_raw_or_ridge_support")

        if hard_reject:
            category = "excluded"
            excluded[cell] = identifier
        elif soft_reasons:
            category = "review"
            review[cell] = identifier
        else:
            category = "safe"
            safe[cell] = identifier
        rows.append({
            "source_instance_id": identifier, "category": category,
            "soma_components": soma_count, "soma_pixels": soma_area,
            "skeleton_pixels": skeleton_area, "skeleton_components": skeleton_components,
            "skeleton_touches_soma": int(touches), "support_fraction_ge_0_5": round(support_fraction, 5),
            "connected_skeleton_fraction": round(connected_skeleton_fraction, 5),
            "touches_image_border": int(border),
            "reasons": ";".join(reasons + soft_reasons), "manual_decision": "", "manual_notes": "",
        })

    # The overlap-selection stage intentionally keeps only one representative
    # in an overlap group for automatic crop export. Keep the non-selected
    # candidates visible as review cases so they can become hard examples later.
    if review_instances is not None:
        selected_ids = set(identifiers)
        broad_ids = [int(value) for value in np.unique(review_instances) if int(value) > 0]
        for identifier in broad_ids:
            if identifier in selected_ids:
                continue
            candidate = review_instances == identifier
            review[candidate] = identifier
            skeleton = candidate & (semantic == 1)
            soma = candidate & (semantic == 2)
            soma_count = int(ndi.label(soma, structure=STRUCTURE_8)[1])
            rows.append({
                "source_instance_id": identifier, "category": "review",
                "soma_components": soma_count, "soma_pixels": int(soma.sum()),
                "skeleton_pixels": int(skeleton.sum()),
                "skeleton_components": int(ndi.label(skeleton, structure=STRUCTURE_8)[1]),
                "skeleton_touches_soma": "", "support_fraction_ge_0_5": "",
                "connected_skeleton_fraction": "", "touches_image_border": int(touches_border(candidate, args.border_margin)),
                "reasons": "not_selected_by_overlap_or_assignment", "manual_decision": "", "manual_notes": "",
            })

    status = np.zeros(semantic.shape, dtype=np.uint8)
    status[safe > 0] = 1
    status[review > 0] = 2
    status[excluded > 0] = 3
    dtype = np.uint16 if (max(identifiers) if identifiers else 0) <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(args.output_dir / "01_cells_safe_for_evo.tif", safe.astype(dtype))
    tifffile.imwrite(args.output_dir / "02_cells_for_manual_review.tif", review.astype(dtype))
    tifffile.imwrite(args.output_dir / "03_cells_excluded_from_export.tif", excluded.astype(dtype))
    tifffile.imwrite(args.output_dir / "04_cell_quality_status_map.tif", status)
    tifffile.imwrite(args.output_dir / "05_combined_raw_ridge_support.tif", support.astype(np.float32))
    save_qc(args.output_dir / "06_cell_quality_gate_qc.png", original, safe, review, excluded, args.qc_max_size)
    with (args.output_dir / "cell_quality_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["source_instance_id"])
        writer.writeheader()
        writer.writerows(rows)
    # This intentionally editable file is the seed for a later hard-case training set.
    with (args.output_dir / "manual_review_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["source_instance_id"])
        writer.writeheader()
        writer.writerows([row for row in rows if row["category"] == "review"])

    counts = {category: int(sum(row["category"] == category for row in rows)) for category in ("safe", "review", "excluded")}
    summary = {
        "script_version": SCRIPT_VERSION, "semantic": str(args.semantic.resolve()),
        "instances": str(args.instances.resolve()), "original": str(args.original.resolve()),
        "review_instances": str(args.review_instances.resolve()) if args.review_instances else None,
        "ridge_evidence": str(args.ridge_evidence.resolve()) if args.ridge_evidence else None,
        "selected_polarity": selected_polarity, "counts": counts,
        "source_instance_count": len(identifiers),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "_script_version.txt").write_text(f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n", encoding="utf-8")
    print("\nCELL QUALITY GATE COMPLETE")
    print(f"Safe: {counts['safe']}; review: {counts['review']}; excluded: {counts['excluded']}")
    print(f"Safe map for Evo export: {args.output_dir / '01_cells_safe_for_evo.tif'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
