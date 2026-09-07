from __future__ import annotations

"""Export pre-assigned safe cell instances without recomputing assignment."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


SCRIPT_VERSION = "assigned-instance-evo-export-v1-2026-08-12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export cell folders directly from 02_cell_instances_safe.tif. "
            "The graph-method instance IDs are preserved and never recomputed."
        )
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--safe-instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--helper-script", type=Path)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--edge-clearance", type=int, default=3)
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--start-number", type=int, default=1)
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def read_2d(path: Path, description: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    array = np.squeeze(tifffile.imread(path))
    if array.ndim != 2:
        raise RuntimeError(f"{description} must be 2-D, found {array.shape}")
    return array


def load_helpers(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Existing Evo exporter helper not found: {path}")
    spec = importlib.util.spec_from_file_location("existing_evo_exporter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell_clearance_ok(mask: np.ndarray, bounds: tuple[int, int, int, int], clearance: int) -> bool:
    y0, y1, x0, x1 = bounds
    crop = mask[y0:y1, x0:x1]
    rows, cols = np.nonzero(crop)
    if rows.size == 0:
        return False
    return bool(
        rows.min() >= clearance
        and cols.min() >= clearance
        and rows.max() < crop.shape[0] - clearance
        and cols.max() < crop.shape[1] - clearance
    )


def main() -> None:
    args = parse_args()
    if args.margin < 0 or args.min_crop_size < 1 or args.edge_clearance < 0:
        raise ValueError("Invalid crop parameters.")
    if args.start_number < 0 or args.digits < 1:
        raise ValueError("Invalid numbering parameters.")

    helper_path = args.helper_script
    if helper_path is None:
        helper_path = Path(__file__).resolve().parent / "r_pipeline" / "export_cells_for_r_pipeline.py"
    helper = load_helpers(helper_path)

    original = read_2d(args.original, "original image")
    semantic = read_2d(args.semantic, "semantic segmentation")
    safe_instances = read_2d(args.safe_instances, "safe instance map").astype(np.uint32)
    if original.shape != semantic.shape or semantic.shape != safe_instances.shape:
        raise RuntimeError(
            f"Shape mismatch: original={original.shape}, semantic={semantic.shape}, "
            f"instances={safe_instances.shape}"
        )
    invalid = set(np.unique(semantic).astype(int).tolist()) - {0, 1, 2}
    if invalid:
        raise RuntimeError(f"Unexpected semantic values: {sorted(invalid)}")
    if np.any((safe_instances > 0) & (semantic == 0)):
        raise RuntimeError("Safe instances contain pixels outside skeleton/soma material.")

    safe_ids = [int(value) for value in np.unique(safe_instances) if int(value) > 0]
    if not safe_ids:
        raise RuntimeError("The safe instance map contains no cells.")

    helper.ensure_empty_output(args.output_dir)
    diagnostics = args.output_dir / "_classification"
    diagnostics.mkdir()
    helper.save_tiff(
        diagnostics / "safe_cell_instances_overview.tif",
        safe_instances,
    )

    manifest_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    output_index = 0

    for source_index, instance_id in enumerate(safe_ids, start=1):
        cell_mask = safe_instances == instance_id
        skeleton_mask = cell_mask & (semantic == 1)
        soma_mask = cell_mask & (semantic == 2)
        rejection = None
        if not np.any(skeleton_mask):
            rejection = "no_skeleton"
        elif not np.any(soma_mask):
            rejection = "no_soma"
        elif helper.touches_border(cell_mask, 1) if hasattr(helper, "touches_border") else (
            np.any(cell_mask[0]) or np.any(cell_mask[-1]) or np.any(cell_mask[:, 0]) or np.any(cell_mask[:, -1])
        ):
            rejection = "touches_source_image_border"

        if rejection is None:
            bounds = helper.crop_bounds(
                cell_mask,
                args.margin,
                args.min_crop_size,
                args.square,
            )
            if not cell_clearance_ok(cell_mask, bounds, args.edge_clearance):
                rejection = "insufficient_crop_clearance"

        if rejection is not None:
            rejected_rows.append(
                {"source_instance_id": instance_id, "reason": rejection}
            )
            print(f"[{source_index}/{len(safe_ids)}] SKIP instance {instance_id}: {rejection}")
            continue

        output_index += 1
        output_number = args.start_number + output_index - 1
        folder_name = f"cell{output_number:0{args.digits}d}"
        cell_dir = args.output_dir / folder_name
        cell_dir.mkdir()
        y0, y1, x0, x1 = bounds

        raw_crop = original[y0:y1, x0:x1]
        skeleton_crop = skeleton_mask[y0:y1, x0:x1]
        soma_crop = soma_mask[y0:y1, x0:x1]
        cell_crop = skeleton_crop | soma_crop
        seg_crop = np.zeros(cell_crop.shape, dtype=np.uint8)
        seg_crop[skeleton_crop] = 1
        seg_crop[soma_crop] = 2

        helper.save_tiff(cell_dir / "raw.tif", raw_crop)
        helper.save_tiff(cell_dir / "skeleton.tif", skeleton_crop.astype(np.uint8))
        helper.save_tiff(cell_dir / "soma.tif", soma_crop.astype(np.uint8))
        helper.save_tiff(cell_dir / "seg.tif", seg_crop)
        helper.save_tiff(cell_dir / "cell_mask.tif", cell_crop.astype(np.uint8))
        helper.write_pixel_csv(cell_dir / "seg.csv", skeleton_crop, x0, y0)
        helper.write_swc(cell_dir / "seg-000.swc", skeleton_crop, soma_crop)

        soma_coordinates = np.argwhere(soma_crop)
        local_y, local_x = soma_coordinates.mean(axis=0)
        bounds_data = {
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
            "source_instance_id": instance_id,
            "skeleton_pixels": int(skeleton_crop.sum()),
            "soma_pixels": int(soma_crop.sum()),
            "bounds": bounds_data,
            "location": location,
        }
        (cell_dir / "bounds.json").write_text(json.dumps(bounds_data, indent=2), encoding="utf-8")
        (cell_dir / "location.json").write_text(json.dumps(location, indent=2), encoding="utf-8")
        (cell_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        manifest_rows.append(
            {
                "folder": folder_name,
                "status": "safe",
                "source_instance_id": instance_id,
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
            f"[{source_index}/{len(safe_ids)}] {folder_name}: source={instance_id}, "
            f"crop={x1-x0}x{y1-y0}, skeleton={int(skeleton_crop.sum())}"
        )

    if not manifest_rows:
        raise RuntimeError("No complete safe cells remained after crop validation.")

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (args.output_dir / "rejected_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_instance_id", "reason"])
        writer.writeheader()
        writer.writerows(rejected_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "source_safe_instances": str(args.safe_instances.resolve()),
        "source_safe_cell_count": len(safe_ids),
        "exported_cell_count": len(manifest_rows),
        "rejected_cell_count": len(rejected_rows),
        "removed_or_reassigned_skeleton_pixels": 0,
        "margin": args.margin,
        "min_crop_size": args.min_crop_size,
        "edge_clearance": args.edge_clearance,
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "_export_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n", encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print(f"Exported {len(manifest_rows)} cells to {args.output_dir}")
    print(f"Rejected incomplete/border cells: {len(rejected_rows)}")
    print("NEXT: run Fiji finalize_cells_with_fiji.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
