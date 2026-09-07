from __future__ import annotations

"""Rasterize manual div10 SWC and ImageJ Soma ROI ZIPs without Fiji.

This script reads the include-only manifest made by
``select_div10_tracings_for_dataset137.py`` and creates a staging ``imagesTr``
and ``labelsTr`` directory. It does not modify source folders or Dataset136.

Labels: 0 = background, 1 = skeleton from seg.swc, 2 = soma from soma.zip.
Soma wins where masks overlap.

Install once in the project virtual environment:

    python -m pip install roifile
"""

import argparse
import csv
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

try:
    from roifile import ROI_TYPE, roiread
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency 'roifile'. Install it in the active project environment:\n"
        "  python -m pip install roifile"
    ) from error

try:
    from skimage.draw import polygon
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency 'scikit-image'. Install it in the active project environment:\n"
        "  python -m pip install scikit-image"
    ) from error


SCRIPT_VERSION = "dataset137-div10-python-rasterizer-v1-2026-08-15"
VOXEL_PATTERN = re.compile(
    r"Voxel\s+separation\s*\(x,y,z\)\s*:\s*"
    r"([^,]+)\s*,\s*([^,]+)\s*,",
    re.IGNORECASE,
)
AREA_TYPES = {
    ROI_TYPE.POLYGON,
    ROI_TYPE.FREEHAND,
    ROI_TYPE.TRACED,
    ROI_TYPE.RECT,
    ROI_TYPE.OVAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--included-manifest",
        type=Path,
        required=True,
        help="included_tracings_manifest.csv from the selection step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty staging output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of included rows to process for a visual pilot.",
    )
    return parser.parse_args()


def require_new_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {path}\n"
            "Choose a new directory to avoid mixing results from different runs."
        )
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "raw_path", "swc_path", "soma_zip_path", "decision"}
    if not rows:
        raise RuntimeError(f"Included manifest contains no rows: {path}")
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    rows = [row for row in rows if row["decision"] == "include"]
    if not rows:
        raise RuntimeError("Manifest contains no included rows.")
    return rows


def read_2d_raw(path: Path) -> np.ndarray:
    raw = np.squeeze(np.asarray(tifffile.imread(path)))
    if raw.ndim != 2:
        raise RuntimeError(f"Raw image must be 2-D, got shape {raw.shape}: {path}")
    if raw.shape[0] <= 0 or raw.shape[1] <= 0:
        raise RuntimeError(f"Raw image has invalid shape {raw.shape}: {path}")
    return raw


def swc_nodes_and_spacing(path: Path) -> tuple[dict[int, tuple[float, float, int]], float, float]:
    nodes: dict[int, tuple[float, float, int]] = {}
    spacing_x: float | None = None
    spacing_y: float | None = None
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = VOXEL_PATTERN.search(line)
                if match:
                    spacing_x = float(match.group(1))
                    spacing_y = float(match.group(2))
                continue
            values = line.split()
            if len(values) < 7:
                raise RuntimeError(f"Invalid SWC row in {path}: {line}")
            node_id = int(float(values[0]))
            nodes[node_id] = (
                float(values[2]),
                float(values[3]),
                int(float(values[6])),
            )
    if not nodes:
        raise RuntimeError(f"No SWC nodes found: {path}")
    if spacing_x is None or spacing_y is None or spacing_x <= 0 or spacing_y <= 0:
        raise RuntimeError(f"No valid XY voxel spacing in SWC header: {path}")
    return nodes, spacing_x, spacing_y


def draw_line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw a one-pixel Bresenham line, clipping out-of-image pixels."""
    height, width = mask.shape
    dx = abs(x1 - x0)
    step_x = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            mask[y0, x0] = True
        if x0 == x1 and y0 == y1:
            return
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += step_x
        if twice_error <= dx:
            error += dx
            y0 += step_y


def skeleton_from_swc(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    nodes, spacing_x, spacing_y = swc_nodes_and_spacing(path)
    coordinates = {
        node_id: (int(round(x_um / spacing_x)), int(round(y_um / spacing_y)))
        for node_id, (x_um, y_um, _parent) in nodes.items()
    }
    skeleton = np.zeros(shape, dtype=bool)
    for node_id, (_x_um, _y_um, parent) in nodes.items():
        x, y = coordinates[node_id]
        if parent in coordinates:
            parent_x, parent_y = coordinates[parent]
            draw_line(skeleton, x, y, parent_x, parent_y)
        elif 0 <= x < shape[1] and 0 <= y < shape[0]:
            skeleton[y, x] = True
    if not skeleton.any():
        raise RuntimeError(f"Rasterized SWC has no in-bounds skeleton pixels: {path}")
    return skeleton, spacing_x, spacing_y


def coordinate_polygons(coordinates: object) -> Iterable[np.ndarray]:
    """Normalize roifile single-/multi-path coordinates to Nx2 polygons."""
    if isinstance(coordinates, np.ndarray):
        if coordinates.ndim == 2 and coordinates.shape[1] >= 2:
            yield coordinates[:, :2]
            return
        if coordinates.ndim == 3 and coordinates.shape[2] >= 2:
            for value in coordinates:
                yield value[:, :2]
            return
    if isinstance(coordinates, (list, tuple)):
        for value in coordinates:
            array = np.asarray(value)
            if array.ndim == 2 and array.shape[1] >= 2:
                yield array[:, :2]
                continue
            raise RuntimeError(f"Unsupported ROI coordinate shape: {array.shape}")
        return
    raise RuntimeError(f"Unsupported ROI coordinates: {type(coordinates).__name__}")


def soma_from_roi_zip(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    read_result = roiread(path)
    rois = read_result if isinstance(read_result, list) else [read_result]
    if not rois:
        raise RuntimeError(f"soma.zip contains no ROI: {path}")
    soma = np.zeros(shape, dtype=bool)
    valid_roi_count = 0
    for roi in rois:
        if bool(getattr(roi, "composite", False)):
            raise RuntimeError(
                f"Composite Soma ROI is not supported safely: {path}. "
                "Do not silently rasterize holes or mixed shapes."
            )
        if roi.roitype not in AREA_TYPES:
            raise RuntimeError(
                f"Unsupported non-area Soma ROI {roi.roitype!s} in {path}"
            )
        polygons = list(coordinate_polygons(roi.coordinates(multi=True)))
        if not polygons:
            raise RuntimeError(f"ROI has no coordinates: {path}")
        for coordinates in polygons:
            if coordinates.shape[0] < 3:
                raise RuntimeError(f"Soma ROI has fewer than three vertices: {path}")
            rows, columns = polygon(
                coordinates[:, 1], coordinates[:, 0], shape=shape
            )
            soma[rows, columns] = True
        valid_roi_count += 1
    if valid_roi_count == 0 or not soma.any():
        raise RuntimeError(f"No soma pixels after reading: {path}")
    return soma, valid_roi_count


def process_row(row: dict[str, str], images_dir: Path, labels_dir: Path) -> dict[str, object]:
    case_id = row["case_id"]
    raw_path = Path(row["raw_path"])
    swc_path = Path(row["swc_path"])
    soma_zip_path = Path(row["soma_zip_path"])
    for source in (raw_path, swc_path, soma_zip_path):
        if not source.is_file():
            raise FileNotFoundError(f"Source file missing: {source}")

    output_image = images_dir / f"{case_id}_0000.tif"
    output_label = labels_dir / f"{case_id}.tif"
    if output_image.exists() or output_label.exists():
        raise RuntimeError(f"Output already exists for case: {case_id}")

    raw = read_2d_raw(raw_path)
    skeleton, spacing_x, spacing_y = skeleton_from_swc(swc_path, raw.shape)
    soma, roi_count = soma_from_roi_zip(soma_zip_path, raw.shape)

    label = np.zeros(raw.shape, dtype=np.uint8)
    label[skeleton] = 1
    label[soma] = 2
    skeleton_pixels = int(np.count_nonzero(label == 1))
    soma_pixels = int(np.count_nonzero(label == 2))
    if skeleton_pixels == 0 or soma_pixels == 0:
        raise RuntimeError(
            f"Invalid rasterized label for {case_id}: "
            f"skeleton={skeleton_pixels}, soma={soma_pixels}"
        )

    shutil.copy2(raw_path, output_image)
    tifffile.imwrite(output_label, label, photometric="minisblack")
    return {
        "case_id": case_id,
        "raw_path": str(raw_path),
        "swc_path": str(swc_path),
        "soma_zip_path": str(soma_zip_path),
        "output_image": str(output_image),
        "output_label": str(output_label),
        "height_px": int(raw.shape[0]),
        "width_px": int(raw.shape[1]),
        "spacing_x_um_per_px": spacing_x,
        "spacing_y_um_per_px": spacing_y,
        "soma_roi_count": roi_count,
        "skeleton_pixels": skeleton_pixels,
        "soma_pixels": soma_pixels,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.included_manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        rows = rows[: args.limit]
    require_new_empty_directory(args.output_dir)
    images_dir = args.output_dir / "imagesTr"
    labels_dir = args.output_dir / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()

    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        try:
            result = process_row(row, images_dir, labels_dir)
            successes.append(result)
            print(
                f"[{index}/{len(rows)}] OK {result['case_id']}: "
                f"{result['width_px']}x{result['height_px']}, "
                f"skeleton={result['skeleton_pixels']}, soma={result['soma_pixels']}"
            )
        except Exception as error:
            failures.append(
                {
                    "case_id": row.get("case_id", ""),
                    "cell_dir": row.get("cell_dir", ""),
                    "reason": str(error),
                }
            )
            print(
                f"[{index}/{len(rows)}] ERROR {row.get('case_id', '')}: {error}",
                file=sys.stderr,
            )

    success_fields = [
        "case_id", "raw_path", "swc_path", "soma_zip_path", "output_image",
        "output_label", "height_px", "width_px", "spacing_x_um_per_px",
        "spacing_y_um_per_px", "soma_roi_count", "skeleton_pixels", "soma_pixels",
    ]
    write_csv(args.output_dir / "rasterization_report.csv", successes, success_fields)
    write_csv(
        args.output_dir / "rasterization_failures.csv",
        failures,
        ["case_id", "cell_dir", "reason"],
    )
    summary = {
        "script_version": SCRIPT_VERSION,
        "included_manifest": str(args.included_manifest),
        "attempted": len(rows),
        "succeeded": len(successes),
        "failed": len(failures),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "rasterization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 72)
    print("DIV10 PYTHON RASTERIZATION COMPLETE")
    print("=" * 72)
    print(f"Succeeded: {len(successes)}")
    print(f"Failed:    {len(failures)}")
    print(f"Output:    {args.output_dir}")
    if failures:
        raise RuntimeError(
            f"Rasterization failed for {len(failures)} cells; "
            "inspect rasterization_failures.csv."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        raise
