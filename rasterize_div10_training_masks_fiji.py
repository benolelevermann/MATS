# This script is executed by Fiji's Jython interpreter, not normal Python.
#
# It converts rows from included_tracings_manifest.csv into nnU-Net-ready
# TIFF pairs. It is intentionally isolated from the Dataset137 assembly step:
# raw data are copied to a staging directory and no existing dataset is edited.
#
# Required environment variables before launching Fiji:
#   DIV10_INCLUDED_MANIFEST = full path to included_tracings_manifest.csv
#   DIV10_RASTER_OUTPUT     = new, empty staging directory

import csv
import os
import re
import shutil
import sys
import traceback

from ij import IJ, ImagePlus
from ij.plugin.frame import RoiManager
from ij.process import ByteProcessor
from java.lang import System


SCRIPT_VERSION = "dataset137-div10-fiji-rasterizer-v1-2026-08-15"
VOXEL_PATTERN = re.compile(
    r"Voxel\s+separation\s*\(x,y,z\)\s*:\s*"
    r"([^,]+)\s*,\s*([^,]+)\s*,",
    re.IGNORECASE,
)


def require_environment(name):
    value = System.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError("Environment variable %s is not set." % name)
    return os.path.abspath(value)


def ensure_new_empty_directory(path):
    if os.path.exists(path) and os.listdir(path):
        raise RuntimeError(
            "Output directory is not empty: %s\n"
            "Choose a new staging directory." % path
        )
    if not os.path.exists(path):
        os.makedirs(path)


def read_manifest(path):
    with open(path, "rb") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Included manifest contains no rows: %s" % path)
    required = set(["case_id", "raw_path", "swc_path", "soma_zip_path", "decision"])
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError("Manifest is missing columns: %s" % sorted(missing))
    rows = [row for row in rows if row["decision"] == "include"]
    if not rows:
        raise RuntimeError("Manifest has no include rows.")
    return rows


def swc_nodes_and_spacing(path):
    nodes = {}
    spacing_x = None
    spacing_y = None
    with open(path, "r") as handle:
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
            parts = line.split()
            if len(parts) < 7:
                raise RuntimeError("Invalid SWC row in %s: %s" % (path, line))
            node_id = int(float(parts[0]))
            nodes[node_id] = {
                "x_um": float(parts[2]),
                "y_um": float(parts[3]),
                "parent": int(float(parts[6])),
            }
    if not nodes:
        raise RuntimeError("No SWC nodes found: %s" % path)
    if spacing_x is None or spacing_y is None or spacing_x <= 0 or spacing_y <= 0:
        raise RuntimeError(
            "Could not read positive x/y voxel separation from SWC header: %s" % path
        )
    return nodes, spacing_x, spacing_y


def set_pixel_if_inside(processor, x, y, value):
    if 0 <= x < processor.getWidth() and 0 <= y < processor.getHeight():
        processor.set(x, y, value)
        return True
    return False


def draw_bresenham(processor, x0, y0, x1, y1, value):
    # Integer line rasterization. The SWC coordinates are already densely
    # sampled by SNT, but this also preserves continuity at diagonal steps.
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        set_pixel_if_inside(processor, x0, y0, value)
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * err
        if twice_error >= dy:
            err += dy
            x0 += sx
        if twice_error <= dx:
            err += dx
            y0 += sy


def skeleton_from_swc(path, width, height):
    nodes, spacing_x, spacing_y = swc_nodes_and_spacing(path)
    skeleton = ByteProcessor(width, height)
    pixel_nodes = {}
    for node_id, node in nodes.items():
        pixel_nodes[node_id] = (
            int(round(node["x_um"] / spacing_x)),
            int(round(node["y_um"] / spacing_y)),
        )
    for node_id, node in nodes.items():
        x, y = pixel_nodes[node_id]
        parent = node["parent"]
        if parent in pixel_nodes:
            parent_x, parent_y = pixel_nodes[parent]
            draw_bresenham(skeleton, x, y, parent_x, parent_y, 1)
        else:
            set_pixel_if_inside(skeleton, x, y, 1)
    return skeleton, spacing_x, spacing_y


def soma_from_roi_zip(path, width, height):
    manager = RoiManager.getInstance2()
    if manager is None:
        manager = RoiManager()
    manager.reset()
    opened = manager.runCommand("Open", path)
    rois = manager.getRoisAsArray()
    if not opened or rois is None or len(rois) == 0:
        manager.reset()
        raise RuntimeError("Could not load an area ROI from: %s" % path)
    soma = ByteProcessor(width, height)
    soma.setValue(1)
    valid_roi_count = 0
    for roi in rois:
        if roi is not None and roi.isArea():
            soma.fill(roi)
            valid_roi_count += 1
    manager.reset()
    if valid_roi_count == 0:
        raise RuntimeError("soma.zip contains no area ROI: %s" % path)
    return soma


def combine_semantic(skeleton, soma):
    width = skeleton.getWidth()
    height = skeleton.getHeight()
    if soma.getWidth() != width or soma.getHeight() != height:
        raise RuntimeError("Skeleton/soma dimensions differ.")
    label = ByteProcessor(width, height)
    skeleton_pixels = 0
    soma_pixels = 0
    for y in range(height):
        for x in range(width):
            if skeleton.get(x, y) != 0:
                label.set(x, y, 1)
                skeleton_pixels += 1
            if soma.get(x, y) != 0:
                label.set(x, y, 2)  # Soma wins wherever both annotations overlap.
                soma_pixels += 1
    if skeleton_pixels == 0:
        raise RuntimeError("Rasterized SWC contains no in-bounds skeleton pixels.")
    if soma_pixels == 0:
        raise RuntimeError("Rasterized soma contains no pixels.")
    return label, skeleton_pixels, soma_pixels


def save_label(path, label):
    image = ImagePlus("semantic_0_1_2", label)
    IJ.saveAsTiff(image, path)
    image.close()
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError("Could not save label TIFF: %s" % path)


def process_row(row, images_dir, labels_dir):
    case_id = row["case_id"]
    raw_path = row["raw_path"]
    swc_path = row["swc_path"]
    soma_zip_path = row["soma_zip_path"]
    for path in (raw_path, swc_path, soma_zip_path):
        if not os.path.isfile(path):
            raise RuntimeError("Source file missing: %s" % path)

    output_image = os.path.join(images_dir, case_id + "_0000.tif")
    output_label = os.path.join(labels_dir, case_id + ".tif")
    if os.path.exists(output_image) or os.path.exists(output_label):
        raise RuntimeError("Output already exists for case %s" % case_id)

    raw = IJ.openImage(raw_path)
    if raw is None:
        raise RuntimeError("Could not open raw TIFF: %s" % raw_path)
    try:
        width = raw.getWidth()
        height = raw.getHeight()
    finally:
        raw.close()
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid raw image geometry: %s" % raw_path)

    skeleton, spacing_x, spacing_y = skeleton_from_swc(swc_path, width, height)
    soma = soma_from_roi_zip(soma_zip_path, width, height)
    label, skeleton_pixels, soma_pixels = combine_semantic(skeleton, soma)

    shutil.copy2(raw_path, output_image)
    save_label(output_label, label)
    return {
        "case_id": case_id,
        "raw_path": raw_path,
        "swc_path": swc_path,
        "soma_zip_path": soma_zip_path,
        "output_image": output_image,
        "output_label": output_label,
        "width_px": width,
        "height_px": height,
        "spacing_x_um_per_px": spacing_x,
        "spacing_y_um_per_px": spacing_y,
        "skeleton_pixels": skeleton_pixels,
        "soma_pixels": soma_pixels,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "wb") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest_path = require_environment("DIV10_INCLUDED_MANIFEST")
    output_root = require_environment("DIV10_RASTER_OUTPUT")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("Manifest does not exist: %s" % manifest_path)
    ensure_new_empty_directory(output_root)
    images_dir = os.path.join(output_root, "imagesTr")
    labels_dir = os.path.join(output_root, "labelsTr")
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    rows = read_manifest(manifest_path)
    successes = []
    failures = []
    for index, row in enumerate(rows, start=1):
        try:
            result = process_row(row, images_dir, labels_dir)
            successes.append(result)
            print(
                "[%d/%d] OK %s: %dx%d, skeleton=%d, soma=%d" % (
                    index, len(rows), result["case_id"], result["width_px"],
                    result["height_px"], result["skeleton_pixels"],
                    result["soma_pixels"],
                )
            )
        except Exception as error:
            failures.append({
                "case_id": row.get("case_id", ""),
                "cell_dir": row.get("cell_dir", ""),
                "reason": str(error),
            })
            print("[%d/%d] ERROR %s: %s" % (index, len(rows), row.get("case_id", ""), error))

    if successes:
        write_csv(
            os.path.join(output_root, "rasterization_report.csv"),
            successes,
            [
                "case_id", "raw_path", "swc_path", "soma_zip_path",
                "output_image", "output_label", "width_px", "height_px",
                "spacing_x_um_per_px", "spacing_y_um_per_px",
                "skeleton_pixels", "soma_pixels",
            ],
        )
    write_csv(
        os.path.join(output_root, "rasterization_failures.csv"),
        failures,
        ["case_id", "cell_dir", "reason"],
    )
    print("\n" + "=" * 72)
    print("DIV10 RASTERIZATION COMPLETE")
    print("=" * 72)
    print("Succeeded: %d" % len(successes))
    print("Failed:    %d" % len(failures))
    print("Output:    %s" % output_root)
    if failures:
        raise RuntimeError("Rasterization failed for %d cells; inspect rasterization_failures.csv" % len(failures))


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    # Fiji/ImageJ can retain background UI threads in headless mode. This script
    # runs in a dedicated process, so explicitly terminate it after reporting.
    System.exit(exit_code)
