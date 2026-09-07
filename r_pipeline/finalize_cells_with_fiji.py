# This script is executed by Fiji's Jython interpreter, not by normal Python.
# It converts every seg-000.swc to a real SNT seg.traces file and creates the
# ImageJ ROI ZIP files required by the copied R pipeline.

import json
import os
import traceback

from ij import IJ
from ij.gui import PointRoi, Roi
from ij.io import RoiEncoder
from ij.plugin.filter import ThresholdToSelection
from ij.process import ImageProcessor
from java.io import (
    ByteArrayOutputStream,
    DataOutputStream,
    FileOutputStream,
)
from java.lang import System
from java.util.zip import ZipEntry, ZipOutputStream
from tracing import PathAndFillManager


SCRIPT_VERSION = "fiji-r-pipeline-finalizer-v3-global-rois-2026-07-24"


def save_roi_zip(path, rois):
    if os.path.exists(path):
        os.remove(path)
    archive = ZipOutputStream(FileOutputStream(path))
    try:
        if not rois:
            raise RuntimeError("No ROI to save: " + path)
        for index, roi in enumerate(rois):
            name = roi.getName()
            if not name:
                name = "roi-%04d" % (index + 1)
            if not name.lower().endswith(".roi"):
                name += ".roi"
            buffer = ByteArrayOutputStream()
            data = DataOutputStream(buffer)
            RoiEncoder(data).write(roi)
            data.flush()
            archive.putNextEntry(ZipEntry(name))
            archive.write(buffer.toByteArray())
            archive.closeEntry()
            data.close()
    finally:
        archive.close()


def soma_roi_from_mask(mask_path):
    image = IJ.openImage(mask_path)
    if image is None:
        raise RuntimeError("Could not open soma mask: " + mask_path)
    try:
        processor = image.getProcessor().duplicate()
        processor.setThreshold(
            1,
            processor.getMax(),
            ImageProcessor.NO_LUT_UPDATE,
        )
        roi = ThresholdToSelection().convert(processor)
        if roi is None:
            raise RuntimeError("Empty soma mask: " + mask_path)
        roi.setName("soma")
        return roi
    finally:
        image.close()


def convert_swc_to_traces(swc_path, traces_path, width, height):
    # The legacy SNT bundled with this Fiji build needs image geometry before
    # an SWC can be imported headlessly.
    manager = PathAndFillManager(
        int(width),
        int(height),
        1,
        1.0,
        1.0,
        1.0,
        "pixel",
    )
    loaded = manager.importSWC(swc_path, True)
    if not loaded:
        raise RuntimeError("SNT could not import SWC: " + swc_path)
    manager.writeXML(traces_path, True)
    if not os.path.isfile(traces_path) or os.path.getsize(traces_path) == 0:
        raise RuntimeError("SNT did not create traces file: " + traces_path)


def process_cell(cell_dir):
    swc_path = os.path.join(cell_dir, "seg-000.swc")
    soma_mask_path = os.path.join(cell_dir, "soma.tif")
    bounds_json_path = os.path.join(cell_dir, "bounds.json")
    location_json_path = os.path.join(cell_dir, "location.json")
    required = [
        swc_path,
        soma_mask_path,
        bounds_json_path,
        location_json_path,
        os.path.join(cell_dir, "raw.tif"),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise RuntimeError("Missing files: " + ", ".join(missing))

    soma_roi = soma_roi_from_mask(soma_mask_path)
    save_roi_zip(os.path.join(cell_dir, "soma.zip"), [soma_roi])

    raw_image = IJ.openImage(os.path.join(cell_dir, "raw.tif"))
    if raw_image is None:
        raise RuntimeError("Could not open raw.tif in: " + cell_dir)
    try:
        width = raw_image.getWidth()
        height = raw_image.getHeight()
    finally:
        raw_image.close()
    traces_path = os.path.join(cell_dir, "seg.traces")
    convert_swc_to_traces(swc_path, traces_path, width, height)

    with open(bounds_json_path, "r") as handle:
        bounds = json.load(handle)
    bounds_roi = Roi(
        int(bounds["x_min"]),
        int(bounds["y_min"]),
        int(bounds["width"]),
        int(bounds["height"]),
    )
    bounds_roi.setName("bounds")
    save_roi_zip(os.path.join(cell_dir, "bounds.zip"), [bounds_roi])

    with open(location_json_path, "r") as handle:
        location = json.load(handle)
    location_roi = PointRoi(
        float(location["global_x"]),
        float(location["global_y"]),
    )
    location_roi.setName("location")
    save_roi_zip(os.path.join(cell_dir, "location.zip"), [location_roi])


def create_global_roi_archives(root, cell_dirs):
    bounds_rois = []
    location_rois = []
    for cell_dir in cell_dirs:
        cell_name = os.path.basename(cell_dir)
        with open(os.path.join(cell_dir, "bounds.json"), "r") as handle:
            bounds = json.load(handle)
        bounds_roi = Roi(
            int(bounds["x_min"]),
            int(bounds["y_min"]),
            int(bounds["width"]),
            int(bounds["height"]),
        )
        bounds_roi.setName(cell_name + "_bounds")
        bounds_rois.append(bounds_roi)

        with open(os.path.join(cell_dir, "location.json"), "r") as handle:
            location = json.load(handle)
        location_roi = PointRoi(
            float(location["global_x"]),
            float(location["global_y"]),
        )
        location_roi.setName(cell_name + "_location")
        location_rois.append(location_roi)

    save_roi_zip(os.path.join(root, "bounds.zip"), bounds_rois)
    save_roi_zip(os.path.join(root, "locations.zip"), location_rois)


def main():
    root = os.environ.get("CELL_EXPORT_ROOT")
    if not root:
        raise RuntimeError(
            "Environment variable CELL_EXPORT_ROOT is not set."
        )
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise RuntimeError("Cell export root does not exist: " + root)

    cell_dirs = sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.startswith("cell")
        and os.path.isdir(os.path.join(root, name))
    )
    if not cell_dirs:
        raise RuntimeError("No cell* folders found in: " + root)

    log_lines = [
        SCRIPT_VERSION,
        "root=" + root,
        "cells=" + str(len(cell_dirs)),
    ]
    failures = []
    for index, cell_dir in enumerate(cell_dirs):
        name = os.path.basename(cell_dir)
        try:
            process_cell(cell_dir)
            line = "[%d/%d] OK %s" % (index + 1, len(cell_dirs), name)
            print(line)
            log_lines.append(line)
        except Exception as error:
            line = "[%d/%d] ERROR %s: %s" % (
                index + 1,
                len(cell_dirs),
                name,
                error,
            )
            print(line)
            log_lines.append(line)
            failures.append((name, traceback.format_exc()))

    if not failures:
        create_global_roi_archives(root, cell_dirs)
        line = "GLOBAL OK bounds.zip and locations.zip (%d ROIs each)" % (
            len(cell_dirs)
        )
        print(line)
        log_lines.append(line)

    log_path = os.path.join(root, "_fiji_finalize_log.txt")
    with open(log_path, "w") as handle:
        handle.write("\n".join(log_lines))
        if failures:
            handle.write("\n\nFAILURES\n")
            for name, details in failures:
                handle.write("\n--- " + name + " ---\n")
                handle.write(details)

    if failures:
        raise RuntimeError(
            "%d of %d cell folders failed. See %s"
            % (len(failures), len(cell_dirs), log_path)
        )
    print("Finalized %d cell folders in %s" % (len(cell_dirs), root))


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    # Fiji/ImageJ 1 can keep non-daemon UI/service threads alive even after a
    # headless Jython script has finished. This script is launched in its own
    # process with --allow-multiple, so explicitly terminating that process is
    # safe and prevents PowerShell from waiting indefinitely.
    System.exit(exit_code)
