from __future__ import annotations

"""Convert NeuroTreeTracer output back into images and tables.

Reads ``traces_flat.mat`` written by run_neurotreetracer.m and produces:

  traced_soma_labels.tif   every traced pixel carries the id of its soma
  traced_neurite_ids.tif   every traced pixel carries a globally unique neurite id
  trace_points.csv         one row per trace point (soma_id, neurite_id, y, x)
  neurite_summary.csv      one row per neurite (length, endpoints, soma)
  overlay.png              traces drawn over the source crop, one colour per soma

The only subtlety is index order: MATLAB linear indices are 1-based and
column-major, so index k in an H x W image is row (k-1) % H and column
(k-1) // H. Getting this wrong transposes the whole result, which is why the
importer verifies that traced pixels actually land on segmented pixels and
reports the hit rate.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import io as sio
from scipy import ndimage as ndi

SCRIPT_VERSION = "neurotreetracer-import-v1-2026-08-19"

# Distinct colours for neighbouring somas; cycled if there are more.
PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
        [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
        [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
    ],
    dtype=np.uint8,
)


def conncomp_order(mask: np.ndarray) -> np.ndarray:
    """Label soma components exactly the way the tracer's connComp.m does.

    connComp.m runs bwconncomp(B,8) and then sort(numPixels,'descend'), so the
    soma_id in the traces is a SIZE RANK, not a raster-scan index: id 1 is the
    largest soma, id 2 the second largest. scipy's ndi.label numbers in raster
    order instead, so comparing the two directly attributes traces to the wrong
    cells. MATLAB's sort is stable, so ties keep bwconncomp's own order, which
    is by first pixel in column-major scan - reproduced here via the minimum
    column-major linear index.

    Returns an image whose non-zero values are the tracer's soma ids.
    """
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return labels.astype(np.uint16)
    height = mask.shape[0]
    stats = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        column_major = xs.astype(np.int64) * height + ys.astype(np.int64)
        stats.append((index, len(ys), int(column_major.min())))
    # size descending, then first-pixel order ascending (stable-sort tie-break)
    stats.sort(key=lambda item: (-item[1], item[2]))
    remap = np.zeros(count + 1, dtype=np.uint16)
    for rank, (index, _size, _first) in enumerate(stats, start=1):
        remap[index] = rank
    return remap[labels]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", type=Path, required=True, help="traces_flat.mat from run_neurotreetracer.m")
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="The neurotreetracer_export_crop.py output dir (for crop_semantic_0-1-2.tif and provenance).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mat = sio.loadmat(args.traces)
    if "traces" not in mat:
        raise RuntimeError(
            f"{args.traces} has no 'traces' variable. Was it written by run_neurotreetracer.m "
            f"with -v7? (scipy cannot read -v7.3/HDF5 files.)"
        )
    traces = np.asarray(mat["traces"], dtype=np.int64)
    image_size = np.asarray(mat["imageSize"]).ravel().astype(int)
    height, width = int(image_size[0]), int(image_size[1])

    semantic_path = args.export_dir / "crop_semantic_0-1-2.tif"
    semantic = np.squeeze(np.asarray(tifffile.imread(semantic_path))).astype(np.uint8)
    if semantic.shape != (height, width):
        raise RuntimeError(
            f"Shape mismatch: traces say {height}x{width}, {semantic_path.name} is {semantic.shape}."
        )

    provenance_path = args.export_dir / "export_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}

    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {out}\nPass --overwrite to replace it.")
    out.mkdir(parents=True, exist_ok=True)

    if traces.size == 0:
        raise RuntimeError(
            "The tracer returned no traces at all. Common causes: every component fell below "
            "runCenterLineParallel's THR=300, or the skeleton was left 1 px wide so seeding "
            "degenerated. Check export_provenance.json."
        )

    soma_ids = traces[:, 0]
    neurite_ids = traces[:, 1]
    linear = traces[:, 2]

    if linear.min() < 1 or linear.max() > height * width:
        raise RuntimeError(
            f"Linear indices out of range for a {height}x{width} image "
            f"(min {linear.min()}, max {linear.max()})."
        )

    # MATLAB: 1-based, column-major.
    rows = (linear - 1) % height
    cols = (linear - 1) // height

    # Check the traces against inputSeg - the image the tracer actually saw -
    # not against the undilated crop_semantic map. Traces legitimately run over
    # pixels that only exist after the skeleton dilation, so scoring them against
    # the original 0/1/2 map understates the hit rate and raises a false alarm.
    input_seg_path = args.export_dir / "inputSeg.mat"
    if input_seg_path.is_file():
        segmented = sio.loadmat(input_seg_path)["inputSeg"].astype(bool)
        reference = "inputSeg.mat (what the tracer traced)"
    else:
        segmented = semantic > 0
        reference = "crop_semantic_0-1-2.tif (inputSeg.mat missing)"
    if segmented.shape != (height, width):
        raise RuntimeError(f"inputSeg shape {segmented.shape} does not match traces {(height, width)}.")
    hit_rate = float(segmented[rows, cols].mean())

    # Also report against the original, undilated prediction for context.
    hit_rate_undilated = float((semantic > 0)[rows, cols].mean())

    soma_label_image = np.zeros((height, width), dtype=np.uint16)
    soma_label_image[rows, cols] = soma_ids.astype(np.uint16)

    # Globally unique neurite id, stable under sorting.
    pair_keys = soma_ids.astype(np.int64) * 100000 + neurite_ids.astype(np.int64)
    unique_pairs, pair_index = np.unique(pair_keys, return_inverse=True)
    neurite_label_image = np.zeros((height, width), dtype=np.uint16)
    neurite_label_image[rows, cols] = (pair_index + 1).astype(np.uint16)

    # Soma instances in the tracer's numbering, so soma_id in every table and
    # image refers to the same physical cell.
    soma_path = args.export_dir / "inputSoma.mat"
    soma_instances = None
    if soma_path.is_file():
        soma_mask = sio.loadmat(soma_path)["inputSoma"].astype(bool)
        soma_instances = conncomp_order(soma_mask)
        tifffile.imwrite(out / "soma_instances_tracer_ids.tif", soma_instances)

    tifffile.imwrite(out / "traced_soma_labels.tif", soma_label_image)
    tifffile.imwrite(out / "traced_neurite_ids.tif", neurite_label_image)

    with (out / "trace_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["soma_id", "neurite_id", "global_neurite_id", "y", "x", "on_segmentation"])
        for s, n, g, y, x in zip(soma_ids, neurite_ids, pair_index + 1, rows, cols):
            writer.writerow([int(s), int(n), int(g), int(y), int(x), int(segmented[y, x])])

    with (out / "neurite_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["global_neurite_id", "soma_id", "neurite_id", "n_points",
             "start_y", "start_x", "end_y", "end_x", "path_length_px", "fraction_on_segmentation"]
        )
        for g, key in enumerate(unique_pairs, start=1):
            sel = pair_index == (g - 1)
            ys, xs = rows[sel], cols[sel]
            steps = np.hypot(np.diff(ys.astype(float)), np.diff(xs.astype(float)))
            writer.writerow(
                [g, int(key // 100000), int(key % 100000), int(sel.sum()),
                 int(ys[0]), int(xs[0]), int(ys[-1]), int(xs[-1]),
                 round(float(steps.sum()), 2), round(float(segmented[ys, xs].mean()), 4)]
            )

    # Overlay: grey segmentation, coloured traces, white soma outlines.
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[semantic == 1] = (70, 70, 70)
    overlay[semantic == 2] = (130, 130, 130)
    colours = PALETTE[(soma_ids - 1) % len(PALETTE)]
    overlay[rows, cols] = colours
    try:
        from PIL import Image

        Image.fromarray(overlay).save(out / "overlay.png")
        overlay_written = True
    except Exception:
        tifffile.imwrite(out / "overlay.tif", overlay)
        overlay_written = False

    n_somas_traced = len(np.unique(soma_ids))
    summary = {
        "script_version": SCRIPT_VERSION,
        "traces_file": str(args.traces.resolve()),
        "image_shape": [height, width],
        "trace_points": int(traces.shape[0]),
        "neurites": int(len(unique_pairs)),
        "somas_with_at_least_one_trace": int(n_somas_traced),
        "somas_exported": provenance.get("somas", {}).get("exported"),
        "on_segmentation_reference": reference,
        "fraction_of_trace_points_on_segmentation": round(hit_rate, 4),
        "fraction_on_undilated_prediction": round(hit_rate_undilated, 4),
        "elapsed_seconds": float(np.asarray(mat["elapsedSec"]).ravel()[0]) if "elapsedSec" in mat else None,
        "export_provenance": provenance,
    }
    (out / "import_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("NEUROTREETRACER OUTPUT IMPORTED")
    print("=" * 72)
    print(f"trace points              : {traces.shape[0]:,}")
    print(f"neurites                  : {len(unique_pairs)}")
    exported = provenance.get("somas", {}).get("exported")
    print(f"somas with >=1 trace      : {n_somas_traced}" + (f" of {exported} exported" if exported else ""))
    print(f"on segmentation           : {100*hit_rate:.1f}%  [{reference}]")
    print(f"on undilated prediction   : {100*hit_rate_undilated:.1f}%  "
          f"(lower by construction: traces may run on dilation-added pixels)")
    if hit_rate < 0.9:
        print("  WARNING                 : a low hit rate against inputSeg means the index order is "
              "wrong (transposed result) or the traces drifted off the segmentation.")
    print(f"written to                : {out}" + ("" if overlay_written else "  (PIL missing, overlay as .tif)"))


if __name__ == "__main__":
    main()
