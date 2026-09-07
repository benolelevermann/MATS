from __future__ import annotations

"""Export an nnU-Net skeleton/soma prediction as NeuroTreeTracer input.

NeuroTreeTracer (Kayasandik et al., Sci Rep 2018, doi:10.1038/s41598-018-24753-w)
starts *after* segmentation: ``runCenterLineParallel(inputSeg, inputSoma, option)``
assumes "the binary segmented image and the soma masks are given as input". Our
nnU-Net already produces exactly those two things, so we skip the paper's own
shearlet denoising / SVM segmentation / directional-ratio soma detection and feed
the network output straight into the tracer.

Two adaptations are required and both are deliberate:

1. Thickness. The tracer seeds neurite centerlines from the Euclidean distance
   transform: seeds are local maxima of ``Df`` and each seed suppresses other
   seeds within radius ``Df(s)``. That logic needs a neurite *body* several
   pixels wide. Our label 1 is a 1-px skeleton (half-width 1.0), whereas the
   reference ``inputSeg.mat`` shipped with the paper has neurite half-width
   ~2.2 px (median). We therefore dilate the skeleton - and only the skeleton -
   to restore a comparable thickness. The soma mask is never dilated.

2. Size. ``createRectangles`` pre-generates 18 x 10 x 360 = 64800 search
   rectangles, each a full image-sized logical mask. Memory is therefore
   64800 * H * W bytes and grows with the crop area, not with the cell count.
   A 512x512 crop already costs ~17 GB. This script refuses to write a crop
   whose predicted footprint exceeds --max-memory-gb.

The script only reads the prediction and writes into --output-dir. It never
modifies the source run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import io as sio
from scipy import ndimage as ndi

SCRIPT_VERSION = "neurotreetracer-export-v1-2026-08-19"

# Constants hard-coded inside the NeuroTreeTracer sources; mirrored here so the
# QC report can predict what the tracer will do with our data.
NTT_RECTANGLE_COUNT = 18 * 10 * 360  # createRectangles.m: lenIncrStep * nBands/2 * 360
NTT_COMPONENT_THRESHOLD = 300  # runCenterLineParallel.m: THR, drops components <= 300 px
NTT_MIN_NEURITE_LENGTH = 3  # traceNeurites.m: minNeuLen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--semantic",
        type=Path,
        required=True,
        help="0/1/2 label map (background/skeleton/soma), e.g. "
        "03_soma_preparation/<img>/04_semantic_soma_filled_and_filtered_0-1-2.tif",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("Y0", "Y1", "X0", "X1"),
        default=None,
        help="Explicit crop bounds (y0 y1 x0 x1, half-open). Omit to auto-select with --auto-size.",
    )
    parser.add_argument(
        "--auto-size",
        type=int,
        default=256,
        help="Side length of the auto-selected square crop (default: 256).",
    )
    parser.add_argument(
        "--dilate-skeleton",
        type=int,
        default=3,
        help="Dilation radius applied to the skeleton class to restore neurite thickness. "
        "Measured on this data: r=2 -> half-width 1.41, r=3 -> 2.00, r=4 -> 2.24, against the "
        "paper's reference of 2.24. Default 3 as a compromise between matching the reference "
        "and not inflating the skeleton (r=4 grows it ~10x). 0 disables.",
    )
    parser.add_argument(
        "--min-soma-px",
        type=int,
        default=200,
        help="Ignore soma components smaller than this (default: 200).",
    )
    parser.add_argument(
        "--keep-border-somas",
        action="store_true",
        help="Keep somas touching the crop border. The paper explicitly ignores them; "
        "off by default for that reason.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=24.0,
        help="Refuse to export if the predicted createRectangles footprint exceeds this (default: 24).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_semantic(path: Path) -> np.ndarray:
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2-D label map, got shape {array.shape}: {path}")
    values = set(np.unique(array).astype(int).tolist())
    if not values <= {0, 1, 2}:
        raise RuntimeError(f"Expected only labels 0/1/2, found {sorted(values)}: {path}")
    if 2 not in values:
        raise RuntimeError(f"No soma pixels (label 2) in {path}")
    return array.astype(np.uint8)


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def soma_attached_skeleton(semantic: np.ndarray) -> np.ndarray:
    """Skeleton pixels that share a connected component with some soma.

    NeuroTreeTracer traces outward from each soma, so detached skeleton debris is
    invisible to it. Scoring candidate crops by this mask - rather than by raw
    skeleton pixel count - is what keeps auto-selection away from debris fields.
    """
    structure = np.ones((3, 3), dtype=bool)
    labels, _ = ndi.label(semantic > 0, structure=structure)
    soma_components = set(np.unique(labels[semantic == 2]).tolist()) - {0}
    if not soma_components:
        return np.zeros_like(semantic, dtype=bool)
    return np.isin(labels, list(soma_components)) & (semantic == 1)


def auto_select_crop(semantic: np.ndarray, size: int, min_soma_px: int) -> tuple[int, int, int, int]:
    """Pick the square window with the most traceable structure.

    Ranked by soma-attached skeleton pixels first, then by the number of
    fully-contained somas. Ranking on raw skeleton pixels instead lands on
    debris fields, where the skeleton is thousands of 1-px specks.
    """
    height, width = semantic.shape
    size = min(size, height, width)
    labels, count = ndi.label(semantic == 2)
    if count == 0:
        raise RuntimeError("No soma components found.")
    objects = ndi.find_objects(labels)
    somas = []
    for index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        area = int(np.count_nonzero(labels[slices] == index))
        if area < min_soma_px:
            continue
        somas.append((slices[0].start, slices[0].stop, slices[1].start, slices[1].stop))
    if not somas:
        raise RuntimeError(f"No soma component reaches --min-soma-px={min_soma_px}.")

    skeleton = soma_attached_skeleton(semantic).astype(np.int64)
    integral = skeleton.cumsum(0).cumsum(1)

    def skeleton_in(y0: int, y1: int, x0: int, x1: int) -> int:
        a = integral[y1 - 1, x1 - 1]
        b = integral[y0 - 1, x1 - 1] if y0 > 0 else 0
        c = integral[y1 - 1, x0 - 1] if x0 > 0 else 0
        d = integral[y0 - 1, x0 - 1] if y0 > 0 and x0 > 0 else 0
        return int(a - b - c + d)

    best = None
    # Centre a candidate window on each soma; that guarantees at least one
    # fully-contained soma per candidate and keeps the search small.
    for sy0, sy1, sx0, sx1 in somas:
        cy = (sy0 + sy1) // 2
        cx = (sx0 + sx1) // 2
        y0 = int(np.clip(cy - size // 2, 0, height - size))
        x0 = int(np.clip(cx - size // 2, 0, width - size))
        y1, x1 = y0 + size, x0 + size
        contained = sum(1 for ay0, ay1, ax0, ax1 in somas if ay0 >= y0 and ay1 <= y1 and ax0 >= x0 and ax1 <= x1)
        score = (skeleton_in(y0, y1, x0, x1), contained)
        if best is None or score > best[0]:
            best = (score, (y0, y1, x0, x1))
    assert best is not None
    return best[1]


def main() -> None:
    args = parse_args()
    semantic_full = read_semantic(args.semantic)
    height, width = semantic_full.shape

    if args.crop is not None:
        y0, y1, x0, x1 = args.crop
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise RuntimeError(f"Crop {args.crop} is outside the image ({height}x{width}).")
        crop_source = "explicit"
    else:
        y0, y1, x0, x1 = auto_select_crop(semantic_full, args.auto_size, args.min_soma_px)
        crop_source = "auto"

    crop = semantic_full[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape

    memory_gb = NTT_RECTANGLE_COUNT * crop_h * crop_w / 1e9
    if memory_gb > args.max_memory_gb:
        raise RuntimeError(
            f"Crop {crop_h}x{crop_w} would make createRectangles allocate about {memory_gb:.1f} GB "
            f"({NTT_RECTANGLE_COUNT} full-size masks), above --max-memory-gb={args.max_memory_gb}.\n"
            f"Use a smaller --auto-size. Guide: 256x256 ~ {NTT_RECTANGLE_COUNT*256*256/1e9:.1f} GB, "
            f"384x384 ~ {NTT_RECTANGLE_COUNT*384*384/1e9:.1f} GB, "
            f"512x512 ~ {NTT_RECTANGLE_COUNT*512*512/1e9:.1f} GB."
        )

    soma = crop == 2
    skeleton = crop == 1

    # Drop soma specks; they would become spurious tracing roots.
    soma_labels, soma_count_raw = ndi.label(soma)
    keep = np.zeros(soma_count_raw + 1, dtype=bool)
    for index in range(1, soma_count_raw + 1):
        keep[index] = np.count_nonzero(soma_labels == index) >= args.min_soma_px
    dropped_small = int(soma_count_raw - keep[1:].sum())

    # The paper ignores cells whose soma overlaps the sub-image boundary.
    dropped_border = 0
    if not args.keep_border_somas:
        border = np.zeros_like(soma, dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        for index in range(1, soma_count_raw + 1):
            if keep[index] and (soma_labels == index)[border].any():
                keep[index] = False
                dropped_border += 1

    soma_kept = keep[soma_labels]
    soma_kept[soma_labels == 0] = False
    soma_count = int(keep[1:].sum())
    if soma_count == 0:
        raise RuntimeError(
            "No soma survives filtering in this crop. Try a different --crop, a larger "
            "--auto-size, a smaller --min-soma-px, or --keep-border-somas."
        )

    # Restore neurite thickness. Only the skeleton is dilated.
    components_before = int(ndi.label(skeleton)[1])
    if args.dilate_skeleton > 0:
        skeleton_thick = ndi.binary_dilation(skeleton, structure=disk(args.dilate_skeleton))
    else:
        skeleton_thick = skeleton.copy()
    components_after = int(ndi.label(skeleton_thick)[1])

    input_seg = (skeleton_thick | soma_kept).astype(np.uint8)
    input_soma = soma_kept.astype(np.uint8)

    # Predict what runCenterLineParallel's THR=300 filter will discard.
    seg_labels, seg_count = ndi.label(input_seg > 0)
    seg_sizes = np.bincount(seg_labels.ravel())
    surviving = np.flatnonzero(seg_sizes > NTT_COMPONENT_THRESHOLD)
    surviving = surviving[surviving != 0]
    kept_mask = np.isin(seg_labels, surviving)
    somas_lost_to_thr = 0
    for index in range(1, soma_count_raw + 1):
        if not keep[index]:
            continue
        if not kept_mask[soma_labels == index].any():
            somas_lost_to_thr += 1

    distance = ndi.distance_transform_edt(input_seg > 0)
    neurite_only = skeleton_thick & ~ndi.binary_dilation(soma_kept, structure=disk(6))
    half_width = float(np.median(distance[neurite_only])) if neurite_only.any() else float("nan")

    # Skeleton the tracer can actually reach: it only ever walks outward from a soma.
    attached = soma_attached_skeleton(crop)
    attached_fraction = float(attached.sum() / max(1, np.count_nonzero(skeleton)))

    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {out}\nPass --overwrite to replace it.")
    out.mkdir(parents=True, exist_ok=True)

    # MATLAB variable names must be exactly inputSeg / inputSoma: Script.m does
    # load('inputSeg') and then uses the variable of the same name.
    sio.savemat(out / "inputSeg.mat", {"inputSeg": input_seg}, do_compression=True)
    sio.savemat(out / "inputSoma.mat", {"inputSoma": input_soma}, do_compression=True)

    tifffile.imwrite(out / "crop_semantic_0-1-2.tif", crop)
    tifffile.imwrite(out / "crop_inputSeg_preview.tif", (input_seg * 255).astype(np.uint8))
    tifffile.imwrite(out / "crop_inputSoma_preview.tif", (input_soma * 255).astype(np.uint8))

    provenance = {
        "script_version": SCRIPT_VERSION,
        "semantic_source": str(args.semantic.resolve()),
        "source_shape": [int(height), int(width)],
        "crop_source": crop_source,
        "crop": {"y_min": int(y0), "y_max_exclusive": int(y1), "x_min": int(x0), "x_max_exclusive": int(x1)},
        "crop_shape": [int(crop_h), int(crop_w)],
        "dilate_skeleton_radius": int(args.dilate_skeleton),
        "min_soma_px": int(args.min_soma_px),
        "keep_border_somas": bool(args.keep_border_somas),
        "somas": {
            "raw_components": int(soma_count_raw),
            "dropped_below_min_px": int(dropped_small),
            "dropped_touching_border": int(dropped_border),
            "exported": int(soma_count),
            "predicted_lost_to_THR_300": int(somas_lost_to_thr),
        },
        "skeleton": {
            "pixels_before_dilation": int(np.count_nonzero(skeleton)),
            "pixels_after_dilation": int(np.count_nonzero(skeleton_thick)),
            "components_before_dilation": components_before,
            "components_after_dilation": components_after,
            "median_neurite_half_width_px": None if np.isnan(half_width) else round(half_width, 2),
            "paper_reference_half_width_px": 2.24,
            "fraction_attached_to_a_soma": round(attached_fraction, 4),
        },
        "neurotreetracer_constants": {
            "rectangle_count": NTT_RECTANGLE_COUNT,
            "component_threshold_px": NTT_COMPONENT_THRESHOLD,
            "min_neurite_length": NTT_MIN_NEURITE_LENGTH,
        },
        "predicted_createRectangles_memory_gb": round(memory_gb, 2),
    }
    (out / "export_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("=" * 72)
    print("NEUROTREETRACER INPUT EXPORTED")
    print("=" * 72)
    print(f"crop ({crop_source})          : y[{y0}:{y1}] x[{x0}:{x1}]  ->  {crop_h}x{crop_w}")
    print(f"somas exported            : {soma_count}"
          f"  (dropped: {dropped_small} too small, {dropped_border} on border)")
    if somas_lost_to_thr:
        print(f"  WARNING                 : {somas_lost_to_thr} of them sit on a component <= "
              f"{NTT_COMPONENT_THRESHOLD} px and will be discarded by runCenterLineParallel.")
    print(f"skeleton attached to soma : {100*attached_fraction:.1f}% of skeleton pixels "
          f"(the rest is unreachable for the tracer)")
    if attached_fraction < 0.4:
        print("  WARNING                 : most of the skeleton is detached from any soma. The tracer "
              "walks outward from somas only, so it will ignore that debris.")
    print(f"skeleton dilation r={args.dilate_skeleton}     : {components_before} -> {components_after} components, "
          f"median neurite half-width {half_width:.2f} px (paper reference 2.24)")
    if components_after < components_before:
        print(f"  note                    : dilation merged {components_before - components_after} skeleton "
              f"components; merged neurites read as crossings to the tracer.")
    print(f"createRectangles memory   : ~{memory_gb:.1f} GB")
    print(f"written to                : {out}")


if __name__ == "__main__":
    main()
