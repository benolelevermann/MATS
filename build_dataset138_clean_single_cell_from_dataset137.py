from __future__ import annotations

"""Create an auditable single-cell recrop dataset from Dataset137.

This tool deliberately has two separate commands:

``prepare``
    Reads every Dataset137 ``imagesTr``/``labelsTr`` pair, proposes a tighter
    crop around the *complete manual target* (skeleton OR soma), flags obvious
    unlabelled neighbouring signal for review, and writes a paginated HTML QC
    gallery plus ``selection_manifest.csv``. It never changes Dataset137 and
    does not create a new nnU-Net dataset yet.

``build``
    Reads the reviewed manifest and writes a new, fresh Dataset138-style
    nnU-Net dataset from rows where ``selected_for_dataset`` is true. The
    source TIFFs and labels are cropped identically, without rescaling. The
    original case IDs are retained so a filtered Dataset137 split can be kept
    for a fair comparison.

The automatic foreign-signal detector is intentionally a *review heuristic*,
not ground truth. A zero/background label must mean known absence, not merely
"not annotated". Therefore review cases default to not selected until a human
explicitly changes ``selected_for_dataset`` in the CSV.
"""

import argparse
import csv
import html
import json
import math
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


SCRIPT_VERSION = "dataset138-clean-single-cell-builder-v1-2026-08-19"
VALID_LABELS = {0, 1, 2}
SKELETON_RGB = np.array([0, 220, 255], dtype=np.float32)
SOMA_RGB = np.array([255, 45, 170], dtype=np.float32)
FOREIGN_RGB = np.array([255, 185, 0], dtype=np.float32)


@dataclass(frozen=True)
class Pair:
    case_id: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Create a QC gallery and editable selection manifest; does not build a dataset.",
    )
    prepare.add_argument("--source-dataset", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--margin-px",
        type=int,
        default=8,
        help="Context retained around the full manual skeleton+soma target (default: 8).",
    )
    prepare.add_argument(
        "--min-crop-size",
        type=int,
        default=0,
        help="Optional minimum height and width (default: 0 = no padding; nnU-Net can pad).",
    )
    prepare.add_argument(
        "--source-edge-review-px",
        type=int,
        default=8,
        help="Flag a target close to its existing source crop edge for review (default: 8).",
    )
    prepare.add_argument(
        "--skeleton-exclusion-radius",
        type=int,
        default=8,
        help="Pixels around annotated skeleton ignored by the foreign-signal heuristic.",
    )
    prepare.add_argument(
        "--soma-exclusion-radius",
        type=int,
        default=14,
        help="Pixels around annotated soma ignored by the foreign-signal heuristic.",
    )
    prepare.add_argument("--smoothing-sigma", type=float, default=1.5)
    prepare.add_argument(
        "--foreground-mad-multiplier",
        type=float,
        default=5.0,
        help="Conservative raw-signal threshold above local background (default: 5.0).",
    )
    prepare.add_argument("--foreign-min-pixels", type=int, default=50)
    prepare.add_argument(
        "--foreign-relative-soma-area",
        type=float,
        default=0.15,
        help="A foreign component must also reach this fraction of target soma area (default: 0.15).",
    )
    prepare.add_argument("--page-size", type=int, default=48)
    prepare.add_argument("--preview-size", type=int, default=220)
    prepare.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
    )
    prepare.add_argument("--overwrite", action="store_true")

    build = commands.add_parser(
        "build",
        help="Build a fresh cropped nnU-Net dataset from an edited selection manifest.",
    )
    build.add_argument("--source-dataset", type=Path, required=True)
    build.add_argument("--selection-manifest", type=Path, required=True)
    build.add_argument("--output-dataset", type=Path, required=True)
    build.add_argument(
        "--source-splits",
        type=Path,
        default=None,
        help="Optional Dataset137 splits_final.json. A filtered copy is written as provenance.",
    )
    build.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_new_output(path: Path, overwrite: bool) -> None:
    """Create only the explicit output path; never touch a source dataset."""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {path}\n"
                "Use a new output directory, or explicitly pass --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_2d(path: Path, description: str) -> np.ndarray:
    try:
        array = tifffile.imread(path)
    except Exception as error:
        raise RuntimeError(f"Could not read {description}: {path}\n{error}") from error
    array = np.squeeze(np.asarray(array))
    if array.ndim != 2:
        raise RuntimeError(f"{description} must be 2-D, got {array.shape}: {path}")
    return array


def collect_pairs(dataset_dir: Path) -> dict[str, Pair]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected imagesTr and labelsTr in {dataset_dir}")
    pairs: dict[str, Pair] = {}
    for image_path in sorted(images_dir.glob("*_0000.tif"), key=lambda path: path.name.lower()):
        case_id = image_path.name[: -len("_0000.tif")]
        label_path = labels_dir / f"{case_id}.tif"
        if not label_path.is_file():
            raise RuntimeError(f"Missing label for {image_path.name}: {label_path}")
        if case_id in pairs:
            raise RuntimeError(f"Duplicate case ID: {case_id}")
        pairs[case_id] = Pair(case_id, image_path, label_path)
    label_ids = {path.stem for path in labels_dir.glob("*.tif")}
    unmatched_labels = label_ids - set(pairs)
    if unmatched_labels:
        preview = ", ".join(sorted(unmatched_labels)[:8])
        raise RuntimeError(f"Labels without matching images, for example: {preview}")
    if not pairs:
        raise RuntimeError(f"No '*_0000.tif' images in {images_dir}")
    return pairs


def disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def validate_pair(pair: Pair, raw: np.ndarray, label: np.ndarray) -> None:
    if raw.shape != label.shape:
        raise RuntimeError(
            f"Shape mismatch for {pair.case_id}: image={raw.shape}, label={label.shape}"
        )
    values = set(np.unique(label).astype(int).tolist())
    invalid = values - VALID_LABELS
    if invalid:
        raise RuntimeError(f"Invalid label values for {pair.case_id}: {sorted(invalid)}")
    if not np.any(label == 1):
        raise RuntimeError(f"No skeleton label pixels: {pair.case_id}")
    if not np.any(label == 2):
        raise RuntimeError(f"No soma label pixels: {pair.case_id}")


def expand_axis(start: int, end: int, limit: int, minimum: int) -> tuple[int, int]:
    """Grow an interval symmetrically where possible, preserving its current content."""
    current = end - start
    if current >= minimum:
        return start, end
    needed = minimum - current
    before = min(start, needed // 2)
    after = min(limit - end, needed - before)
    start -= before
    end += after
    remaining = minimum - (end - start)
    if remaining > 0:
        grow_before = min(start, remaining)
        start -= grow_before
        remaining -= grow_before
    if remaining > 0:
        end += min(limit - end, remaining)
    return start, end


def crop_bounds(target: np.ndarray, margin_px: int, min_crop_size: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(target)
    if not len(ys):
        raise RuntimeError("Cannot crop an empty target mask.")
    height, width = target.shape
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(height, int(ys.max()) + 1 + margin_px)
    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(width, int(xs.max()) + 1 + margin_px)
    y0, y1 = expand_axis(y0, y1, height, min_crop_size)
    x0, x1 = expand_axis(x0, x1, width, min_crop_size)
    return y0, y1, x0, x1


def min_mask_edge_distance(mask: np.ndarray) -> int:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return -1
    height, width = mask.shape
    return int(min(ys.min(), xs.min(), height - 1 - ys.max(), width - 1 - xs.max()))


def foreign_signal_mask(
    raw: np.ndarray,
    label: np.ndarray,
    skeleton_exclusion_radius: int,
    soma_exclusion_radius: int,
    smoothing_sigma: float,
    foreground_mad_multiplier: float,
    foreign_min_pixels: int,
    foreign_relative_soma_area: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Flag large bright components not explained by the manual target label.

    The result is deliberately conservative and only drives an automatic
    *review* status. It must never be interpreted as a segmentation mask.
    """
    skeleton = label == 1
    soma = label == 2
    support = ndi.binary_dilation(skeleton, structure=disk(skeleton_exclusion_radius))
    support |= ndi.binary_dilation(soma, structure=disk(soma_exclusion_radius))

    smooth = ndi.gaussian_filter(np.asarray(raw, dtype=np.float32), sigma=smoothing_sigma)
    background_values = smooth[~support]
    if background_values.size < 64:
        background_values = smooth.ravel()
    median = float(np.median(background_values))
    mad = float(np.median(np.abs(background_values - median)))
    robust_sigma = 1.4826 * mad
    percentile_floor = float(np.percentile(background_values, 99.0))
    threshold = max(median + foreground_mad_multiplier * robust_sigma, percentile_floor)
    # Strict greater-than is important for flat backgrounds: with a median/MAD
    # threshold equal to the constant background value, >= would incorrectly
    # label every background pixel as a candidate neighbour.
    bright = (smooth > threshold) & ~support

    # Join the adjacent pixels of a compact bright neighbour, but avoid a
    # broad morphology operation that could turn fine background noise into a
    # candidate cell.
    bright = ndi.binary_closing(bright, structure=disk(1))
    components, count = ndi.label(bright, structure=np.ones((3, 3), dtype=np.uint8))
    soma_area = int(np.count_nonzero(soma))
    significant_min_area = max(
        int(foreign_min_pixels),
        int(math.ceil(max(0.0, foreign_relative_soma_area) * max(1, soma_area))),
    )

    foreign = np.zeros_like(bright, dtype=bool)
    component_areas: list[int] = []
    for identifier in range(1, count + 1):
        component = components == identifier
        area = int(np.count_nonzero(component))
        if area >= significant_min_area:
            foreign |= component
            component_areas.append(area)
    return foreign, {
        "foreign_detection_threshold": threshold,
        "foreign_background_median": median,
        "foreign_background_robust_sigma": robust_sigma,
        "foreign_min_component_pixels": significant_min_area,
        "foreign_component_count": len(component_areas),
        "foreign_largest_component_pixels": max(component_areas, default=0),
        "foreign_total_component_pixels": int(np.count_nonzero(foreign)),
    }


def preview_stride(shape: tuple[int, int], max_size: int) -> int:
    return max(1, math.ceil(max(shape) / max(1, max_size)))


def downsample_label(label: np.ndarray, stride: int) -> np.ndarray:
    """Block-max labels so a one-pixel skeleton remains visible in QC only."""
    if stride == 1:
        return np.asarray(label, dtype=np.uint8)
    height, width = label.shape
    out_height = math.ceil(height / stride)
    out_width = math.ceil(width / stride)
    padded = np.zeros((out_height * stride, out_width * stride), dtype=np.uint8)
    padded[:height, :width] = np.asarray(label, dtype=np.uint8)
    blocks = padded.reshape(out_height, stride, out_width, stride)
    result = np.zeros((out_height, out_width), dtype=np.uint8)
    result[np.any(blocks == 1, axis=(1, 3))] = 1
    result[np.any(blocks == 2, axis=(1, 3))] = 2
    return result


def downsample_mask(mask: np.ndarray, stride: int) -> np.ndarray:
    if stride == 1:
        return np.asarray(mask, dtype=bool)
    height, width = mask.shape
    out_height = math.ceil(height / stride)
    out_width = math.ceil(width / stride)
    padded = np.zeros((out_height * stride, out_width * stride), dtype=bool)
    padded[:height, :width] = mask
    blocks = padded.reshape(out_height, stride, out_width, stride)
    return np.any(blocks, axis=(1, 3))


def normalize_raw(raw: np.ndarray) -> np.ndarray:
    raw_float = np.asarray(raw, dtype=np.float32)
    low, high = np.percentile(raw_float, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(raw_float.min()), float(raw_float.max())
    if high <= low:
        high = low + 1.0
    return np.uint8(np.clip((raw_float - low) / (high - low), 0.0, 1.0) * 255.0)


def overlay(raw_gray: np.ndarray, label: np.ndarray, foreign: np.ndarray | None = None) -> np.ndarray:
    rgb = np.repeat(np.asarray(raw_gray)[..., None], 3, axis=2).astype(np.float32)
    skeleton = label == 1
    soma = label == 2
    for mask, color, alpha in ((skeleton, SKELETON_RGB, 0.88), (soma, SOMA_RGB, 0.82)):
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    if foreign is not None and np.any(foreign):
        rgb[foreign] = 0.35 * rgb[foreign] + 0.65 * FOREIGN_RGB
    return np.uint8(np.clip(rgb, 0, 255))


def preview_panel(raw: np.ndarray, label: np.ndarray, foreign: np.ndarray | None, preview_size: int) -> np.ndarray:
    stride = preview_stride(raw.shape, preview_size)
    raw_small = normalize_raw(raw[::stride, ::stride])
    label_small = downsample_label(label, stride)
    foreign_small = downsample_mask(foreign, stride) if foreign is not None else None
    if raw_small.shape != label_small.shape:
        raise RuntimeError("Preview raw/label shape mismatch.")
    return overlay(raw_small, label_small, foreign_small)


def make_preview(
    raw: np.ndarray,
    label: np.ndarray,
    crop_raw: np.ndarray,
    crop_label: np.ndarray,
    foreign: np.ndarray,
    preview_size: int,
) -> np.ndarray:
    original = preview_panel(raw, label, None, preview_size)
    cropped = preview_panel(crop_raw, crop_label, foreign, preview_size)
    output_height = max(original.shape[0], cropped.shape[0])

    def pad_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == output_height:
            return image
        pad = output_height - image.shape[0]
        return np.pad(image, ((0, pad), (0, 0), (0, 0)), constant_values=245)

    divider = np.full((output_height, 4, 3), 242, dtype=np.uint8)
    return np.concatenate([pad_height(original), divider, pad_height(cropped)], axis=1)


def status_from_metrics(
    target_source_edge_distance: int,
    source_edge_review_px: int,
    foreign_component_count: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if target_source_edge_distance <= 0:
        return "rejected", ["target_touches_existing_source_crop_edge"]
    if target_source_edge_distance < source_edge_review_px:
        reasons.append("target_near_existing_source_crop_edge")
    if foreign_component_count:
        reasons.append("possible_unlabelled_bright_neighbour")
    if reasons:
        return "review", reasons
    return "accepted", reasons


def analyse_pair(pair: Pair, args: argparse.Namespace, assets_dir: Path) -> dict[str, Any]:
    raw = read_2d(pair.image_path, "raw image")
    label = read_2d(pair.label_path, "manual label")
    validate_pair(pair, raw, label)
    target = label > 0
    y0, y1, x0, x1 = crop_bounds(target, args.margin_px, args.min_crop_size)
    crop_raw = raw[y0:y1, x0:x1]
    crop_label = label[y0:y1, x0:x1]
    target_source_edge_distance = min_mask_edge_distance(target)
    target_crop_edge_distance = min_mask_edge_distance(crop_label > 0)
    foreign, foreign_metrics = foreign_signal_mask(
        crop_raw,
        crop_label,
        args.skeleton_exclusion_radius,
        args.soma_exclusion_radius,
        args.smoothing_sigma,
        args.foreground_mad_multiplier,
        args.foreign_min_pixels,
        args.foreign_relative_soma_area,
    )
    auto_status, reasons = status_from_metrics(
        target_source_edge_distance,
        args.source_edge_review_px,
        int(foreign_metrics["foreign_component_count"]),
    )
    asset_path = assets_dir / f"{pair.case_id}.webp"
    Image.fromarray(
        make_preview(raw, label, crop_raw, crop_label, foreign, args.preview_size)
    ).save(asset_path, "WEBP", quality=88, method=4)

    return {
        "case_id": pair.case_id,
        "selected_for_dataset": "1" if auto_status == "accepted" else "0",
        "manual_note": "",
        "auto_status": auto_status,
        "auto_reasons": ";".join(reasons),
        "source_image_path": str(pair.image_path),
        "source_label_path": str(pair.label_path),
        "source_height_px": int(raw.shape[0]),
        "source_width_px": int(raw.shape[1]),
        "raw_dtype": str(raw.dtype),
        "crop_y_min": y0,
        "crop_y_max_exclusive": y1,
        "crop_x_min": x0,
        "crop_x_max_exclusive": x1,
        "crop_height_px": int(y1 - y0),
        "crop_width_px": int(x1 - x0),
        "target_source_edge_distance_px": target_source_edge_distance,
        "target_crop_edge_distance_px": target_crop_edge_distance,
        "skeleton_pixels": int(np.count_nonzero(label == 1)),
        "soma_pixels": int(np.count_nonzero(label == 2)),
        "crop_skeleton_pixels": int(np.count_nonzero(crop_label == 1)),
        "crop_soma_pixels": int(np.count_nonzero(crop_label == 2)),
        "asset": f"assets/{asset_path.name}",
        **foreign_metrics,
    }


MANIFEST_FIELDS = [
    "case_id",
    "selected_for_dataset",
    "manual_note",
    "auto_status",
    "auto_reasons",
    "source_image_path",
    "source_label_path",
    "source_height_px",
    "source_width_px",
    "raw_dtype",
    "crop_y_min",
    "crop_y_max_exclusive",
    "crop_x_min",
    "crop_x_max_exclusive",
    "crop_height_px",
    "crop_width_px",
    "target_source_edge_distance_px",
    "target_crop_edge_distance_px",
    "skeleton_pixels",
    "soma_pixels",
    "crop_skeleton_pixels",
    "crop_soma_pixels",
    "foreign_detection_threshold",
    "foreign_background_median",
    "foreign_background_robust_sigma",
    "foreign_min_component_pixels",
    "foreign_component_count",
    "foreign_largest_component_pixels",
    "foreign_total_component_pixels",
    "asset",
]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def status_badge(status: str) -> str:
    safe = html.escape(status)
    return f'<span class="badge badge-{safe}">{safe}</span>'


def navigation(page_number: int, page_count: int, prefix: str = "") -> str:
    entries = []
    for number in range(1, page_count + 1):
        if number == page_number:
            entries.append(f'<span class="current">{number}</span>')
        else:
            entries.append(f'<a href="{prefix}page_{number:03d}.html">{number}</a>')
    return '<nav class="nav">' + "".join(entries) + "</nav>"


def common_css() -> str:
    return """
body { margin:0; background:#f5f7fb; color:#142033; font-family:Arial,sans-serif; }
main { max-width:2200px; margin:auto; padding:22px; }
h1,h2 { margin:0 0 10px; } .intro { max-width:1100px; color:#4c5d73; line-height:1.45; }
.stats { display:flex; gap:12px; flex-wrap:wrap; margin:18px 0; }.stat { background:#fff; border:1px solid #dce3ee; border-radius:9px; padding:12px 15px; min-width:145px; }
.stat strong { display:block; font-size:24px; }.nav { display:flex; gap:6px; flex-wrap:wrap; margin:18px 0; }.nav a,.nav span { border:1px solid #ccd7e6; padding:6px 9px; border-radius:5px; background:#fff; text-decoration:none; color:#174b71; }.nav .current { background:#174b71; color:#fff; border-color:#174b71; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:14px; }.card { background:#fff; border:1px solid #dce3ee; border-radius:9px; overflow:hidden; padding:9px; }.card img { width:100%; display:block; background:#111; }.card h3 { margin:8px 0 5px; font-size:16px; }.meta { color:#526177; font-size:12px; line-height:1.4; word-break:break-word; }.badge { float:right; padding:3px 7px; border-radius:12px; color:#fff; font-size:12px; }.badge-accepted { background:#18834b; }.badge-review { background:#b16a00; }.badge-rejected { background:#b12626; }.note { background:#fff3ce; border-left:4px solid #d59d09; padding:10px 12px; margin:14px 0; color:#5c4700; }.legend { margin:12px 0; color:#526177; }.cyan { color:#009fbd; font-weight:bold; }.magenta { color:#d50089; font-weight:bold; }.orange { color:#c77b00; font-weight:bold; }
"""


def write_index(output_dir: Path, summary: dict[str, Any], page_count: int) -> None:
    counts = summary["auto_status_counts"]
    # The index lives one directory above the rendered pages.
    page_links = navigation(0, page_count, prefix="pages/")
    index = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dataset138 single-cell recrop QC</title><style>{common_css()}</style></head><body><main>
<h1>Dataset138 clean single-cell recrop QC</h1>
<p class="intro">Every card compares the original Dataset137 pair (left) with its proposed tighter crop (right). Cyan is the manual skeleton, magenta is the manual soma, and orange is a conservative raw-image heuristic for possible unlabelled neighbouring signal. Orange is a review cue, not a new label.</p>
<p class="note"><strong>Before building Dataset138:</strong> inspect <code>selection_manifest.csv</code>. Only rows with <code>selected_for_dataset = 1</code> are used by the build command. You may promote a good review case to 1 or deselect any accepted case. Dataset137 is never modified.</p>
<div class="stats"><div class="stat"><strong>{summary['total_cases']:,}</strong>source cases</div><div class="stat"><strong>{counts.get('accepted', 0):,}</strong>auto accepted</div><div class="stat"><strong>{counts.get('review', 0):,}</strong>review</div><div class="stat"><strong>{counts.get('rejected', 0):,}</strong>rejected</div></div>
<p class="legend"><span class="cyan">Cyan</span> = manual skeleton; <span class="magenta">magenta</span> = manual soma; <span class="orange">orange</span> = possible unlabelled neighbouring raw signal. Source target labels are never modified.</p>
{page_links}
<p><a href="selection_manifest.csv">Download/edit selection_manifest.csv</a> &nbsp;|&nbsp; <a href="recrop_summary.json">recrop_summary.json</a></p>
</main></body></html>"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")


def write_pages(output_dir: Path, rows: list[dict[str, Any]], page_size: int, summary: dict[str, Any]) -> int:
    pages_dir = output_dir / "pages"
    pages_dir.mkdir()
    ordered = sorted(rows, key=lambda row: ({"review": 0, "rejected": 1, "accepted": 2}.get(row["auto_status"], 3), row["case_id"]))
    groups = list(chunks(ordered, page_size))
    for page_number, page_rows in enumerate(groups, start=1):
        cards = []
        for row in page_rows:
            reason = html.escape(row["auto_reasons"] or "none")
            cards.append(
                f'''<article class="card"><img loading="lazy" src="../{html.escape(row['asset'])}" alt="{html.escape(row['case_id'])} original and proposed recrop"><h3>{html.escape(row['case_id'])}{status_badge(row['auto_status'])}</h3><div class="meta">original {row['source_width_px']} x {row['source_height_px']} px → crop {row['crop_width_px']} x {row['crop_height_px']} px</div><div class="meta">skeleton {row['crop_skeleton_pixels']:,} px | soma {row['crop_soma_pixels']:,} px | target source-edge distance {row['target_source_edge_distance_px']} px</div><div class="meta">foreign candidates {row['foreign_component_count']} | largest {row['foreign_largest_component_pixels']:,} px | reason: {reason}</div><div class="meta">selected_for_dataset: {html.escape(row['selected_for_dataset'])}</div></article>'''
            )
        document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dataset138 recrop QC page {page_number}</title><style>{common_css()}</style></head><body><main>
<h1>Dataset138 clean single-cell recrop QC — page {page_number}/{len(groups)}</h1><p class="intro">Left = original Dataset137 raw + manual label. Right = proposed crop. Orange only indicates a possible unlabelled neighbour from raw intensity.</p>
<p><a href="../index.html">← overview</a> &nbsp;|&nbsp; <a href="../selection_manifest.csv">selection manifest</a></p>{navigation(page_number, len(groups))}
<section class="grid">{''.join(cards)}</section>{navigation(page_number, len(groups))}
</main></body></html>"""
        (pages_dir / f"page_{page_number:03d}.html").write_text(document, encoding="utf-8")
    return len(groups)


def prepare(args: argparse.Namespace) -> None:
    if args.margin_px < 0 or args.min_crop_size < 0:
        raise ValueError("--margin-px and --min-crop-size must be >=0.")
    if args.preview_size < 32 or args.page_size < 1 or args.workers < 1:
        raise ValueError("--preview-size, --page-size and --workers must be positive.")
    pairs = collect_pairs(args.source_dataset)
    require_new_output(args.output_dir, args.overwrite)
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir()
    print(f"Preparing recrop QC for {len(pairs):,} Dataset137 cases...")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyse_pair, pair, args, assets_dir): case_id
            for case_id, pair in pairs.items()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:  # collect all failures for a useful audit
                failures.append(f"{case_id}: {error}")
            if index % 100 == 0 or index == len(futures):
                print(f"Analysed {index:,}/{len(futures):,} cases")
    if failures:
        (args.output_dir / "analysis_failures.txt").write_text("\n".join(failures), encoding="utf-8")
        raise RuntimeError(
            f"Recrop QC failed for {len(failures)} cases. See {args.output_dir / 'analysis_failures.txt'}"
        )
    rows.sort(key=lambda row: row["case_id"])
    write_csv(args.output_dir / "selection_manifest.csv", rows, MANIFEST_FIELDS)
    status_counts = Counter(row["auto_status"] for row in rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "source_dataset": str(args.source_dataset),
        "total_cases": len(rows),
        "auto_status_counts": dict(sorted(status_counts.items())),
        "auto_selected_count": sum(row["selected_for_dataset"] == "1" for row in rows),
        "settings": {
            "margin_px": args.margin_px,
            "min_crop_size": args.min_crop_size,
            "source_edge_review_px": args.source_edge_review_px,
            "skeleton_exclusion_radius": args.skeleton_exclusion_radius,
            "soma_exclusion_radius": args.soma_exclusion_radius,
            "smoothing_sigma": args.smoothing_sigma,
            "foreground_mad_multiplier": args.foreground_mad_multiplier,
            "foreign_min_pixels": args.foreign_min_pixels,
            "foreign_relative_soma_area": args.foreign_relative_soma_area,
        },
    }
    (args.output_dir / "recrop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    page_count = write_pages(args.output_dir, rows, args.page_size, summary)
    write_index(args.output_dir, summary, page_count)
    print("\n" + "=" * 72)
    print("SINGLE-CELL RECROP QC COMPLETE")
    print("=" * 72)
    print(f"Output:          {args.output_dir}")
    print(f"Auto accepted:   {status_counts.get('accepted', 0):,}")
    print(f"Review:          {status_counts.get('review', 0):,}")
    print("Next: inspect index.html, then edit selection_manifest.csv before build.")


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        missing = {"case_id", "selected_for_dataset", "crop_y_min", "crop_y_max_exclusive", "crop_x_min", "crop_x_max_exclusive"} - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Selection manifest lacks columns: {', '.join(sorted(missing))}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def selected(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "accept", "accepted", "keep"}


def parse_bound(row: dict[str, str], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid {name} for {row.get('case_id', '<unknown>')}: {row.get(name)!r}") from error


def validate_selection(rows: list[dict[str, str]], pairs: dict[str, Pair]) -> list[dict[str, str]]:
    selected_rows = [row for row in rows if selected(row["selected_for_dataset"])]
    if not selected_rows:
        raise RuntimeError("No rows are selected. Set selected_for_dataset=1 for accepted cases.")
    seen: set[str] = set()
    for row in selected_rows:
        case_id = row["case_id"]
        if case_id in seen:
            raise RuntimeError(f"Duplicate selected case ID in manifest: {case_id}")
        seen.add(case_id)
        if case_id not in pairs:
            raise RuntimeError(f"Selected case is not in the source dataset: {case_id}")
        y0, y1 = parse_bound(row, "crop_y_min"), parse_bound(row, "crop_y_max_exclusive")
        x0, x1 = parse_bound(row, "crop_x_min"), parse_bound(row, "crop_x_max_exclusive")
        if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
            raise RuntimeError(f"Invalid crop bounds for {case_id}")
    return selected_rows


def filtered_splits(source_splits: Path, selected_ids: set[str]) -> list[dict[str, list[str]]]:
    data = json.loads(source_splits.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Source splits JSON must contain a list of split dictionaries.")
    result = []
    for number, split in enumerate(data):
        train = [str(case_id) for case_id in split.get("train", []) if str(case_id) in selected_ids]
        val = [str(case_id) for case_id in split.get("val", []) if str(case_id) in selected_ids]
        overlap = set(train) & set(val)
        if overlap:
            raise RuntimeError(f"Filtered split {number} has train/val overlap: {sorted(overlap)[:4]}")
        if not train or not val:
            raise RuntimeError(
                f"Filtered split {number} has an empty train or validation set; do not use it."
            )
        result.append({"train": train, "val": val})
    return result


def build(args: argparse.Namespace) -> None:
    source_json_path = args.source_dataset / "dataset.json"
    if not source_json_path.is_file():
        raise FileNotFoundError(source_json_path)
    source_json = json.loads(source_json_path.read_text(encoding="utf-8"))
    if source_json.get("labels") != {"background": 0, "skeleton": 1, "soma": 2}:
        raise RuntimeError("Source dataset does not have the expected labels 0/1/2.")
    pairs = collect_pairs(args.source_dataset)
    manifest_rows = read_manifest(args.selection_manifest)
    selected_rows = validate_selection(manifest_rows, pairs)
    selected_ids = {row["case_id"] for row in selected_rows}
    splits = None
    if args.source_splits is not None:
        if not args.source_splits.is_file():
            raise FileNotFoundError(args.source_splits)
        splits = filtered_splits(args.source_splits, selected_ids)

    # Validate every selected crop before writing a single output file.
    prepared: list[tuple[dict[str, str], Pair, np.ndarray, np.ndarray]] = []
    print(f"Validating {len(selected_rows):,} selected cases before dataset creation...")
    for index, row in enumerate(sorted(selected_rows, key=lambda item: item["case_id"]), start=1):
        pair = pairs[row["case_id"]]
        raw = read_2d(pair.image_path, "source raw image")
        label = read_2d(pair.label_path, "source label")
        validate_pair(pair, raw, label)
        y0, y1 = parse_bound(row, "crop_y_min"), parse_bound(row, "crop_y_max_exclusive")
        x0, x1 = parse_bound(row, "crop_x_min"), parse_bound(row, "crop_x_max_exclusive")
        if y1 > raw.shape[0] or x1 > raw.shape[1]:
            raise RuntimeError(f"Crop bounds are outside source image for {pair.case_id}")
        crop_raw = raw[y0:y1, x0:x1]
        crop_label = label[y0:y1, x0:x1]
        validate_pair(pair, crop_raw, crop_label)
        prepared.append((row, pair, crop_raw, crop_label))
        if index % 100 == 0 or index == len(selected_rows):
            print(f"Validated {index:,}/{len(selected_rows):,}")

    require_new_output(args.output_dataset, args.overwrite)
    images_out = args.output_dataset / "imagesTr"
    labels_out = args.output_dataset / "labelsTr"
    images_out.mkdir()
    labels_out.mkdir()
    report_rows: list[dict[str, Any]] = []
    for index, (row, pair, crop_raw, crop_label) in enumerate(prepared, start=1):
        tifffile.imwrite(images_out / f"{pair.case_id}_0000.tif", crop_raw)
        tifffile.imwrite(labels_out / f"{pair.case_id}.tif", np.asarray(crop_label, dtype=np.uint8))
        report_rows.append(
            {
                **row,
                "output_image": str(images_out / f"{pair.case_id}_0000.tif"),
                "output_label": str(labels_out / f"{pair.case_id}.tif"),
                "output_height_px": int(crop_raw.shape[0]),
                "output_width_px": int(crop_raw.shape[1]),
                "output_skeleton_pixels": int(np.count_nonzero(crop_label == 1)),
                "output_soma_pixels": int(np.count_nonzero(crop_label == 2)),
            }
        )
        if index % 100 == 0 or index == len(prepared):
            print(f"Wrote {index:,}/{len(prepared):,} cropped pairs")

    dataset_json = {
        "channel_names": {"0": "image"},
        "labels": {"background": 0, "skeleton": 1, "soma": 2},
        "numTraining": len(prepared),
        "file_ending": ".tif",
    }
    (args.output_dataset / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    report_fields = list(MANIFEST_FIELDS) + [
        "output_image",
        "output_label",
        "output_height_px",
        "output_width_px",
        "output_skeleton_pixels",
        "output_soma_pixels",
    ]
    write_csv(args.output_dataset / "dataset138_clean_single_cell_build_report.csv", report_rows, report_fields)
    shutil.copy2(args.selection_manifest, args.output_dataset / "selection_manifest_used.csv")
    if splits is not None:
        (args.output_dataset / "splits_final_filtered_from_dataset137.json").write_text(
            json.dumps(splits, indent=2), encoding="utf-8"
        )
    readme = f"""Dataset built by {SCRIPT_VERSION}\n\nSource dataset: {args.source_dataset}\nSelected cases: {len(prepared)}\nCase IDs were retained from the source dataset.\n\nImportant: This folder contains a filtered source split only as provenance. After\nplanning/preprocessing this new dataset, copy the file\n`splits_final_filtered_from_dataset137.json` into the new nnUNet_preprocessed\ndataset directory as `splits_final.json` if you want to preserve the source\ntrain/validation membership for every retained case.\n"""
    (args.output_dataset / "README_DATASET138.txt").write_text(readme, encoding="utf-8")
    summary = {
        "script_version": SCRIPT_VERSION,
        "source_dataset": str(args.source_dataset),
        "selection_manifest": str(args.selection_manifest),
        "output_dataset": str(args.output_dataset),
        "selected_cases": len(prepared),
        "source_case_count": len(pairs),
        "source_splits": str(args.source_splits) if args.source_splits else None,
    }
    (args.output_dataset / "dataset138_build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 72)
    print("CLEAN SINGLE-CELL DATASET BUILD COMPLETE")
    print("=" * 72)
    print(f"Selected cases: {len(prepared):,}/{len(pairs):,}")
    print(f"Dataset:        {args.output_dataset}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "build":
        build(args)
    else:  # argparse prevents this, but keep a defensive message.
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
