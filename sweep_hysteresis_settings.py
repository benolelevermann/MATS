from __future__ import annotations

"""Empirically compare hysteresis settings on one and the same set of cells.

Generalises sweep_hysteresis_tlow.py from a single T_low axis to a full grid of
settings. One nnU-Net probability archive is loaded once, then every requested
setting is applied to it:

  argmax          plain nnU-Net argmax, the baseline the network produces
  adaptive alpha  Otsu-derived T+/T- exactly as apply_adaptive_hysteresis.py and
                  therefore as the current web pipeline uses them
  T_high/T_low    any fixed threshold pair, seeded at T_high and grown to T_low

Soma pixels always stay the unchanged argmax soma, so the soma labelling - and
with it the identity and the crop window of every cell - is identical in every
setting. That is what makes the side-by-side comparison meaningful.

Outputs below --output-dir:

  settings_sweep.csv              one row per setting, image-level metrics
  settings_cell_comparison.csv    one row per cell and setting
  settings_sweep_summary.json     machine readable summary of the whole sweep
  settings_sweep_report.html      the comparison report to open in the browser
  <setting>/preview.png           full-image overlay per setting
  <setting>/*.tif                 only with --write-full-tifs (~100 MB each)
  cell_crops/soma_XXXX/*.png      the identical crop of one cell per setting

Use --soma-ids to pin the compared cells, so a later sweep with other settings
shows exactly the same cells again.
"""

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from apply_adaptive_hysteresis import (
    CONNECTIVITY_8,
    SKELETON,
    SOMA,
    bracket_thresholds,
    hysteresis,
    load_probabilities,
    measure,
    otsu_with_variance_curve,
)

SCRIPT_VERSION = "hysteresis-settings-sweep-v1-2026-08-25"

KEPT_COLOR = (28, 211, 166)
ADDED_COLOR = (255, 190, 54)
REMOVED_COLOR = (230, 74, 65)
TARGET_SOMA_COLOR = (238, 83, 154)
OTHER_SOMA_COLOR = (151, 91, 214)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probabilities", type=Path, required=True, help="nnU-Net .npz probability archive.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Raw TIFF used as background of every overlay.")
    parser.add_argument(
        "--t-high",
        type=float,
        nargs="+",
        default=[0.60, 0.40],
        help="Seed thresholds. Combined with every --t-low into a grid (default: 0.60 0.40).",
    )
    parser.add_argument(
        "--t-low",
        type=float,
        nargs="+",
        default=[0.30, 0.20, 0.10],
        help="Growth thresholds. Pairs with T_low >= T_high are skipped (default: 0.30 0.20 0.10).",
    )
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="T_HIGH:T_LOW",
        help="Explicit threshold pair, repeatable. Added on top of the grid.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="*",
        default=[1.0 / 3.0],
        help="Adaptive Otsu variants to include as reference (default: 0.3333 = web pipeline). "
             "Pass --alpha with no value to omit them.",
    )
    parser.add_argument("--no-argmax", action="store_true", help="Omit the plain argmax reference column.")
    parser.add_argument("--min-soma-area", type=int, default=20, help="Soma components below this are noise.")
    parser.add_argument("--min-skeleton-px", type=int, default=8, help="Skeleton pixels a cell needs to count.")
    parser.add_argument(
        "--soma-ids",
        type=int,
        nargs="+",
        help="Compare exactly these soma ids. Overrides --max-cells and keeps the cell set stable "
             "across sweeps of the same image.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=60,
        help="Number of evenly spread cells to render (0 = all valid somas, default: 60).",
    )
    parser.add_argument("--cell-crop-size", type=int, default=320, help="Side length of the comparison crops.")
    parser.add_argument("--max-preview-size", type=int, default=1100, help="Long edge of the full-image previews.")
    parser.add_argument(
        "--write-full-tifs",
        action="store_true",
        help="Also write semantic/skeleton/added/removed TIFFs per setting (~100 MB per setting).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_setting(text: str) -> tuple[float, float]:
    parts = text.replace(",", ":").split(":")
    if len(parts) != 2:
        raise ValueError(f"--setting expects T_HIGH:T_LOW, got {text!r}.")
    return float(parts[0]), float(parts[1])


def threshold_token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def build_variants(args: argparse.Namespace) -> list[dict[str, object]]:
    """Argmax and adaptive references first, then the fixed grid, coarse to fine."""
    variants: list[dict[str, object]] = []
    if not args.no_argmax:
        variants.append(
            {
                "key": "argmax",
                "kind": "argmax",
                "label": "Argmax (nnU-Net)",
                "console_label": "Argmax (nnU-Net)",
                "t_high": None,
                "t_low": None,
            }
        )
    for alpha in args.alpha or []:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"--alpha must be in (0,1), got {alpha}.")
        variants.append(
            {
                "key": f"adaptive_a{threshold_token(alpha)}",
                "kind": "adaptive",
                "label": f"Adaptiv α={alpha:.2f}",
                "console_label": f"Adaptiv alpha={alpha:.2f}",
                "alpha": float(alpha),
                "t_high": None,
                "t_low": None,
            }
        )

    pairs: list[tuple[float, float]] = []
    for t_high in args.t_high:
        for t_low in args.t_low:
            if t_low < t_high:
                pairs.append((float(t_high), float(t_low)))
    pairs.extend(parse_setting(text) for text in args.setting)

    seen: set[tuple[float, float]] = set()
    ordered: list[tuple[float, float]] = []
    for t_high, t_low in sorted(pairs, key=lambda item: (-item[0], -item[1])):
        if not 0.0 < t_high <= 1.0:
            raise ValueError(f"T_high must be in (0,1], got {t_high}.")
        if not 0.0 <= t_low < t_high:
            raise ValueError(f"T_low must be in [0,T_high), got {t_low} with T_high={t_high}.")
        rounded = (round(t_high, 4), round(t_low, 4))
        if rounded not in seen:
            seen.add(rounded)
            ordered.append(rounded)
    for t_high, t_low in ordered:
        variants.append(
            {
                "key": f"th{threshold_token(t_high)}_tl{threshold_token(t_low)}",
                "kind": "fixed",
                "label": f"T↑ {t_high:.2f} / T↓ {t_low:.2f}",
                "console_label": f"T+ {t_high:.2f} / T- {t_low:.2f}",
                "t_high": t_high,
                "t_low": t_low,
            }
        )
    if not variants:
        raise ValueError("No settings to compare. Provide --t-high/--t-low, --setting or keep a reference column.")
    return variants


def read_2d(path: Path, label: str) -> np.ndarray:
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{label} must be a 2-D TIFF, got {array.shape}: {path}")
    return array


def normalize_u8(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    return np.round(np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def preview_size(shape: tuple[int, int], maximum: int) -> tuple[int, int]:
    height, width = shape
    scale = min(1.0, maximum / max(height, width))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0


def write_full_preview(
    path: Path,
    raw_preview: np.ndarray,
    size: tuple[int, int],
    skeleton: np.ndarray,
    soma: np.ndarray,
    added: np.ndarray,
    removed: np.ndarray,
) -> None:
    gray = np.asarray(Image.fromarray(raw_preview).resize(size, Image.Resampling.LANCZOS))
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[resize_mask(skeleton, size)] = KEPT_COLOR
    rgb[resize_mask(added, size)] = ADDED_COLOR
    rgb[resize_mask(removed, size)] = REMOVED_COLOR
    rgb[resize_mask(soma, size)] = TARGET_SOMA_COLOR
    Image.fromarray(rgb).save(path, optimize=True)


def write_cell_preview(
    path: Path,
    raw_crop: np.ndarray,
    skeleton: np.ndarray,
    target_soma: np.ndarray,
    other_somas: np.ndarray,
    added: np.ndarray,
    removed: np.ndarray,
) -> None:
    rgb = np.repeat(raw_crop[..., None], 3, axis=2)
    rgb[skeleton] = KEPT_COLOR
    rgb[added] = ADDED_COLOR
    rgb[removed] = REMOVED_COLOR
    rgb[other_somas] = OTHER_SOMA_COLOR
    rgb[target_soma] = TARGET_SOMA_COLOR
    Image.fromarray(rgb).save(path, optimize=True)


def crop_bounds(center_y: int, center_x: int, shape: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    crop_height = min(size, shape[0])
    crop_width = min(size, shape[1])
    y0 = min(max(0, center_y - crop_height // 2), shape[0] - crop_height)
    x0 = min(max(0, center_x - crop_width // 2), shape[1] - crop_width)
    return y0, y0 + crop_height, x0, x0 + crop_width


def soma_representatives(labels: np.ndarray, identifiers: np.ndarray) -> np.ndarray:
    objects = ndi.find_objects(labels)
    representatives: list[tuple[int, int]] = []
    for identifier in identifiers.tolist():
        slices = objects[identifier - 1]
        if slices is None:
            raise RuntimeError(f"Soma component {identifier} has no bounding box.")
        local = np.argwhere(labels[slices] == identifier)[0]
        representatives.append((int(local[0] + slices[0].start), int(local[1] + slices[1].start)))
    return np.asarray(representatives, dtype=np.int64)


def soma_centers(labels: np.ndarray, identifiers: np.ndarray) -> np.ndarray:
    objects = ndi.find_objects(labels)
    centers: list[tuple[int, int]] = []
    for identifier in identifiers.tolist():
        slices = objects[identifier - 1]
        if slices is None:
            raise RuntimeError(f"Soma component {identifier} has no bounding box.")
        local = np.argwhere(labels[slices] == identifier)
        center = np.rint(
            local.mean(axis=0) + np.asarray([slices[0].start, slices[1].start], dtype=np.float64)
        ).astype(np.int64)
        centers.append((int(center[0]), int(center[1])))
    return np.asarray(centers, dtype=np.int64)


def topology_analysis(
    skeleton: np.ndarray,
    valid_soma: np.ndarray,
    representatives: np.ndarray,
    minimum_skeleton_pixels: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Group skeleton and valid somas into material components and classify every soma."""
    material_labels, material_count = ndi.label(skeleton | valid_soma, structure=CONNECTIVITY_8)
    skeleton_per_material = np.bincount(material_labels[skeleton], minlength=material_count + 1)
    if representatives.size:
        soma_material = material_labels[representatives[:, 0], representatives[:, 1]]
        soma_per_material = np.bincount(soma_material, minlength=material_count + 1)
        somas_with_skeleton = int(np.sum(skeleton_per_material[soma_material] >= minimum_skeleton_pixels))
        somas_in_conflict = int(np.sum(soma_per_material[soma_material] > 1))
    else:
        soma_material = np.empty(0, dtype=np.int64)
        soma_per_material = np.zeros(material_count + 1, dtype=np.int64)
        somas_with_skeleton = 0
        somas_in_conflict = 0
    isolated = int(np.count_nonzero((soma_per_material == 1) & (skeleton_per_material >= minimum_skeleton_pixels)))
    orphan_selection = (soma_per_material == 0) & (skeleton_per_material > 0)
    total_skeleton = int(skeleton.sum())
    orphan_px = int(skeleton_per_material[orphan_selection].sum())
    metrics: dict[str, object] = {
        "material_components": int(material_count),
        "isolated_candidates": isolated,
        "conflict_groups": int(np.count_nonzero(soma_per_material > 1)),
        "somas_in_conflict": somas_in_conflict,
        "orphan_components": int(np.count_nonzero(orphan_selection)),
        "orphan_skeleton_px": orphan_px,
        "orphan_skeleton_pct": round(100.0 * orphan_px / total_skeleton, 2) if total_skeleton else 0.0,
        "somas_with_min_skeleton": somas_with_skeleton,
    }

    statuses: list[dict[str, object]] = []
    for component_id in soma_material.tolist():
        skeleton_pixels = int(skeleton_per_material[component_id])
        soma_count = int(soma_per_material[component_id])
        if skeleton_pixels < minimum_skeleton_pixels:
            status, label = "short", "zu wenig Skeleton"
        elif soma_count == 1:
            status, label = "isolated", "isoliert"
        else:
            status, label = "conflict", f"Konflikt ({soma_count} Somata)"
        statuses.append(
            {
                "status": status,
                "status_label": label,
                "component_skeleton_px": skeleton_pixels,
                "somas_in_component": soma_count,
            }
        )
    if isolated:
        isolated_sizes = skeleton_per_material[(soma_per_material == 1) & (skeleton_per_material >= minimum_skeleton_pixels)]
        metrics["mean_skeleton_px_per_isolated"] = round(float(isolated_sizes.mean()), 1)
    else:
        metrics["mean_skeleton_px_per_isolated"] = 0.0
    return metrics, statuses


def select_cells(valid_soma_ids: np.ndarray, requested: list[int] | None, maximum: int) -> np.ndarray:
    if requested:
        missing = sorted(set(requested) - set(valid_soma_ids.tolist()))
        if missing:
            raise ValueError(f"Requested soma ids are not valid somas in this image: {missing}")
        return np.asarray([identifier for identifier in valid_soma_ids.tolist() if identifier in set(requested)],
                          dtype=np.int64)
    if maximum <= 0 or valid_soma_ids.size <= maximum:
        return valid_soma_ids
    indices = np.unique(np.round(np.linspace(0, valid_soma_ids.size - 1, maximum)).astype(int))
    return valid_soma_ids[indices]


def write_csv(path: Path, rows: list[dict[str, object]], fallback_header: str) -> None:
    if not rows:
        path.write_text(fallback_header + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


REPORT_CSS = """
body{margin:0;background:#f2f0e9;color:#18201d;font-family:system-ui,-apple-system,Segoe UI,sans-serif}
main{width:min(1900px,97vw);margin:34px auto 90px}
h1{font:44px/1.1 Georgia,serif;margin:0 0 10px}
.intro{color:#5f6a64;max-width:960px;line-height:1.6;margin:0 0 6px}
.intro code{background:#e6e4dc;padding:1px 5px;border-radius:3px;font-size:13px}
table{width:100%;border-collapse:collapse;background:#fff;margin:24px 0;font-size:12px}
th,td{padding:9px 11px;border:1px solid #d8d6ce;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:#e9e7df;position:sticky;top:0}
td.best{background:#d9f4e6;font-weight:650}
tr.reference td{background:#f6f2e4}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:18px 0}
.legend i{width:11px;height:11px;display:inline-block;margin-right:5px;vertical-align:-1px}
.toolbar{position:sticky;top:0;z-index:6;background:#18201d;color:#fff;padding:12px 16px;margin:26px 0 14px;border-radius:4px;
  display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.toolbar label{font-size:13px}
.toolbar select{font:inherit;padding:6px 9px;border:0;border-radius:3px}
.toolbar output{margin-left:auto;color:#cfd8d3;font-size:13px}
.columns{display:flex;flex-wrap:wrap;gap:10px;background:#fff;border:1px solid #d8d6ce;padding:12px 14px;margin-bottom:16px;font-size:12px}
.columns strong{width:100%;font-size:12px;color:#5f6a64;font-weight:600}
.columns label{display:inline-flex;align-items:center;gap:5px;background:#f4f3ee;border:1px solid #dfddd5;border-radius:3px;padding:4px 8px;cursor:pointer}
.cell-list{display:grid;gap:18px}
.cell-card{background:#fff;border:1px solid #d8d6ce;padding:16px}
.cell-card header{display:flex;align-items:baseline;gap:16px;margin-bottom:12px;flex-wrap:wrap}
.cell-card h3{font:25px Georgia,serif;margin:0}
.cell-card header p{color:#5f6a64;font-size:12px;margin:0}
.comparison{display:flex;gap:10px;overflow-x:auto;padding-bottom:6px}
figure{margin:0;flex:0 0 200px;background:#f4f3ee;border:1px solid #dfddd5}
figure[hidden]{display:none}
figure img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#111}
figcaption{display:grid;grid-template-columns:1fr auto;align-items:center;gap:5px;padding:8px 9px;font-size:12px}
figcaption small{grid-column:1/-1;color:#5f6a64}
.badge{border-radius:999px;padding:3px 7px;font-size:10px;white-space:nowrap}
.badge.isolated{background:#d9f4e6;color:#17623d}
.badge.conflict{background:#ffe0d7;color:#8c2918}
.badge.short{background:#ece9e0;color:#665f52}
.badge.raw{background:#e4e8e6;color:#45514b}
.gain{color:#17623d}.loss{color:#8c2918}
details{margin-top:34px}
summary{cursor:pointer;font-weight:650;padding:14px 0}
.overview-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}
.overview-grid article{background:#fff;border:1px solid #d8d6ce;padding:16px}
.overview-grid h2{font:23px Georgia,serif;margin:0 0 12px}
.overview-grid img{display:block;width:100%;background:#111}
.overview-grid p{color:#5f6a64;font-size:12px;margin:12px 0 0}
@media(max-width:760px){table{display:block;overflow:auto}.toolbar{flex-direction:column;align-items:flex-start}.toolbar output{margin-left:0}}
"""

REPORT_JS = """
const filter=document.getElementById('cell-filter');
const sorter=document.getElementById('cell-sort');
const list=document.querySelector('.cell-list');
const cards=[...document.querySelectorAll('.cell-card')];
const counter=document.getElementById('visible-count');
function applyFilter(){
  let shown=0;
  for(const card of cards){
    const mode=filter.value;
    const visible=mode==='all'
      ||(mode==='change'&&card.dataset.change==='1')
      ||(mode==='isolated'&&card.dataset.isolated==='1')
      ||(mode==='never'&&card.dataset.isolated==='0')
      ||(mode==='conflict'&&card.dataset.conflict==='1');
    card.hidden=!visible;
    if(visible)shown++;
  }
  counter.textContent=shown+' von '+cards.length+' Zellen sichtbar';
}
function applySort(){
  const mode=sorter.value;
  const ordered=[...cards].sort((a,b)=>{
    if(mode==='gain')return Number(b.dataset.gain)-Number(a.dataset.gain);
    if(mode==='change')return (Number(b.dataset.change)-Number(a.dataset.change))||(Number(a.dataset.soma)-Number(b.dataset.soma));
    return Number(a.dataset.soma)-Number(b.dataset.soma);
  });
  for(const card of ordered)list.appendChild(card);
}
function applyColumns(){
  const active=new Set([...document.querySelectorAll('.columns input:checked')].map(box=>box.value));
  for(const figure of document.querySelectorAll('figure[data-variant]')){
    figure.hidden=!active.has(figure.dataset.variant);
  }
}
filter.addEventListener('change',applyFilter);
sorter.addEventListener('change',applySort);
for(const box of document.querySelectorAll('.columns input'))box.addEventListener('change',applyColumns);
applyFilter();applySort();applyColumns();
"""

BEST_HIGH = {"isolated_candidates", "somas_with_min_skeleton", "attached_to_soma_pct", "mean_skeleton_px_per_isolated"}
BEST_LOW = {"skeleton_components", "orphan_components", "orphan_skeleton_pct", "somas_in_conflict"}

METRIC_COLUMNS = [
    ("label", "Einstellung", None),
    ("skeleton_px", "Skelettpixel", "{:,}"),
    ("added_px", "+ vs. Argmax", "{:+,}"),
    ("removed_px", "− vs. Argmax", "{:,}"),
    ("skeleton_components", "Komponenten", "{:,}"),
    ("attached_to_soma_pct", "am Soma", "{:.1f}%"),
    ("somas_with_min_skeleton", "Somata mit Skelett", "{:,}"),
    ("isolated_candidates", "isolierte Einzelzellen", "{:,}"),
    ("mean_skeleton_px_per_isolated", "Ø Skelett je Zelle", "{:.0f}"),
    ("somas_in_conflict", "Somata in Konflikt", "{:,}"),
    ("conflict_groups", "Konfliktgruppen", "{:,}"),
    ("orphan_skeleton_pct", "Skelett ohne Soma", "{:.1f}%"),
]


def format_metric(value: object, pattern: str | None) -> str:
    if pattern is None:
        return html.escape(str(value))
    return pattern.format(value)


def write_html_report(
    path: Path,
    case: str,
    rows: list[dict[str, object]],
    variants: list[dict[str, object]],
    reference_key: str,
    cells: list[dict[str, object]],
    crop_size: int,
    min_skeleton_px: int,
) -> None:
    best_values: dict[str, float] = {}
    for column, _, _ in METRIC_COLUMNS:
        if column in BEST_HIGH:
            best_values[column] = max(float(row[column]) for row in rows)
        elif column in BEST_LOW:
            best_values[column] = min(float(row[column]) for row in rows)

    metric_rows = []
    for row in rows:
        cells_html = []
        for column, _, pattern in METRIC_COLUMNS:
            value = row[column]
            classes = " class=\"best\"" if column in best_values and float(value) == best_values[column] else ""
            cells_html.append(f"<td{classes}>{format_metric(value, pattern)}</td>")
        reference = " class=\"reference\"" if row["key"] == reference_key else ""
        metric_rows.append(f"<tr{reference}>{''.join(cells_html)}</tr>")
    header_cells = "".join(f"<th>{html.escape(title)}</th>" for _, title, _ in METRIC_COLUMNS)

    column_boxes = "".join(
        f"<label><input type=\"checkbox\" value=\"{html.escape(str(variant['key']))}\" checked> "
        f"{html.escape(str(variant['label']))}</label>"
        for variant in variants
    )

    overview_cards = "".join(
        "<article>"
        f"<h2>{html.escape(str(row['label']))}</h2>"
        f"<a href=\"{html.escape(str(row['preview']))}\" target=\"_blank\">"
        f"<img loading=\"lazy\" src=\"{html.escape(str(row['preview']))}\" alt=\"{html.escape(str(row['label']))}\"></a>"
        f"<p>{int(row['isolated_candidates'])} isolierte Einzelzellen · "
        f"{int(row['conflict_groups'])} Konfliktgruppen · "
        f"{float(row['orphan_skeleton_pct']):.1f}% Skelett ohne Soma</p>"
        "</article>"
        for row in rows
    )

    card_parts: list[str] = []
    for cell in cells:
        figures = [
            "<figure>"
            f"<a href=\"{html.escape(str(cell['raw_preview']))}\" target=\"_blank\">"
            f"<img loading=\"lazy\" src=\"{html.escape(str(cell['raw_preview']))}\" "
            f"alt=\"Rohbild Soma {int(cell['soma_id'])}\"></a>"
            "<figcaption><strong>Rohbild</strong><span class=\"badge raw\">ohne Overlay</span></figcaption>"
            "</figure>"
        ]
        for entry in cell["variants"]:
            status = html.escape(str(entry["status"]))
            delta = int(entry["delta_vs_reference"])
            delta_class = "gain" if delta > 0 else ("loss" if delta < 0 else "")
            delta_text = f"<span class=\"{delta_class}\">{delta:+,} vs. Referenz</span>" if delta else "wie Referenz"
            figures.append(
                f"<figure data-variant=\"{html.escape(str(entry['key']))}\">"
                f"<a href=\"{html.escape(str(entry['preview']))}\" target=\"_blank\">"
                f"<img loading=\"lazy\" src=\"{html.escape(str(entry['preview']))}\" "
                f"alt=\"Soma {int(cell['soma_id'])}, {html.escape(str(entry['label']))}\"></a>"
                "<figcaption>"
                f"<strong>{html.escape(str(entry['label']))}</strong>"
                f"<span class=\"badge {status}\">{html.escape(str(entry['status_label']))}</span>"
                f"<small>{int(entry['component_skeleton_px']):,} Skelettpixel · {delta_text}</small>"
                "</figcaption></figure>"
            )
        flags: list[str] = []
        if cell["status_changes"]:
            flags.append("Status ändert sich")
        if cell["isolated_any"]:
            flags.append("mindestens einmal isoliert")
        else:
            flags.append("nie isoliert")
        if cell["conflict_any"]:
            flags.append("mindestens einmal Konflikt")
        card_parts.append(
            f"<article class=\"cell-card\" data-soma=\"{int(cell['soma_id'])}\" "
            f"data-change=\"{int(bool(cell['status_changes']))}\" "
            f"data-isolated=\"{int(bool(cell['isolated_any']))}\" "
            f"data-conflict=\"{int(bool(cell['conflict_any']))}\" "
            f"data-gain=\"{int(cell['max_gain'])}\">"
            "<header>"
            f"<h3>Soma {int(cell['soma_id']):04d}</h3>"
            f"<p>Zentrum y={int(cell['center_y'])}, x={int(cell['center_x'])} · "
            f"{html.escape(' · '.join(flags))}</p>"
            "</header>"
            f"<div class=\"comparison\">{''.join(figures)}</div>"
            "</article>"
        )

    changed = sum(bool(cell["status_changes"]) for cell in cells)
    isolated_any = sum(bool(cell["isolated_any"]) for cell in cells)
    never_isolated = len(cells) - isolated_any
    conflict_any = sum(bool(cell["conflict_any"]) for cell in cells)
    reference_label = next(str(variant["label"]) for variant in variants if variant["key"] == reference_key)

    document = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Hysterese-Einstellungen – {html.escape(case)}</title>
<style>{REPORT_CSS}</style></head><body><main>
<h1>Hysterese-Vergleich: {html.escape(case)}</h1>
<p class="intro">Jede Zeile zeigt dieselbe Zelle als identischen {crop_size}×{crop_size}-Crop unter allen
{len(variants)} Einstellungen. Somata stammen immer aus dem unveränderten Argmax, deshalb sind Zellidentität und
Bildausschnitt über alle Einstellungen hinweg gleich. Referenzspalte für die Zuwachs-Angaben ist
<code>{html.escape(reference_label)}</code>. Eine Zelle gilt als vorhanden ab {min_skeleton_px} Skelettpixeln.</p>
<p class="intro">Gesucht ist die Einstellung mit vielen <em>isolierten Einzelzellen</em> und viel Skelett je Zelle,
ohne dass die Zahl der <em>Somata in Konflikt</em> stark steigt – letzteres bedeutet, dass Nachbarzellen
zusammenwachsen.</p>
<div class="legend"><span><i style="background:#1cd3a6"></i>beibehaltenes Skeleton</span>
<span><i style="background:#ffbe36"></i>neu hinzugefügt gegenüber Argmax</span>
<span><i style="background:#e64a41"></i>gegenüber Argmax entfernt</span>
<span><i style="background:#ee539a"></i>Ziel-Soma</span>
<span><i style="background:#975bd6"></i>anderes Soma</span></div>
<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
<div class="columns"><strong>Spalten ein-/ausblenden</strong>{column_boxes}</div>
<div class="toolbar">
<label for="cell-filter">Zellen:</label>
<select id="cell-filter">
<option value="all">alle ({len(cells)})</option>
<option value="change">Status ändert sich ({changed})</option>
<option value="isolated">mindestens einmal isoliert ({isolated_any})</option>
<option value="never">nie isoliert ({never_isolated})</option>
<option value="conflict">mindestens einmal Konflikt ({conflict_any})</option>
</select>
<label for="cell-sort">Sortierung:</label>
<select id="cell-sort">
<option value="soma">Soma-ID</option>
<option value="gain">größter Skelettzuwachs</option>
<option value="change">Statusänderungen zuerst</option>
</select>
<output id="visible-count"></output></div>
<section class="cell-list">{''.join(card_parts)}</section>
<details><summary>Vollbild-Übersichten anzeigen</summary><section class="overview-grid">{overview_cards}</section></details>
<script>{REPORT_JS}</script></main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = parse_args()
    variants = build_variants(args)
    if args.min_soma_area < 1 or args.min_skeleton_px < 1:
        raise ValueError("--min-soma-area and --min-skeleton-px must both be positive.")
    if args.cell_crop_size < 32:
        raise ValueError("--cell-crop-size must be at least 32.")
    if args.max_preview_size < 128:
        raise ValueError("--max-preview-size must be at least 128.")

    output_root = args.output_dir
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output_root}\nPass --overwrite to replace files.")
    output_root.mkdir(parents=True, exist_ok=True)
    cells_root = output_root / "cell_crops"
    cells_root.mkdir(parents=True, exist_ok=True)

    case = args.probabilities.stem
    probabilities = load_probabilities(args.probabilities)
    argmax = np.argmax(probabilities, axis=0).astype(np.uint8)
    gray = np.clip(probabilities[SKELETON] * 255.0, 0, 255).astype(np.uint8)
    del probabilities
    soma = argmax == SOMA
    argmax_skeleton = argmax == SKELETON
    shape = argmax.shape

    raw = read_2d(args.raw, "Raw image") if args.raw else np.zeros(shape, dtype=np.uint8)
    if raw.shape != shape:
        raise RuntimeError(f"Shape mismatch: raw={raw.shape}, probabilities={shape}")
    raw_preview = normalize_u8(raw)
    del raw
    full_preview_size = preview_size(shape, args.max_preview_size)

    soma_labels, soma_count = ndi.label(soma, structure=CONNECTIVITY_8)
    soma_areas = np.bincount(soma_labels.ravel(), minlength=soma_count + 1)
    valid_soma_ids = np.flatnonzero(soma_areas >= args.min_soma_area)
    valid_soma_ids = valid_soma_ids[valid_soma_ids > 0]
    if valid_soma_ids.size == 0:
        raise RuntimeError(f"No soma component reaches --min-soma-area={args.min_soma_area}.")
    valid_soma = np.isin(soma_labels, valid_soma_ids)
    representatives = soma_representatives(soma_labels, valid_soma_ids)
    centers = soma_centers(soma_labels, valid_soma_ids)

    selected_ids = select_cells(valid_soma_ids, args.soma_ids, args.max_cells)
    position_of = {identifier: index for index, identifier in enumerate(valid_soma_ids.tolist())}
    selected_positions = [position_of[identifier] for identifier in selected_ids.tolist()]
    windows = {
        identifier: crop_bounds(int(centers[position][0]), int(centers[position][1]), shape, args.cell_crop_size)
        for identifier, position in zip(selected_ids.tolist(), selected_positions)
    }

    print(f"{case}: {shape[0]}x{shape[1]} px, {valid_soma_ids.size} gültige Somata, "
          f"{len(selected_ids)} davon im Vergleich, {len(variants)} Einstellungen", flush=True)

    for identifier in selected_ids.tolist():
        y0, y1, x0, x1 = windows[identifier]
        cell_root = cells_root / f"soma_{identifier:04d}"
        cell_root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raw_preview[y0:y1, x0:x1]).save(cell_root / "raw.png", optimize=True)

    baseline = measure(argmax_skeleton, soma)
    argmax_px = max(1, int(argmax_skeleton.sum()))
    rows: list[dict[str, object]] = []
    per_cell: dict[int, list[dict[str, object]]] = {identifier: [] for identifier in selected_ids.tolist()}

    for index, variant in enumerate(variants, start=1):
        kind = str(variant["kind"])
        if kind == "argmax":
            skeleton = argmax_skeleton.copy()
            high_effective = low_effective = None
        else:
            if kind == "adaptive":
                otsu_threshold, variance = otsu_with_variance_curve(gray)
                high_u8, low_u8 = bracket_thresholds(otsu_threshold, variance, float(variant["alpha"]))
                variant["otsu_T"] = round(otsu_threshold / 255.0, 4)
            else:
                high_u8 = int(round(float(variant["t_high"]) * 255.0))
                low_u8 = int(round(float(variant["t_low"]) * 255.0))
            skeleton = hysteresis(gray, high_u8, low_u8, forbidden=soma)
            high_effective = round(high_u8 / 255.0, 6)
            low_effective = round(low_u8 / 255.0, 6)
            variant["t_high_effective"] = high_effective
            variant["t_low_effective"] = low_effective

        added = skeleton & ~argmax_skeleton
        removed = argmax_skeleton & ~skeleton
        metrics = measure(skeleton, soma)
        topology, statuses = topology_analysis(skeleton, valid_soma, representatives, args.min_skeleton_px)

        variant_root = output_root / str(variant["key"])
        variant_root.mkdir(parents=True, exist_ok=True)
        if args.write_full_tifs:
            semantic = np.zeros(shape, dtype=np.uint8)
            semantic[skeleton] = SKELETON
            semantic[soma] = SOMA
            tifffile.imwrite(variant_root / "semantic_0-1-2.tif", semantic)
            tifffile.imwrite(variant_root / "skeleton.tif", skeleton.astype(np.uint8))
            tifffile.imwrite(variant_root / "added_vs_argmax.tif", added.astype(np.uint8))
            tifffile.imwrite(variant_root / "removed_vs_argmax.tif", removed.astype(np.uint8))
            del semantic
        write_full_preview(
            variant_root / "preview.png", raw_preview, full_preview_size, skeleton, soma, added, removed
        )

        for identifier, position in zip(selected_ids.tolist(), selected_positions):
            y0, y1, x0, x1 = windows[identifier]
            window = (slice(y0, y1), slice(x0, x1))
            preview_path = cells_root / f"soma_{identifier:04d}" / f"{variant['key']}.png"
            write_cell_preview(
                preview_path,
                raw_preview[window],
                skeleton[window],
                soma_labels[window] == identifier,
                valid_soma[window] & (soma_labels[window] != identifier),
                added[window],
                removed[window],
            )
            per_cell[identifier].append(
                {
                    "key": variant["key"],
                    "label": variant["label"],
                    **statuses[position],
                    "preview": preview_path.relative_to(output_root).as_posix(),
                }
            )

        row: dict[str, object] = {
            "case": case,
            "key": variant["key"],
            "label": variant["label"],
            "console_label": variant["console_label"],
            "kind": kind,
            "t_high": variant.get("t_high_effective", high_effective),
            "t_low": variant.get("t_low_effective", low_effective),
            "alpha": variant.get("alpha"),
            "otsu_T": variant.get("otsu_T"),
            "valid_somas": int(valid_soma_ids.size),
            "skeleton_px": metrics["skeleton_px"],
            "added_px": int(added.sum()),
            "removed_px": int(removed.sum()),
            "preserved_argmax_pct": round(100.0 * int((skeleton & argmax_skeleton).sum()) / argmax_px, 3),
            "skeleton_components": metrics["components"],
            "largest_component": metrics["largest_component"],
            "attached_to_soma_pct": metrics["attached_to_soma_pct"],
            **topology,
            "preview": f"{variant['key']}/preview.png",
        }
        rows.append(row)
        print(
            f"[{index}/{len(variants)}] {variant['console_label']}: "
            f"Skelett {metrics['skeleton_px']:,} px, Komponenten {metrics['components']:,}, "
            f"am Soma {metrics['attached_to_soma_pct']:.1f}%, isoliert {topology['isolated_candidates']}, "
            f"Konflikt-Somata {topology['somas_in_conflict']}, ohne Soma {topology['orphan_skeleton_pct']:.1f}%",
            flush=True,
        )
        del skeleton, added, removed

    reference_key = str(rows[0]["key"])
    for row in rows:
        if row["kind"] == "adaptive":
            reference_key = str(row["key"])
            break

    cells: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for identifier, position in zip(selected_ids.tolist(), selected_positions):
        entries = per_cell[identifier]
        reference_px = next(
            int(entry["component_skeleton_px"]) for entry in entries if entry["key"] == reference_key
        )
        for entry in entries:
            entry["delta_vs_reference"] = int(entry["component_skeleton_px"]) - reference_px
        y0, y1, x0, x1 = windows[identifier]
        statuses = {str(entry["status"]) for entry in entries}
        cell = {
            "soma_id": identifier,
            "center_y": int(centers[position][0]),
            "center_x": int(centers[position][1]),
            "bounds": [y0, y1, x0, x1],
            "raw_preview": (cells_root / f"soma_{identifier:04d}" / "raw.png").relative_to(output_root).as_posix(),
            "variants": entries,
            "status_changes": len(statuses) > 1,
            "isolated_any": "isolated" in statuses,
            "conflict_any": "conflict" in statuses,
            "max_gain": max(int(entry["delta_vs_reference"]) for entry in entries),
        }
        cells.append(cell)
        for entry in entries:
            cell_rows.append(
                {
                    "case": case,
                    "soma_id": identifier,
                    "center_y": cell["center_y"],
                    "center_x": cell["center_x"],
                    "y_min": y0,
                    "y_max_exclusive": y1,
                    "x_min": x0,
                    "x_max_exclusive": x1,
                    "setting": entry["key"],
                    "label": entry["label"],
                    "status": entry["status"],
                    "status_label": entry["status_label"],
                    "component_skeleton_px": entry["component_skeleton_px"],
                    "somas_in_component": entry["somas_in_component"],
                    "delta_vs_reference": entry["delta_vs_reference"],
                    "preview": entry["preview"],
                }
            )

    write_csv(output_root / "settings_sweep.csv", rows, "case,key,label")
    write_csv(output_root / "settings_cell_comparison.csv", cell_rows, "case,soma_id,setting,status")
    summary = {
        "script_version": SCRIPT_VERSION,
        "probabilities": str(args.probabilities.resolve()),
        "raw": str(args.raw.resolve()) if args.raw else None,
        "shape": [int(shape[0]), int(shape[1])],
        "min_soma_area": args.min_soma_area,
        "min_skeleton_px": args.min_skeleton_px,
        "cell_crop_size": args.cell_crop_size,
        "reference_setting": reference_key,
        "settings": [
            {key: variant.get(key) for key in ("key", "label", "kind", "t_high", "t_low", "alpha")}
            for variant in variants
        ],
        "compared_soma_ids": [int(identifier) for identifier in selected_ids.tolist()],
        "cell_comparisons": {
            "count": len(cells),
            "status_changes": sum(bool(cell["status_changes"]) for cell in cells),
            "isolated_at_least_once": sum(bool(cell["isolated_any"]) for cell in cells),
            "conflict_at_least_once": sum(bool(cell["conflict_any"]) for cell in cells),
            "csv": "settings_cell_comparison.csv",
            "directory": "cell_crops",
        },
        "argmax": baseline,
        "rows": rows,
    }
    (output_root / "settings_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = output_root / "settings_sweep_report.html"
    write_html_report(
        report_path, case, rows, variants, reference_key, cells, args.cell_crop_size, args.min_skeleton_px
    )

    ranking = sorted(rows, key=lambda row: (-int(row["isolated_candidates"]), int(row["somas_in_conflict"])))
    print("\n" + "=" * 96)
    print("Rangliste nach isolierten Einzelzellen (bei Gleichstand weniger Konflikt-Somata zuerst):")
    for place, row in enumerate(ranking[:5], start=1):
        print(
            f"  {place}. {row['console_label']:<28} isoliert {int(row['isolated_candidates']):>4}"
            f"  |  Konflikt-Somata {int(row['somas_in_conflict']):>4}"
            f"  |  Ø Skelett/Zelle {float(row['mean_skeleton_px_per_isolated']):>6.0f}"
            f"  |  ohne Soma {float(row['orphan_skeleton_pct']):>5.1f}%"
        )
    print(f"\nVergleichene Somata: {', '.join(str(int(i)) for i in selected_ids.tolist()[:12])}"
          f"{' ...' if len(selected_ids) > 12 else ''}")
    print("Fuer denselben Zellsatz im naechsten Sweep: --soma-ids "
          f"{' '.join(str(int(i)) for i in selected_ids.tolist())}")
    print(f"\nBericht: {report_path.resolve()}")


if __name__ == "__main__":
    main()
