from __future__ import annotations

"""Empirically compare fixed-low-threshold hysteresis variants.

The nnU-Net probability archive is loaded once. For every requested T_low, the
skeleton is seeded at a fixed T_high and grown through pixels down to T_low.
Soma pixels remain the unchanged Dataset-138 argmax, exactly as in the normal
postprocessing pipeline.

Outputs per T_low:
  semantic_0-1-2.tif       final background/skeleton/soma map
  skeleton.tif             binary final skeleton
  added_vs_argmax.tif      pixels added to the argmax skeleton
  removed_vs_argmax.tif    argmax skeleton pixels not retained by hysteresis
  preview.png              downsampled quality-control overlay

The output root also contains tlow_sweep.csv, tlow_sweep_summary.json and an
HTML comparison report. The report shows the same soma-centred cell crop for
all thresholds side by side. Matching crop PNGs are written below cell_crops/.
"""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from apply_adaptive_hysteresis import (
    CONNECTIVITY_8,
    SOMA,
    SKELETON,
    hysteresis,
    load_probabilities,
    measure,
)


SCRIPT_VERSION = "hysteresis-tlow-sweep-v2-2026-08-25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probabilities", type=Path, required=True, help="nnU-Net .npz probability archive.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--t-high",
        type=float,
        default=0.60,
        help="Fixed confident skeleton seed threshold in [0,1] (default: 0.60).",
    )
    parser.add_argument(
        "--t-low",
        type=float,
        nargs="+",
        default=[0.30, 0.25, 0.20, 0.15],
        help="One or more weak growth thresholds in [0,T_high) (default: 0.30 0.25 0.20 0.15).",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        help="Optional raw TIFF used as the background of the preview overlays.",
    )
    parser.add_argument("--min-soma-area", type=int, default=20)
    parser.add_argument("--min-skeleton-px", type=int, default=8)
    parser.add_argument("--max-preview-size", type=int, default=1100)
    parser.add_argument(
        "--cell-crop-size",
        type=int,
        default=320,
        help="Side length of the identical soma-centred comparison crops (default: 320).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_thresholds(t_high: float, t_lows: list[float]) -> list[float]:
    if not 0.0 < t_high <= 1.0:
        raise ValueError(f"T_high must be in (0,1], got {t_high}.")
    if not t_lows:
        raise ValueError("At least one T_low is required.")
    unique: list[float] = []
    for value in t_lows:
        if not 0.0 <= value < t_high:
            raise ValueError(f"Every T_low must be in [0,T_high), got {value} with T_high={t_high}.")
        if value not in unique:
            unique.append(value)
    return unique


def threshold_name(value: float) -> str:
    return f"tlow_{value:.3f}".replace(".", "p")


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
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    ) > 0


def write_preview(
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
    skeleton_small = resize_mask(skeleton, size)
    soma_small = resize_mask(soma, size)
    added_small = resize_mask(added, size)
    removed_small = resize_mask(removed, size)
    rgb[skeleton_small] = (28, 211, 166)
    rgb[added_small] = (255, 190, 54)
    rgb[removed_small] = (230, 74, 65)
    rgb[soma_small] = (238, 83, 154)
    Image.fromarray(rgb).save(path, optimize=True)


def crop_bounds(
    center_y: int,
    center_x: int,
    shape: tuple[int, int],
    size: int,
) -> tuple[int, int, int, int]:
    crop_height = min(size, shape[0])
    crop_width = min(size, shape[1])
    y0 = min(max(0, center_y - crop_height // 2), shape[0] - crop_height)
    x0 = min(max(0, center_x - crop_width // 2), shape[1] - crop_width)
    return y0, y0 + crop_height, x0, x0 + crop_width


def write_raw_crop(path: Path, raw_preview: np.ndarray) -> None:
    Image.fromarray(raw_preview).save(path, optimize=True)


def write_cell_preview(
    path: Path,
    raw_preview: np.ndarray,
    skeleton: np.ndarray,
    target_soma: np.ndarray,
    other_somas: np.ndarray,
    added: np.ndarray,
    removed: np.ndarray,
) -> None:
    rgb = np.repeat(raw_preview[..., None], 3, axis=2)
    rgb[skeleton] = (28, 211, 166)
    rgb[added] = (255, 190, 54)
    rgb[removed] = (230, 74, 65)
    rgb[other_somas] = (151, 91, 214)
    rgb[target_soma] = (238, 83, 154)
    Image.fromarray(rgb).save(path, optimize=True)


def soma_representatives(labels: np.ndarray, identifiers: np.ndarray) -> np.ndarray:
    objects = ndi.find_objects(labels)
    representatives: list[tuple[int, int]] = []
    for identifier in identifiers.tolist():
        slices = objects[identifier - 1]
        if slices is None:
            raise RuntimeError(f"Soma component {identifier} has no bounding box.")
        local = np.argwhere(labels[slices] == identifier)[0]
        representatives.append(
            (int(local[0] + slices[0].start), int(local[1] + slices[1].start))
        )
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
            local.mean(axis=0)
            + np.asarray([slices[0].start, slices[1].start], dtype=np.float64)
        ).astype(np.int64)
        centers.append((int(center[0]), int(center[1])))
    return np.asarray(centers, dtype=np.int64)


def topology_analysis(
    skeleton: np.ndarray,
    valid_soma: np.ndarray,
    representatives: np.ndarray,
    minimum_skeleton_pixels: int,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    material_labels, material_count = ndi.label(skeleton | valid_soma, structure=CONNECTIVITY_8)
    skeleton_per_material = np.bincount(
        material_labels[skeleton], minlength=material_count + 1
    )
    if representatives.size:
        soma_material = material_labels[representatives[:, 0], representatives[:, 1]]
        soma_per_material = np.bincount(soma_material, minlength=material_count + 1)
        somas_with_skeleton = int(
            np.sum(skeleton_per_material[soma_material] >= minimum_skeleton_pixels)
        )
    else:
        soma_material = np.empty(0, dtype=np.int64)
        soma_per_material = np.zeros(material_count + 1, dtype=np.int64)
        somas_with_skeleton = 0
    isolated = int(
        np.count_nonzero(
            (soma_per_material == 1) & (skeleton_per_material >= minimum_skeleton_pixels)
        )
    )
    conflicts = int(np.count_nonzero(soma_per_material > 1))
    orphan_selection = (soma_per_material == 0) & (skeleton_per_material > 0)
    metrics = {
        "material_components": int(material_count),
        "isolated_candidates": isolated,
        "conflict_groups": conflicts,
        "orphan_components": int(np.count_nonzero(orphan_selection)),
        "orphan_skeleton_px": int(skeleton_per_material[orphan_selection].sum()),
        "somas_with_min_skeleton": somas_with_skeleton,
    }
    soma_statuses: list[dict[str, object]] = []
    for component_id in soma_material.tolist():
        skeleton_pixels = int(skeleton_per_material[component_id])
        soma_count = int(soma_per_material[component_id])
        if skeleton_pixels < minimum_skeleton_pixels:
            status = "short"
            label = "zu wenig Skeleton"
        elif soma_count == 1:
            status = "isolated"
            label = "isoliert"
        else:
            status = "conflict"
            label = f"Konflikt ({soma_count} Somata)"
        soma_statuses.append(
            {
                "status": status,
                "status_label": label,
                "skeleton_px": skeleton_pixels,
                "somas_in_component": soma_count,
            }
        )
    return metrics, soma_statuses


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_cell_comparisons(
    output_root: Path,
    case: str,
    raw_preview: np.ndarray,
    argmax_skeleton: np.ndarray,
    soma_labels: np.ndarray,
    valid_soma: np.ndarray,
    valid_soma_ids: np.ndarray,
    centers: np.ndarray,
    variants: list[dict[str, object]],
    crop_size: int,
) -> list[dict[str, object]]:
    cells_root = output_root / "cell_crops"
    cells_root.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    total = len(valid_soma_ids)
    for position, (soma_id, center) in enumerate(
        zip(valid_soma_ids.tolist(), centers.tolist()), start=1
    ):
        if position == 1 or position % 25 == 0 or position == total:
            print(f"    cell crops: {position}/{total}", flush=True)
        center_y, center_x = int(center[0]), int(center[1])
        y0, y1, x0, x1 = crop_bounds(
            center_y, center_x, raw_preview.shape, crop_size
        )
        crop_slice = (slice(y0, y1), slice(x0, x1))
        cell_root = cells_root / f"soma_{soma_id:04d}"
        cell_root.mkdir(parents=True, exist_ok=True)
        raw_path = cell_root / "raw.png"
        raw_crop = raw_preview[crop_slice]
        write_raw_crop(raw_path, raw_crop)
        target_soma = soma_labels[crop_slice] == soma_id
        other_somas = valid_soma[crop_slice] & ~target_soma
        argmax_crop = argmax_skeleton[crop_slice]
        variant_cells: list[dict[str, object]] = []
        for variant in variants:
            skeleton = np.asarray(variant["skeleton"])[crop_slice]
            added = skeleton & ~argmax_crop
            removed = argmax_crop & ~skeleton
            variant_name = str(variant["name"])
            preview_path = cell_root / f"{variant_name}.png"
            write_cell_preview(
                preview_path,
                raw_crop,
                skeleton,
                target_soma,
                other_somas,
                added,
                removed,
            )
            status = dict(variant["soma_statuses"][position - 1])
            cell_variant: dict[str, object] = {
                "t_low": float(variant["t_low"]),
                **status,
                "preview": preview_path.relative_to(output_root).as_posix(),
            }
            variant_cells.append(cell_variant)
            csv_rows.append(
                {
                    "case": case,
                    "soma_id": soma_id,
                    "center_y": center_y,
                    "center_x": center_x,
                    "y_min": y0,
                    "y_max_exclusive": y1,
                    "x_min": x0,
                    "x_max_exclusive": x1,
                    **cell_variant,
                }
            )
        status_kinds = {str(item["status"]) for item in variant_cells}
        cells.append(
            {
                "soma_id": soma_id,
                "center_y": center_y,
                "center_x": center_x,
                "bounds": [y0, y1, x0, x1],
                "raw_preview": raw_path.relative_to(output_root).as_posix(),
                "variants": variant_cells,
                "status_changes": len(status_kinds) > 1,
                "isolated_any": any(item["status"] == "isolated" for item in variant_cells),
            }
        )
    cells.sort(
        key=lambda item: (
            not bool(item["status_changes"]),
            not bool(item["isolated_any"]),
            int(item["soma_id"]),
        )
    )
    if csv_rows:
        write_csv(output_root / "tlow_cell_comparison.csv", csv_rows)
    else:
        (output_root / "tlow_cell_comparison.csv").write_text(
            "case,soma_id,t_low,status,status_label\n", encoding="utf-8"
        )
    return cells


def write_html_report(
    path: Path,
    case: str,
    rows: list[dict[str, object]],
    t_high: float,
    cells: list[dict[str, object]],
    crop_size: int,
) -> None:
    metric_rows = "".join(
        "<tr>"
        f"<td>{float(row['t_low']):.3f}</td>"
        f"<td>{int(row['skeleton_px']):,}</td>"
        f"<td>+{int(row['added_px']):,} / -{int(row['removed_px']):,}</td>"
        f"<td>{int(row['skeleton_components']):,}</td>"
        f"<td>{float(row['attached_to_soma_pct']):.1f}%</td>"
        f"<td>{int(row['somas_with_min_skeleton'])}</td>"
        f"<td>{int(row['isolated_candidates'])}</td>"
        f"<td>{int(row['conflict_groups'])}</td>"
        f"<td>{int(row['orphan_components']):,}</td>"
        "</tr>"
        for row in rows
    )
    overview_cards = "".join(
        "<article>"
        f"<h2>T<sub>low</sub> = {float(row['t_low']):.3f}</h2>"
        f"<img src=\"{html.escape(str(row['preview']))}\" alt=\"T low {float(row['t_low']):.3f}\">"
        f"<p>{int(row['isolated_candidates'])} isolierte Kandidaten · "
        f"{int(row['conflict_groups'])} Konfliktgruppen · "
        f"{int(row['somas_with_min_skeleton'])} Somata mit Skelett</p>"
        "</article>"
        for row in rows
    )
    cell_cards_parts: list[str] = []
    for cell in cells:
        comparisons = [
            "<figure>"
            f"<img loading=\"lazy\" src=\"{html.escape(str(cell['raw_preview']))}\" alt=\"Raw crop Soma {int(cell['soma_id'])}\">"
            "<figcaption><strong>Raw</strong><span class=\"badge raw\">ohne Overlay</span></figcaption>"
            "</figure>"
        ]
        for variant in cell["variants"]:
            status = html.escape(str(variant["status"]))
            comparisons.append(
                "<figure>"
                f"<img loading=\"lazy\" src=\"{html.escape(str(variant['preview']))}\" "
                f"alt=\"Soma {int(cell['soma_id'])}, T low {float(variant['t_low']):.3f}\">"
                "<figcaption>"
                f"<strong>T<sub>low</sub> {float(variant['t_low']):.3f}</strong>"
                f"<span class=\"badge {status}\">{html.escape(str(variant['status_label']))}</span>"
                f"<small>{int(variant['skeleton_px']):,} Skeletonpixel</small>"
                "</figcaption></figure>"
            )
        flags: list[str] = []
        if cell["status_changes"]:
            flags.append("Status ändert sich")
        if cell["isolated_any"]:
            flags.append("mindestens einmal isoliert")
        flag_text = " · ".join(flags) if flags else "bei keiner Schwelle isoliert"
        cell_cards_parts.append(
            f"<article class=\"cell-card\" data-change=\"{int(bool(cell['status_changes']))}\" "
            f"data-isolated=\"{int(bool(cell['isolated_any']))}\">"
            "<header>"
            f"<h3>Soma {int(cell['soma_id']):04d}</h3>"
            f"<p>Zentrum y={int(cell['center_y'])}, x={int(cell['center_x'])} · {html.escape(flag_text)}</p>"
            "</header>"
            f"<div class=\"comparison\">{''.join(comparisons)}</div>"
            "</article>"
        )
    cell_cards = "".join(cell_cards_parts)
    changed_count = sum(bool(cell["status_changes"]) for cell in cells)
    isolated_count = sum(bool(cell["isolated_any"]) for cell in cells)
    document = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>T_low Sweep – {html.escape(case)}</title>
<style>
body{{margin:0;background:#f2f0e9;color:#18201d;font-family:system-ui,sans-serif}}
main{{width:min(1800px,96vw);margin:38px auto 80px}}h1{{font:44px Georgia,serif;margin-bottom:8px}}
.intro{{color:#65706a;max-width:900px;line-height:1.55}}table{{width:100%;border-collapse:collapse;background:#fff;margin:26px 0}}
th,td{{padding:10px 12px;border:1px solid #d8d6ce;text-align:right;font-size:12px}}th:first-child,td:first-child{{text-align:left}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:20px 0}}.legend i{{width:11px;height:11px;display:inline-block;margin-right:5px}}
.toolbar{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:12px;background:#18201d;color:#fff;padding:12px 16px;margin:28px 0 16px;border-radius:4px}}
.toolbar select{{font:inherit;padding:7px 10px;border:0;border-radius:3px}}.toolbar output{{margin-left:auto;color:#cfd8d3;font-size:13px}}
.cell-list{{display:grid;gap:18px}}.cell-card{{background:#fff;border:1px solid #d8d6ce;padding:16px}}
.cell-card header{{display:flex;align-items:baseline;gap:16px;margin-bottom:12px}}.cell-card h3{{font:25px Georgia,serif;margin:0}}
.cell-card header p{{color:#65706a;font-size:12px;margin:0}}.comparison{{display:grid;grid-template-columns:repeat({len(rows) + 1},minmax(190px,1fr));gap:10px;overflow-x:auto}}
figure{{margin:0;min-width:190px;background:#f4f3ee;border:1px solid #dfddd5}}figure img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#111}}
figcaption{{display:grid;grid-template-columns:1fr auto;align-items:center;gap:5px;padding:9px;font-size:12px}}figcaption small{{grid-column:1/-1;color:#65706a}}
.badge{{border-radius:999px;padding:3px 7px;font-size:10px;white-space:nowrap}}.badge.isolated{{background:#d9f4e6;color:#17623d}}
.badge.conflict{{background:#ffe0d7;color:#8c2918}}.badge.short{{background:#ece9e0;color:#665f52}}.badge.raw{{background:#e4e8e6;color:#45514b}}
details{{margin-top:32px}}summary{{cursor:pointer;font-weight:650;padding:14px 0}}.overview-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
.overview-grid article{{background:#fff;border:1px solid #d8d6ce;padding:16px}}.overview-grid h2{{font:25px Georgia,serif;margin:0 0 12px}}
.overview-grid img{{display:block;width:100%;background:#111}}.overview-grid p{{color:#65706a;font-size:12px;margin:12px 0 0}}
@media(max-width:700px){{table{{display:block;overflow:auto}}.cell-card header{{display:block}}.toolbar{{align-items:flex-start;flex-direction:column}}.toolbar output{{margin-left:0}}}}
</style></head><body><main><h1>T<sub>low</sub>-Vergleich: {html.escape(case)}</h1>
<p class="intro">T<sub>high</sub> bleibt bei {t_high:.3f}. Jede Zeile zeigt dieselbe Soma-Position als {crop_size}×{crop_size}-Pixel-Crop. So lässt sich direkt erkennen, welche niedrige Schwelle die Zelle vervollständigt und ab wann sie fälschlich mit Nachbarzellen verbunden wird.</p>
<div class="legend"><span><i style="background:#1cd3a6"></i>beibehaltenes Skeleton</span><span><i style="background:#ffbe36"></i>neu hinzugefügt</span><span><i style="background:#e64a41"></i>entfernt</span><span><i style="background:#ee539a"></i>Ziel-Soma</span><span><i style="background:#975bd6"></i>anderes Soma</span></div>
<table><thead><tr><th>T_low</th><th>Skelettpixel</th><th>+ / − vs. Argmax</th><th>Komponenten</th><th>am Soma</th><th>Somata ≥ Minimum</th><th>isoliert</th><th>Konflikte</th><th>Fragmente</th></tr></thead><tbody>{metric_rows}</tbody></table>
<div class="toolbar"><label for="cell-filter">Zellen anzeigen:</label><select id="cell-filter"><option value="all">alle {len(cells)} Somata</option><option value="change">nur Statusänderungen ({changed_count})</option><option value="isolated">mindestens einmal isoliert ({isolated_count})</option></select><output id="visible-count"></output></div>
<section class="cell-list">{cell_cards}</section>
<details><summary>Vollbild-Übersichten anzeigen</summary><section class="overview-grid">{overview_cards}</section></details>
<script>
const select=document.getElementById('cell-filter');const cards=[...document.querySelectorAll('.cell-card')];const count=document.getElementById('visible-count');
function filterCells(){{let shown=0;for(const card of cards){{const visible=select.value==='all'||(select.value==='change'&&card.dataset.change==='1')||(select.value==='isolated'&&card.dataset.isolated==='1');card.hidden=!visible;if(visible)shown++;}}count.textContent=shown+' Zellen sichtbar';}}
select.addEventListener('change',filterCells);filterCells();
</script></main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    t_lows = validate_thresholds(args.t_high, args.t_low)
    if args.min_soma_area < 1 or args.min_skeleton_px < 1:
        raise ValueError("Minimum soma area and skeleton pixels must both be positive.")
    if args.max_preview_size < 128:
        raise ValueError("--max-preview-size must be at least 128.")
    if args.cell_crop_size < 32:
        raise ValueError("--cell-crop-size must be at least 32.")
    output_root = args.output_dir
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output_root}\nPass --overwrite to replace files.")
    output_root.mkdir(parents=True, exist_ok=True)

    probabilities = load_probabilities(args.probabilities)
    argmax = np.argmax(probabilities, axis=0).astype(np.uint8)
    soma = argmax == SOMA
    argmax_skeleton = argmax == SKELETON
    gray = np.clip(probabilities[SKELETON] * 255.0, 0, 255).astype(np.uint8)
    raw = read_2d(args.raw, "Raw image") if args.raw else np.zeros(argmax.shape, dtype=np.uint8)
    if raw.shape != argmax.shape:
        raise RuntimeError(f"Shape mismatch: raw={raw.shape}, probabilities={argmax.shape}")
    raw_preview = normalize_u8(raw)
    size = preview_size(argmax.shape, args.max_preview_size)

    soma_labels, soma_count = ndi.label(soma, structure=CONNECTIVITY_8)
    soma_areas = np.bincount(soma_labels.ravel(), minlength=soma_count + 1)
    valid_soma_ids = np.flatnonzero(soma_areas >= args.min_soma_area)
    valid_soma_ids = valid_soma_ids[valid_soma_ids > 0]
    valid_soma = np.isin(soma_labels, valid_soma_ids)
    representatives = soma_representatives(soma_labels, valid_soma_ids)
    centers = soma_centers(soma_labels, valid_soma_ids)
    baseline = measure(argmax_skeleton, soma)

    rows: list[dict[str, object]] = []
    variants: list[dict[str, object]] = []
    high_u8 = int(round(args.t_high * 255.0))
    for index, t_low in enumerate(t_lows, start=1):
        print(f"[{index}/{len(t_lows)}] T_high={args.t_high:.3f}, T_low={t_low:.3f}", flush=True)
        low_u8 = int(round(t_low * 255.0))
        skeleton = hysteresis(gray, high_u8, low_u8, forbidden=soma)
        added = skeleton & ~argmax_skeleton
        removed = argmax_skeleton & ~skeleton
        semantic = np.zeros(argmax.shape, dtype=np.uint8)
        semantic[skeleton] = SKELETON
        semantic[soma] = SOMA
        variant_root = output_root / threshold_name(t_low)
        variant_root.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(variant_root / "semantic_0-1-2.tif", semantic)
        tifffile.imwrite(variant_root / "skeleton.tif", skeleton.astype(np.uint8))
        tifffile.imwrite(variant_root / "added_vs_argmax.tif", added.astype(np.uint8))
        tifffile.imwrite(variant_root / "removed_vs_argmax.tif", removed.astype(np.uint8))
        write_preview(
            variant_root / "preview.png", raw_preview, size, skeleton, soma, added, removed
        )
        metrics = measure(skeleton, soma)
        topology, soma_statuses = topology_analysis(
            skeleton, valid_soma, representatives, args.min_skeleton_px
        )
        row: dict[str, object] = {
            "case": args.probabilities.stem,
            "t_high_requested": args.t_high,
            "t_high_effective": round(high_u8 / 255.0, 6),
            "t_low": t_low,
            "t_low_effective": round(low_u8 / 255.0, 6),
            "valid_somas": int(len(valid_soma_ids)),
            "skeleton_px": metrics["skeleton_px"],
            "added_px": int(added.sum()),
            "removed_px": int(removed.sum()),
            "preserved_argmax_pct": round(
                100.0 * int((skeleton & argmax_skeleton).sum()) / max(1, int(argmax_skeleton.sum())), 3
            ),
            "skeleton_components": metrics["components"],
            "largest_component": metrics["largest_component"],
            "attached_to_soma_pct": metrics["attached_to_soma_pct"],
            **topology,
            "preview": f"{threshold_name(t_low)}/preview.png",
            "semantic": f"{threshold_name(t_low)}/semantic_0-1-2.tif",
        }
        rows.append(row)
        variants.append(
            {
                "name": threshold_name(t_low),
                "t_low": t_low,
                "skeleton": skeleton.copy(),
                "soma_statuses": soma_statuses,
            }
        )
        print(
            f"    skeleton={metrics['skeleton_px']:,}, components={metrics['components']:,}, "
            f"isolated={topology['isolated_candidates']}, conflicts={topology['conflict_groups']}, "
            f"somas_with_skeleton={topology['somas_with_min_skeleton']}/{len(valid_soma_ids)}",
            flush=True,
        )

    write_csv(output_root / "tlow_sweep.csv", rows)
    print("\nWriting soma-centred side-by-side crops...", flush=True)
    cells = write_cell_comparisons(
        output_root,
        args.probabilities.stem,
        raw_preview,
        argmax_skeleton,
        soma_labels,
        valid_soma,
        valid_soma_ids,
        centers,
        variants,
        args.cell_crop_size,
    )
    summary = {
        "script_version": SCRIPT_VERSION,
        "probabilities": str(args.probabilities.resolve()),
        "raw": str(args.raw.resolve()) if args.raw else None,
        "shape": [int(argmax.shape[0]), int(argmax.shape[1])],
        "t_high": args.t_high,
        "t_lows": t_lows,
        "min_soma_area": args.min_soma_area,
        "min_skeleton_px": args.min_skeleton_px,
        "cell_crop_size": args.cell_crop_size,
        "cell_comparisons": {
            "count": len(cells),
            "status_changes": sum(bool(cell["status_changes"]) for cell in cells),
            "isolated_at_least_once": sum(bool(cell["isolated_any"]) for cell in cells),
            "csv": "tlow_cell_comparison.csv",
            "directory": "cell_crops",
        },
        "argmax": baseline,
        "rows": rows,
    }
    (output_root / "tlow_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_html_report(
        output_root / "tlow_sweep_report.html",
        args.probabilities.stem,
        rows,
        args.t_high,
        cells,
        args.cell_crop_size,
    )
    print(f"\nComparison written to: {output_root.resolve()}")
    print(f"Open: {(output_root / 'tlow_sweep_report.html').resolve()}")


if __name__ == "__main__":
    main()
