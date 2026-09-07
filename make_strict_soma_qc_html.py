from __future__ import annotations

"""Create a mask-faithful HTML gallery for a strict soma quality-gate run.

The gallery is intentionally based on the actual TIFF input masks. It is a
review tool only: it never changes the quality-gate maps or segmentation.

Colours:
    cyan    selected skeleton
    magenta selected soma
    red     a foreign semantic soma inside the planned crop
    yellow  robust distance-transform soma core(s)
"""

import argparse
import csv
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


SCRIPT_VERSION = "strict-soma-qc-html-v1-2026-08-16"
COLORS = {
    "skeleton": np.asarray([0, 220, 255], dtype=np.float32),
    "soma": np.asarray([255, 45, 180], dtype=np.float32),
    "foreign_soma": np.asarray([245, 70, 55], dtype=np.float32),
    "cores": np.asarray([255, 235, 0], dtype=np.float32),
}


@dataclass(frozen=True)
class Cell:
    identifier: int
    category: str
    mask: np.ndarray
    row: dict[str, str]
    bounds: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mask-faithful HTML review gallery for strict soma QC."
    )
    parser.add_argument(
        "--quality-dir",
        type=Path,
        required=True,
        help="Output folder produced by quality_gate_cells_for_evo_soma_strict.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-flagged", action="store_true", help="Render review and excluded candidates only.")
    parser.add_argument("--thumbnail-size", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_2d(path: Path, label: str) -> np.ndarray:
    require_file(path, label)
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{label} must be 2-D after squeeze, got {array.shape}: {path}")
    return array


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def robust_unit_scale(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    values = image[np.isfinite(image)]
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    lower, upper = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def crop_bounds_from_row(row: dict[str, str], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    x0 = parse_int(row.get("crop_x_min"))
    y0 = parse_int(row.get("crop_y_min"))
    x1 = parse_int(row.get("crop_x_max_exclusive"))
    y1 = parse_int(row.get("crop_y_max_exclusive"))
    if None in (x0, y0, x1, y1):
        return None
    assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
    if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
        return None
    return y0, y1, x0, x1


def load_cells(quality_dir: Path, shape: tuple[int, int]) -> list[Cell]:
    report = quality_dir / "cell_quality_report.csv"
    require_file(report, "cell quality report")
    maps = {
        "safe": read_2d(quality_dir / "01_cells_safe_for_evo.tif", "safe instance map"),
        "review": read_2d(quality_dir / "02_cells_for_manual_review.tif", "review instance map"),
        "excluded": read_2d(quality_dir / "03_cells_excluded_from_export.tif", "excluded instance map"),
    }
    if any(value.shape != shape for value in maps.values()):
        raise RuntimeError("Quality maps must have the same shape as the raw/semantic image.")
    cells: list[Cell] = []
    with report.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            category = (row.get("category") or "").strip()
            identifier = parse_int(row.get("source_instance_id"))
            if category not in maps or identifier is None or identifier <= 0:
                continue
            bounds = crop_bounds_from_row(row, shape)
            # Non-selected broad assignment cases do not have a crop proposal;
            # they stay documented in CSV but cannot be shown as a crop card.
            if bounds is None:
                continue
            mask = maps[category] == identifier
            if not mask.any():
                continue
            cells.append(Cell(identifier, category, mask, row, bounds))
    order = {"excluded": 0, "review": 1, "safe": 2}
    return sorted(cells, key=lambda cell: (order[cell.category], cell.identifier))


def overlay(canvas: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if np.any(mask):
        canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * color


def mark_cores(canvas: np.ndarray, markers: np.ndarray) -> None:
    ys, xs = np.nonzero(markers)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 2), min(canvas.shape[0], y + 3)
        x0, x1 = max(0, x - 2), min(canvas.shape[1], x + 3)
        canvas[y0:y1, x0:x1] = COLORS["cores"]


def build_assets(
    cell: Cell,
    original: np.ndarray,
    semantic: np.ndarray,
    core_markers: np.ndarray,
    assets: Path,
    thumbnail_size: int,
) -> tuple[str, str, int, int]:
    y0, y1, x0, x1 = cell.bounds
    raw = original[y0:y1, x0:x1]
    local_cell = cell.mask[y0:y1, x0:x1]
    local_semantic = semantic[y0:y1, x0:x1]
    local_cores = core_markers[y0:y1, x0:x1] == cell.identifier
    skeleton = local_cell & (local_semantic == 1)
    soma = local_cell & (local_semantic == 2)
    foreign_soma = (local_semantic == 2) & ~local_cell

    raw_rgb = np.repeat((robust_unit_scale(raw) * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    overlay_rgb = raw_rgb.copy()
    overlay(overlay_rgb, skeleton, COLORS["skeleton"], 0.92)
    overlay(overlay_rgb, soma, COLORS["soma"], 0.80)
    overlay(overlay_rgb, foreign_soma, COLORS["foreign_soma"], 0.88)
    mark_cores(overlay_rgb, local_cores)
    mask_rgb = np.zeros_like(raw_rgb)
    mask_rgb[skeleton] = COLORS["skeleton"]
    mask_rgb[soma] = COLORS["soma"]
    mask_rgb[foreign_soma] = COLORS["foreign_soma"]
    mark_cores(mask_rgb, local_cores)

    stem = f"{cell.category}_{cell.identifier:05d}"
    raw_name, overlay_name, mask_name = f"{stem}_raw.png", f"{stem}_overlay.png", f"{stem}_mask.png"
    Image.fromarray(np.clip(raw_rgb, 0, 255).astype(np.uint8)).save(assets / raw_name, optimize=True)
    Image.fromarray(np.clip(overlay_rgb, 0, 255).astype(np.uint8)).save(assets / overlay_name, optimize=True)
    Image.fromarray(np.clip(mask_rgb, 0, 255).astype(np.uint8)).save(assets / mask_name, optimize=True)
    return raw_name, overlay_name, mask_name, int(max(raw.shape))


def reason_text(cell: Cell) -> str:
    hard = (cell.row.get("hard_reasons") or "").replace(";", "; ")
    review = (cell.row.get("review_reasons") or "").replace(";", "; ")
    if hard and review:
        return f"Hard: {hard} | Review: {review}"
    return hard or review or "Passed all automatic QC rules"


def metric(cell: Cell, name: str) -> str:
    value = (cell.row.get(name) or "").strip()
    return value if value else "–"


def card(cell: Cell, raw: str, overlay_img: str, mask: str) -> str:
    identifier = cell.identifier
    title = f"Instance {identifier} · {cell.category}"
    reason = reason_text(cell)
    return f"""<article class='card {html.escape(cell.category)}' data-category='{html.escape(cell.category)}'>
  <h2>{html.escape(title)}</h2>
  <button class='image-button' type='button' data-raw='assets/{html.escape(raw)}' data-overlay='assets/{html.escape(overlay_img)}' data-mask='assets/{html.escape(mask)}' data-title='{html.escape(title)}'>
    <img src='assets/{html.escape(overlay_img)}' alt='{html.escape(title)}'>
  </button>
  <p class='reason'>{html.escape(reason)}</p>
  <dl>
    <dt>Soma px</dt><dd>{html.escape(metric(cell, 'soma_pixels'))}</dd>
    <dt>Strong cores</dt><dd>{html.escape(metric(cell, 'strong_soma_core_count'))}</dd>
    <dt>Core radius</dt><dd>{html.escape(metric(cell, 'soma_core_radius_px'))} px</dd>
    <dt>Solidity</dt><dd>{html.escape(metric(cell, 'soma_solidity'))}</dd>
    <dt>Foreign somata</dt><dd>{html.escape(metric(cell, 'foreign_soma_components_in_crop'))}</dd>
    <dt>Foreign IDs</dt><dd>{html.escape(metric(cell, 'foreign_instance_ids_in_crop'))}</dd>
    <dt>Skeleton px</dt><dd>{html.escape(metric(cell, 'skeleton_pixels'))}</dd>
  </dl>
</article>"""


def write_html(output: Path, cards: list[str], counts: dict[str, int], quality_dir: Path) -> None:
    card_html = "\n".join(cards)
    output.write_text(
        f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Strict soma QC review</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: Canvas; color: CanvasText; }}
main {{ max-width: 1640px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 6px; font-size: 1.55rem; }}
.intro {{ margin: 0 0 16px; max-width: 1100px; color: color-mix(in srgb, CanvasText 72%, Canvas 28%); }}
.counts, .controls {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 14px 0; }}
.chip, .filter {{ border: 1px solid color-mix(in srgb, CanvasText 30%, Canvas 70%); border-radius: 999px; padding: 6px 11px; font: inherit; background: Canvas; color: CanvasText; }}
.chip.safe {{ border-color: #15946e; }} .chip.review {{ border-color: #c78300; }} .chip.excluded {{ border-color: #c43b32; }}
.filter {{ cursor: pointer; }} .filter[aria-pressed='false'] {{ opacity: .48; text-decoration: line-through; }}
.legend {{ font-size: .86rem; margin: 0 0 18px; color: color-mix(in srgb, CanvasText 70%, Canvas 30%); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid color-mix(in srgb, CanvasText 23%, Canvas 77%); border-radius: 10px; overflow: hidden; background: color-mix(in srgb, Canvas 94%, CanvasText 6%); }}
.card.review {{ border-top: 5px solid #d39100; }} .card.excluded {{ border-top: 5px solid #d34a3f; }} .card.safe {{ border-top: 5px solid #22a979; }}
.card h2 {{ margin: 0; padding: 10px 12px 8px; font-size: .98rem; }}
.image-button {{ padding: 0; width: 100%; border: 0; background: #000; cursor: zoom-in; display: block; }}
.image-button img {{ display: block; width: 100%; max-height: 340px; object-fit: contain; image-rendering: pixelated; }}
.reason {{ min-height: 2.65em; margin: 10px 12px; font-size: .83rem; overflow-wrap: anywhere; }}
dl {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; margin: 10px 12px 13px; font-size: .78rem; }}
dt {{ color: color-mix(in srgb, CanvasText 62%, Canvas 38%); }} dd {{ margin: 0; overflow-wrap: anywhere; }}
dialog {{ width: min(96vw, 1100px); height: min(92vh, 1000px); padding: 14px; border: 1px solid color-mix(in srgb, CanvasText 35%, Canvas 65%); background: Canvas; color: CanvasText; }}
dialog::backdrop {{ background: rgb(0 0 0 / .72); }} .dialog-head {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:10px; }}
.dialog-head p {{ margin:0; overflow-wrap:anywhere; }} .dialog-body {{ height:calc(100% - 48px); display:grid; place-items:center; overflow:auto; background:#000; }}
.dialog-body img {{ max-width:100%; max-height:100%; image-rendering:pixelated; }}
@media (max-width: 650px) {{ main {{ padding: 14px; }} }}
</style>
</head>
<body>
<main>
  <h1>Strict soma / multi-cell crop selection</h1>
  <p class='intro'>This report uses the actual raw image and exact QC mask pixels. It does not dilate, skeletonize, split, or modify any segmentation. Cyan: selected skeleton; magenta: selected soma; red: another semantic soma in the proposed crop; yellow: robust distance-transform core.</p>
  <div class='counts'>
    <span class='chip safe'>Safe / export: {counts['safe']}</span>
    <span class='chip review'>Review / not exported: {counts['review']}</span>
    <span class='chip excluded'>Excluded / not exported: {counts['excluded']}</span>
  </div>
  <div class='controls' aria-label='Category filters'>
    <button class='filter' data-category='safe' aria-pressed='true'>Safe</button>
    <button class='filter' data-category='review' aria-pressed='true'>Review</button>
    <button class='filter' data-category='excluded' aria-pressed='true'>Excluded</button>
  </div>
  <p class='legend'>Source report: {html.escape(str(quality_dir / 'cell_quality_report.csv'))}</p>
  <section class='grid'>{card_html}</section>
</main>
<dialog id='dialog'><div class='dialog-head'><p id='dialog-title'></p><div><button id='show-raw'>Raw</button> <button id='show-overlay'>Overlay</button> <button id='show-mask'>Masks</button> <button id='close'>Close</button></div></div><div class='dialog-body'><img id='dialog-image' alt='QC crop'></div></dialog>
<script>
(() => {{
  const visible = new Set(['safe','review','excluded']);
  document.querySelectorAll('.filter').forEach(button => button.addEventListener('click', () => {{
    const category = button.dataset.category;
    if (visible.has(category)) visible.delete(category); else visible.add(category);
    button.setAttribute('aria-pressed', String(visible.has(category)));
    document.querySelectorAll('.card').forEach(card => card.hidden = !visible.has(card.dataset.category));
  }}));
  const dialog = document.getElementById('dialog'); const image = document.getElementById('dialog-image');
  let paths = {{}};
  document.querySelectorAll('.image-button').forEach(button => button.addEventListener('click', () => {{
    paths = {{raw: button.dataset.raw, overlay: button.dataset.overlay, mask: button.dataset.mask}};
    image.src = paths.overlay; document.getElementById('dialog-title').textContent = button.dataset.title; dialog.showModal();
  }}));
  document.getElementById('show-raw').onclick = () => image.src = paths.raw;
  document.getElementById('show-overlay').onclick = () => image.src = paths.overlay;
  document.getElementById('show-mask').onclick = () => image.src = paths.mask;
  document.getElementById('close').onclick = () => dialog.close();
}})();
</script>
</body>
</html>""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.thumbnail_size < 64:
        raise ValueError("--thumbnail-size must be at least 64.")
    quality_dir = args.quality_dir.resolve()
    summary_path = quality_dir / "run_summary.json"
    require_file(summary_path, "Quality gate run summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        original_path = Path(summary["original"])
        semantic_path = Path(summary["semantic"])
    except KeyError as error:
        raise RuntimeError("run_summary.json does not contain original/semantic paths.") from error
    original = read_2d(original_path, "Original raw image")
    semantic = read_2d(semantic_path, "Semantic segmentation")
    core_markers = read_2d(quality_dir / "06_soma_core_markers.tif", "Soma core marker map")
    if not (original.shape == semantic.shape == core_markers.shape):
        raise RuntimeError("Raw, semantic and core-marker images must share a shape.")
    cells = load_cells(quality_dir, original.shape)
    if args.only_flagged:
        cells = [cell for cell in cells if cell.category != "safe"]
    if not cells:
        raise RuntimeError("No renderable cells matched the requested category selection.")
    prepare_output(args.output_dir, args.overwrite)
    assets = args.output_dir / "assets"
    assets.mkdir()
    cards: list[str] = []
    manifest_rows: list[dict[str, str]] = []
    counts = {"safe": 0, "review": 0, "excluded": 0}
    for cell in cells:
        raw, overlay_img, mask, max_dimension = build_assets(cell, original, semantic, core_markers, assets, args.thumbnail_size)
        cards.append(card(cell, raw, overlay_img, mask))
        counts[cell.category] += 1
        manifest_rows.append({
            "source_instance_id": str(cell.identifier),
            "category": cell.category,
            "hard_reasons": cell.row.get("hard_reasons", ""),
            "review_reasons": cell.row.get("review_reasons", ""),
            "raw_asset": f"assets/{raw}",
            "overlay_asset": f"assets/{overlay_img}",
            "mask_asset": f"assets/{mask}",
            "max_dimension_px": str(max_dimension),
        })
    with (args.output_dir / "gallery_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_html(args.output_dir / "index.html", cards, counts, quality_dir)
    (args.output_dir / "gallery_summary.json").write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "quality_dir": str(quality_dir),
        "rendered_counts": counts,
        "display_note": "Masks use exact TIFF pixels; raw images are percentile-normalized for browser display only.",
    }, indent=2), encoding="utf-8")
    print("=" * 72)
    print("STRICT SOMA QC HTML GALLERY COMPLETE")
    print("=" * 72)
    print(f"Safe: {counts['safe']}; review: {counts['review']}; excluded: {counts['excluded']}")
    print(f"HTML: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
