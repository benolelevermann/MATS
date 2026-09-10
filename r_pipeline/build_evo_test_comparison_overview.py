#!/usr/bin/env python3
"""Build the permanent manual/automatic div10_CC comparison gallery.

Every manual cell is shown, even when no automatic counterpart exists yet.
Pairing is conservative: source overview plus a one-to-one soma-position match.
The skeleton panels render only parent-child edges from the exact SWC files that
enter the Evo pipeline; no display-only soma connection is added.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from roifile import ImagejRoi


SCRIPT_VERSION = "evo-test-comparison-v2-manual-swc-scale-2026-09-10"


@dataclass(frozen=True)
class ManualCell:
    manual_id: str
    number: int
    cell_dir: Path
    treatment: str
    source_image: str
    source_key: str
    x: float
    y: float


@dataclass(frozen=True)
class AutomaticCell:
    automatic_id: str
    cell_dir: Path
    source_image: str
    source_key: str
    x: float
    y: float


@dataclass(frozen=True)
class Match:
    manual: ManualCell
    automatic: AutomaticCell
    distance_px: float


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    test_root = project_root / "EvoTest" / "div10_CC"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=test_root)
    parser.add_argument(
        "--automatic-root",
        type=Path,
        action="append",
        default=[],
        help="Root containing safe_cells/cell* or cell* directly; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=test_root / "comparison_overview",
    )
    parser.add_argument("--maximum-distance-px", type=float, default=20.0)
    parser.add_argument(
        "--manual-pixel-size-um",
        type=float,
        default=0.406249892061169,
        help="Pixel calibration used by the manual SNT SWCs.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing generated overview after a successful build.",
    )
    return parser.parse_args()


def normalize_source(value: str) -> str:
    # ``Path.stem`` mistakes the decimal point in names such as ``zoom1.6``
    # for a file extension when the automatic case has no trailing ``.tif``.
    stem = re.sub(r"\.(?:tif|tiff|png|jpg|jpeg)$", "", Path(value).name, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def manual_number(raw_id: str) -> int:
    match = re.search(r"_(\d+)_\d+$", raw_id)
    if not match:
        raise ValueError(f"Cannot determine manual cell number from {raw_id!r}")
    return int(match.group(1))


def read_point_roi(path: Path) -> tuple[float, float]:
    rois = ImagejRoi.fromfile(path)
    if len(rois) != 1:
        raise ValueError(f"Expected one ROI in {path}, got {len(rois)}")
    coordinates = rois[0].coordinates()
    if len(coordinates) != 1:
        raise ValueError(f"Expected one point in {path}, got {len(coordinates)}")
    return float(coordinates[0][0]), float(coordinates[0][1])


def load_manual(test_root: Path) -> list[ManualCell]:
    assignment_path = test_root / "Decrypted" / "treatment_assignment_full.csv"
    cell_root = test_root / "tracings_M237_totrace_Encrypted"
    cells: list[ManualCell] = []
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            number = manual_number(row["id"])
            cell_dir = cell_root / f"cell{number}"
            x, y = read_point_roi(cell_dir / "location.zip")
            source = row.get("matched_original_name", "")
            cells.append(
                ManualCell(
                    manual_id=row["id"],
                    number=number,
                    cell_dir=cell_dir,
                    treatment=row.get("treatment", ""),
                    source_image=source,
                    source_key=normalize_source(source),
                    x=x,
                    y=y,
                )
            )
    return sorted(cells, key=lambda cell: cell.number)


def candidate_cell_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (root, root / "safe_cells"):
        if base.is_dir():
            candidates.extend(path for path in base.glob("cell*") if path.is_dir())
    return sorted(set(candidates))


def load_automatic(roots: Iterable[Path]) -> list[AutomaticCell]:
    cells: list[AutomaticCell] = []
    seen_paths: set[Path] = set()
    for root in roots:
        for cell_dir in candidate_cell_dirs(root):
            resolved = cell_dir.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            review_path = cell_dir / "review.json"
            location_path = cell_dir / "location.json"
            swc_path = cell_dir / "seg-000.swc"
            if not review_path.is_file() or not location_path.is_file() or not swc_path.is_file():
                continue
            review = json.loads(review_path.read_text(encoding="utf-8"))
            location = json.loads(location_path.read_text(encoding="utf-8"))
            source = str(review.get("case", ""))
            cells.append(
                AutomaticCell(
                    automatic_id=cell_dir.name,
                    cell_dir=cell_dir,
                    source_image=source,
                    source_key=normalize_source(source),
                    x=float(location["global_x"]),
                    y=float(location["global_y"]),
                )
            )
    return cells


def pair_cells(
    manual_cells: Iterable[ManualCell],
    automatic_cells: Iterable[AutomaticCell],
    maximum_distance_px: float,
) -> tuple[dict[str, Match], list[AutomaticCell]]:
    manual_cells = list(manual_cells)
    automatic_cells = list(automatic_cells)
    candidates = sorted(
        (
            math.hypot(manual.x - automatic.x, manual.y - automatic.y),
            manual.number,
            automatic.automatic_id,
            manual,
            automatic,
        )
        for manual in manual_cells
        for automatic in automatic_cells
        if manual.source_key and manual.source_key == automatic.source_key
    )
    used_manual: set[str] = set()
    used_automatic: set[Path] = set()
    matches: dict[str, Match] = {}
    for distance, _number, _automatic_id, manual, automatic in candidates:
        if distance > maximum_distance_px:
            continue
        automatic_key = automatic.cell_dir.resolve()
        if manual.manual_id in used_manual or automatic_key in used_automatic:
            continue
        matches[manual.manual_id] = Match(manual, automatic, distance)
        used_manual.add(manual.manual_id)
        used_automatic.add(automatic_key)
    unmatched = [
        cell for cell in automatic_cells if cell.cell_dir.resolve() not in used_automatic
    ]
    return matches, unmatched


def read_swc(path: Path) -> tuple[dict[int, tuple[float, float]], list[tuple[int, int]]]:
    nodes: dict[int, tuple[float, float]] = {}
    parents: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 7:
                continue
            node_id = int(float(fields[0]))
            nodes[node_id] = (float(fields[2]), float(fields[3]))
            parents[node_id] = int(float(fields[6]))
    edges = [
        (node_id, parent_id)
        for node_id, parent_id in parents.items()
        if parent_id in nodes
    ]
    if not nodes:
        raise ValueError(f"No SWC nodes in {path}")
    return nodes, edges


def normalized_rgb(path: Path) -> Image.Image:
    array = np.squeeze(np.asarray(tifffile.imread(path))).astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size:
        low, high = np.percentile(finite, [1.0, 99.7])
        if high <= low:
            high = low + 1.0
        gray = np.round(np.clip((array - low) / (high - low), 0, 1) * 255).astype(
            np.uint8
        )
    else:
        gray = np.zeros(array.shape, dtype=np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def fit_for_web(image: Image.Image, maximum_side: int = 640) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
    return copy


def save_webp(image: Image.Image, path: Path) -> None:
    fit_for_web(image).save(path, "WEBP", quality=86, method=4)


def manual_annotation(cell: ManualCell, coordinate_scale: float) -> Image.Image:
    image = normalized_rgb(cell.cell_dir / "raw.tif")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for roi in ImagejRoi.fromfile(cell.cell_dir / "soma.zip"):
        points = [tuple(map(float, point)) for point in roi.coordinates()]
        if len(points) >= 3:
            draw.polygon(points, fill=(255, 54, 164, 180))
    nodes, edges = read_swc(cell.cell_dir / "seg.swc")
    for node_id, parent_id in edges:
        parent_x, parent_y = nodes[parent_id]
        node_x, node_y = nodes[node_id]
        draw.line(
            (
                parent_x * coordinate_scale,
                parent_y * coordinate_scale,
                node_x * coordinate_scale,
                node_y * coordinate_scale,
            ),
            fill=(0, 238, 255, 255),
            width=1,
        )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def write_trace_svg(
    path: Path,
    nodes: dict[int, tuple[float, float]],
    edges: list[tuple[int, int]],
) -> None:
    xs = [point[0] for point in nodes.values()]
    ys = [point[1] for point in nodes.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 1.0) * 1.14
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    stroke = max(span / 190.0, 0.12)
    root_ids = set(nodes).difference(node_id for node_id, _parent_id in edges)
    lines = "".join(
        f'<line x1="{nodes[parent_id][0]:.5f}" y1="{nodes[parent_id][1]:.5f}" '
        f'x2="{nodes[node_id][0]:.5f}" y2="{nodes[node_id][1]:.5f}" />'
        for node_id, parent_id in edges
    )
    roots = "".join(
        f'<circle cx="{nodes[root_id][0]:.5f}" cy="{nodes[root_id][1]:.5f}" '
        f'r="{stroke * 2.0:.5f}" />'
        for root_id in sorted(root_ids)
    )
    document = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{center_x - span / 2:.5f} {center_y - span / 2:.5f} {span:.5f} {span:.5f}" '
        'preserveAspectRatio="xMidYMid meet">'
        '<rect width="100%" height="100%" fill="#070b0f" />'
        f'<g fill="none" stroke="#00eefa" stroke-width="{stroke:.5f}" '
        f'stroke-linecap="round" stroke-linejoin="round">{lines}</g>'
        f'<g fill="#f2b84b">{roots}</g></svg>'
    )
    path.write_text(document, encoding="utf-8")


def save_existing_preview(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        save_webp(image.convert("RGB"), destination)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict[str, object]:
    manual_cells = load_manual(args.test_root)
    automatic_cells = load_automatic(args.automatic_root)
    matches, unmatched_automatic = pair_cells(
        manual_cells, automatic_cells, args.maximum_distance_px
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []
    cards: list[str] = []
    for index, manual in enumerate(manual_cells, start=1):
        prefix = f"cell_{manual.number:04d}"
        manual_raw = f"{prefix}_manual_raw.webp"
        manual_annotation_name = f"{prefix}_manual_annotation.webp"
        manual_trace = f"{prefix}_manual_evo_input.svg"
        save_webp(normalized_rgb(manual.cell_dir / "raw.tif"), assets / manual_raw)
        save_webp(
            manual_annotation(manual, 1.0 / args.manual_pixel_size_um),
            assets / manual_annotation_name,
        )
        manual_nodes, manual_edges = read_swc(manual.cell_dir / "seg.swc")
        write_trace_svg(assets / manual_trace, manual_nodes, manual_edges)

        match = matches.get(manual.manual_id)
        if match:
            automatic = match.automatic
            auto_raw = f"{prefix}_automatic_raw.webp"
            auto_prediction = f"{prefix}_automatic_prediction.webp"
            auto_trace = f"{prefix}_automatic_evo_input.svg"
            raw_preview = automatic.cell_dir / "raw_preview.png"
            prediction_preview = automatic.cell_dir / "prediction_preview.png"
            if not raw_preview.is_file():
                save_webp(normalized_rgb(automatic.cell_dir / "raw.tif"), assets / auto_raw)
            else:
                save_existing_preview(raw_preview, assets / auto_raw)
            if prediction_preview.is_file():
                save_existing_preview(prediction_preview, assets / auto_prediction)
            else:
                save_webp(normalized_rgb(automatic.cell_dir / "prediction.tif"), assets / auto_prediction)
            auto_nodes, auto_edges = read_swc(automatic.cell_dir / "seg-000.swc")
            write_trace_svg(assets / auto_trace, auto_nodes, auto_edges)
            automatic_html = f"""
              <figure><img loading="lazy" src="assets/{auto_raw}" alt="Automatisches Rohbild"><figcaption>Automatisch · Rohbild</figcaption></figure>
              <figure><img loading="lazy" src="assets/{auto_prediction}" alt="Netz-141-Prediction"><figcaption>Automatisch · Netz-141-Prediction</figcaption></figure>
              <figure><img loading="lazy" src="assets/{auto_trace}" alt="Automatisches Evo-Eingabe-SWC"><figcaption>Automatisch · exaktes Evo-Eingabe-SWC</figcaption></figure>
            """
            paired = "yes"
            automatic_id = automatic.automatic_id
            distance = f"{match.distance_px:.3f}"
        else:
            automatic_html = "".join(
                '<figure class="missing"><div>Noch nicht automatisch getraced</div><figcaption>Automatischer Gegenpart fehlt</figcaption></figure>'
                for _ in range(3)
            )
            paired = "no"
            automatic_id = ""
            distance = ""

        rows.append(
            {
                "manual_id": manual.manual_id,
                "cell_number": manual.number,
                "treatment": manual.treatment,
                "source_image": manual.source_image,
                "global_x": f"{manual.x:.3f}",
                "global_y": f"{manual.y:.3f}",
                "paired": paired,
                "automatic_id": automatic_id,
                "pair_distance_px": distance,
            }
        )
        pair_label = (
            f"↔ {html.escape(automatic_id)} · {float(distance):.2f} px"
            if paired == "yes"
            else "noch ohne automatischen Gegenpart"
        )
        cards.append(
            f"""
            <article class="cell-card" data-paired="{paired}" data-treatment="{html.escape(manual.treatment.lower())}" data-search="{html.escape((manual.manual_id + ' ' + manual.source_image + ' ' + manual.treatment).lower())}">
              <header><strong>Zelle {manual.number} · {html.escape(manual.treatment)}</strong><span>{pair_label}</span><small>{html.escape(manual.source_image)}</small></header>
              <div class="panels">
                <figure><img loading="lazy" src="assets/{manual_raw}" alt="Manuelles Rohbild"><figcaption>Manuell · Rohbild</figcaption></figure>
                <figure><img loading="lazy" src="assets/{manual_annotation_name}" alt="Manuelle Annotation"><figcaption>Manuell · Annotation (Soma + Tracing)</figcaption></figure>
                <figure><img loading="lazy" src="assets/{manual_trace}" alt="Manuelles Evo-Eingabe-SWC"><figcaption>Manuell · exaktes Evo-Eingabe-SWC</figcaption></figure>
                {automatic_html}
              </div>
            </article>
            """
        )
        if index % 25 == 0 or index == len(manual_cells):
            print(f"Rendered {index}/{len(manual_cells)} manual cells", flush=True)

    write_csv(args.output_dir / "comparison_manifest.csv", rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "manual_cells": len(manual_cells),
        "automatic_cells": len(automatic_cells),
        "safe_pairs": len(matches),
        "manual_without_automatic": len(manual_cells) - len(matches),
        "automatic_without_manual": len(unmatched_automatic),
        "maximum_distance_px": args.maximum_distance_px,
        "manual_pixel_size_um": args.manual_pixel_size_um,
        "automatic_roots": [str(path.resolve()) for path in args.automatic_root],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    treatments = sorted({cell.treatment for cell in manual_cells})
    treatment_buttons = "".join(
        f'<button data-filter="treatment" data-value="{html.escape(value.lower())}">{html.escape(value)}</button>'
        for value in treatments
    )
    document = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>div10_CC · permanenter Testdatensatz</title><style>
:root{{--bg:#0c1218;--card:#151e27;--line:#2b3b48;--text:#eef5f6;--muted:#9db0b9;--cyan:#00eefa;--pink:#ff36a4;--gold:#f2b84b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}}main{{width:min(1900px,calc(100% - 28px));margin:auto;padding:24px 0 50px}}h1{{font:38px Georgia,serif;margin:0}}.lead{{max-width:1100px;color:var(--muted)}}.summary{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}}.metric{{border:1px solid var(--line);background:var(--card);padding:10px 15px;border-radius:9px}}.metric strong{{display:block;color:var(--cyan);font-size:22px}}.toolbar{{position:sticky;top:0;z-index:4;background:#0c1218ee;border-bottom:1px solid var(--line);padding:12px 0;display:flex;gap:8px;flex-wrap:wrap}}button,input{{border:1px solid #405462;background:#17232d;color:var(--text);border-radius:7px;padding:8px 11px}}input{{min-width:280px}}button.active{{border-color:var(--cyan);color:var(--cyan)}}.cell-card{{border:1px solid var(--line);background:var(--card);border-radius:11px;overflow:hidden;margin:15px 0}}.cell-card header{{display:grid;grid-template-columns:auto auto 1fr;gap:15px;padding:11px 14px;border-bottom:1px solid var(--line);align-items:center}}header strong{{color:var(--cyan)}}header span,header small{{color:var(--muted)}}header small{{text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.panels{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1px;background:var(--line)}}figure{{margin:0;background:#070b0f;min-width:0}}figure img,figure>div{{display:block;width:100%;height:300px;object-fit:contain}}figure>div{{display:grid;place-items:center;color:#71838c;padding:20px;text-align:center}}figcaption{{background:var(--card);color:var(--muted);padding:9px 11px;min-height:47px}}.note{{border-left:3px solid var(--gold);padding-left:12px}}@media(max-width:1200px){{.panels{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:720px){{.panels{{grid-template-columns:1fr}}.cell-card header{{grid-template-columns:1fr}}header small{{text-align:left}}}}
</style></head><body><main><h1>div10_CC · dauerhafter Testdatensatz</h1>
<p class="lead">Alle {len(manual_cells)} manuell getracten Zellen bleiben in dieser Übersicht erhalten. Jede Zeile steht für genau eine manuelle Zelle. Wenn dieselbe Zelle automatisch getraced wurde, steht sie anhand von Übersichtsbild und Soma-Position direkt rechts daneben.</p>
<p class="lead note">Die SWC-Felder zeigen ausschließlich die echten Parent-Child-Kanten der Datei, die in die Evo-Pipeline gelangt. Es werden für die Darstellung keine Linien ergänzt. Orange markiert den vorhandenen SWC-Root. Das Feld „manuelle Annotation“ ist keine Netz-Prediction, sondern die manuelle Referenz in derselben Farbgebung.</p>
<div class="summary"><div class="metric"><strong>{len(manual_cells)}</strong>manuell</div><div class="metric"><strong>{len(matches)}</strong>sicher gepaart</div><div class="metric"><strong>{len(manual_cells)-len(matches)}</strong>noch ohne Automatik</div></div>
<div class="toolbar"><input id="search" type="search" placeholder="Zelle, Bild oder Bedingung suchen …"><button class="active" data-filter="paired" data-value="all">Alle</button><button data-filter="paired" data-value="yes">Nur gepaart</button><button data-filter="paired" data-value="no">Nur ungepaart</button>{treatment_buttons}</div>
<section id="cards">{''.join(cards)}</section></main><script>
let paired='all',treatment='all';const cards=[...document.querySelectorAll('.cell-card')],search=document.getElementById('search');
function apply(){{const q=search.value.trim().toLowerCase();cards.forEach(c=>{{const okPair=paired==='all'||c.dataset.paired===paired;const okTreatment=treatment==='all'||c.dataset.treatment===treatment;const okSearch=!q||c.dataset.search.includes(q);c.hidden=!(okPair&&okTreatment&&okSearch)}})}}
document.querySelectorAll('button[data-filter]').forEach(b=>b.onclick=()=>{{const kind=b.dataset.filter;document.querySelectorAll(`button[data-filter="${{kind}}"]`).forEach(x=>x.classList.remove('active'));b.classList.add('active');if(kind==='paired')paired=b.dataset.value;else treatment=(treatment===b.dataset.value?'all':b.dataset.value);apply()}});search.oninput=apply;
</script></body></html>"""
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.maximum_distance_px <= 0:
        raise ValueError("--maximum-distance-px must be positive")
    if args.manual_pixel_size_um <= 0:
        raise ValueError("--manual-pixel-size-um must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.replace:
        raise FileExistsError(
            f"Output is not empty: {args.output_dir}. Add --replace to update it."
        )
    final_output = args.output_dir.resolve()
    if args.replace and final_output.exists() and any(final_output.iterdir()):
        final_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{final_output.name}.building-", dir=final_output.parent
            )
        )
        staging_output = temporary_parent / final_output.name
        previous_output = final_output.parent / f".{final_output.name}.previous"
        if previous_output.exists():
            raise FileExistsError(
                f"Previous-update recovery directory exists: {previous_output}"
            )
        args.output_dir = staging_output
        try:
            summary = build(args)
            final_output.replace(previous_output)
            try:
                staging_output.replace(final_output)
            except Exception:
                previous_output.replace(final_output)
                raise
            shutil.rmtree(previous_output)
        finally:
            if temporary_parent.exists():
                shutil.rmtree(temporary_parent)
        args.output_dir = final_output
    else:
        summary = build(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Overview: {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
