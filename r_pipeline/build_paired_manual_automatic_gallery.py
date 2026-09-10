#!/usr/bin/env python3
"""Build a conservative one-to-one gallery of manual and automatic cells.

Cells are paired only by their location ROI in the original overview image.
The trace panels contain exactly the parent-child edges found in the supplied
final SWC files; no soma connection or other display-only edge is added.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
from roifile import ImagejRoi


@dataclass(frozen=True)
class CellLocation:
    cell_id: str
    cell_dir: Path
    x: float
    y: float


@dataclass(frozen=True)
class Pair:
    manual: CellLocation
    automatic: CellLocation
    distance_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-assignments", type=Path, required=True)
    parser.add_argument("--manual-cell-root", type=Path, required=True)
    parser.add_argument("--manual-swc-dir", type=Path, required=True)
    parser.add_argument("--automatic-cell-dir", type=Path, required=True)
    parser.add_argument("--automatic-swc-dir", type=Path, required=True)
    parser.add_argument("--automatic-job-id", required=True)
    parser.add_argument(
        "--automatic-pixel-size-um",
        type=float,
        default=0.2875008,
        help="Pixel size used to map final automatic SWC coordinates to raw pixels.",
    )
    parser.add_argument("--source-report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--maximum-distance-px",
        type=float,
        default=20.0,
        help="Maximum overview-image distance for a safe match (default: 20).",
    )
    return parser.parse_args()


def read_location_roi(path: Path) -> tuple[float, float]:
    rois = ImagejRoi.fromfile(path)
    if len(rois) != 1:
        raise ValueError(f"Expected one location ROI in {path}, got {len(rois)}")
    coordinates = rois[0].coordinates()
    if len(coordinates) != 1:
        raise ValueError(f"Expected one point in {path}, got {len(coordinates)}")
    return float(coordinates[0][0]), float(coordinates[0][1])


def manual_cell_number(raw_id: str) -> int:
    match = re.search(r"_(\d+)_\d+$", raw_id)
    if not match:
        raise ValueError(f"Cannot determine manual cell number from {raw_id!r}")
    return int(match.group(1))


def load_manual_cells(assignments: Path, cell_root: Path) -> list[CellLocation]:
    cells: list[CellLocation] = []
    with assignments.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_id = row["id"]
            cell_dir = cell_root / f"cell{manual_cell_number(raw_id)}"
            x, y = read_location_roi(cell_dir / "location.zip")
            cells.append(CellLocation(raw_id, cell_dir, x, y))
    return cells


def load_automatic_cells(cell_root: Path, job_id: str) -> list[CellLocation]:
    cells: list[CellLocation] = []
    for cell_dir in sorted(cell_root.glob("cell*")):
        review_path = cell_dir / "review.json"
        location_path = cell_dir / "location.json"
        if not review_path.is_file() or not location_path.is_file():
            continue
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("job_id") != job_id:
            continue
        location = json.loads(location_path.read_text(encoding="utf-8"))
        cells.append(
            CellLocation(
                cell_dir.name,
                cell_dir,
                float(location["global_x"]),
                float(location["global_y"]),
            )
        )
    return cells


def pair_cells(
    manual: Iterable[CellLocation],
    automatic: Iterable[CellLocation],
    maximum_distance_px: float,
) -> tuple[list[Pair], list[CellLocation], list[CellLocation]]:
    manual = list(manual)
    automatic = list(automatic)
    candidates = sorted(
        (
            math.hypot(manual_cell.x - automatic_cell.x, manual_cell.y - automatic_cell.y),
            manual_cell,
            automatic_cell,
        )
        for manual_cell in manual
        for automatic_cell in automatic
    )
    used_manual: set[str] = set()
    used_automatic: set[str] = set()
    pairs: list[Pair] = []
    for distance, manual_cell, automatic_cell in candidates:
        if distance > maximum_distance_px:
            break
        if manual_cell.cell_id in used_manual or automatic_cell.cell_id in used_automatic:
            continue
        pairs.append(Pair(manual_cell, automatic_cell, distance))
        used_manual.add(manual_cell.cell_id)
        used_automatic.add(automatic_cell.cell_id)
    pairs.sort(key=lambda pair: manual_cell_number(pair.manual.cell_id))
    unmatched_manual = [cell for cell in manual if cell.cell_id not in used_manual]
    unmatched_automatic = [cell for cell in automatic if cell.cell_id not in used_automatic]
    return pairs, unmatched_manual, unmatched_automatic


def load_report_asset_map(report_dir: Path) -> dict[str, str]:
    manifest_path = report_dir / "cell_image_manifest.csv"
    asset_map: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            kind = "manual" if row["source_kind"] == "Manuell" else "automatic"
            filename = f"{index:04d}_{kind}_raw.png"
            path = report_dir / "cell_images" / "assets" / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            asset_map[row["raw_id"]] = filename
    return asset_map


def read_swc(path: Path) -> tuple[dict[int, tuple[float, float]], list[tuple[int, int]]]:
    nodes: dict[int, tuple[float, float]] = {}
    parent_ids: dict[int, int] = {}
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
            parent_ids[node_id] = int(float(fields[6]))
    edges = [
        (node_id, parent_id)
        for node_id, parent_id in parent_ids.items()
        if parent_id in nodes
    ]
    if not nodes:
        raise ValueError(f"No SWC nodes found in {path}")
    return nodes, edges


def trace_extent(nodes: dict[int, tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in nodes.values()]
    ys = [point[1] for point in nodes.values()]
    return min(xs), min(ys), max(xs), max(ys)


def write_trace_svg(
    path: Path,
    nodes: dict[int, tuple[float, float]],
    edges: list[tuple[int, int]],
    shared_span: float,
) -> None:
    x_min, y_min, x_max, y_max = trace_extent(nodes)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    view_min_x = center_x - shared_span / 2.0
    view_min_y = center_y - shared_span / 2.0
    stroke_width = max(shared_span / 170.0, 0.08)
    root_ids = set(nodes).difference(node_id for node_id, _ in edges)
    lines = [
        (
            f'<line x1="{nodes[parent_id][0]:.6f}" y1="{nodes[parent_id][1]:.6f}" '
            f'x2="{nodes[node_id][0]:.6f}" y2="{nodes[node_id][1]:.6f}" />'
        )
        for node_id, parent_id in edges
    ]
    roots = [
        f'<circle cx="{nodes[root_id][0]:.6f}" cy="{nodes[root_id][1]:.6f}" r="{stroke_width * 1.8:.6f}" />'
        for root_id in sorted(root_ids)
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="'
        f'{view_min_x:.6f} {view_min_y:.6f} {shared_span:.6f} {shared_span:.6f}" '
        'preserveAspectRatio="xMidYMid meet">'
        '<rect width="100%" height="100%" fill="#080c10" />'
        f'<g fill="none" stroke="#18dce8" stroke-width="{stroke_width:.6f}" '
        'stroke-linecap="round" stroke-linejoin="round">'
        + "".join(lines)
        + '</g><g fill="#f2b84b">'
        + "".join(roots)
        + "</g></svg>"
    )
    path.write_text(svg, encoding="utf-8")


def write_trace_overlay(
    raw_image_path: Path,
    output_path: Path,
    nodes: dict[int, tuple[float, float]],
    edges: list[tuple[int, int]],
    coordinate_scale: float,
) -> None:
    image = Image.open(raw_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for node_id, parent_id in edges:
        parent = nodes[parent_id]
        node = nodes[node_id]
        draw.line(
            (
                parent[0] * coordinate_scale,
                parent[1] * coordinate_scale,
                node[0] * coordinate_scale,
                node[1] * coordinate_scale,
            ),
            fill=(24, 220, 232),
            width=1,
        )
    image.save(output_path)


def swc_total_length(
    nodes: dict[int, tuple[float, float]], edges: list[tuple[int, int]]
) -> float:
    return sum(
        math.hypot(
            nodes[node_id][0] - nodes[parent_id][0],
            nodes[node_id][1] - nodes[parent_id][1],
        )
        for node_id, parent_id in edges
    )


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(args: argparse.Namespace) -> tuple[int, int, int]:
    manual_cells = load_manual_cells(args.manual_assignments, args.manual_cell_root)
    automatic_cells = load_automatic_cells(args.automatic_cell_dir, args.automatic_job_id)
    pairs, unmatched_manual, unmatched_automatic = pair_cells(
        manual_cells, automatic_cells, args.maximum_distance_px
    )
    asset_map = load_report_asset_map(args.source_report_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []
    cards: list[str] = []
    max_observed_distance = max((pair.distance_px for pair in pairs), default=0.0)
    for index, pair in enumerate(pairs, start=1):
        manual_swc = args.manual_swc_dir / f"{pair.manual.cell_id}.swc"
        automatic_swc = args.automatic_swc_dir / f"{pair.automatic.cell_id}.swc"
        manual_nodes, manual_edges = read_swc(manual_swc)
        automatic_nodes, automatic_edges = read_swc(automatic_swc)
        extents = [trace_extent(manual_nodes), trace_extent(automatic_nodes)]
        shared_span = max(
            max(x_max - x_min, y_max - y_min)
            for x_min, y_min, x_max, y_max in extents
        )
        shared_span = max(shared_span * 1.12, 1.0)
        manual_trace_name = f"pair_{index:02d}_manual_trace.svg"
        automatic_trace_name = f"pair_{index:02d}_automatic_trace.svg"
        automatic_overlay_name = f"pair_{index:02d}_automatic_overlay.png"
        automatic_prediction_name = f"pair_{index:02d}_network_prediction.png"
        write_trace_svg(assets_dir / manual_trace_name, manual_nodes, manual_edges, shared_span)
        write_trace_svg(
            assets_dir / automatic_trace_name,
            automatic_nodes,
            automatic_edges,
            shared_span,
        )
        automatic_raw_asset = (
            args.source_report_dir
            / "cell_images"
            / "assets"
            / asset_map[pair.automatic.cell_id]
        )
        write_trace_overlay(
            automatic_raw_asset,
            assets_dir / automatic_overlay_name,
            automatic_nodes,
            automatic_edges,
            1.0 / args.automatic_pixel_size_um,
        )
        prediction_preview = pair.automatic.cell_dir / "prediction_preview.png"
        if not prediction_preview.is_file():
            raise FileNotFoundError(prediction_preview)
        Image.open(prediction_preview).convert("RGB").save(
            assets_dir / automatic_prediction_name
        )
        manual_length = swc_total_length(manual_nodes, manual_edges)
        automatic_length = swc_total_length(automatic_nodes, automatic_edges)
        rows.append(
            {
                "pair": index,
                "manual_id": pair.manual.cell_id,
                "automatic_id": pair.automatic.cell_id,
                "manual_global_x": f"{pair.manual.x:.3f}",
                "manual_global_y": f"{pair.manual.y:.3f}",
                "automatic_global_x": f"{pair.automatic.x:.3f}",
                "automatic_global_y": f"{pair.automatic.y:.3f}",
                "distance_px": f"{pair.distance_px:.3f}",
                "manual_trace_length": f"{manual_length:.3f}",
                "automatic_trace_length": f"{automatic_length:.3f}",
            }
        )
        manual_raw = f"../cell_images/assets/{asset_map[pair.manual.cell_id]}"
        automatic_raw = f"../cell_images/assets/{asset_map[pair.automatic.cell_id]}"
        cards.append(
            f"""
            <article class="pair-card">
              <header><span>Paar {index:02d}</span><strong>{html.escape(pair.manual.cell_id)} ↔ {html.escape(pair.automatic.cell_id)}</strong><small>Positionsabstand {pair.distance_px:.2f} px</small></header>
              <div class="pair-grid">
                <figure><img src="{manual_raw}" alt="Manuelles Rohbild"><figcaption>Manuelles Rohbild</figcaption></figure>
                <figure><img src="assets/{manual_trace_name}" alt="Manueller finaler SWC"><figcaption>Manueller finaler SWC · Länge {manual_length:.1f}</figcaption></figure>
                <figure><img src="{automatic_raw}" alt="Automatisches Rohbild"><figcaption>Automatisches Rohbild</figcaption></figure>
                <figure><img src="assets/{automatic_prediction_name}" alt="Prediction von Netz 141"><figcaption>Netz 141 · Prediction vor Hysterese und Zell-Extraktion</figcaption></figure>
                <figure><img src="assets/{automatic_overlay_name}" alt="Automatisches Tracing auf dem Rohbild"><figcaption>Automatisches finales Tracing auf Rohbild · exakt 1 px</figcaption></figure>
                <figure><img src="assets/{automatic_trace_name}" alt="Automatischer finaler SWC"><figcaption>Automatischer finaler SWC · Länge {automatic_length:.1f}</figcaption></figure>
              </div>
            </article>
            """
        )

    write_csv(
        args.output_dir / "paired_cells.csv",
        rows,
        [
            "pair",
            "manual_id",
            "automatic_id",
            "manual_global_x",
            "manual_global_y",
            "automatic_global_x",
            "automatic_global_y",
            "distance_px",
            "manual_trace_length",
            "automatic_trace_length",
        ],
    )
    write_csv(
        args.output_dir / "unmatched_manual_cells.csv",
        ({"manual_id": cell.cell_id, "global_x": cell.x, "global_y": cell.y} for cell in unmatched_manual),
        ["manual_id", "global_x", "global_y"],
    )
    document = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gleiche Zellen: manuell ↔ automatisch</title>
<style>
:root{{--bg:#0d1218;--card:#151d25;--line:#2b3a47;--text:#edf5f7;--muted:#9bb0ba;--cyan:#18dce8;--gold:#f2b84b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,Segoe UI,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}} h1{{margin:.1em 0}} .intro{{color:var(--muted);max-width:1000px;margin-bottom:22px}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0 26px}} .metric{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px}}
.metric strong{{display:block;font-size:24px;color:var(--cyan)}} .pair-card{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:0 0 18px;overflow:hidden}}
.pair-card header{{display:grid;grid-template-columns:90px 1fr auto;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line)}}
.pair-card header span{{color:var(--cyan);font-weight:700}} .pair-card header small{{color:var(--muted)}} .pair-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1px;background:var(--line)}}
figure{{margin:0;background:#080c10;min-width:0}} figure img{{width:100%;height:310px;display:block;object-fit:contain;image-rendering:auto}} figcaption{{background:var(--card);padding:9px 12px;color:var(--muted)}}
.legend{{border-left:3px solid var(--gold);padding-left:12px}} a{{color:var(--cyan)}} @media(max-width:1100px){{.pair-grid{{grid-template-columns:1fr 1fr}}.pair-card header{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p><a href="../index.html">← Vollständiger Gruppenvergleich</a></p>
<h1>Identische Zellen direkt gegenübergestellt</h1>
<p class="intro">Die Zuordnung basiert ausschließlich auf dem Soma-Mittelpunkt im gemeinsamen Übersichtsbild, nicht auf ähnlicher Trace-Form. Es werden nur eindeutige Paare bis {args.maximum_distance_px:.0f} px Abstand gezeigt. Cyan sind ausschließlich die echten Kanten der jeweiligen finalen <code>swc_final</code>-Datei; der orange Punkt markiert nur deren vorhandene Wurzel. Es wurden keine Linien ergänzt.</p>
<div class="summary"><div class="metric"><strong>{len(pairs)}</strong>sichere Paare</div><div class="metric"><strong>{len(unmatched_manual)}</strong>manuell ohne sicheren Gegenpart</div><div class="metric"><strong>{max_observed_distance:.1f} px</strong>größter Paarabstand</div></div>
<p class="intro legend">Beide Trace-Panels eines Paares verwenden denselben Maßstab. Dadurch sind Form und Ausdehnung direkt vergleichbar.</p>
{''.join(cards)}
</main></body></html>"""
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    return len(pairs), len(unmatched_manual), len(unmatched_automatic)


def main() -> int:
    args = parse_args()
    if args.maximum_distance_px <= 0:
        raise ValueError("--maximum-distance-px must be positive")
    if args.automatic_pixel_size_um <= 0:
        raise ValueError("--automatic-pixel-size-um must be positive")
    pair_count, unmatched_manual_count, unmatched_automatic_count = build_report(args)
    print(
        json.dumps(
            {
                "pairs": pair_count,
                "unmatched_manual": unmatched_manual_count,
                "unmatched_automatic": unmatched_automatic_count,
                "output": str(args.output_dir / "index.html"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
