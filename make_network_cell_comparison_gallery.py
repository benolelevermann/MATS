from __future__ import annotations

"""Match exported cells by soma position and render a Dataset138/139 gallery."""

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.optimize import linear_sum_assignment

from cell_pipeline_web.pipeline import _normalize_u8


SKELETON = np.asarray((0, 225, 255), dtype=np.uint8)
SOMA = np.asarray((245, 70, 170), dtype=np.uint8)
ADDED = np.asarray((255, 190, 54), dtype=np.uint8)
REMOVED = np.asarray((235, 70, 65), dtype=np.uint8)


@dataclass(frozen=True)
class Cell:
    network: str
    case: str
    folder: str
    root: Path
    center_y: float
    center_x: float
    x0: int
    y0: int
    x1: int
    y1: int
    skeleton_pixels: int
    soma_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--pred-138", type=Path, required=True)
    parser.add_argument("--pred-139", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-match-distance", type=float, default=40.0)
    parser.add_argument("--minimum-crop-size", type=int, default=192)
    return parser.parse_args()


def read2d(path: Path) -> np.ndarray:
    return np.squeeze(np.asarray(tifffile.imread(path)))


def find_prediction(folder: Path, case: str) -> Path:
    for name in (f"{case}.tif", f"{case}_138.tif", f"{case}_139.tif"):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Keine Prediction fuer {case} in {folder}")


def load_cells(root: Path, network: str, case: str) -> list[Cell]:
    result: list[Cell] = []
    case_root = root / f"cells_net{network}" / case
    for folder in sorted(case_root.glob("cell*")):
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        bounds = metadata["bounds"]
        location = metadata["location"]
        result.append(
            Cell(
                network=network,
                case=case,
                folder=folder.name,
                root=folder,
                center_y=float(location["global_y"]),
                center_x=float(location["global_x"]),
                x0=int(bounds["x_min"]),
                y0=int(bounds["y_min"]),
                x1=int(bounds["x_max_exclusive"]),
                y1=int(bounds["y_max_exclusive"]),
                skeleton_pixels=int(metadata["skeleton_pixels"]),
                soma_pixels=int(metadata["soma_pixels"]),
            )
        )
    return result


def match_cells(
    cells138: list[Cell], cells139: list[Cell], maximum_distance: float
) -> list[tuple[str, Cell | None, Cell | None, float | None]]:
    if not cells138:
        return [("only139", None, cell, None) for cell in cells139]
    if not cells139:
        return [("only138", cell, None, None) for cell in cells138]
    a = np.asarray([(cell.center_y, cell.center_x) for cell in cells138], dtype=np.float64)
    b = np.asarray([(cell.center_y, cell.center_x) for cell in cells139], dtype=np.float64)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, columns = linear_sum_assignment(distances)
    used138: set[int] = set()
    used139: set[int] = set()
    result: list[tuple[str, Cell | None, Cell | None, float | None]] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        distance = float(distances[row, column])
        if distance <= maximum_distance:
            used138.add(row)
            used139.add(column)
            result.append(("matched", cells138[row], cells139[column], distance))
    result.extend(("only138", cell, None, None) for i, cell in enumerate(cells138) if i not in used138)
    result.extend(("only139", None, cell, None) for i, cell in enumerate(cells139) if i not in used139)
    return sorted(
        result,
        key=lambda item: (
            {"matched": 0, "only138": 1, "only139": 2}[item[0]],
            (item[1] or item[2]).center_y,
            (item[1] or item[2]).center_x,
        ),
    )


def square_bounds(
    cell138: Cell | None, cell139: Cell | None, shape: tuple[int, int], minimum: int
) -> tuple[int, int, int, int]:
    cells = [cell for cell in (cell138, cell139) if cell is not None]
    x0 = min(cell.x0 for cell in cells)
    y0 = min(cell.y0 for cell in cells)
    x1 = max(cell.x1 for cell in cells)
    y1 = max(cell.y1 for cell in cells)
    side = max(minimum, x1 - x0, y1 - y0)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    side = min(side, shape[0], shape[1])
    left = max(0, min(round(center_x - side / 2), shape[1] - side))
    top = max(0, min(round(center_y - side / 2), shape[0] - side))
    return int(top), int(top + side), int(left), int(left + side)


def overlay(gray: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[semantic == 1] = SKELETON
    rgb[semantic == 2] = SOMA
    return rgb


def hysteresis_overlay(gray: np.ndarray, prediction: np.ndarray, postprocessed: np.ndarray) -> np.ndarray:
    rgb = overlay(gray, postprocessed)
    added = (postprocessed == 1) & (prediction != 1)
    removed = (prediction == 1) & (postprocessed != 1)
    rgb[added] = ADDED
    rgb[removed] = REMOVED
    return rgb


def projected_cell(cell: Cell | None, bounds: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = bounds
    canvas = np.zeros((bottom - top, right - left), dtype=np.uint8)
    if cell is None:
        return canvas
    semantic = read2d(cell.root / "seg.tif").astype(np.uint8)
    gy0, gy1 = max(top, cell.y0), min(bottom, cell.y1)
    gx0, gx1 = max(left, cell.x0), min(right, cell.x1)
    if gy1 <= gy0 or gx1 <= gx0:
        return canvas
    canvas[gy0 - top : gy1 - top, gx0 - left : gx1 - left] = semantic[
        gy0 - cell.y0 : gy1 - cell.y0, gx0 - cell.x0 : gx1 - cell.x0
    ]
    return canvas


def save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, optimize=True)


def write_case_page(
    path: Path,
    case: str,
    cards: list[str],
    matched: int,
    only138: int,
    only139: int,
) -> None:
    body = "\n".join(cards)
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(case)} – Zellvergleich</title>
<style>
body{{font:14px/1.45 Segoe UI,Arial,sans-serif;margin:22px;background:#101318;color:#edf1f5}}
a{{color:#7fc8ff}} button{{margin:3px;padding:7px 11px;border:1px solid #46515e;border-radius:6px;background:#20262d;color:#edf1f5;cursor:pointer}}
.card{{background:#191e24;border:1px solid #303842;border-radius:9px;margin:16px 0;padding:12px}}
.card h2{{font-size:15px;margin:0 0 8px}} .network{{display:grid;grid-template-columns:75px repeat(4,minmax(190px,1fr));gap:8px;align-items:start;margin-top:8px}}
.network>strong{{padding-top:8px}} figure{{margin:0}} img{{width:100%;display:block;background:#000;border-radius:5px;image-rendering:auto}} figcaption{{color:#b9c2cc;padding-top:3px;font-size:12px}}
.matched{{border-left:4px solid #48c78e}} .only138{{border-left:4px solid #ffb84d}} .only139{{border-left:4px solid #7aa7ff}}
.muted{{color:#aab4be}} @media(max-width:900px){{.network{{grid-template-columns:1fr 1fr}}}}
</style></head><body><p><a href="index.html">← Übersicht</a></p><h1>{html.escape(case)}</h1>
<p>{matched} zugeordnete Zellen · {only138} nur Netz 138 · {only139} nur Netz 139</p>
<p><button onclick="show('all')">Alle</button><button onclick="show('matched')">Zugeordnet</button><button onclick="show('only138')">Nur 138</button><button onclick="show('only139')">Nur 139</button></p>
<p class="muted">Cyan: Skelett · Magenta: Soma · Gelb: durch Hysterese ergänzt · Rot: durch Hysterese entfernt.</p>{body}
<script>function show(s){{document.querySelectorAll('.card').forEach(c=>c.style.display=(s==='all'||c.dataset.status===s)?'block':'none')}};</script></body></html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_root = args.output_dir / "assets"
    assets_root.mkdir(exist_ok=True)
    cases = sorted(path.name.removesuffix("_0000.tif") for path in args.inputs.glob("*_0000.tif"))
    all_rows: list[dict[str, object]] = []
    index_rows: list[str] = []

    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case}", flush=True)
        raw = read2d(args.inputs / f"{case}_0000.tif")
        pred138 = read2d(find_prediction(args.pred_138, case)).astype(np.uint8)
        pred139 = read2d(find_prediction(args.pred_139, case)).astype(np.uint8)
        hyst138 = read2d(args.comparison_root / "hysteresis_net138" / f"{case}_adaptive_hysteresis_0-1-2.tif").astype(np.uint8)
        hyst139 = read2d(args.comparison_root / "hysteresis_net139" / f"{case}_adaptive_hysteresis_0-1-2.tif").astype(np.uint8)
        cells138 = load_cells(args.comparison_root, "138", case)
        cells139 = load_cells(args.comparison_root, "139", case)
        pairs = match_cells(cells138, cells139, args.max_match_distance)
        case_assets = assets_root / case
        case_assets.mkdir(exist_ok=True)
        cards: list[str] = []
        counts = {key: sum(1 for status, *_ in pairs if status == key) for key in ("matched", "only138", "only139")}

        for pair_index, (status, cell138, cell139, distance) in enumerate(pairs, start=1):
            bounds = square_bounds(cell138, cell139, raw.shape, args.minimum_crop_size)
            top, bottom, left, right = bounds
            selection = np.s_[top:bottom, left:right]
            gray = _normalize_u8(raw[selection])
            prefix = f"cell_{pair_index:04d}"
            files = {
                "raw": case_assets / f"{prefix}_raw.png",
                "138_prediction": case_assets / f"{prefix}_138_prediction.png",
                "138_hysteresis": case_assets / f"{prefix}_138_hysteresis.png",
                "138_cell": case_assets / f"{prefix}_138_cell.png",
                "139_prediction": case_assets / f"{prefix}_139_prediction.png",
                "139_hysteresis": case_assets / f"{prefix}_139_hysteresis.png",
                "139_cell": case_assets / f"{prefix}_139_cell.png",
            }
            save_png(files["raw"], gray)
            save_png(files["138_prediction"], overlay(gray, pred138[selection]))
            save_png(files["138_hysteresis"], hysteresis_overlay(gray, pred138[selection], hyst138[selection]))
            save_png(files["138_cell"], overlay(gray, projected_cell(cell138, bounds)))
            save_png(files["139_prediction"], overlay(gray, pred139[selection]))
            save_png(files["139_hysteresis"], hysteresis_overlay(gray, pred139[selection], hyst139[selection]))
            save_png(files["139_cell"], overlay(gray, projected_cell(cell139, bounds)))
            relative = {key: value.relative_to(args.output_dir).as_posix() for key, value in files.items()}
            distance_text = f" · Soma-Abstand {distance:.1f} px" if distance is not None else ""
            title138 = cell138.folder if cell138 else "keine exportierte Zelle"
            title139 = cell139.folder if cell139 else "keine exportierte Zelle"
            cards.append(
                f'<article class="card {status}" data-status="{status}"><h2>#{pair_index:04d} · {status}{distance_text}</h2>'
                f'<div class="network"><strong>Netz 138</strong><figure><img loading="lazy" src="{relative["raw"]}"><figcaption>Rohbild</figcaption></figure>'
                f'<figure><img loading="lazy" src="{relative["138_prediction"]}"><figcaption>Prediction</figcaption></figure>'
                f'<figure><img loading="lazy" src="{relative["138_hysteresis"]}"><figcaption>Hysterese</figcaption></figure>'
                f'<figure><img loading="lazy" src="{relative["138_cell"]}"><figcaption>Einzelzelle: {html.escape(title138)}</figcaption></figure></div>'
                f'<div class="network"><strong>Netz 139</strong><figure><img loading="lazy" src="{relative["raw"]}"><figcaption>Rohbild</figcaption></figure>'
                f'<figure><img loading="lazy" src="{relative["139_prediction"]}"><figcaption>Prediction</figcaption></figure>'
                f'<figure><img loading="lazy" src="{relative["139_hysteresis"]}"><figcaption>Hysterese</figcaption></figure>'
                f'<figure><img loading="lazy" src="{relative["139_cell"]}"><figcaption>Einzelzelle: {html.escape(title139)}</figcaption></figure></div></article>'
            )
            all_rows.append(
                {
                    "case": case,
                    "comparison_id": pair_index,
                    "status": status,
                    "net138_cell": cell138.folder if cell138 else "",
                    "net139_cell": cell139.folder if cell139 else "",
                    "soma_distance_px": round(distance, 3) if distance is not None else "",
                    "net138_skeleton_px": cell138.skeleton_pixels if cell138 else "",
                    "net139_skeleton_px": cell139.skeleton_pixels if cell139 else "",
                    "y_min": top,
                    "x_min": left,
                    "size": bottom - top,
                }
            )

        page_name = f"{case}.html"
        write_case_page(args.output_dir / page_name, case, cards, counts["matched"], counts["only138"], counts["only139"])
        index_rows.append(
            f'<tr><td><a href="{html.escape(page_name)}">{html.escape(case)}</a></td>'
            f'<td>{len(cells138)}</td><td>{len(cells139)}</td><td>{counts["matched"]}</td>'
            f'<td>{counts["only138"]}</td><td>{counts["only139"]}</td></tr>'
        )

    with (args.output_dir / "cell_matches.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    index = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Zellvergleich 138 vs. 139</title>
<style>body{{font:14px/1.5 Segoe UI,Arial,sans-serif;margin:28px;background:#101318;color:#edf1f5}}a{{color:#7fc8ff}}table{{border-collapse:collapse;background:#191e24}}th,td{{padding:9px 13px;border-bottom:1px solid #303842;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head>
<body><h1>Einzelzellvergleich: Netz 138 gegen Netz 139</h1><p>Adaptive Hysterese mit alpha=1/3; nur eindeutig isolierte Ein-Soma-Komponenten. Zuordnung anhand der Soma-Position, maximal {args.max_match_distance:g} px.</p>
<table><thead><tr><th>Übersichtsbild</th><th>Zellen 138</th><th>Zellen 139</th><th>Zugeordnet</th><th>Nur 138</th><th>Nur 139</th></tr></thead><tbody>{''.join(index_rows)}</tbody></table></body></html>"""
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"Geschrieben: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
