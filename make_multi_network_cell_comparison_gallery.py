from __future__ import annotations

"""Match extracted cells by soma position and render an N-network gallery."""

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
    status: str
    training_safe_without_ignore: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--network", nargs=2, action="append", required=True, metavar=("ID", "PREDICTION_DIR")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-match-distance", type=float, default=40.0)
    parser.add_argument("--minimum-crop-size", type=int, default=192)
    return parser.parse_args()


def read2d(path: Path) -> np.ndarray:
    return np.squeeze(np.asarray(tifffile.imread(path)))


def find_prediction(folder: Path, case: str, network: str) -> Path:
    for name in (f"{case}.tif", f"{case}_{network}.tif"):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Keine Prediction fuer {case} (Netz {network}) in {folder}")


def load_cells(root: Path, network: str, case: str) -> list[Cell]:
    result: list[Cell] = []
    case_root = root / f"cells_net{network}" / case
    for folder in sorted(case_root.glob("cell*")):
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        bounds = metadata["bounds"]
        location = metadata["location"]
        qc = metadata.get("qc", {})
        training_safe = qc.get("training_safe_without_ignore")
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
                status=str(metadata.get("status", "")),
                training_safe_without_ignore=(
                    bool(training_safe) if training_safe is not None else None
                ),
            )
        )
    return result


def group_center(group: dict[str, Cell]) -> tuple[float, float]:
    return (
        float(np.mean([cell.center_y for cell in group.values()])),
        float(np.mean([cell.center_x for cell in group.values()])),
    )


def match_cells(
    cells_by_network: dict[str, list[Cell]], networks: list[str], maximum_distance: float
) -> list[dict[str, Cell]]:
    groups: list[dict[str, Cell]] = [{networks[0]: cell} for cell in cells_by_network[networks[0]]]
    for network in networks[1:]:
        cells = cells_by_network[network]
        if not groups:
            groups.extend({network: cell} for cell in cells)
            continue
        if not cells:
            continue
        a = np.asarray([group_center(group) for group in groups], dtype=np.float64)
        b = np.asarray([(cell.center_y, cell.center_x) for cell in cells], dtype=np.float64)
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        used: set[int] = set()
        for row, column in zip(rows.tolist(), columns.tolist()):
            if float(distances[row, column]) <= maximum_distance:
                groups[row][network] = cells[column]
                used.add(column)
        groups.extend({network: cell} for index, cell in enumerate(cells) if index not in used)
    return sorted(
        groups,
        key=lambda group: (
            0 if len(group) == len(networks) else 1,
            group_center(group)[0],
            group_center(group)[1],
        ),
    )


def square_bounds(group: dict[str, Cell], shape: tuple[int, int], minimum: int) -> tuple[int, int, int, int]:
    cells = list(group.values())
    x0, y0 = min(cell.x0 for cell in cells), min(cell.y0 for cell in cells)
    x1, y1 = max(cell.x1 for cell in cells), max(cell.y1 for cell in cells)
    side = min(max(minimum, x1 - x0, y1 - y0), shape[0], shape[1])
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
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
    rgb[(postprocessed == 1) & (prediction != 1)] = ADDED
    rgb[(prediction == 1) & (postprocessed != 1)] = REMOVED
    return rgb


def projected_cell(cell: Cell | None, bounds: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = bounds
    canvas = np.zeros((bottom - top, right - left), dtype=np.uint8)
    if cell is None:
        return canvas
    semantic = read2d(cell.root / "seg.tif").astype(np.uint8)
    gy0, gy1 = max(top, cell.y0), min(bottom, cell.y1)
    gx0, gx1 = max(left, cell.x0), min(right, cell.x1)
    if gy1 > gy0 and gx1 > gx0:
        canvas[gy0 - top : gy1 - top, gx0 - left : gx1 - left] = semantic[
            gy0 - cell.y0 : gy1 - cell.y0, gx0 - cell.x0 : gx1 - cell.x0
        ]
    return canvas


def save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, optimize=True)


def main() -> None:
    args = parse_args()
    networks = [str(item[0]) for item in args.network]
    prediction_roots = {str(item[0]): Path(item[1]) for item in args.network}
    if len(networks) != len(set(networks)):
        raise RuntimeError(f"Netz-IDs duerfen nicht doppelt vorkommen: {networks}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_root = args.output_dir / "assets"
    assets_root.mkdir(exist_ok=True)
    cases = sorted(path.name.removesuffix("_0000.tif") for path in args.inputs.glob("*_0000.tif"))
    all_rows: list[dict[str, object]] = []
    index_rows: list[str] = []

    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case}", flush=True)
        raw = read2d(args.inputs / f"{case}_0000.tif")
        predictions = {
            network: read2d(find_prediction(prediction_roots[network], case, network)).astype(np.uint8)
            for network in networks
        }
        hysteresis_maps = {
            network: read2d(
                args.comparison_root / f"hysteresis_net{network}" / f"{case}_adaptive_hysteresis_0-1-2.tif"
            ).astype(np.uint8)
            for network in networks
        }
        cells = {network: load_cells(args.comparison_root, network, case) for network in networks}
        groups = match_cells(cells, networks, args.max_match_distance)
        case_assets = assets_root / case
        case_assets.mkdir(exist_ok=True)
        cards: list[str] = []
        common = sum(len(group) == len(networks) for group in groups)

        for group_index, group in enumerate(groups, start=1):
            bounds = square_bounds(group, raw.shape, args.minimum_crop_size)
            top, bottom, left, right = bounds
            selection = np.s_[top:bottom, left:right]
            gray = _normalize_u8(raw[selection])
            prefix = f"cell_{group_index:04d}"
            raw_path = case_assets / f"{prefix}_raw.png"
            save_png(raw_path, gray)
            raw_relative = raw_path.relative_to(args.output_dir).as_posix()
            rows: list[str] = []
            csv_row: dict[str, object] = {
                "case": case,
                "comparison_id": group_index,
                "present_networks": ",".join(network for network in networks if network in group),
                "missing_networks": ",".join(network for network in networks if network not in group),
                "y_min": top,
                "x_min": left,
                "size": bottom - top,
            }
            center_points = np.asarray([(cell.center_y, cell.center_x) for cell in group.values()])
            spread = float(np.max(np.linalg.norm(center_points[:, None] - center_points[None, :], axis=2)))
            csv_row["max_soma_distance_px"] = round(spread, 3)
            for network in networks:
                cell = group.get(network)
                pred_path = case_assets / f"{prefix}_net{network}_prediction.png"
                hyst_path = case_assets / f"{prefix}_net{network}_hysteresis.png"
                cell_path = case_assets / f"{prefix}_net{network}_cell.png"
                save_png(pred_path, overlay(gray, predictions[network][selection]))
                save_png(hyst_path, hysteresis_overlay(gray, predictions[network][selection], hysteresis_maps[network][selection]))
                save_png(cell_path, overlay(gray, projected_cell(cell, bounds)))
                if cell is None:
                    title = "keine extrahierte Zelle"
                else:
                    status_names = {
                        "accepted_high_confidence": "hohe Graph-Sicherheit",
                        "review_medium_confidence": "manuell prüfen",
                    }
                    suffix = status_names.get(cell.status, "")
                    title = cell.folder + (f" · {suffix}" if suffix else "")
                rows.append(
                    f'<div class="network"><strong>Netz {network}</strong>'
                    f'<figure><img loading="lazy" src="{raw_relative}"><figcaption>Rohbild</figcaption></figure>'
                    f'<figure><img loading="lazy" src="{pred_path.relative_to(args.output_dir).as_posix()}"><figcaption>Prediction</figcaption></figure>'
                    f'<figure><img loading="lazy" src="{hyst_path.relative_to(args.output_dir).as_posix()}"><figcaption>Hysterese</figcaption></figure>'
                    f'<figure><img loading="lazy" src="{cell_path.relative_to(args.output_dir).as_posix()}"><figcaption>{html.escape(title)}</figcaption></figure></div>'
                )
                csv_row[f"net{network}_cell"] = cell.folder if cell else ""
                csv_row[f"net{network}_skeleton_px"] = cell.skeleton_pixels if cell else ""
                csv_row[f"net{network}_soma_px"] = cell.soma_pixels if cell else ""
                csv_row[f"net{network}_status"] = cell.status if cell else ""
                csv_row[f"net{network}_training_safe_without_ignore"] = (
                    cell.training_safe_without_ignore if cell else ""
                )
            status = "common" if len(group) == len(networks) else "different"
            missing = [network for network in networks if network not in group]
            status_text = "von allen Netzen extrahiert" if not missing else f"fehlt bei Netz {', '.join(missing)}"
            cards.append(
                f'<article id="comparison-{group_index:04d}" class="card {status}" data-status="{status}"><h2>#{group_index:04d} · {status_text} · Soma-Abstand max. {spread:.1f} px</h2>{"".join(rows)}</article>'
            )
            all_rows.append(csv_row)

        case_page = args.output_dir / f"{case}.html"
        case_page.write_text(
            f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(case)}</title>
<style>body{{font:14px/1.45 Segoe UI,Arial,sans-serif;margin:22px;background:#101318;color:#edf1f5}}a{{color:#7fc8ff}}button{{margin:3px;padding:7px 11px;border:1px solid #46515e;border-radius:6px;background:#20262d;color:#edf1f5;cursor:pointer}}.card{{background:#191e24;border:1px solid #303842;border-radius:9px;margin:16px 0;padding:12px}}.card h2{{font-size:15px;margin:0 0 8px}}.network{{display:grid;grid-template-columns:75px repeat(4,minmax(175px,1fr));gap:8px;align-items:start;margin-top:8px}}.network>strong{{padding-top:8px}}figure{{margin:0}}img{{width:100%;display:block;background:#000;border-radius:5px}}figcaption{{color:#b9c2cc;padding-top:3px;font-size:12px}}.common{{border-left:4px solid #48c78e}}.different{{border-left:4px solid #ffb84d}}.muted{{color:#aab4be}}@media(max-width:900px){{.network{{grid-template-columns:1fr 1fr}}}}</style></head>
<body><p><a href="index.html">← Übersicht</a></p><h1>{html.escape(case)}</h1><p>{len(groups)} Zellpositionen · {common} von allen Netzen · {len(groups)-common} mit unterschiedlicher Extraktion</p><p><button onclick="show('all')">Alle</button><button onclick="show('common')">Alle Netze</button><button onclick="show('different')">Unterschiede</button></p><p class="muted">Cyan: Skelett · Magenta: Soma · Gelb: durch Hysterese ergänzt · Rot: entfernt. Eine fehlende Einzelzelle bedeutet, dass die Hysterese-Struktur nicht als isolierte Ein-Soma-Zelle exportierbar war.</p>{''.join(cards)}<script>function show(s){{document.querySelectorAll('.card').forEach(c=>c.style.display=(s==='all'||c.dataset.status===s)?'block':'none')}}</script></body></html>""",
            encoding="utf-8",
        )
        count_cells = "".join(f"<td>{len(cells[network])}</td>" for network in networks)
        index_rows.append(
            f'<tr><td><a href="{case_page.name}">{html.escape(case)}</a></td>{count_cells}<td>{common}</td><td>{len(groups)-common}</td></tr>'
        )

    fields: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (args.output_dir / "cell_matches.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    network_headers = "".join(f"<th>Zellen Netz {network}</th>" for network in networks)
    (args.output_dir / "index.html").write_text(
        f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Zellvergleich {' / '.join(networks)}</title><style>body{{font:14px/1.5 Segoe UI,Arial,sans-serif;margin:28px;background:#101318;color:#edf1f5}}a{{color:#7fc8ff}}table{{border-collapse:collapse;background:#191e24}}th,td{{padding:9px 13px;border-bottom:1px solid #303842;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head><body><h1>Einzelzellvergleich: Netze {' · '.join(networks)}</h1><p>Adaptive Hysterese mit alpha=1/3; exportiert werden nur eindeutig isolierte Ein-Soma-Komponenten. Zellen werden anhand der Soma-Position bis maximal {args.max_match_distance:g} px zugeordnet.</p><table><thead><tr><th>Übersichtsbild</th>{network_headers}<th>alle Netze</th><th>unterschiedlich</th></tr></thead><tbody>{''.join(index_rows)}</tbody></table></body></html>""",
        encoding="utf-8",
    )
    print(f"Geschrieben: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
