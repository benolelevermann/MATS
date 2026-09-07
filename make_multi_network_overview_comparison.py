from __future__ import annotations

"""Create a visual/numeric prediction and hysteresis comparison for N networks."""

import argparse
import csv
import html
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from make_objectivetest_overview import compute_stats, densest_window, read2d, stretch


SKELETON_RGB = np.asarray((0, 225, 255), dtype=np.uint8)
SOMA_RGB = np.asarray((245, 70, 170), dtype=np.uint8)
ADDED_RGB = np.asarray((255, 190, 54), dtype=np.uint8)
REMOVED_RGB = np.asarray((235, 70, 65), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--network",
        nargs=3,
        action="append",
        required=True,
        metavar=("ID", "PREDICTION_DIR", "HYSTERESIS_DIR"),
    )
    parser.add_argument("--cell-summary", type=Path)
    parser.add_argument("--cell-gallery", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=1000)
    parser.add_argument("--overview-width", type=int, default=720)
    return parser.parse_args()


def find_prediction(folder: Path, case: str, network: str) -> Path:
    for name in (f"{case}.tif", f"{case}_{network}.tif"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Keine Prediction fuer {case} (Netz {network}) in {folder}")


def render_overlay(raw: np.ndarray, semantic: np.ndarray | None, *, overview: bool) -> np.ndarray:
    gray = stretch(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if semantic is None:
        return rgb
    skeleton = semantic == 1
    soma = semantic == 2
    if overview:
        skeleton = ndi.binary_dilation(skeleton, iterations=max(1, int(np.ceil(max(raw.shape) / 1800))))
    rgb[skeleton] = SKELETON_RGB
    rgb[soma] = SOMA_RGB
    return rgb


def render_hysteresis(
    raw: np.ndarray, prediction: np.ndarray, hysteresis: np.ndarray, *, overview: bool
) -> np.ndarray:
    rgb = render_overlay(raw, hysteresis, overview=overview)
    added = (hysteresis == 1) & (prediction != 1)
    removed = (prediction == 1) & (hysteresis != 1)
    if overview:
        iterations = max(1, int(np.ceil(max(raw.shape) / 1800)))
        added = ndi.binary_dilation(added, iterations=iterations)
        removed = ndi.binary_dilation(removed, iterations=iterations)
    rgb[added] = ADDED_RGB
    rgb[removed] = REMOVED_RGB
    return rgb


def save_jpeg(path: Path, array: np.ndarray, width: int) -> None:
    image = Image.fromarray(array)
    if image.width > width:
        image = image.resize(
            (width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS
        )
    image.save(path, format="JPEG", quality=88, optimize=True)


def load_cell_counts(path: Path | None) -> dict[tuple[str, str], dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["network"]), str(row["case"])): row for row in rows}


def main() -> None:
    args = parse_args()
    networks = [(str(network), Path(pred), Path(hyst)) for network, pred, hyst in args.network]
    ids = [network for network, _, _ in networks]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Netz-IDs duerfen nicht doppelt vorkommen: {ids}")
    cases = sorted(path.name.removesuffix("_0000.tif") for path in args.inputs.glob("*_0000.tif"))
    if not cases:
        raise RuntimeError(f"Keine *_0000.tif-Bilder in {args.inputs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    cell_counts = load_cell_counts(args.cell_summary)
    metric_rows: list[dict[str, object]] = []
    case_sections: list[str] = []

    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case}", flush=True)
        raw = read2d(args.inputs / f"{case}_0000.tif")
        predictions: dict[str, np.ndarray] = {}
        hysteresis_maps: dict[str, np.ndarray] = {}
        for network, pred_root, hyst_root in networks:
            pred = read2d(find_prediction(pred_root, case, network)).astype(np.uint8)
            hyst = read2d(hyst_root / f"{case}_adaptive_hysteresis_0-1-2.tif").astype(np.uint8)
            for label, value in (("Prediction", pred), ("Hysterese", hyst)):
                if value.shape != raw.shape:
                    raise RuntimeError(
                        f"Formfehler {case}, Netz {network}, {label}: {value.shape} statt {raw.shape}"
                    )
                values = set(np.unique(value).astype(int).tolist())
                if not values <= {0, 1, 2}:
                    raise RuntimeError(f"Ungueltige Labels {case}, Netz {network}, {label}: {values}")
            predictions[network] = pred
            hysteresis_maps[network] = hyst

            extraction = cell_counts.get((network, case), {})
            for stage, semantic in (("prediction", pred), ("hysteresis", hyst)):
                row: dict[str, object] = {"case": case, "network": network, "stage": stage}
                row.update(asdict(compute_stats(semantic)))
                row["extracted_cells"] = extraction.get("exported_cell_count", "")
                row["conflict_groups"] = extraction.get("conflict_groups", "")
                row["discarded_without_soma"] = extraction.get("discarded_components_without_soma", "")
                metric_rows.append(row)

        case_assets = assets / case
        case_assets.mkdir(exist_ok=True)
        reference = hysteresis_maps[ids[-1]]
        size = min(args.crop_size, *raw.shape)
        y0, x0 = densest_window(reference, size)
        selection = np.s_[y0 : y0 + size, x0 : x0 + size]

        panels: list[str] = []
        raw_overview = case_assets / "overview_raw.jpg"
        raw_crop = case_assets / "crop_raw.jpg"
        save_jpeg(raw_overview, render_overlay(raw, None, overview=True), args.overview_width)
        save_jpeg(raw_crop, render_overlay(raw[selection], None, overview=False), args.overview_width)
        panels.append(
            f'<figure><img src="{raw_overview.relative_to(args.output_dir).as_posix()}"><figcaption>Rohbild</figcaption></figure>'
        )
        crop_panels = [
            f'<figure><img src="{raw_crop.relative_to(args.output_dir).as_posix()}"><figcaption>Rohbild</figcaption></figure>'
        ]
        for network in ids:
            pred_overview = case_assets / f"overview_net{network}_prediction.jpg"
            hyst_overview = case_assets / f"overview_net{network}_hysteresis.jpg"
            pred_crop = case_assets / f"crop_net{network}_prediction.jpg"
            hyst_crop = case_assets / f"crop_net{network}_hysteresis.jpg"
            save_jpeg(pred_overview, render_overlay(raw, predictions[network], overview=True), args.overview_width)
            save_jpeg(
                hyst_overview,
                render_hysteresis(raw, predictions[network], hysteresis_maps[network], overview=True),
                args.overview_width,
            )
            save_jpeg(
                pred_crop,
                render_overlay(raw[selection], predictions[network][selection], overview=False),
                args.overview_width,
            )
            save_jpeg(
                hyst_crop,
                render_hysteresis(
                    raw[selection], predictions[network][selection], hysteresis_maps[network][selection], overview=False
                ),
                args.overview_width,
            )
            panels.extend(
                (
                    f'<figure><img src="{pred_overview.relative_to(args.output_dir).as_posix()}"><figcaption>Netz {network}: Prediction</figcaption></figure>',
                    f'<figure><img src="{hyst_overview.relative_to(args.output_dir).as_posix()}"><figcaption>Netz {network}: Hysterese</figcaption></figure>',
                )
            )
            crop_panels.extend(
                (
                    f'<figure><img src="{pred_crop.relative_to(args.output_dir).as_posix()}"><figcaption>Netz {network}: Prediction</figcaption></figure>',
                    f'<figure><img src="{hyst_crop.relative_to(args.output_dir).as_posix()}"><figcaption>Netz {network}: Hysterese</figcaption></figure>',
                )
            )

        case_sections.append(
            f'<section><h2>{html.escape(case)}</h2><p>Gesamtbild (Skelett nur in der Anzeige verbreitert)</p>'
            f'<div class="panels">{"".join(panels)}</div><p>Unveraenderter 1-px-Ausschnitt: '
            f'{size} × {size} px bei y={y0}, x={x0}</p><div class="panels">{"".join(crop_panels)}</div></section>'
        )

    fields = list(metric_rows[0])
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    table_rows: list[str] = []
    for row in metric_rows:
        table_rows.append(
            f'<tr><td>{html.escape(str(row["case"]))}</td><td>{row["network"]}</td><td>{row["stage"]}</td>'
            f'<td>{int(row["skeleton_px"]):,}</td><td>{int(row["skeleton_components"]):,}</td>'
            f'<td>{float(row["skeleton_on_soma_pct"]):.1f}%</td><td>{int(row["soma_instances"]):,}</td>'
            f'<td>{row["extracted_cells"]}</td><td>{row["conflict_groups"]}</td></tr>'
        )
    cell_link = ""
    if args.cell_gallery is not None:
        relative = Path(args.cell_gallery).resolve().relative_to(args.output_dir.resolve().parent).as_posix()
        cell_link = f'<p><a class="button" href="../{html.escape(relative)}">Einzelzellvergleich öffnen →</a></p>'

    document = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Netze {' / '.join(ids)}</title>
<style>
body{{font:14px/1.45 Segoe UI,Arial,sans-serif;margin:24px;background:#101318;color:#edf1f5}}
h1{{margin-bottom:4px}} h2{{margin-top:0}} p{{color:#b7c0ca}} section{{margin:24px 0;padding:16px;background:#191e24;border-radius:10px}}
.panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}} figure{{margin:0}} img{{display:block;width:100%;background:#000;border-radius:6px}} figcaption{{padding:5px 2px;color:#cbd3db}}
.table{{overflow:auto}} table{{border-collapse:collapse;width:100%;background:#191e24}} th,td{{padding:7px 9px;border-bottom:1px solid #303842;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{color:#c9d2dc}}
.cyan{{color:rgb(0,225,255)}} .magenta{{color:rgb(245,70,170)}} .yellow{{color:rgb(255,190,54)}} .red{{color:rgb(235,70,65)}} .button{{display:inline-block;padding:9px 13px;background:#1769aa;color:white;border-radius:6px;text-decoration:none}}
</style></head><body><h1>Netzvergleich {' · '.join(ids)}</h1>
<p><span class="cyan">Cyan: Skelett</span> · <span class="magenta">Magenta: Soma</span> · <span class="yellow">Gelb: durch Hysterese ergänzt</span> · <span class="red">Rot: entfernt</span>. Alle Netze sehen dieselben unveränderten Bilder.</p>{cell_link}
<div class="table"><table><thead><tr><th>Bild</th><th>Netz</th><th>Stufe</th><th>Skelettpixel</th><th>Komponenten</th><th>am Soma</th><th>Somata ≥200 px</th><th>extrahierte Zellen</th><th>Konfliktgruppen</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
{''.join(case_sections)}</body></html>"""
    output = args.output_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    print(f"Geschrieben: {output}")


if __name__ == "__main__":
    main()
