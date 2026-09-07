from __future__ import annotations

"""Create a compact visual and numeric comparison of Dataset138 and Dataset139."""

import argparse
import base64
import csv
import html
import io
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from make_objectivetest_overview import compute_stats, densest_window, iou, read2d, stretch


SKELETON_RGB = np.asarray((0, 225, 255), dtype=np.uint8)
SOMA_RGB = np.asarray((255, 60, 175), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--pred-138", type=Path, required=True)
    parser.add_argument("--pred-139", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=1000)
    return parser.parse_args()


def find_prediction(folder: Path, case: str) -> Path:
    candidates = (folder / f"{case}.tif", folder / f"{case}_138.tif", folder / f"{case}_139.tif")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Keine Prediction fuer {case} in {folder}")


def render_overlay(raw: np.ndarray, pred: np.ndarray | None, *, overview: bool) -> np.ndarray:
    gray = stretch(raw)
    rgb = np.dstack((gray, gray, gray))
    if pred is None:
        return rgb

    skeleton = pred == 1
    soma = pred == 2
    if overview:
        # The labels really remain 1 px wide. Dilation is display-only so that
        # skeletons survive the strong downsampling of full overview images.
        iterations = max(1, int(np.ceil(max(raw.shape) / 1800)))
        skeleton = ndi.binary_dilation(skeleton, iterations=iterations)
    rgb[skeleton] = SKELETON_RGB
    rgb[soma] = SOMA_RGB
    return rgb


def data_uri(array: np.ndarray, width: int, quality: int = 86) -> str:
    image = Image.fromarray(array)
    if image.width > width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def fmt(value: object, digits: int = 0) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return f"{value}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(path.name.removesuffix("_0000.tif") for path in args.inputs.glob("*_0000.tif"))
    if not cases:
        raise RuntimeError(f"Keine *_0000.tif-Bilder in {args.inputs}")

    rows: list[dict[str, object]] = []
    cards: list[str] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case}", flush=True)
        raw = read2d(args.inputs / f"{case}_0000.tif")
        pred138 = read2d(find_prediction(args.pred_138, case))
        pred139 = read2d(find_prediction(args.pred_139, case))
        if raw.shape != pred138.shape or raw.shape != pred139.shape:
            raise RuntimeError(f"Unterschiedliche Formen bei {case}: {raw.shape}, {pred138.shape}, {pred139.shape}")
        for name, pred in (("138", pred138), ("139", pred139)):
            values = set(np.unique(pred).tolist())
            if not values <= {0, 1, 2}:
                raise RuntimeError(f"Ungueltige Labels in Netz {name}, {case}: {sorted(values)}")

        stats138 = asdict(compute_stats(pred138))
        stats139 = asdict(compute_stats(pred139))
        row: dict[str, object] = {"case": case}
        row.update({f"net138_{key}": value for key, value in stats138.items()})
        row.update({f"net139_{key}": value for key, value in stats139.items()})
        for key in stats138:
            row[f"delta_{key}"] = round(float(stats139[key]) - float(stats138[key]), 4)
        row["iou_skeleton"] = iou(pred138, pred139, 1)
        row["iou_soma"] = iou(pred138, pred139, 2)
        row["agreement_pct"] = round(100.0 * float((pred138 == pred139).mean()), 3)
        rows.append(row)

        reference = np.where(pred139 == 2, 2, pred138)
        y0, x0 = densest_window(reference, args.crop_size)
        size = min(args.crop_size, *raw.shape)
        slices = np.s_[y0 : y0 + size, x0 : x0 + size]
        overview_panels = (
            ("Rohbild", data_uri(render_overlay(raw, None, overview=True), 520)),
            ("Netz 138", data_uri(render_overlay(raw, pred138, overview=True), 520)),
            ("Netz 139 (final)", data_uri(render_overlay(raw, pred139, overview=True), 520)),
        )
        crop_panels = (
            ("Rohbild", data_uri(render_overlay(raw[slices], None, overview=False), 520)),
            ("Netz 138", data_uri(render_overlay(raw[slices], pred138[slices], overview=False), 520)),
            ("Netz 139 (final)", data_uri(render_overlay(raw[slices], pred139[slices], overview=False), 520)),
        )

        def panels(items: tuple[tuple[str, str], ...]) -> str:
            return "".join(
                f'<figure><img src="{uri}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'
                for label, uri in items
            )

        cards.append(
            f'<section><h2>{html.escape(case)}</h2>'
            f'<p>Gesamtes Bild – Skelett nur fuer die Anzeige verbreitert</p><div class="panels">{panels(overview_panels)}</div>'
            f'<p>Unveraenderter 1-px-Ausschnitt: {size} x {size} px bei y={y0}, x={x0}</p>'
            f'<div class="panels">{panels(crop_panels)}</div></section>'
        )

    fields = list(rows[0])
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['case']))}</td>"
            f"<td>{fmt(row['net138_skeleton_px'])}</td><td>{fmt(row['net139_skeleton_px'])}</td>"
            f"<td>{fmt(row['delta_skeleton_px'])}</td>"
            f"<td>{fmt(row['net138_skeleton_components'])}</td><td>{fmt(row['net139_skeleton_components'])}</td>"
            f"<td>{fmt(row['net138_skeleton_on_soma_pct'], 1)} %</td>"
            f"<td>{fmt(row['net139_skeleton_on_soma_pct'], 1)} %</td>"
            f"<td>{fmt(row['net138_soma_instances'])}</td><td>{fmt(row['net139_soma_instances'])}</td>"
            f"<td>{fmt(row['iou_skeleton'], 3)}</td><td>{fmt(row['iou_soma'], 3)}</td>"
            "</tr>"
        )

    document = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Netz 138 vs. Netz 139</title>
<style>
body{{font:14px/1.45 Segoe UI,Arial,sans-serif;margin:24px;background:#111418;color:#edf1f5}}
h1{{margin-bottom:4px}} h2{{margin-top:0}} p{{color:#b7c0ca}} section{{margin:24px 0;padding:16px;background:#191e24;border-radius:10px}}
.panels{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:10px}} figure{{margin:0}} img{{display:block;width:100%;background:#000;border-radius:6px}} figcaption{{padding:5px 2px;color:#cbd3db}}
.table{{overflow:auto}} table{{border-collapse:collapse;width:100%;background:#191e24}} th,td{{padding:7px 9px;border-bottom:1px solid #303842;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{color:#c9d2dc}}
.legend{{color:#b7c0ca}} .cyan{{color:rgb(0,225,255)}} .magenta{{color:rgb(255,80,185)}}
@media(max-width:950px){{.panels{{grid-template-columns:1fr}}}}
</style></head><body><h1>Netz 138 gegen Netz 139</h1>
<p class="legend"><span class="cyan">Cyan: Skelett</span> · <span class="magenta">Magenta: Soma</span>. Identische Rohbilder, Fold 0; Netz 139 verwendet checkpoint_final.pth.</p>
<div class="table"><table><thead><tr><th>Bild</th><th>Skel. 138</th><th>Skel. 139</th><th>Delta</th><th>Komponenten 138</th><th>Komponenten 139</th><th>am Soma 138</th><th>am Soma 139</th><th>Somata 138</th><th>Somata 139</th><th>IoU Skel.</th><th>IoU Soma</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
{''.join(cards)}</body></html>"""
    html_path = args.output_dir / "comparison_138_vs_139.html"
    html_path.write_text(document, encoding="utf-8")
    print(f"Geschrieben: {html_path}")


if __name__ == "__main__":
    main()
