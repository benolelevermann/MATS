from __future__ import annotations

"""Build a per-image comparison of the Dataset137 and Dataset138 networks.

Both networks were run on the SAME flat inference inputs, so every case name maps
1:1 between them. For each case this script computes a small set of segmentation
statistics per network, the agreement between the two, and renders a crop showing
raw / 137 / 138 side by side. Output is a CSV plus a self-contained HTML gallery.

The headline metric is `skeleton_on_soma_pct`: the share of skeleton pixels that
share a connected component with a soma. Raw skeleton mass says little on its own
because detached debris counts toward it; the attached share separates plausible
neurites from noise, which is what matters when comparing objectives.
"""

import argparse
import base64
import csv
import html
import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

SCRIPT_VERSION = "objectivetest-overview-v1-2026-08-19"
CONN = np.ones((3, 3), dtype=bool)
SOMA_MIN_PX = 200

SKELETON_RGB = (0, 225, 255)
SOMA_RGB = (255, 60, 175)


@dataclass
class Stats:
    skeleton_px: int
    soma_px: int
    soma_instances: int
    skeleton_components: int
    largest_skeleton_component: int
    skeleton_on_soma_pct: float
    foreground_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True, help="Flat folder of <case>_0000.tif inputs.")
    parser.add_argument("--pred-137", type=Path, required=True, help="Folder of Dataset137 <case>.tif predictions.")
    parser.add_argument("--pred-138", type=Path, nargs="+", required=True,
                        help="One or more folders holding Dataset138 <case>.tif predictions.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=900)
    parser.add_argument("--thumb-width", type=int, default=430)
    parser.add_argument("--exclude", nargs="*", default=[], help="Case names to skip.")
    return parser.parse_args()


def read2d(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as handle:
        shape = handle.series[0].shape
        if len(shape) != 2:
            raise RuntimeError(f"Not a 2-D image {shape}: {path}")
    return np.squeeze(np.asarray(tifffile.imread(path)))


def compute_stats(pred: np.ndarray) -> Stats:
    skeleton = pred == 1
    soma = pred == 2
    n_skel = int(skeleton.sum())

    soma_labels, _ = ndi.label(soma, structure=CONN)
    soma_sizes = np.bincount(soma_labels.ravel())[1:] if soma_labels.max() else np.array([], dtype=int)

    skel_labels, skel_count = ndi.label(skeleton, structure=CONN)
    skel_sizes = np.bincount(skel_labels.ravel())[1:] if skel_count else np.array([0])

    # Skeleton sharing a component with any soma: the part a soma-rooted method could use.
    attached = 0
    if soma.any() and n_skel:
        both, _ = ndi.label(pred > 0, structure=CONN)
        soma_components = set(np.unique(both[soma]).tolist()) - {0}
        if soma_components:
            attached = int((np.isin(both, list(soma_components)) & skeleton).sum())

    return Stats(
        skeleton_px=n_skel,
        soma_px=int(soma.sum()),
        soma_instances=int((soma_sizes >= SOMA_MIN_PX).sum()),
        skeleton_components=int(skel_count),
        largest_skeleton_component=int(skel_sizes.max()) if skel_count else 0,
        skeleton_on_soma_pct=round(100.0 * attached / n_skel, 2) if n_skel else 0.0,
        foreground_pct=round(100.0 * float((pred > 0).mean()), 4),
    )


def iou(a: np.ndarray, b: np.ndarray, value: int) -> float:
    am, bm = a == value, b == value
    union = int((am | bm).sum())
    return round(float((am & bm).sum() / union), 4) if union else float("nan")


def densest_window(pred: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of the size x size window with the most soma pixels."""
    height, width = pred.shape
    size = min(size, height, width)
    small = (pred == 2).astype(np.float32)
    # Coarse search on a downsampled map keeps this cheap on 90 Mpx images.
    factor = max(1, size // 32)
    reduced = small[:: factor, :: factor]
    window = max(1, size // factor)
    integral = reduced.cumsum(0).cumsum(1)

    def box(y: int, x: int) -> float:
        y1, x1 = y + window - 1, x + window - 1
        total = integral[y1, x1]
        if y:
            total -= integral[y - 1, x1]
        if x:
            total -= integral[y1, x - 1]
        if y and x:
            total += integral[y - 1, x - 1]
        return float(total)

    best = None
    for y in range(0, reduced.shape[0] - window, max(1, window // 3)):
        for x in range(0, reduced.shape[1] - window, max(1, window // 3)):
            value = box(y, x)
            if best is None or value > best[0]:
                best = (value, y, x)
    if best is None:
        return 0, 0
    y0 = min(max(0, best[1] * factor), height - size)
    x0 = min(max(0, best[2] * factor), width - size)
    return y0, x0


def stretch(raw: np.ndarray) -> np.ndarray:
    values = raw.astype(np.float32)
    lo, hi = np.percentile(values, [1.0, 99.5])
    scaled = np.clip((values - lo) / max(1e-6, hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def overlay(gray: np.ndarray, pred: np.ndarray | None) -> np.ndarray:
    image = np.dstack([gray] * 3)
    if pred is not None:
        image[pred == 1] = SKELETON_RGB
        image[pred == 2] = SOMA_RGB
    return image


def to_data_uri(image: np.ndarray, width: int) -> str:
    pil = Image.fromarray(image)
    if pil.width != width:
        pil = pil.resize((width, max(1, round(pil.height * width / pil.width))), Image.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def find_prediction(folders: list[Path], case: str) -> Path | None:
    for folder in folders:
        direct = folder / f"{case}.tif"
        if direct.is_file():
            return direct
        for candidate in folder.rglob(f"{case}.tif"):
            return candidate
    return None


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name[: -len("_0000.tif")] for p in args.inputs.glob("*_0000.tif"))
    cases = [c for c in cases if c not in set(args.exclude)]
    if not cases:
        raise RuntimeError(f"No <case>_0000.tif inputs in {args.inputs}")

    rows: list[dict] = []
    cards: list[dict] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case}", flush=True)
        raw_path = args.inputs / f"{case}_0000.tif"
        p137 = find_prediction([args.pred_137], case)
        p138 = find_prediction(list(args.pred_138), case)

        row: dict = {"case": case}
        preds: dict[str, np.ndarray | None] = {}
        for tag, path in (("137", p137), ("138", p138)):
            if path is None:
                print(f"    WARNUNG: keine {tag}-Prediction gefunden")
                preds[tag] = None
                continue
            pred = read2d(path)
            preds[tag] = pred
            stats = compute_stats(pred)
            row.update({f"{tag}_{k}": v for k, v in asdict(stats).items()})

        a, b = preds.get("137"), preds.get("138")
        if a is not None and b is not None:
            if a.shape != b.shape:
                print(f"    WARNUNG: Formen unterschiedlich {a.shape} vs {b.shape} - Vergleich uebersprungen")
            else:
                row["iou_skeleton"] = iou(a, b, 1)
                row["iou_soma"] = iou(a, b, 2)
                row["agreement_pct"] = round(100.0 * float((a == b).mean()), 3)
                for key in ("skeleton_px", "soma_px", "soma_instances",
                            "largest_skeleton_component", "skeleton_on_soma_pct"):
                    if f"137_{key}" in row and f"138_{key}" in row:
                        row[f"delta_{key}"] = round(row[f"138_{key}"] - row[f"137_{key}"], 2)
        rows.append(row)

        # crop panels
        reference = b if b is not None else a
        raw = read2d(raw_path)
        if reference is not None and reference.shape == raw.shape:
            y0, x0 = densest_window(reference, args.crop_size)
        else:
            y0, x0 = 0, 0
        size = min(args.crop_size, raw.shape[0], raw.shape[1])
        gray = stretch(raw[y0:y0 + size, x0:x0 + size])
        panels = {"raw": to_data_uri(overlay(gray, None), args.thumb_width)}
        for tag, pred in (("137", a), ("138", b)):
            if pred is not None and pred.shape == raw.shape:
                panels[tag] = to_data_uri(overlay(gray, pred[y0:y0 + size, x0:x0 + size]), args.thumb_width)
        cards.append({"case": case, "crop": [int(y0), int(x0), int(size)], "panels": panels, "row": row})
        del raw, a, b, preds

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out / "objectivetest_overview.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (out / "objectivetest_overview.json").write_text(
        json.dumps({"script_version": SCRIPT_VERSION, "rows": rows}, indent=2), encoding="utf-8")

    write_html(out / "objectivetest_overview.html", rows, cards)
    print(f"\nGeschrieben: {out}")


def fmt(value, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "&ndash;"
    if isinstance(value, float) and digits:
        return f"{value:,.{digits}f}".replace(",", "&thinsp;")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", "&thinsp;")
    return html.escape(str(value))


def delta_cell(row: dict, key: str, digits: int = 0) -> str:
    value = row.get(f"delta_{key}")
    if value is None:
        return "<td class='num'>&ndash;</td>"
    cls = "up" if value > 0 else ("down" if value < 0 else "")
    sign = "+" if value > 0 else ""
    text = f"{sign}{value:,.{digits}f}".replace(",", "&thinsp;")
    return f"<td class='num {cls}'>{text}</td>"


def write_html(path: Path, rows: list[dict], cards: list[dict]) -> None:
    head = """<meta charset="utf-8"><title>Objective test: Dataset137 vs Dataset138</title>
<style>
:root{--bg:#ffffff;--fg:#14171a;--mut:#5b6570;--line:#e3e7eb;--card:#f7f9fb;--up:#0a7d3f;--down:#b3261e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#101315;--fg:#e8ecef;--mut:#9aa5b1;--line:#262c31;--card:#171b1f;--up:#5ddb95;--down:#ff8a80}}
:root[data-theme=dark]{--bg:#101315;--fg:#e8ecef;--mut:#9aa5b1;--line:#262c31;--card:#171b1f;--up:#5ddb95;--down:#ff8a80}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:28px}
h1{font-size:21px;margin:0 0 4px} h2{font-size:16px;margin:34px 0 10px}
p.sub{color:var(--mut);margin:0 0 20px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:900px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:var(--card);font-weight:600;position:sticky;top:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.up{color:var(--up)} td.down{color:var(--down)}
tr:hover td{background:var(--card)}
.card{border:1px solid var(--line);border-radius:8px;margin:16px 0;overflow:hidden}
.card h3{margin:0;padding:10px 14px;background:var(--card);font-size:14px;border-bottom:1px solid var(--line)}
.panels{display:flex;flex-wrap:wrap;gap:8px;padding:10px}
.panel{flex:1 1 300px;min-width:0}
.panel img{width:100%;height:auto;display:block;border-radius:4px;background:#000}
.panel span{display:block;color:var(--mut);font-size:12px;padding:4px 2px}
.legend{color:var(--mut);font-size:12px;margin:6px 0 0}
.key{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:baseline;margin-right:4px}
</style>"""

    body = ["<h1>Objektivtest: Dataset137 gegen Dataset138</h1>",
            "<p class='sub'>Beide Netze auf identischen Eingaben, Trainer "
            "<code>nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x</code>, fold 0. "
            "&bdquo;Skelett am Soma&ldquo; = Anteil der Skelettpixel, die mit einem Soma in derselben "
            "Zusammenhangskomponente liegen &mdash; die aussagekr&auml;ftigste Spalte, weil blo&szlig;e "
            "Skelettmasse auch Debris sein kann.</p>"]

    body.append("<h2>Kennzahlen</h2><div class='wrap'><table><thead><tr>"
                "<th>Bild</th>"
                "<th class='num'>Skelett 137</th><th class='num'>Skelett 138</th><th class='num'>&Delta;</th>"
                "<th class='num'>Soma-Inst. 137</th><th class='num'>Soma-Inst. 138</th><th class='num'>&Delta;</th>"
                "<th class='num'>gr&ouml;&szlig;te Komp. 137</th><th class='num'>gr&ouml;&szlig;te Komp. 138</th><th class='num'>&Delta;</th>"
                "<th class='num'>am Soma 137</th><th class='num'>am Soma 138</th><th class='num'>&Delta;</th>"
                "<th class='num'>IoU Skel.</th><th class='num'>IoU Soma</th>"
                "</tr></thead><tbody>")
    for row in rows:
        body.append("<tr><td>" + html.escape(row["case"]) + "</td>"
                    + f"<td class='num'>{fmt(row.get('137_skeleton_px'))}</td>"
                    + f"<td class='num'>{fmt(row.get('138_skeleton_px'))}</td>"
                    + delta_cell(row, "skeleton_px")
                    + f"<td class='num'>{fmt(row.get('137_soma_instances'))}</td>"
                    + f"<td class='num'>{fmt(row.get('138_soma_instances'))}</td>"
                    + delta_cell(row, "soma_instances")
                    + f"<td class='num'>{fmt(row.get('137_largest_skeleton_component'))}</td>"
                    + f"<td class='num'>{fmt(row.get('138_largest_skeleton_component'))}</td>"
                    + delta_cell(row, "largest_skeleton_component")
                    + f"<td class='num'>{fmt(row.get('137_skeleton_on_soma_pct'),1)}&thinsp;%</td>"
                    + f"<td class='num'>{fmt(row.get('138_skeleton_on_soma_pct'),1)}&thinsp;%</td>"
                    + delta_cell(row, "skeleton_on_soma_pct", 1)
                    + f"<td class='num'>{fmt(row.get('iou_skeleton'),3)}</td>"
                    + f"<td class='num'>{fmt(row.get('iou_soma'),3)}</td></tr>")
    body.append("</tbody></table></div>")

    body.append("<h2>Bildvergleich</h2>"
                "<p class='legend'><span class='key' style='background:rgb(0,225,255)'></span>Skelett"
                "&nbsp;&nbsp;<span class='key' style='background:rgb(255,60,175)'></span>Soma"
                "&nbsp;&nbsp;&mdash; jeweils der somareichste Ausschnitt des Bildes.</p>")
    for card in cards:
        row = card["row"]
        y0, x0, size = card["crop"]
        body.append(f"<div class='card'><h3>{html.escape(card['case'])} "
                    f"<span style='color:var(--mut);font-weight:400'>&mdash; Ausschnitt {size}&times;{size} "
                    f"bei y{y0} x{x0}</span></h3><div class='panels'>")
        for tag, label in (("raw", "Rohbild"), ("137", "Dataset137"), ("138", "Dataset138")):
            if tag in card["panels"]:
                body.append(f"<div class='panel'><img src='{card['panels'][tag]}' alt='{label}'>"
                            f"<span>{label}</span></div>")
        body.append("</div></div>")

    path.write_text(head + "\n" + "\n".join(body), encoding="utf-8")


if __name__ == "__main__":
    main()
