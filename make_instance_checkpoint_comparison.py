from __future__ import annotations

"""Build a static side-by-side review of two separator checkpoints."""

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--label-a", default="Epoche 64")
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-b", default="Epoche 80")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(run: Path) -> dict[str, object]:
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def status_text(status: str) -> str:
    return {
        "accepted_separation": "sicher freigegeben",
        "rejected_uncertain": "unsicher – nicht exportiert",
        "unresolved_too_large": "nicht verarbeitet – zu groß",
    }.get(status, status)


def save_difference(
    path: Path,
    raw_image_path: Path,
    changed: np.ndarray,
) -> None:
    raw = np.asarray(Image.open(raw_image_path).convert("L"), dtype=np.uint8)
    rgb = np.repeat(raw[..., None], 3, axis=2).astype(np.float32)
    rgb[changed] = 0.18 * rgb[changed] + 0.82 * np.asarray(
        (255, 60, 90), dtype=np.float32
    )
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    assets = output_dir / "assets"
    assets.mkdir()
    summary_a = load_summary(run_a)
    summary_b = load_summary(run_b)
    conflicts_a = {int(row["component_id"]): row for row in summary_a["conflicts"]}
    conflicts_b = {int(row["component_id"]): row for row in summary_b["conflicts"]}
    if set(conflicts_a) != set(conflicts_b):
        raise ValueError("The runs do not contain the same conflict components")

    primary_a = np.asarray(tifffile.imread(run_a / "instance_primary.tif"))
    secondary_a = np.asarray(tifffile.imread(run_a / "instance_secondary.tif"))
    primary_b = np.asarray(tifffile.imread(run_b / "instance_primary.tif"))
    secondary_b = np.asarray(tifffile.imread(run_b / "instance_secondary.tif"))
    if primary_a.shape != primary_b.shape:
        raise ValueError("Instance maps have different shapes")

    cards: list[str] = []
    rows: list[dict[str, object]] = []
    for component_id in sorted(conflicts_a):
        row_a = conflicts_a[component_id]
        row_b = conflicts_b[component_id]
        y0, y1, x0, x1 = (int(value) for value in row_a["crop_yxyx"])
        changed = (
            (primary_a[y0:y1, x0:x1] != primary_b[y0:y1, x0:x1])
            | (secondary_a[y0:y1, x0:x1] != secondary_b[y0:y1, x0:x1])
        )
        changed_pixels = int(np.count_nonzero(changed))
        status_a = str(row_a["status"])
        status_b = str(row_b["status"])
        decision_changed = status_a != status_b
        base = f"conflict_{component_id:03d}"
        raw_name = f"{base}_raw.png"
        semantic_name = f"{base}_semantic.png"
        a_name = f"{base}_epoch64.png"
        b_name = f"{base}_epoch80.png"
        difference_name = f"{base}_difference.png"
        shutil.copy2(run_a / "assets" / f"{base}_raw.png", assets / raw_name)
        shutil.copy2(
            run_a / "assets" / f"{base}_semantic.png", assets / semantic_name
        )
        shutil.copy2(
            run_a / "assets" / f"{base}_instances.png", assets / a_name
        )
        shutil.copy2(
            run_b / "assets" / f"{base}_instances.png", assets / b_name
        )
        save_difference(assets / difference_name, assets / raw_name, changed)
        confidence_a = float(row_a.get("mean_confidence", 0.0))
        confidence_b = float(row_b.get("mean_confidence", 0.0))
        change_class = "decision-change" if decision_changed else "same-decision"
        accepted_a = status_a == "accepted_separation"
        accepted_b = status_b == "accepted_separation"
        cards.append(
            f"""
            <section class="card {change_class}" id="conflict-{component_id}"
                     data-changed="{str(decision_changed).lower()}"
                     data-a="{str(accepted_a).lower()}" data-b="{str(accepted_b).lower()}">
              <h2>Konflikt {component_id} · {len(row_a['soma_ids'])} Zellen</h2>
              <div class="decisions">
                <span class="pill {status_a}">{html.escape(args.label_a)}: {html.escape(status_text(status_a))}</span>
                <span class="pill {status_b}">{html.escape(args.label_b)}: {html.escape(status_text(status_b))}</span>
                <span>{changed_pixels:,} unterschiedlich zugeordnete Pixel</span>
              </div>
              <div class="panels">
                <figure><img src="assets/{raw_name}"><figcaption>Rohbild</figcaption></figure>
                <figure><img src="assets/{semantic_name}"><figcaption>Netz 141 + Hysterese</figcaption></figure>
                <figure><img src="assets/{a_name}"><figcaption>{html.escape(args.label_a)} · Sicherheit {confidence_a:.1%}</figcaption></figure>
                <figure><img src="assets/{b_name}"><figcaption>{html.escape(args.label_b)} · Sicherheit {confidence_b:.1%}</figcaption></figure>
                <figure><img src="assets/{difference_name}"><figcaption>Unterschiede rot markiert</figcaption></figure>
              </div>
            </section>
            """
        )
        rows.append(
            {
                "component_id": component_id,
                "status_a": status_a,
                "status_b": status_b,
                "decision_changed": decision_changed,
                "changed_assignment_pixels": changed_pixels,
            }
        )

    changed_decisions = sum(bool(row["decision_changed"]) for row in rows)
    document = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Epoche 64 gegen Epoche 80</title>
<style>
body{{margin:0;background:#10161d;color:#e8eef5;font:15px system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#10161df2;padding:18px 24px;border-bottom:1px solid #34404d}}
main{{max-width:1900px;margin:auto;padding:20px}} h1{{margin:0 0 10px}}
button{{border:1px solid #526170;border-radius:7px;background:#202c37;color:#eef5fb;padding:8px 12px;margin-right:6px;cursor:pointer}}
button.active{{background:#217b69;border-color:#2bd2ae}} .card{{background:#17212b;margin:0 0 22px;padding:16px;border-radius:10px}}
.decision-change{{border-left:6px solid #ffb341}} .same-decision{{border-left:6px solid #526170}}
.decisions{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px}} .pill{{padding:5px 9px;border-radius:999px;background:#303b47}}
.accepted_separation{{background:#176653}} .rejected_uncertain{{background:#78531d}}
.panels{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}} figure{{margin:0}}
img{{width:100%;height:310px;object-fit:contain;background:#05080b}} figcaption{{padding:7px;color:#c1cfdb}}
@media(max-width:1300px){{.panels{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<header><h1>{html.escape(args.label_a)} gegen {html.escape(args.label_b)}</h1>
<div>{len(rows)} Mehrzellgruppen · bei {changed_decisions} Gruppen unterschiedliche Sicherheitsentscheidung</div>
<p><button class="active" data-filter="all">Alle</button><button data-filter="changed">Nur unterschiedliche Entscheidung</button><button data-filter="a">Nur in {html.escape(args.label_a)} sicher</button><button data-filter="b">Nur in {html.escape(args.label_b)} sicher</button></p></header>
<main>{''.join(cards)}</main>
<script>
for(const button of document.querySelectorAll('button')){{button.onclick=()=>{{
 document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));button.classList.add('active');
 const filter=button.dataset.filter;for(const card of document.querySelectorAll('.card')){{
  card.hidden=filter==='changed'?card.dataset.changed!=='true':filter==='a'?card.dataset.a!=='true':filter==='b'?card.dataset.b!=='true':false;
 }}
}};
}}
</script></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "run_a": str(run_a),
                "run_b": str(run_b),
                "label_a": args.label_a,
                "label_b": args.label_b,
                "conflict_components": len(rows),
                "changed_safety_decisions": changed_decisions,
                "components": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
