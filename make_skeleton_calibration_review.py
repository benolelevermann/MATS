from __future__ import annotations

"""Create an offline, named calibration page for connectivity variants.

Unlike the formal blinded A/B page, this report deliberately shows variant
names. It is restricted to the fixed 12-case calibration manifest and records
which setting has the best continuity plus whether each candidate introduces an
obvious false connection. The page recommends the conservative first candidate
when continuity counts tie.
"""

import argparse
import json
from pathlib import Path

from make_blinded_skeleton_ab_review import (
    difference_image,
    find_tiff,
    normalize_gray,
    read_2d,
    relative_url,
    safe_identifier,
    save_png,
    semantic_overlay,
)
from skeleton_connectivity_experiments import read_manifest


REPORT_VERSION = "skeleton-connectivity-calibration-v1-2026-08-27"
DEFAULT_DATASET = "Dataset138_cleanSingleCell_soma_skeleton_recrop"


def parse_named_directory(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"--candidate expects NAME=PATH, got {text!r}")
    # Candidate labels may themselves contain '=' (for example T_low=0.20),
    # so the final separator belongs to the NAME=PATH interface.
    name, path_text = text.rsplit("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Candidate name is empty: {text!r}")
    return name, Path(path_text).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--base-name", default="R0 aktuelle Baseline")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named candidate prediction directory; repeat for every setting.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_report(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.resolve()
    raw = project_root / "nnUNet_raw" / DEFAULT_DATASET
    images_dir = (args.images_dir or raw / "imagesTr").resolve()
    labels_dir = (args.labels_dir or raw / "labelsTr").resolve()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    candidates = [parse_named_directory(text) for text in args.candidate]
    names = [name for name, _ in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique")
    for label, directory in (
        ("images", images_dir),
        ("labels", labels_dir),
        ("base predictions", base_dir),
        *((f"candidate {name}", directory) for name, directory in candidates),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")

    report_path = output_dir / "calibration.html"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Calibration report already exists: {report_path}. Use --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for order, case_id in enumerate(read_manifest(args.manifest), start=1):
        image = read_2d(find_tiff(images_dir, f"{case_id}_0000"))
        target = read_2d(find_tiff(labels_dir, case_id))
        base = read_2d(find_tiff(base_dir, case_id))
        variants = [(name, read_2d(find_tiff(directory, case_id))) for name, directory in candidates]
        shapes = {image.shape, target.shape, base.shape, *(variant.shape for _, variant in variants)}
        if len(shapes) != 1:
            raise ValueError(f"Shape mismatch in calibration case {case_id}: {sorted(shapes)}")

        gray = normalize_gray(image)
        assets = output_dir / "assets" / safe_identifier(case_id)
        original_path = assets / "original.png"
        target_path = assets / "ground_truth.png"
        base_path = assets / "base.png"
        save_png(original_path, gray)
        save_png(target_path, semantic_overlay(gray, target))
        save_png(base_path, semantic_overlay(gray, base))
        candidate_assets = []
        for index, (name, prediction) in enumerate(variants):
            token = safe_identifier(f"candidate-{index + 1}-{name}")
            overlay_path = assets / f"{token}.png"
            difference_path = assets / f"{token}-difference.png"
            save_png(overlay_path, semantic_overlay(gray, prediction))
            save_png(difference_path, difference_image(base, prediction))
            candidate_assets.append(
                {
                    "key": f"candidate_{index + 1}",
                    "name": name,
                    "overlay": relative_url(overlay_path, output_dir),
                    "difference": relative_url(difference_path, output_dir),
                }
            )
        cases.append(
            {
                "order": order,
                "case_id": case_id,
                "original": relative_url(original_path, output_dir),
                "ground_truth": relative_url(target_path, output_dir),
                "base": relative_url(base_path, output_dir),
                "candidates": candidate_assets,
            }
        )

    return {
        "version": REPORT_VERSION,
        "report_id": safe_identifier(
            f"calibration-{args.base_name}-{'-'.join(names)}-{args.manifest.resolve()}"
        ),
        "base_name": args.base_name,
        "base_directory": str(base_dir),
        "candidate_order": [
            {"key": f"candidate_{index + 1}", "name": name, "directory": str(directory)}
            for index, (name, directory) in enumerate(candidates)
        ],
        "manifest": str(args.manifest.resolve()),
        "cases": cases,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skeleton-Kalibrierung</title>
<style>
:root { color-scheme: dark; --bg:#0d1117; --panel:#151b24; --line:#303846; --cyan:#25d9e8; --warn:#ffb238; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:#edf2f7; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
main { width:min(98vw,1900px); margin:0 auto; padding:20px; }
h1,h2,p { margin-top:0; }
.intro,.case,.summary { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:18px; }
.image-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:10px; }
.tile { background:#090c11; border:1px solid var(--line); border-radius:9px; overflow:hidden; }
.tile strong { display:block; padding:8px 10px; }
.tile img { width:100%; display:block; image-rendering:pixelated; }
.ratings { margin-top:14px; display:grid; gap:10px; }
.continuity { display:flex; flex-wrap:wrap; gap:9px; align-items:center; }
label.choice { border:1px solid var(--line); border-radius:8px; padding:7px 10px; cursor:pointer; }
label.choice:has(input:checked) { border-color:var(--cyan); background:#103139; }
.false-grid { display:flex; flex-wrap:wrap; gap:12px; }
.false-grid label { color:#ffdba0; }
textarea { width:100%; min-height:58px; background:#0b1017; color:#edf2f7; border:1px solid var(--line); border-radius:8px; padding:8px; }
button { background:#1f6feb; color:white; border:0; border-radius:8px; padding:10px 14px; cursor:pointer; font-weight:650; margin-right:8px; }
button.secondary { background:#303846; }
.counts { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }
.count { border:1px solid var(--line); border-radius:9px; padding:10px; display:grid; gap:3px; }
.recommendation { margin-top:12px; padding:12px; border-left:4px solid var(--warn); background:#2b2312; }
.legend { color:#aab5c3; }
</style>
</head>
<body>
<main>
<section class="intro">
  <h1>Kalibrierung: Skeleton-Kontinuität</h1>
  <p>Diese zwölf Zellen sind nicht Teil des formalen 80er-Panels. Wähle pro Zelle die beste Kontinuität und markiere jede Variante, die eine offensichtliche Fehlverbindung erzeugt.</p>
  <p class="legend">Skeleton cyan, Soma magenta. In den Differenzbildern ist nur die jeweilige Variante blau und nur die Basis orange.</p>
  <button id="export">CSV exportieren</button><button class="secondary" id="reset">Bewertungen zurücksetzen</button>
</section>
<section class="summary"><h2>Zusammenfassung</h2><div class="counts" id="counts"></div><div class="recommendation" id="recommendation"></div></section>
<div id="cases"></div>
</main>
<script>
const REPORT = __REPORT_JSON__;
const STORAGE_KEY = `skeleton-calibration:${REPORT.report_id}`;
const state = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{"ratings":{}}');
if (!state.ratings) state.ratings = {};
const persist = () => localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
const escapeCsv = value => { const text=String(value ?? ''); return /[;"\r\n]/.test(text) ? `"${text.replaceAll('"','""')}"` : text; };

function imageTile(title, source) {
  const tile=document.createElement('div'); tile.className='tile';
  tile.innerHTML=`<strong>${title}</strong><img src="${source}" alt="${title}">`;
  return tile;
}

function rating(caseId) {
  if (!state.ratings[caseId]) state.ratings[caseId]={};
  return state.ratings[caseId];
}

function renderCases() {
  const root=document.querySelector('#cases'); root.replaceChildren();
  for (const item of REPORT.cases) {
    const card=document.createElement('section'); card.className='case';
    const title=document.createElement('h2'); title.textContent=`${item.order}/12 – Zelle ${item.case_id}`; card.append(title);
    const grid=document.createElement('div'); grid.className='image-grid';
    grid.append(imageTile('Original',item.original),imageTile('Ground Truth',item.ground_truth),imageTile(REPORT.base_name,item.base));
    for (const candidate of item.candidates) {
      grid.append(imageTile(candidate.name,candidate.overlay),imageTile(`${candidate.name} – Differenz zur Basis`,candidate.difference));
    }
    card.append(grid);
    const controls=document.createElement('div'); controls.className='ratings';
    const continuity=document.createElement('div'); continuity.className='continuity';
    continuity.append(Object.assign(document.createElement('strong'),{textContent:'Beste Kontinuität:'}));
    const choices=[['base',REPORT.base_name],['tie','kein Unterschied'],...REPORT.candidate_order.map(c=>[c.key,c.name])];
    for (const [key,label] of choices) {
      const wrapper=document.createElement('label'); wrapper.className='choice';
      const input=document.createElement('input'); input.type='radio'; input.name=`continuity-${item.case_id}`; input.value=key; input.checked=rating(item.case_id).continuity===key;
      input.addEventListener('change',()=>{ rating(item.case_id).continuity=key; persist(); renderSummary(); });
      wrapper.append(input,document.createTextNode(` ${label}`)); continuity.append(wrapper);
    }
    controls.append(continuity);
    const falseGrid=document.createElement('div'); falseGrid.className='false-grid';
    falseGrid.append(Object.assign(document.createElement('strong'),{textContent:'Offensichtliche Fehlverbindung:'}));
    for (const candidate of REPORT.candidate_order) {
      const wrapper=document.createElement('label'); const input=document.createElement('input'); input.type='checkbox'; input.checked=Boolean(rating(item.case_id)[`false_${candidate.key}`]);
      input.addEventListener('change',()=>{ rating(item.case_id)[`false_${candidate.key}`]=input.checked; persist(); renderSummary(); });
      wrapper.append(input,document.createTextNode(` ${candidate.name}`)); falseGrid.append(wrapper);
    }
    controls.append(falseGrid);
    const notes=document.createElement('textarea'); notes.placeholder='Optionaler Kommentar'; notes.value=rating(item.case_id).comment || '';
    notes.addEventListener('input',()=>{ rating(item.case_id).comment=notes.value; persist(); }); controls.append(notes);
    card.append(controls); root.append(card);
  }
}

function totals() {
  const result={base:{continuity:0,false:0},tie:{continuity:0,false:0}};
  for (const candidate of REPORT.candidate_order) result[candidate.key]={continuity:0,false:0,name:candidate.name};
  for (const item of REPORT.cases) {
    const row=rating(item.case_id); if (result[row.continuity]) result[row.continuity].continuity += 1;
    for (const candidate of REPORT.candidate_order) if (row[`false_${candidate.key}`]) result[candidate.key].false += 1;
  }
  return result;
}

function recommendation(result) {
  const eligible=REPORT.candidate_order.filter(c=>result[c.key].false<=2);
  if (!eligible.length) return 'Empfehlung: Basis behalten – alle Kandidaten haben mehr als zwei markierte Fehlverbindungen.';
  let best=eligible[0];
  for (const candidate of eligible.slice(1)) if (result[candidate.key].continuity > result[best.key].continuity) best=candidate;
  if (result.base.continuity >= result[best.key].continuity || result[best.key].continuity===0) return `Empfehlung: ${REPORT.base_name} behalten.`;
  return `Empfehlung nach der festgelegten Regel: ${best.name}.`;
}

function renderSummary() {
  const result=totals(); const root=document.querySelector('#counts'); root.replaceChildren();
  const entries=[['base',REPORT.base_name],...REPORT.candidate_order.map(c=>[c.key,c.name])];
  for (const [key,name] of entries) { const box=document.createElement('div'); box.className='count'; box.innerHTML=`<strong>${name}</strong><span>Kontinuitäts-Siege: ${result[key].continuity}</span>${key==='base'?'':`<span>Fälle mit Fehlverbindung: ${result[key].false}</span>`}`; root.append(box); }
  document.querySelector('#recommendation').textContent=recommendation(result);
}

document.querySelector('#export').addEventListener('click',()=>{
  const headers=['order','case_id','best_continuity',...REPORT.candidate_order.map(c=>`false_${c.key}`),'comment'];
  const lines=[headers.join(';')];
  for (const item of REPORT.cases) { const row=rating(item.case_id); const values=[item.order,item.case_id,row.continuity||'',...REPORT.candidate_order.map(c=>row[`false_${c.key}`]?'yes':'no'),row.comment||'']; lines.push(values.map(escapeCsv).join(';')); }
  const blob=new Blob([lines.join('\r\n')],{type:'text/csv;charset=utf-8'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=`${REPORT.report_id}-calibration.csv`; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.querySelector('#reset').addEventListener('click',()=>{ if (confirm('Alle Kalibrierungsbewertungen löschen?')) { localStorage.removeItem(STORAGE_KEY); location.reload(); } });
renderCases(); renderSummary();
</script>
</body>
</html>'''


def write_report(args: argparse.Namespace, report: dict[str, object]) -> Path:
    output_dir = args.output_dir.resolve()
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    html = HTML_TEMPLATE.replace(
        "__REPORT_JSON__",
        json.dumps(report, ensure_ascii=False).replace("</", "<\\/"),
    )
    path = output_dir / "calibration.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    report = build_report(args)
    path = write_report(args, report)
    print(f"Calibration page: {path}")
    print(f"Cases: {len(report['cases'])}")


if __name__ == "__main__":
    main()
