from __future__ import annotations

"""Create a paginated visual overview of every Dataset137 training pair.

The gallery reads nnU-Net's actual ``imagesTr`` and ``labelsTr`` directories.
It does not use a sample list, a split file, or model outputs. Each card shows
the raw training image beside the corresponding manual skeleton/soma label.
"""

import argparse
import csv
import html
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


SCRIPT_VERSION = "dataset137-training-overview-v1-2026-08-19"
SKELETON_RGB = np.array([0, 220, 255], dtype=np.float32)
SOMA_RGB = np.array([255, 45, 170], dtype=np.float32)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_path: Path
    label_path: Path
    source: str
    treatment: str
    original_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--build-report",
        type=Path,
        default=None,
        help="Optional dataset137_build_report.csv. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--div10-provenance",
        type=Path,
        default=None,
        help="Optional div10_selection_provenance.csv. Auto-detected if omitted.",
    )
    parser.add_argument("--page-size", type=int, default=80)
    parser.add_argument("--preview-size", type=int, default=180)
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if key not in (rows.fieldnames or []):
            raise RuntimeError(f"CSV is missing the '{key}' column: {path}")
        return {str(row[key]): {name: str(value or "") for name, value in row.items()} for row in rows}


def resolve_optional_path(given: Path | None, candidate: Path) -> Path | None:
    if given is not None:
        if not given.is_file():
            raise FileNotFoundError(f"Optional provenance file was not found: {given}")
        return given
    return candidate if candidate.is_file() else None


def collect_cases(dataset_dir: Path, build_report: Path | None, provenance: Path | None) -> list[CaseRecord]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise RuntimeError(f"Expected imagesTr and labelsTr in: {dataset_dir}")

    report_rows = read_csv_by_key(build_report, "case_id") if build_report else {}
    provenance_rows = read_csv_by_key(provenance, "case_id") if provenance else {}
    image_paths = sorted(images_dir.glob("*_0000.tif"), key=lambda path: path.name.lower())
    if not image_paths:
        raise RuntimeError(f"No '*_0000.tif' training images found in: {images_dir}")

    cases: list[CaseRecord] = []
    missing_labels: list[str] = []
    for image_path in image_paths:
        case_id = image_path.name[: -len("_0000.tif")]
        label_path = labels_dir / f"{case_id}.tif"
        if not label_path.is_file():
            missing_labels.append(case_id)
            continue
        report = report_rows.get(case_id, {})
        provenance_row = provenance_rows.get(case_id, {})
        source = report.get("source") or ("div10_added" if case_id.startswith("div10_") else "unknown")
        treatment = provenance_row.get("treatment", "")
        original_name = provenance_row.get("original_name", "")
        cases.append(
            CaseRecord(
                case_id=case_id,
                image_path=image_path,
                label_path=label_path,
                source=source,
                treatment=treatment,
                original_name=original_name,
            )
        )
    if missing_labels:
        preview = ", ".join(missing_labels[:8])
        raise RuntimeError(f"Missing labels for {len(missing_labels)} training images, for example: {preview}")
    return cases


def read_2d(path: Path, description: str) -> np.ndarray:
    try:
        array = tifffile.imread(path)
    except Exception as error:
        raise RuntimeError(f"Could not read {description}: {path}\n{error}") from error
    array = np.squeeze(np.asarray(array))
    if array.ndim != 2:
        raise RuntimeError(f"{description} must be 2-D after squeeze; got {array.shape}: {path}")
    return array


def preview_stride(shape: tuple[int, int], max_size: int) -> int:
    return max(1, math.ceil(max(shape) / max(max_size, 1)))


def downsample_labels(label: np.ndarray, stride: int) -> np.ndarray:
    """Downsample labels without losing thin skeleton pixels; soma has priority."""
    if stride == 1:
        return np.asarray(label, dtype=np.uint8)
    height, width = label.shape
    output_height = math.ceil(height / stride)
    output_width = math.ceil(width / stride)
    padded_height = output_height * stride
    padded_width = output_width * stride
    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[:height, :width] = np.asarray(label, dtype=np.uint8)
    blocks = padded.reshape(output_height, stride, output_width, stride)
    output = np.zeros((output_height, output_width), dtype=np.uint8)
    output[np.any(blocks == 1, axis=(1, 3))] = 1
    output[np.any(blocks == 2, axis=(1, 3))] = 2
    return output


def normalize_raw(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    low, high = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return np.uint8(np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0)


def make_overlay(gray: np.ndarray, label: np.ndarray, alpha: float = 0.86) -> np.ndarray:
    base = np.repeat(np.asarray(gray)[..., None], 3, axis=2).astype(np.float32)
    foreground = label > 0
    colors = np.zeros_like(base)
    colors[label == 1] = SKELETON_RGB
    colors[label == 2] = SOMA_RGB
    base[foreground] = (1.0 - alpha) * base[foreground] + alpha * colors[foreground]
    return np.uint8(np.clip(base, 0, 255))


def make_pair_preview(raw: np.ndarray, label: np.ndarray, preview_size: int) -> np.ndarray:
    stride = preview_stride(raw.shape, preview_size)
    raw_preview = normalize_raw(raw[::stride, ::stride])
    label_preview = downsample_labels(label, stride)
    if raw_preview.shape != label_preview.shape:
        raise RuntimeError(f"Preview shape mismatch: raw={raw_preview.shape}, label={label_preview.shape}")
    raw_rgb = np.repeat(raw_preview[..., None], 3, axis=2)
    overlay = make_overlay(raw_preview, label_preview)
    divider = np.full((raw_rgb.shape[0], 3, 3), 236, dtype=np.uint8)
    return np.concatenate([raw_rgb, divider, overlay], axis=1)


def write_preview(case: CaseRecord, assets_dir: Path, preview_size: int) -> dict[str, Any]:
    raw = read_2d(case.image_path, "Training image")
    label = read_2d(case.label_path, "Training label")
    if raw.shape != label.shape:
        raise RuntimeError(f"Shape mismatch for {case.case_id}: image={raw.shape}, label={label.shape}")
    label_values = set(np.unique(label).astype(int).tolist())
    if not label_values.issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected label values for {case.case_id}: {sorted(label_values)}")

    asset_path = assets_dir / f"{case.case_id}.webp"
    Image.fromarray(make_pair_preview(raw, label, preview_size)).save(asset_path, "WEBP", quality=88, method=4)
    return {
        "case_id": case.case_id,
        "source": case.source,
        "treatment": case.treatment,
        "original_name": case.original_name,
        "image_path": str(case.image_path),
        "label_path": str(case.label_path),
        "height_px": int(raw.shape[0]),
        "width_px": int(raw.shape[1]),
        "raw_dtype": str(raw.dtype),
        "skeleton_pixels": int(np.count_nonzero(label == 1)),
        "soma_pixels": int(np.count_nonzero(label == 2)),
        "foreground_pixels": int(np.count_nonzero(label > 0)),
        "asset": f"assets/{asset_path.name}",
    }


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        # This is deliberately restricted to the user-supplied output folder.
        for child in path.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def html_page(
    page_records: list[dict[str, Any]],
    page_number: int,
    page_count: int,
    root: Path,
    summary: dict[str, Any],
) -> str:
    cards: list[str] = []
    for record in page_records:
        treatment = html.escape(record["treatment"] or "-")
        original_name = html.escape(record["original_name"] or "-")
        cards.append(
            f"""
<article class=\"card\">
  <img loading=\"lazy\" src=\"../{html.escape(record['asset'])}\" alt=\"{html.escape(record['case_id'])}: raw and label overlay\">
  <div class=\"label\"><strong>{html.escape(record['case_id'])}</strong></div>
  <div class=\"meta\">{record['width_px']} x {record['height_px']} px | skel {record['skeleton_pixels']:,} | soma {record['soma_pixels']:,}</div>
  <div class=\"meta\">source: {html.escape(record['source'])} | treatment: {treatment}</div>
  <div class=\"name\" title=\"{original_name}\">{original_name}</div>
</article>"""
        )
    nav: list[str] = []
    for number in range(1, page_count + 1):
        if number == page_number:
            nav.append(f"<span class=\"current\">{number}</span>")
        else:
            nav.append(f"<a href=\"page_{number:03d}.html\">{number}</a>")
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Dataset137 training data - page {page_number}</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f4f6fa; }}
main {{ max-width: 2200px; margin: auto; padding: 22px; }}
h1 {{ margin: 0; }} .note {{ color:#536174; line-height:1.45; }}
.legend {{ margin:14px 0; }} .cyan {{ color:#009fb8; font-weight:bold; }} .magenta {{ color:#d7008d; font-weight:bold; }}
.nav {{ display:flex; flex-wrap:wrap; gap:7px; margin:18px 0; align-items:center; }}
.nav a, .nav span {{ padding:6px 9px; border-radius:5px; border:1px solid #ced8e6; text-decoration:none; color:#1b4965; background:#fff; }}
.nav .current {{ background:#1b4965; color:white; border-color:#1b4965; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:13px; }}
.card {{ background:#fff; border:1px solid #dce2ed; border-radius:9px; padding:9px; box-shadow:0 1px 3px #15233a12; overflow:hidden; }}
.card img {{ width:100%; display:block; image-rendering:auto; background:#111; }} .label {{ margin-top:7px; }}
.meta, .name {{ color:#536174; font-size:12px; line-height:1.35; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
</style></head><body><main>
<h1>Dataset137: actual training image-label pairs</h1>
<p class=\"note\">Page {page_number} of {page_count}. This gallery reads the exact nnU-Net <code>imagesTr</code> and <code>labelsTr</code> files used for Dataset137. Each card: raw image on the left, manual label overlay on the right. No prediction or postprocessing is shown.</p>
<p class=\"legend\"><span class=\"cyan\">Cyan</span>: skeleton (label 1) | <span class=\"magenta\">Magenta</span>: soma (label 2) | Total: {summary['total_cases']:,} cases</p>
<div class=\"nav\"><a href=\"../index.html\">Summary</a>{''.join(nav)}</div>
<section class=\"grid\">{''.join(cards)}</section>
<div class=\"nav\"><a href=\"../index.html\">Summary</a>{''.join(nav)}</div>
</main></body></html>"""


def summary_html(summary: dict[str, Any], page_count: int) -> str:
    sources = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{count:,}</td></tr>"
        for name, count in sorted(summary["source_counts"].items())
    )
    treatments = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{count:,}</td></tr>"
        for name, count in sorted(summary["treatment_counts"].items())
        if name
    ) or "<tr><td>-</td><td>0</td></tr>"
    pages = "".join(f"<a href=\"pages/page_{number:03d}.html\">Page {number}</a>" for number in range(1, page_count + 1))
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Dataset137 training-data overview</title>
<style>
body {{ margin:0; font-family:Arial,sans-serif; color:#172033; background:#f4f6fa; }} main {{ max-width:1300px; margin:auto; padding:32px; }}
h1 {{ margin:0 0 8px; }} .note {{ color:#536174; line-height:1.5; max-width:1000px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:22px 0; }} .card, section {{ background:#fff; border:1px solid #dce2ed; border-radius:10px; padding:18px; box-shadow:0 1px 3px #15233a12; }}
.number {{ font-size:30px; font-weight:700; color:#1b4965; }} table {{ border-collapse:collapse; width:100%; }} td,th {{ border-bottom:1px solid #e8ecf2; padding:9px; text-align:left; }} th:last-child,td:last-child {{ text-align:right; }}
.pages {{ display:flex; flex-wrap:wrap; gap:9px; }} .pages a {{ padding:9px 12px; background:#1b4965; color:white; border-radius:6px; text-decoration:none; }} code {{ background:#e9eef5; padding:2px 4px; border-radius:4px; }}
</style></head><body><main>
<h1>Dataset137 training-data overview</h1>
<p class=\"note\">This report contains every image-label pair actually present in <code>Dataset137_allcells_soma_skeleton_plus_div10_nonBleb_nonDMSO</code> at generation time. The gallery is paginated so all samples remain practical to inspect in a browser.</p>
<div class=\"cards\">
 <div class=\"card\"><div class=\"number\">{summary['total_cases']:,}</div><div>complete image-label pairs</div></div>
 <div class=\"card\"><div class=\"number\">{summary['base_cases']:,}</div><div>carried over from Dataset136</div></div>
 <div class=\"card\"><div class=\"number\">{summary['div10_cases']:,}</div><div>included div10 additions</div></div>
 <div class=\"card\"><div class=\"number\">0 / 1 / 2</div><div>background / skeleton / soma labels</div></div>
</div>
<section><h2>Dataset composition</h2><table><thead><tr><th>Source</th><th>Cases</th></tr></thead><tbody>{sources}</tbody></table></section>
<section><h2>Included div10 treatments</h2><table><thead><tr><th>Treatment</th><th>Cases</th></tr></thead><tbody>{treatments}</tbody></table><p class=\"note\">Only included cases are shown. Blebbistatin and DMSO were excluded before Dataset137 was assembled; dilutedDMSO was retained.</p></section>
<section><h2>Open the complete gallery</h2><p class=\"note\">Every card has the raw training image left and its manual label overlay right. Skeleton is cyan; soma is magenta.</p><div class=\"pages\">{pages}</div></section>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    if args.page_size < 1:
        raise ValueError("--page-size must be positive")
    if args.preview_size < 64:
        raise ValueError("--preview-size must be at least 64")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    dataset_dir = args.dataset_dir.resolve()
    build_report = resolve_optional_path(args.build_report, dataset_dir / "dataset137_build_report.csv")
    provenance = resolve_optional_path(args.div10_provenance, dataset_dir / "div10_selection_provenance.csv")
    cases = collect_cases(dataset_dir, build_report, provenance)
    ensure_output_directory(args.output_dir, args.overwrite)
    assets_dir = args.output_dir / "assets"
    pages_dir = args.output_dir / "pages"
    assets_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset137 gallery: rendering {len(cases):,} exact training pairs with {args.workers} worker(s).")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(write_preview, case, assets_dir, args.preview_size): case
            for case in cases
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                failures.append({"case_id": case.case_id, "error": str(error)})
            if completed == len(cases) or completed % 100 == 0:
                print(f"Rendered {completed:,}/{len(cases):,} cases")
    if failures:
        failure_path = args.output_dir / "render_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "error"])
            writer.writeheader()
            writer.writerows(failures)
        raise RuntimeError(f"Could not render {len(failures)} cases. See: {failure_path}")

    records.sort(key=lambda record: (record["source"], record["treatment"], record["case_id"].lower()))
    source_counts = Counter(record["source"] for record in records)
    treatment_counts = Counter(record["treatment"] for record in records if record["treatment"])
    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset_dir": str(dataset_dir),
        "dataset_json": json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8")),
        "total_cases": len(records),
        "base_cases": sum(1 for record in records if record["source"] == "Dataset136_existing"),
        "div10_cases": sum(1 for record in records if record["case_id"].startswith("div10_")),
        "source_counts": dict(source_counts),
        "treatment_counts": dict(treatment_counts),
        "build_report": str(build_report) if build_report else "",
        "div10_provenance": str(provenance) if provenance else "",
    }
    fields = list(records[0])
    with (args.output_dir / "training_case_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    page_count = math.ceil(len(records) / args.page_size)
    for page_index in range(page_count):
        page_number = page_index + 1
        start = page_index * args.page_size
        end = start + args.page_size
        (pages_dir / f"page_{page_number:03d}.html").write_text(
            html_page(records[start:end], page_number, page_count, args.output_dir, summary), encoding="utf-8"
        )
    (args.output_dir / "index.html").write_text(summary_html(summary, page_count), encoding="utf-8")
    print(f"Dataset137 training gallery created: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
