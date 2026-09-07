from __future__ import annotations

"""Audit MATS patch exports and build a compact, self-contained HTML gallery.

Every export manifest is read, but only the newest revision of each
``dataset_case_id`` is shown. The TIFF data are never modified.
"""

import argparse
import csv
import html
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage


SCRIPT_VERSION = "mats-annotation-overview-v1-2026-09-03"
SKELETON_RGB = np.array([0, 224, 244], dtype=np.float32)
SOMA_RGB = np.array([250, 55, 164], dtype=np.float32)
IGNORE_RGB = np.array([255, 190, 48], dtype=np.float32)
ALLOWED_LABELS = {0, 1, 2, 3}


@dataclass(frozen=True)
class ExportRecord:
    data: dict[str, Any]
    batch_dir: Path
    batch_name: str
    batch_created_at: str
    history_count: int
    label_version_count: int

    @property
    def case_id(self) -> str:
        return str(self.data["dataset_case_id"])

    @property
    def image_path(self) -> Path:
        return self.batch_dir / str(self.data["image_file"])

    @property
    def label_path(self) -> Path:
        return self.batch_dir / str(self.data["label_file"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-size", type=int, default=240)
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def revision_key(entry: tuple[dict[str, Any], Path, str]) -> tuple[int, str, str]:
    record, _batch_dir, created_at = entry
    try:
        revision = int(record.get("verified_revision", -1))
    except (TypeError, ValueError):
        revision = -1
    return revision, str(record.get("verified_at", "")), created_at


def collect_latest_records(exports_dir: Path) -> tuple[list[ExportRecord], dict[str, Any]]:
    manifests = sorted(exports_dir.glob("*/annotation_manifest.json"), key=lambda path: path.parent.name)
    if not manifests:
        raise FileNotFoundError(f"Keine annotation_manifest.json unter {exports_dir} gefunden")

    history: dict[str, list[tuple[dict[str, Any], Path, str]]] = defaultdict(list)
    batch_rows: list[dict[str, Any]] = []
    label_mapping: dict[str, int] | None = None
    total_rows = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels = {str(key): int(value) for key, value in manifest.get("labels", {}).items()}
        if label_mapping is None:
            label_mapping = labels
        elif labels != label_mapping:
            raise RuntimeError(f"Uneinheitliche Klassenbelegung in {manifest_path}")
        created_at = str(manifest.get("created_at", ""))
        records = list(manifest.get("records", []))
        total_rows += len(records)
        image_ids = sorted({str(record.get("image_id", "")) for record in records})
        batch_rows.append(
            {
                "batch": manifest_path.parent.name,
                "created_at": created_at,
                "rows": len(records),
                "images": image_ids,
            }
        )
        for record in records:
            case_id = str(record.get("dataset_case_id", ""))
            if not case_id:
                raise RuntimeError(f"Datensatz ohne dataset_case_id in {manifest_path}")
            history[case_id].append((record, manifest_path.parent, created_at))

    latest: list[ExportRecord] = []
    for case_id, versions in history.items():
        data, batch_dir, created_at = max(versions, key=revision_key)
        latest.append(
            ExportRecord(
                data=data,
                batch_dir=batch_dir,
                batch_name=batch_dir.name,
                batch_created_at=created_at,
                history_count=len(versions),
                label_version_count=len({str(item[0].get("label_sha256", "")) for item in versions}),
            )
        )
    latest.sort(
        key=lambda record: (
            str(record.data.get("image_id", "")),
            int(record.data.get("row_index", 0)),
            int(record.data.get("column_index", 0)),
        )
    )
    metadata = {
        "manifests": len(manifests),
        "exported_rows": total_rows,
        "unique_cases": len(latest),
        "superseded_rows": total_rows - len(latest),
        "label_mapping": label_mapping or {},
        "batches": batch_rows,
    }
    return latest, metadata


def read_2d(path: Path, description: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{description} fehlt: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{description} ist nach squeeze nicht 2-D: {path} -> {array.shape}")
    return array


def normalize_raw(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.7])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return np.uint8(np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0)


def downsample_labels(label: np.ndarray, stride: int) -> np.ndarray:
    """Max-pool class masks so that one-pixel skeletons remain visible."""
    if stride <= 1:
        return np.asarray(label, dtype=np.uint8)
    height, width = label.shape
    out_h, out_w = math.ceil(height / stride), math.ceil(width / stride)
    padded = np.zeros((out_h * stride, out_w * stride), dtype=np.uint8)
    padded[:height, :width] = np.asarray(label, dtype=np.uint8)
    blocks = padded.reshape(out_h, stride, out_w, stride)
    output = np.zeros((out_h, out_w), dtype=np.uint8)
    # Skeleton first, then soma and ignore with increasing display priority.
    output[np.any(blocks == 1, axis=(1, 3))] = 1
    output[np.any(blocks == 2, axis=(1, 3))] = 2
    output[np.any(blocks == 3, axis=(1, 3))] = 3
    return output


def make_pair_preview(raw: np.ndarray, label: np.ndarray, preview_size: int) -> np.ndarray:
    stride = max(1, math.ceil(max(raw.shape) / max(1, preview_size)))
    gray = normalize_raw(raw)[::stride, ::stride]
    labels = downsample_labels(label, stride)
    if gray.shape != labels.shape:
        raise RuntimeError(f"Interner Preview-Shape-Fehler: {gray.shape} != {labels.shape}")
    raw_rgb = np.repeat(gray[..., None], 3, axis=2)
    overlay = raw_rgb.astype(np.float32)
    colors = np.zeros_like(overlay)
    colors[labels == 1] = SKELETON_RGB
    colors[labels == 2] = SOMA_RGB
    colors[labels == 3] = IGNORE_RGB
    foreground = labels > 0
    overlay[foreground] = 0.12 * overlay[foreground] + 0.88 * colors[foreground]
    divider = np.full((gray.shape[0], 4, 3), 228, dtype=np.uint8)
    return np.concatenate([raw_rgb, divider, np.uint8(np.clip(overlay, 0, 255))], axis=1)


def component_count(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))[1])


def touched_edges(mask: np.ndarray) -> str:
    touched: list[str] = []
    if np.any(mask[0, :]):
        touched.append("oben")
    if np.any(mask[-1, :]):
        touched.append("unten")
    if np.any(mask[:, 0]):
        touched.append("links")
    if np.any(mask[:, -1]):
        touched.append("rechts")
    return ", ".join(touched)


def inspect_record(record: ExportRecord, assets_dir: Path, preview_size: int) -> dict[str, Any]:
    raw = read_2d(record.image_path, "Bild")
    label = read_2d(record.label_path, "Label")
    expected_shape = tuple(int(value) for value in record.data.get("shape", []))
    actual_values = {int(value) for value in np.unique(label)}
    counts = {value: int(np.count_nonzero(label == value)) for value in sorted(actual_values | ALLOWED_LABELS)}
    manifest_counts = {int(key): int(value) for key, value in record.data.get("class_pixel_counts", {}).items()}

    problems: list[str] = []
    notes: list[str] = []
    if raw.shape != label.shape:
        problems.append(f"Bild/Label-Form verschieden ({raw.shape} vs. {label.shape})")
    if expected_shape and label.shape != expected_shape:
        problems.append(f"Manifest-Form {expected_shape} stimmt nicht")
    invalid = sorted(actual_values - ALLOWED_LABELS)
    if invalid:
        problems.append(f"Unbekannte Klassen: {invalid}")
    if manifest_counts and any(counts.get(key, 0) != value for key, value in manifest_counts.items()):
        problems.append("Pixelzahlen weichen vom Manifest ab")
    if counts.get(1, 0) == 0 and counts.get(2, 0) == 0:
        notes.append("komplett ohne Label")
    elif counts.get(1, 0) == 0:
        notes.append("Soma ohne Skelett")
    elif counts.get(2, 0) == 0:
        notes.append("Skelett ohne Soma")
    if counts.get(3, 0) > 0:
        notes.append("Ignore-Pixel vorhanden")
    if label.shape != (512, 512):
        notes.append("Randpatch mit kleinerer Form")
    skeleton_edges = touched_edges(label == 1)
    soma_edges = touched_edges(label == 2)
    if skeleton_edges:
        notes.append("Skelett beruehrt Patchrand")
    if soma_edges:
        notes.append("Soma beruehrt Patchrand")

    asset_name = f"{record.case_id}.webp"
    Image.fromarray(make_pair_preview(raw, label, preview_size)).save(
        assets_dir / asset_name, "WEBP", quality=90, method=4
    )
    return {
        "case_id": record.case_id,
        "image_id": str(record.data.get("image_id", "")),
        "row": int(record.data.get("row_index", 0)),
        "column": int(record.data.get("column_index", 0)),
        "y0": int(record.data.get("y0", 0)),
        "y1": int(record.data.get("y1", raw.shape[0])),
        "x0": int(record.data.get("x0", 0)),
        "x1": int(record.data.get("x1", raw.shape[1])),
        "height": int(raw.shape[0]),
        "width": int(raw.shape[1]),
        "raw_dtype": str(raw.dtype),
        "label_dtype": str(label.dtype),
        "background_pixels": counts.get(0, 0),
        "skeleton_pixels": counts.get(1, 0),
        "soma_pixels": counts.get(2, 0),
        "ignore_pixels": counts.get(3, 0),
        "skeleton_components": component_count(label == 1),
        "soma_components": component_count(label == 2),
        "skeleton_edges": skeleton_edges,
        "soma_edges": soma_edges,
        "problems": problems,
        "notes": notes,
        "batch": record.batch_name,
        "verified_revision": int(record.data.get("verified_revision", -1)),
        "verified_at": str(record.data.get("verified_at", "")),
        "history_count": record.history_count,
        "label_version_count": record.label_version_count,
        "raw_source": str(record.data.get("raw_source", "")),
        "image_path": str(record.image_path),
        "label_path": str(record.label_path),
        "asset": f"assets/{asset_name}",
    }


def image_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["image_id"]].append(record)
    summaries: list[dict[str, Any]] = []
    for image_id, items in sorted(grouped.items()):
        coordinates = {(item["row"], item["column"]) for item in items}
        max_row = max(item["row"] for item in items)
        max_column = max(item["column"] for item in items)
        expected = {(row, column) for row in range(max_row + 1) for column in range(max_column + 1)}
        summaries.append(
            {
                "image_id": image_id,
                "patches": len(items),
                "grid_rows": max_row + 1,
                "grid_columns": max_column + 1,
                "mosaic_height": max(item["y1"] for item in items),
                "mosaic_width": max(item["x1"] for item in items),
                "missing_tiles": sorted(expected - coordinates),
                "blank": sum(item["skeleton_pixels"] == 0 and item["soma_pixels"] == 0 for item in items),
                "without_soma": sum(item["skeleton_pixels"] > 0 and item["soma_pixels"] == 0 for item in items),
                "without_skeleton": sum(item["skeleton_pixels"] == 0 and item["soma_pixels"] > 0 for item in items),
                "edge_sized": sum((item["height"], item["width"]) != (512, 512) for item in items),
                "skeleton_pixels": sum(item["skeleton_pixels"] for item in items),
                "soma_pixels": sum(item["soma_pixels"] for item in items),
                "problems": sum(bool(item["problems"]) for item in items),
                "raw_source": items[0]["raw_source"],
            }
        )
    return summaries


def audit_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "case_id", "image_id", "row", "column", "y0", "y1", "x0", "x1", "height", "width",
        "raw_dtype", "label_dtype", "background_pixels", "skeleton_pixels", "soma_pixels", "ignore_pixels",
        "skeleton_components", "soma_components", "skeleton_edges", "soma_edges", "problems", "notes",
        "batch", "verified_revision", "verified_at", "history_count", "label_version_count", "image_path", "label_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            row["problems"] = " | ".join(record["problems"])
            row["notes"] = " | ".join(record["notes"])
            writer.writerow(row)


def escape(value: Any) -> str:
    return html.escape(str(value))


def render_html(
    records: list[dict[str, Any]], image_stats: list[dict[str, Any]], metadata: dict[str, Any]
) -> str:
    total_skeleton = sum(record["skeleton_pixels"] for record in records)
    total_soma = sum(record["soma_pixels"] for record in records)
    blank = sum(record["skeleton_pixels"] == 0 and record["soma_pixels"] == 0 for record in records)
    problems = sum(bool(record["problems"]) for record in records)
    revised = sum(record["label_version_count"] > 1 for record in records)

    source_rows = []
    for index, item in enumerate(image_stats):
        proposed_role = "Training" if index < 2 else "Validierung (Empfehlung)"
        completeness = "vollstaendig" if not item["missing_tiles"] else f"{len(item['missing_tiles'])} fehlen"
        source_rows.append(
            f"<tr><td><strong>{escape(item['image_id'])}</strong></td><td>{item['patches']}</td>"
            f"<td>{item['grid_columns']} x {item['grid_rows']}</td>"
            f"<td>{item['mosaic_width']:,} x {item['mosaic_height']:,} px</td>"
            f"<td>{item['skeleton_pixels']:,}</td><td>{item['soma_pixels']:,}</td>"
            f"<td>{item['blank']}</td><td>{item['without_soma']}</td><td>{item['edge_sized']}</td>"
            f"<td>{completeness}</td><td>{proposed_role}</td></tr>"
        )

    maps = []
    records_by_image = defaultdict(list)
    for record in records:
        records_by_image[record["image_id"]].append(record)
    for item in image_stats:
        cells = []
        lookup = {(record["row"], record["column"]): record for record in records_by_image[item["image_id"]]}
        for row in range(item["grid_rows"]):
            for column in range(item["grid_columns"]):
                record = lookup.get((row, column))
                if record is None:
                    cells.append('<div class="tile missing" title="nicht exportiert">fehlt</div>')
                    continue
                if record["problems"]:
                    state = "problem"
                elif record["skeleton_pixels"] == 0 and record["soma_pixels"] == 0:
                    state = "blank"
                elif record["soma_pixels"] == 0:
                    state = "nosoma"
                else:
                    state = "labelled"
                title = (
                    f"{record['case_id']} | Skeleton {record['skeleton_pixels']:,} | "
                    f"Soma {record['soma_pixels']:,}"
                )
                cells.append(
                    f'<a class="tile {state}" href="#{escape(record["case_id"])}" title="{escape(title)}">'
                    f"{row},{column}</a>"
                )
        maps.append(
            f'<section class="map-card"><h3>{escape(item["image_id"])}</h3>'
            f'<div class="tile-grid" style="--columns:{item["grid_columns"]}">{"".join(cells)}</div></section>'
        )

    cards = []
    for record in records:
        states = ["all", record["image_id"]]
        if record["problems"]:
            states.append("problem")
        if record["skeleton_pixels"] == 0 and record["soma_pixels"] == 0:
            states.append("blank")
        if record["skeleton_pixels"] > 0 and record["soma_pixels"] == 0:
            states.append("nosoma")
        if record["skeleton_pixels"] == 0 and record["soma_pixels"] > 0:
            states.append("noskeleton")
        if (record["height"], record["width"]) != (512, 512):
            states.append("edge")
        if record["ignore_pixels"]:
            states.append("ignore")
        badges = []
        for problem in record["problems"]:
            badges.append(f'<span class="badge bad">{escape(problem)}</span>')
        for note in record["notes"]:
            badges.append(f'<span class="badge">{escape(note)}</span>')
        if not badges:
            badges.append('<span class="badge good">formal unauffaellig</span>')
        cards.append(
            f'<article class="card" id="{escape(record["case_id"])}" '
            f'data-state="{escape(" ".join(states))}" data-search="{escape(record["case_id"].lower())}">'
            f'<img loading="lazy" src="{escape(record["asset"])}" alt="Rohbild und Label-Overlay">'
            f'<div class="body"><h3>{escape(record["case_id"])}</h3>'
            f'<div class="meta">Patch {record["column"]},{record["row"]} · '
            f'{record["width"]} x {record["height"]} px · Revision {record["verified_revision"]}</div>'
            f'<div class="counts"><span><i class="cyan"></i>Skeleton {record["skeleton_pixels"]:,}</span>'
            f'<span><i class="pink"></i>Soma {record["soma_pixels"]:,}</span>'
            f'<span>Komponenten {record["skeleton_components"]}/{record["soma_components"]}</span></div>'
            f'<div class="badges">{"".join(badges)}</div>'
            f'<details><summary>Technische Details</summary><div class="details">'
            f'Koordinaten x={record["x0"]}:{record["x1"]}, y={record["y0"]}:{record["y1"]}<br>'
            f'Export: {escape(record["batch"])}<br>Historie: {record["history_count"]} Exporte, '
            f'{record["label_version_count"]} Label-Version(en)<br>'
            f'Skelett am Rand: {escape(record["skeleton_edges"] or "nein")}<br>'
            f'Soma am Rand: {escape(record["soma_edges"] or "nein")}</div></details></div></article>'
        )

    options = "".join(f'<option value="{escape(item["image_id"])}">{escape(item["image_id"])}</option>' for item in image_stats)
    generated = datetime.now(timezone.utc).astimezone().strftime("%d.%m.%Y %H:%M")
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MATS-Trainingsdaten · 146 neueste Patches</title>
<style>
:root{{--bg:#f3f6f4;--paper:#fff;--ink:#15211c;--muted:#64736b;--line:#d7e0db;--cyan:#00dce8;--pink:#f5379f;--gold:#f4b83f;--green:#2d8f63;--red:#bf3f48}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1540px;margin:auto;padding:28px}} h1{{font-size:clamp(28px,4vw,48px);margin:.15em 0}} h2{{margin-top:34px}} .lead{{max-width:980px;color:#425049;font-size:17px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:24px 0}} .stat{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:15px}} .stat strong{{display:block;font-size:26px}} .stat span{{color:var(--muted)}}
.notice{{background:#fff8e8;border:1px solid #ead099;border-radius:14px;padding:15px 18px;margin:18px 0}} .ok{{background:#edf8f2;border-color:#b9dfca}}
.table-wrap{{overflow:auto;background:var(--paper);border:1px solid var(--line);border-radius:14px}} table{{width:100%;border-collapse:collapse;min-width:1050px}} th,td{{padding:11px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--muted);font-size:12px;text-transform:uppercase}}
.maps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}} .map-card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:14px}} .map-card h3{{margin:0 0 10px}} .tile-grid{{display:grid;grid-template-columns:repeat(var(--columns),1fr);gap:4px}} .tile{{aspect-ratio:1;display:grid;place-items:center;text-decoration:none;color:#173328;border-radius:6px;background:#bfe9d4;font-size:11px}} .tile:hover{{outline:3px solid #234}} .tile.blank{{background:#e5e8e6}} .tile.nosoma{{background:#b9e8ef}} .tile.problem,.tile.missing{{background:#f5b9bd}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);margin:9px 0}} .legend span::before{{content:"";display:inline-block;width:11px;height:11px;border-radius:3px;background:#bfe9d4;margin-right:5px}} .legend .blank-key::before{{background:#e5e8e6}} .legend .nosoma-key::before{{background:#b9e8ef}} .legend .problem-key::before{{background:#f5b9bd}}
.toolbar{{position:sticky;top:0;z-index:5;background:rgba(243,246,244,.95);backdrop-filter:blur(10px);display:flex;gap:10px;flex-wrap:wrap;padding:12px 0;border-bottom:1px solid var(--line)}} select,input{{padding:10px 12px;border:1px solid #aebcb5;border-radius:9px;background:white;font:inherit}} input{{min-width:260px}} .shown{{margin-left:auto;padding:10px;color:var(--muted)}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px;margin-top:18px}} .card{{background:var(--paper);border:1px solid var(--line);border-radius:15px;overflow:hidden;box-shadow:0 3px 14px #1c362508}} .card img{{display:block;width:100%;height:230px;object-fit:contain;background:#101512}} .body{{padding:14px}} .body h3{{font-size:15px;margin:0 0 5px;overflow-wrap:anywhere}} .meta,.details{{color:var(--muted);font-size:13px}} .counts{{display:flex;gap:13px;flex-wrap:wrap;margin:9px 0;font-size:13px}} i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}} .cyan{{background:var(--cyan)}} .pink{{background:var(--pink)}} .badges{{display:flex;gap:5px;flex-wrap:wrap}} .badge{{font-size:11px;padding:3px 7px;background:#eef1ef;border-radius:99px}} .badge.good{{background:#dff3e8;color:#176642}} .badge.bad{{background:#fde1e2;color:#8f222a}} details{{margin-top:10px}} summary{{cursor:pointer;color:#39564a;font-size:13px}} .hidden{{display:none!important}} footer{{color:var(--muted);margin:38px 0}}
@media(max-width:650px){{main{{padding:17px}} .shown{{width:100%;margin-left:0}} .gallery{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="meta">MATS Annotation Audit · erzeugt {generated}</p><h1>Manuell annotierte Übersichtsbilder</h1>
<p class="lead">Links steht jeweils das Rohbild, rechts das manuelle Label als Overlay. Cyan = 1-px-Skelett, Magenta = Soma, Gelb = Ignore. Die Galerie verwendet pro Position nur die neueste verifizierte Revision.</p>
<div class="stats"><div class="stat"><strong>{len(records)}</strong><span>eindeutige Patches</span></div><div class="stat"><strong>{len(image_stats)}</strong><span>Übersichtsbilder</span></div><div class="stat"><strong>{total_skeleton:,}</strong><span>Skelettpixel</span></div><div class="stat"><strong>{total_soma:,}</strong><span>Somapixel</span></div><div class="stat"><strong>{blank}</strong><span>leere Labelpatches</span></div><div class="stat"><strong>{problems}</strong><span>formale Fehler</span></div></div>
<div class="notice ok"><strong>Versionsprüfung:</strong> In {metadata['manifests']} Exportordnern liegen insgesamt {metadata['exported_rows']} Einträge. Davon sind {metadata['superseded_rows']} ältere Zwischenstände und werden nicht doppelt gezählt. Bei {revised} Positionen änderte sich das Label zwischen den Exporten.</div>
<div class="notice"><strong>Empfehlung fürs Fine-Tuning:</strong> Die Patches zuerst wieder zu vollständigen Labels zusammensetzen. So entstehen keine künstlichen Bildränder an den 512-px-Nähten. Mosaic 1 und 2 können trainieren, Mosaic 3 bleibt eine echte Validierung; Mosaic 4 bleibt zusätzlich ein vollständig blindes Testbild.</div>
<h2>Abdeckung und vorgeschlagene Trennung</h2><div class="table-wrap"><table><thead><tr><th>Bild</th><th>Patches</th><th>Raster</th><th>Gesamtgröße</th><th>Skeleton px</th><th>Soma px</th><th>leer</th><th>ohne Soma</th><th>Randpatches</th><th>Abdeckung</th><th>Rolle</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></div>
<h2>Patchkarte</h2><div class="legend"><span>Skeleton + Soma</span><span class="blank-key">ohne Label</span><span class="nosoma-key">ohne Soma</span><span class="problem-key">Fehler/fehlt</span></div><div class="maps">{''.join(maps)}</div>
<h2>Einzelprüfung</h2><div class="toolbar"><select id="imageFilter"><option value="all">Alle Übersichtsbilder</option>{options}</select><select id="stateFilter"><option value="all">Alle Zustände</option><option value="problem">Nur formale Fehler</option><option value="blank">Ohne Label</option><option value="nosoma">Skelett ohne Soma</option><option value="noskeleton">Soma ohne Skelett</option><option value="edge">Kleinere Randpatches</option><option value="ignore">Mit Ignore</option></select><input id="search" type="search" placeholder="Patch-ID suchen"><span class="shown" id="shown"></span></div>
<div class="gallery" id="gallery">{''.join(cards)}</div><footer>Audit-Version {SCRIPT_VERSION}. Alle Quelldateien bleiben unverändert. Vollständige Werte stehen in <code>patch_audit.csv</code>.</footer>
<script>
const cards=[...document.querySelectorAll('.card')], imageFilter=document.querySelector('#imageFilter'), stateFilter=document.querySelector('#stateFilter'), search=document.querySelector('#search'), shown=document.querySelector('#shown');
function applyFilters(){{let count=0;const image=imageFilter.value,state=stateFilter.value,q=search.value.trim().toLowerCase();for(const card of cards){{const states=card.dataset.state.split(' ');const visible=(image==='all'||states.includes(image))&&(state==='all'||states.includes(state))&&(!q||card.dataset.search.includes(q));card.classList.toggle('hidden',!visible);if(visible)count++;}}shown.textContent=count+' von '+cards.length;}}
imageFilter.addEventListener('change',applyFilters);stateFilter.addEventListener('change',applyFilters);search.addEventListener('input',applyFilters);applyFilters();
</script></main></body></html>"""


def prepare_output(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Ausgabeordner ist nicht leer: {output_dir}. Optional --overwrite verwenden.")
        # Only the explicit report directory is replaced; source exports are untouched.
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir()
    return assets_dir


def main() -> None:
    args = parse_args()
    exports_dir = args.exports_dir.resolve()
    output_dir = args.output_dir.resolve()
    records, metadata = collect_latest_records(exports_dir)
    assets_dir = prepare_output(output_dir, args.overwrite)

    audited: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(inspect_record, record, assets_dir, args.preview_size): record.case_id
            for record in records
        }
        for future in as_completed(futures):
            try:
                audited.append(future.result())
            except Exception as error:
                raise RuntimeError(f"Prüfung fehlgeschlagen für {futures[future]}: {error}") from error
    audited.sort(key=lambda item: (item["image_id"], item["row"], item["column"]))
    image_stats = image_summaries(audited)
    audit_csv(output_dir / "patch_audit.csv", audited)

    summary = {
        "script_version": SCRIPT_VERSION,
        "exports_dir": str(exports_dir),
        "output_dir": str(output_dir),
        **metadata,
        "images": image_stats,
        "formal_problem_cases": [record["case_id"] for record in audited if record["problems"]],
        "blank_label_cases": [
            record["case_id"] for record in audited
            if record["skeleton_pixels"] == 0 and record["soma_pixels"] == 0
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        render_html(audited, image_stats, metadata), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
