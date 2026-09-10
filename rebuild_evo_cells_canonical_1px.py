from __future__ import annotations

"""Rebuild reviewed Evo cells with one canonical skeleton and an A/B report."""

import argparse
import csv
import html
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from canonical_skeleton import canonicalize_semantic, canonicalize_skeleton
from r_pipeline.export_cells_for_r_pipeline import save_tiff, write_pixel_csv, write_swc


SCRIPT_VERSION = "canonical-evo-rebuild-v3-snt-shared-root-2026-09-10"
UNCHANGED_FILES = (
    "raw.tif",
    "prediction.tif",
    "soma.tif",
    "raw_preview.png",
    "prediction_preview.png",
    "review.json",
    "bounds.json",
    "location.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-soma-gap-px", type=float, default=3.0)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy unchanged TIFFs instead of space-saving NTFS hardlinks.",
    )
    return parser.parse_args()


def read_2d(path: Path) -> np.ndarray:
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D TIFF, got {array.shape}: {path}")
    return array


def link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def normalized_rgb(raw: np.ndarray) -> np.ndarray:
    values = raw.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        gray = np.zeros(raw.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1.0, 99.7])
        if high <= low:
            high = low + 1.0
        gray = np.round(np.clip((values - low) / (high - low), 0, 1) * 255).astype(
            np.uint8
        )
    return np.repeat(gray[..., None], 3, axis=2)


def overlay_mask(raw: np.ndarray, skeleton: np.ndarray, soma: np.ndarray) -> Image.Image:
    rgb = normalized_rgb(raw).astype(np.float32)
    rgb[skeleton] = 0.30 * rgb[skeleton] + 0.70 * np.asarray([0, 238, 255])
    rgb[soma] = 0.25 * rgb[soma] + 0.75 * np.asarray([255, 54, 164])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def read_swc(path: Path) -> dict[int, tuple[int, float, float, int]]:
    nodes: dict[int, tuple[int, float, float, int]] = {}
    if not path.is_file():
        return nodes
    for line_text in path.read_text(encoding="utf-8").splitlines():
        if not line_text.strip() or line_text.startswith("#"):
            continue
        fields = line_text.split()
        if len(fields) < 7:
            continue
        nodes[int(float(fields[0]))] = (
            int(float(fields[1])),
            float(fields[2]),
            float(fields[3]),
            int(float(fields[6])),
        )
    return nodes


def overlay_swc(
    raw: np.ndarray,
    swc_path: Path,
    soma: np.ndarray,
) -> Image.Image:
    image = Image.fromarray(normalized_rgb(raw))
    draw = ImageDraw.Draw(image)
    nodes = read_swc(swc_path)
    for node_id, (node_type, x, y, parent_id) in nodes.items():
        if parent_id not in nodes:
            continue
        _, parent_x, parent_y, _ = nodes[parent_id]
        draw.line((parent_x, parent_y, x, y), fill=(0, 238, 255), width=1)
    soma_rows, soma_columns = np.nonzero(soma)
    if soma_rows.size:
        draw.ellipse(
            (
                float(np.mean(soma_columns)) - 3,
                float(np.mean(soma_rows)) - 3,
                float(np.mean(soma_columns)) + 3,
                float(np.mean(soma_rows)) + 3,
            ),
            fill=(255, 54, 164),
        )
    return image


def save_preview(image: Image.Image, path: Path) -> None:
    image.thumbnail((420, 420), Image.Resampling.LANCZOS)
    image.convert("RGB").save(path, quality=88, optimize=True)


def rebuild_cell(
    source: Path,
    destination: Path,
    max_soma_gap_px: float,
    copy_files: bool,
) -> dict[str, object]:
    required = (
        "raw.tif",
        "skeleton.tif",
        "soma.tif",
        "seg.tif",
        "metadata.json",
        "bounds.json",
        "location.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source.name}: missing {', '.join(missing)}")

    raw = read_2d(source / "raw.tif")
    original_skeleton = read_2d(source / "skeleton.tif").astype(bool)
    soma = read_2d(source / "soma.tif").astype(bool)
    canonical, report = canonicalize_skeleton(
        original_skeleton,
        soma=soma,
        max_soma_gap_px=max_soma_gap_px,
    )
    status = "safe" if report.detached_components_after == 0 else "review"
    destination.mkdir(parents=True, exist_ok=False)

    for name in UNCHANGED_FILES:
        source_file = source / name
        if source_file.is_file():
            link_or_copy(source_file, destination / name, copy_files)

    semantic = np.zeros(raw.shape, dtype=np.uint8)
    semantic[canonical] = 1
    semantic[soma] = 2
    save_tiff(destination / "skeleton.tif", canonical.astype(np.uint8))
    save_tiff(destination / "seg.tif", semantic)
    save_tiff(destination / "cell_mask.tif", (canonical | soma).astype(np.uint8))

    postprocessed_source = source / "postprocessed.tif"
    if postprocessed_source.is_file():
        canonical_postprocessed, _ = canonicalize_semantic(
            read_2d(postprocessed_source),
            max_soma_gap_px=max_soma_gap_px,
        )
        save_tiff(destination / "postprocessed.tif", canonical_postprocessed)

    bounds = json.loads((source / "bounds.json").read_text(encoding="utf-8"))
    write_pixel_csv(
        destination / "seg.csv",
        canonical,
        int(bounds["x_min"]),
        int(bounds["y_min"]),
    )
    swc_report = None
    if status == "safe":
        swc_report = write_swc(destination / "seg-000.swc", canonical, soma)

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["script_version"] = SCRIPT_VERSION
    metadata["source_cell"] = str(source.resolve())
    metadata["status"] = (
        "accepted_single_cell_canonical_1px"
        if status == "safe"
        else "canonical_1px_requires_review"
    )
    metadata["skeleton_pixels"] = int(canonical.sum())
    metadata["canonical_1px"] = report.to_dict()
    metadata["swc"] = swc_report
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    overlay_mask(raw, canonical, soma).save(destination / "preview.png")
    return {
        "cell": source.name,
        "status": status,
        **report.to_dict(),
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
    }


def build_report(
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, object]],
) -> Path:
    preview_root = output_root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, row in enumerate(rows, start=1):
        name = str(row["cell"])
        source = source_root / name
        destination = Path(str(row["output"]))
        raw = read_2d(source / "raw.tif")
        old_skeleton = read_2d(source / "skeleton.tif").astype(bool)
        soma = read_2d(source / "soma.tif").astype(bool)
        new_skeleton = read_2d(destination / "skeleton.tif").astype(bool)
        paths = {
            "raw": preview_root / f"{name}_raw.jpg",
            "old_mask": preview_root / f"{name}_old_mask.jpg",
            "old_swc": preview_root / f"{name}_old_swc.jpg",
            "new_mask": preview_root / f"{name}_new_mask.jpg",
            "new_swc": preview_root / f"{name}_new_swc.jpg",
        }
        save_preview(Image.fromarray(normalized_rgb(raw)), paths["raw"])
        save_preview(overlay_mask(raw, old_skeleton, soma), paths["old_mask"])
        save_preview(
            overlay_swc(
                raw,
                source / "seg-000.swc",
                soma,
            ),
            paths["old_swc"],
        )
        save_preview(overlay_mask(raw, new_skeleton, soma), paths["new_mask"])
        if (destination / "seg-000.swc").is_file():
            new_swc = overlay_swc(
                raw,
                destination / "seg-000.swc",
                soma,
            )
        else:
            new_swc = overlay_mask(raw, new_skeleton, soma)
            ImageDraw.Draw(new_swc).text((8, 8), "REVIEW: detached", fill=(255, 190, 0))
        save_preview(new_swc, paths["new_swc"])
        relative = {key: path.relative_to(output_root).as_posix() for key, path in paths.items()}
        cards.append(
            f"""
            <article class="card" data-status="{html.escape(str(row['status']))}">
              <header><b>{index:03d} · {html.escape(name)}</b><span class="{row['status']}">{row['status']}</span></header>
              <div class="images">
                <figure><img loading="lazy" src="{relative['raw']}"><figcaption>Rohbild</figcaption></figure>
                <figure><img loading="lazy" src="{relative['old_mask']}"><figcaption>Bisherige Maske</figcaption></figure>
                <figure><img loading="lazy" src="{relative['old_swc']}"><figcaption>Bisheriges SWC</figcaption></figure>
                <figure><img loading="lazy" src="{relative['new_mask']}"><figcaption>Kanonisch 1 px</figcaption></figure>
                <figure><img loading="lazy" src="{relative['new_swc']}"><figcaption>Neues SWC · tatsächlicher Pipeline-Baum</figcaption></figure>
              </div>
              <p>{row['input_pixels']} → {row['output_pixels']} Skeletonpixel · {row['pixels_removed_by_thinning']} entfernt · {row['soma_gap_pixels_added']} Soma-Gap-Pixel ergänzt · {row['detached_components_after']} abgetrennte Komponenten</p>
            </article>"""
        )

    safe_count = sum(row["status"] == "safe" for row in rows)
    review_count = len(rows) - safe_count
    analyzed_index = next(
        (
            output_root / folder / "analyzed_skeletons" / "index.html"
            for folder in ("scevoview_exact", "scevoview_normalized_v2")
            if (output_root / folder / "analyzed_skeletons" / "index.html").is_file()
        ),
        None,
    )
    analyzed_link = ""
    if analyzed_index is not None:
        analyzed_link = (
            f'<p><a class="analysis-link" href="{analyzed_index.relative_to(output_root).as_posix()}">'
            "Tatsächlich von scEvoView analysierte SNT-Traces öffnen</a></p>"
        )
    index_path = output_root / "index.html"
    index_path.write_text(
        f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MATS · 1-px-Skeleton A/B</title><style>
body{{margin:0;background:#111820;color:#edf5f2;font:14px system-ui,sans-serif}}main{{width:min(1900px,calc(100% - 28px));margin:auto}}h1{{font:42px Georgia,serif;margin:28px 0 6px}}.lead{{color:#aab8b5;max-width:1000px}}.analysis-link{{display:inline-block;color:#07120f;background:#58e0b7;padding:9px 13px;border-radius:6px;text-decoration:none;font-weight:700}}nav{{position:sticky;top:0;background:#111820ee;padding:12px 0;z-index:3}}button{{border:1px solid #4b5b61;background:#1b2730;color:white;padding:8px 13px;margin-right:7px;border-radius:7px}}.card{{background:#19232c;margin:15px 0;padding:12px;border-left:4px solid #36d5ad}}.card header{{display:flex;justify-content:space-between;font-size:16px}}.images{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:10px}}figure{{margin:0}}img{{width:100%;height:auto;background:#05080a}}figcaption{{color:#aab8b5;margin-top:4px}}.safe{{color:#4ee1a8}}.review{{color:#ffc857}}p{{color:#aab8b5}}@media(max-width:1000px){{.images{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>Kanonisches 1-px-Skeleton</h1><p class="lead">Dieselben {len(rows)} freigegebenen Zellen. Alt bleibt unverändert; neu verwendet eine einzige 1-px-Maske für TIFF, CSV und SWC. Das neue SWC wird ohne zusätzliche Darstellung oder Glättung genauso gezeigt, wie es anschließend in die scEvoView-Pipeline gelangt.</p>{analyzed_link}<nav><button onclick="filterCards('all')">Alle {len(rows)}</button><button onclick="filterCards('safe')">Sicher {safe_count}</button><button onclick="filterCards('review')">Review {review_count}</button></nav>{''.join(cards)}</main>
<script>function filterCards(s){{document.querySelectorAll('.card').forEach(c=>c.style.display=(s==='all'||c.dataset.status===s)?'block':'none')}}</script></body></html>""",
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if args.max_soma_gap_px < 0:
        raise ValueError("--max-soma-gap-px must be >= 0.")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    safe_root = output_root / "safe_cells"
    review_root = output_root / "review_cells"
    safe_root.mkdir(parents=True)
    review_root.mkdir(parents=True)

    rows = []
    cells = sorted(path for path in source_root.glob("cell*") if path.is_dir())
    for index, source in enumerate(cells, start=1):
        temporary = output_root / "_staging" / source.name
        row = rebuild_cell(
            source,
            temporary,
            max_soma_gap_px=args.max_soma_gap_px,
            copy_files=args.copy_files,
        )
        target_root = safe_root if row["status"] == "safe" else review_root
        destination = target_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
        row["output"] = str(destination.resolve())
        rows.append(row)
        print(f"[{index}/{len(cells)}] {source.name}: {row['status']}", flush=True)
    staging = output_root / "_staging"
    if staging.exists():
        staging.rmdir()

    fields = list(rows[0].keys())
    with (output_root / "canonical_1px_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "source": str(source_root),
        "output": str(output_root),
        "cells": len(rows),
        "safe_cells": sum(row["status"] == "safe" for row in rows),
        "review_cells": sum(row["status"] == "review" for row in rows),
        "max_soma_gap_px": args.max_soma_gap_px,
        "space_saving_hardlinks": not args.copy_files,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = build_report(source_root, output_root, rows)
    print(json.dumps(summary, indent=2))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
