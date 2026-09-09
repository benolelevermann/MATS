"""Create browser-friendly raw/export previews for a tracing comparison report."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_image(array: np.ndarray) -> Image.Image:
    data = np.asarray(array)
    data = np.squeeze(data)
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {data.shape}")
    values = data.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, (1.0, 99.8))
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            scaled = np.zeros(values.shape, dtype=np.uint8)
        else:
            scaled = np.clip((values - low) / (high - low), 0, 1)
            scaled = np.round(scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def load_preview(path: Path) -> Image.Image:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return normalize_image(tifffile.imread(path))
    with Image.open(path) as image:
        return image.convert("RGB")


def read_swc(path: Path) -> list[dict[str, float]]:
    nodes: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", ".").split()
            if len(fields) < 7:
                continue
            nodes.append(
                {
                    "node": int(float(fields[0])),
                    "type": int(float(fields[1])),
                    "x": float(fields[2]),
                    "y": float(fields[3]),
                    "radius": float(fields[5]),
                    "parent": int(float(fields[6])),
                }
            )
    return nodes


def overlay_swc(raw: Image.Image, swc_path: Path, coordinate_scale: float) -> Image.Image:
    image = raw.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    nodes = read_swc(swc_path)
    for node in nodes:
        node["x"] *= coordinate_scale
        node["y"] *= coordinate_scale
        node["radius"] *= coordinate_scale
    by_id = {int(node["node"]): node for node in nodes}
    child_counts: dict[int, int] = {}
    for node in nodes:
        parent = int(node["parent"])
        if parent >= 0:
            child_counts[parent] = child_counts.get(parent, 0) + 1
            parent_node = by_id.get(parent)
            if parent_node is not None:
                draw.line(
                    (
                        node["x"],
                        node["y"],
                        parent_node["x"],
                        parent_node["y"],
                    ),
                    fill=(0, 220, 235, 255),
                    width=2,
                )
    for node in nodes:
        x, y = node["x"], node["y"]
        if int(node["type"]) == 1 or int(node["parent"]) < 0:
            radius = max(4.0, min(25.0, node["radius"]))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(237, 76, 145, 190),
                outline=(255, 116, 180, 255),
                width=2,
            )
        elif child_counts.get(int(node["node"]), 0) > 1:
            draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=(255, 209, 102, 255))
    return Image.alpha_composite(image, overlay).convert("RGB")


def save_web_image(image: Image.Image, path: Path) -> None:
    image.thumbnail((560, 560), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    cards_by_source: dict[str, list[str]] = {}
    for index, row in enumerate(rows, start=1):
        source = row["source_kind"]
        raw_source = Path(row["raw_file"])
        export_source = Path(row["export_preview_file"]) if row.get("export_preview_file") else None
        swc_source = Path(row["overlay_swc_file"]) if row.get("overlay_swc_file") else None
        if not raw_source.is_file():
            raise FileNotFoundError(raw_source)

        stem = f"{index:04d}_{'manual' if source == 'Manuell' else 'automatic'}"
        raw_target = assets_dir / f"{stem}_raw.png"
        export_target = assets_dir / f"{stem}_export.png"
        raw_image = load_preview(raw_source)
        save_web_image(raw_image.copy(), raw_target)
        if export_source is not None and export_source.is_file():
            save_web_image(load_preview(export_source), export_target)
            export_caption = "Final exportierte Zelle"
        elif swc_source is not None and swc_source.is_file():
            coordinate_scale = float(row.get("overlay_coordinate_scale") or 1.0)
            save_web_image(overlay_swc(raw_image, swc_source, coordinate_scale), export_target)
            export_caption = "Manuelles SWC auf Rohbild"
        else:
            shutil.copy2(raw_target, export_target)
            export_caption = "Exportansicht nicht vorhanden"

        cell_id = html.escape(row["raw_id"])
        cards_by_source.setdefault(source, []).append(
            "".join(
                [
                    '<article class="cell-image-card">',
                    f"<h3>{cell_id}</h3>",
                    '<div class="cell-image-pair">',
                    "<figure>",
                    f'<a href="cell_images/assets/{raw_target.name}"><img loading="lazy" src="cell_images/assets/{raw_target.name}" alt="Rohbild {cell_id}"></a>',
                    "<figcaption>Rohbild</figcaption></figure>",
                    "<figure>",
                    f'<a href="cell_images/assets/{export_target.name}"><img loading="lazy" src="cell_images/assets/{export_target.name}" alt="Export {cell_id}"></a>',
                    f"<figcaption>{html.escape(export_caption)}</figcaption></figure>",
                    "</div></article>",
                ]
            )
        )

    sections: list[str] = []
    for source in ("Manuell", "Automatisch"):
        cards = cards_by_source.get(source, [])
        if not cards:
            continue
        sections.extend(
            [
                f'<section class="cell-source"><div class="cell-source-head"><span>{html.escape(source)}</span><strong>{len(cards)} Zellen</strong></div>',
                '<div class="cell-image-grid">',
                *cards,
                "</div></section>",
            ]
        )
    (output_dir / "gallery_fragment.html").unlink(missing_ok=True)
    (output_dir / "gallery_fragment.inc").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"CHECK: {len(rows)} raw/export cell pairs generated in {output_dir}")


if __name__ == "__main__":
    main()
