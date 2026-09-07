from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erstellt eine HTML-Übersicht aller exportierten Zellen."
    )
    parser.add_argument(
        "--cells-dir",
        type=Path,
        required=True,
        help="Ordner, der die cell0001-, cell0002- usw. Ordner enthält.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ausgabe-HTML. Standard: <cells-dir>/cell_overview.html",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=320,
        help="Maximale Kantenlänge der Vorschaubilder.",
    )
    return parser.parse_args()


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [1.0, 99.5])

    if high <= low:
        low = float(finite.min())
        high = float(finite.max())

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = np.clip(
        (image - low) / (high - low),
        0.0,
        1.0,
    )

    return np.round(normalized * 255).astype(np.uint8)


def read_2d_tiff(path: Path) -> np.ndarray:
    image = np.squeeze(np.asarray(tifffile.imread(path)))

    if image.ndim != 2:
        raise RuntimeError(
            f"2D-TIFF erwartet, gefunden {image.shape}: {path}"
        )

    return image


def make_preview(
    raw_path: Path,
    seg_path: Path,
    output_path: Path,
    thumbnail_size: int,
) -> tuple[int, int, int, int]:
    raw = read_2d_tiff(raw_path)
    seg = read_2d_tiff(seg_path)

    if raw.shape != seg.shape:
        raise RuntimeError(
            f"Shape stimmt nicht überein: raw={raw.shape}, seg={seg.shape}"
        )

    gray = normalize_grayscale(raw)
    raw_rgb = np.repeat(gray[..., None], 3, axis=2)

    overlay = raw_rgb.astype(np.float32)

    skeleton = seg == 1
    soma = seg == 2

    # Skeleton: grün
    overlay[skeleton] = (
        0.25 * overlay[skeleton]
        + 0.75 * np.asarray([0, 255, 60], dtype=np.float32)
    )

    # Soma: magenta
    overlay[soma] = (
        0.25 * overlay[soma]
        + 0.75 * np.asarray([255, 0, 255], dtype=np.float32)
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    raw_image = Image.fromarray(raw_rgb)
    overlay_image = Image.fromarray(overlay)

    raw_image.thumbnail(
        (thumbnail_size, thumbnail_size),
        Image.Resampling.LANCZOS,
    )
    overlay_image.thumbnail(
        (thumbnail_size, thumbnail_size),
        Image.Resampling.LANCZOS,
    )

    panel_width = raw_image.width + overlay_image.width
    panel_height = max(raw_image.height, overlay_image.height)

    panel = Image.new(
        "RGB",
        (panel_width, panel_height),
        color=(20, 20, 20),
    )

    panel.paste(raw_image, (0, 0))
    panel.paste(overlay_image, (raw_image.width, 0))

    panel.save(output_path, quality=92)

    return (
        int(raw.shape[1]),
        int(raw.shape[0]),
        int(skeleton.sum()),
        int(soma.sum()),
    )


def numeric_cell_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)

    if match:
        return int(match.group(1)), path.name

    return 10**12, path.name


def relative_link(target: Path, html_path: Path) -> str:
    try:
        return target.relative_to(html_path.parent).as_posix()
    except ValueError:
        return target.resolve().as_uri()


def main() -> None:
    args = parse_args()

    cells_dir = args.cells_dir.resolve()

    if not cells_dir.exists():
        raise FileNotFoundError(cells_dir)

    output_html = (
        args.output.resolve()
        if args.output is not None
        else cells_dir / "cell_overview.html"
    )

    assets_dir = output_html.parent / "_cell_gallery_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    cell_dirs = sorted(
        [
            path
            for path in cells_dir.iterdir()
            if path.is_dir() and re.fullmatch(r"cell\d+", path.name)
        ],
        key=numeric_cell_key,
    )

    cards: list[str] = []
    exported = 0
    skipped = 0
    total_skeleton_pixels = 0
    total_soma_pixels = 0

    for cell_dir in cell_dirs:
        raw_path = cell_dir / "raw.tif"
        seg_path = cell_dir / "seg.tif"

        if not raw_path.exists() or not seg_path.exists():
            print(
                f"ÜBERSPRUNGEN: {cell_dir.name}: "
                "raw.tif oder seg.tif fehlt."
            )
            skipped += 1
            continue

        preview_path = assets_dir / f"{cell_dir.name}.jpg"

        try:
            (
                crop_width,
                crop_height,
                skeleton_pixels,
                soma_pixels,
            ) = make_preview(
                raw_path=raw_path,
                seg_path=seg_path,
                output_path=preview_path,
                thumbnail_size=args.thumbnail_size,
            )

        except Exception as error:
            print(f"ÜBERSPRUNGEN: {cell_dir.name}: {error}")
            skipped += 1
            continue

        metadata_path = cell_dir / "metadata.json"
        source_soma_id = ""

        if metadata_path.exists():
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                source_soma_id = metadata.get("source_soma_id", "")
            except Exception:
                source_soma_id = ""

        total_skeleton_pixels += skeleton_pixels
        total_soma_pixels += soma_pixels
        exported += 1

        links: list[str] = []

        for filename in (
            "raw.tif",
            "skeleton.tif",
            "soma.tif",
            "seg.tif",
            "seg.csv",
            "seg-000.swc",
            "metadata.json",
        ):
            file_path = cell_dir / filename

            if file_path.exists():
                links.append(
                    f'<a href="{html.escape(relative_link(file_path, output_html))}">'
                    f"{html.escape(filename)}</a>"
                )

        search_text = (
            f"{cell_dir.name} soma {source_soma_id} "
            f"skeleton {skeleton_pixels} crop {crop_width} {crop_height}"
        ).lower()

        cards.append(
            f"""
            <article class="card" data-search="{html.escape(search_text)}">
                <a href="{html.escape(relative_link(seg_path, output_html))}">
                    <img
                        src="{html.escape(relative_link(preview_path, output_html))}"
                        alt="{html.escape(cell_dir.name)}"
                        loading="lazy"
                    >
                </a>

                <div class="content">
                    <h2>{html.escape(cell_dir.name)}</h2>

                    <dl>
                        <dt>Soma-ID</dt>
                        <dd>{html.escape(str(source_soma_id or "–"))}</dd>

                        <dt>Crop</dt>
                        <dd>{crop_width} × {crop_height}</dd>

                        <dt>Skeletonpixel</dt>
                        <dd>{skeleton_pixels:,}</dd>

                        <dt>Somapixel</dt>
                        <dd>{soma_pixels:,}</dd>
                    </dl>

                    <div class="links">
                        {" · ".join(links)}
                    </div>
                </div>
            </article>
            """
        )

        print(
            f"[{exported}] {cell_dir.name}: "
            f"{crop_width}x{crop_height}, "
            f"Skeleton={skeleton_pixels}, Soma={soma_pixels}"
        )

    html_text = f"""<!doctype html>
<html lang="de">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >
    <title>Übersicht exportierter Zellen</title>

    <style>
        :root {{
            color-scheme: dark;
            font-family: Arial, Helvetica, sans-serif;
        }}

        body {{
            margin: 0;
            background: #111318;
            color: #eef1f5;
        }}

        header {{
            position: sticky;
            top: 0;
            z-index: 10;
            padding: 18px 24px;
            background: rgba(17, 19, 24, 0.96);
            border-bottom: 1px solid #343943;
            backdrop-filter: blur(8px);
        }}

        h1 {{
            margin: 0 0 12px;
            font-size: 24px;
        }}

        .summary {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            margin-bottom: 14px;
            color: #c9ced8;
        }}

        .summary strong {{
            color: white;
        }}

        input {{
            width: min(500px, calc(100% - 24px));
            padding: 11px 12px;
            border: 1px solid #424955;
            border-radius: 8px;
            background: #20242c;
            color: white;
            font-size: 16px;
        }}

        main {{
            display: grid;
            grid-template-columns:
                repeat(auto-fill, minmax(310px, 1fr));
            gap: 18px;
            padding: 22px;
        }}

        .card {{
            overflow: hidden;
            background: #1b1f27;
            border: 1px solid #343943;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.28);
        }}

        .card img {{
            display: block;
            width: 100%;
            height: auto;
            background: black;
        }}

        .content {{
            padding: 14px;
        }}

        h2 {{
            margin: 0 0 12px;
            font-size: 19px;
        }}

        dl {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 5px 15px;
            margin: 0 0 14px;
        }}

        dt {{
            color: #adb4c0;
        }}

        dd {{
            margin: 0;
            font-variant-numeric: tabular-nums;
        }}

        .links {{
            line-height: 1.7;
            font-size: 13px;
        }}

        a {{
            color: #74b9ff;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .legend {{
            font-size: 13px;
            color: #c9ced8;
        }}

        .green {{
            color: #00ff3c;
        }}

        .magenta {{
            color: #ff00ff;
        }}

        .hidden {{
            display: none;
        }}
    </style>
</head>

<body>
<header>
    <h1>Übersicht exportierter Zellen</h1>

    <div class="summary">
        <span><strong>{exported}</strong> Zellen</span>
        <span><strong>{skipped}</strong> übersprungen</span>
        <span>
            <strong>{total_skeleton_pixels:,}</strong>
            Skeletonpixel
        </span>
        <span>
            <strong>{total_soma_pixels:,}</strong>
            Somapixel
        </span>
    </div>

    <div class="legend">
        Jeweils links: Original · rechts:
        <span class="green">Skeleton</span> +
        <span class="magenta">Soma</span>
    </div>

    <p>
        <input
            id="search"
            type="search"
            placeholder="Zelle oder Soma-ID suchen …"
            autocomplete="off"
        >
    </p>
</header>

<main id="gallery">
    {"".join(cards)}
</main>

<script>
    const search = document.getElementById("search");
    const cards = Array.from(
        document.querySelectorAll(".card")
    );

    search.addEventListener("input", () => {{
        const query = search.value.trim().toLowerCase();

        for (const card of cards) {{
            const text = card.dataset.search || "";
            card.classList.toggle(
                "hidden",
                query && !text.includes(query)
            );
        }}
    }});
</script>
</body>
</html>
"""

    output_html.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_html.write_text(
        html_text,
        encoding="utf-8",
    )

    print()
    print(f"HTML erstellt: {output_html}")
    print(f"Vorschaubilder: {assets_dir}")
    print()
    print("Im Browser öffnen mit:")
    print(f'Start-Process "{output_html}"')


if __name__ == "__main__":
    main()
