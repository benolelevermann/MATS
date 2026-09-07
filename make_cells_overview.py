from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFilter, ImageFont


SCRIPT_VERSION = "cell-crop-overview-v1-2026-07-27"
SKELETON_COLOR = np.array([0, 220, 255], dtype=np.float32)
SOMA_COLOR = np.array([255, 45, 170], dtype=np.float32)


@dataclass(frozen=True)
class CellRecord:
    folder: Path
    condition: str
    cell_id: str
    raw_path: Path
    skeleton_path: Path
    soma_path: Path

    @property
    def display_name(self) -> str:
        return f"{self.condition}/{self.cell_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a contact sheet of exported cell crops. Raw intensity is "
            "shown in grayscale, skeleton in cyan and soma in magenta."
        )
    )
    parser.add_argument(
        "--cells-root",
        type=Path,
        required=True,
        help=(
            "Root containing cell folders. The script searches recursively "
            "for raw.tif."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for overview PNGs and overview_index.csv.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=0,
        help=(
            "Columns in the all-cells overview. Use 0 for automatic "
            "landscape layout (default: 0)."
        ),
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=140,
        help="Square image area per cell in pixels (default: 140).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=8,
        help="Gap between cells in pixels (default: 8).",
    )
    parser.add_argument(
        "--skeleton-thickness",
        type=int,
        default=1,
        help=(
            "Visual dilation radius after resizing. This changes only the "
            "overview, never skeleton.tif (default: 1)."
        ),
    )
    parser.add_argument(
        "--skeleton-alpha",
        type=float,
        default=0.95,
        help="Skeleton overlay opacity from 0 to 1 (default: 0.95).",
    )
    parser.add_argument(
        "--soma-alpha",
        type=float,
        default=0.48,
        help="Soma overlay opacity from 0 to 1 (default: 0.48).",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="Lower raw-image display percentile (default: 1).",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.5,
        help="Upper raw-image display percentile (default: 99.5).",
    )
    parser.add_argument(
        "--make-pages",
        action="store_true",
        help="Also create larger paginated detail sheets.",
    )
    parser.add_argument(
        "--page-columns",
        type=int,
        default=6,
        help="Columns per detail page (default: 6).",
    )
    parser.add_argument(
        "--page-rows",
        type=int,
        default=4,
        help="Rows per detail page (default: 4).",
    )
    parser.add_argument(
        "--page-thumbnail-size",
        type=int,
        default=280,
        help="Image area per cell on detail pages (default: 280).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N cells; 0 means all cells (default: 0).",
    )
    return parser.parse_args()


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def compact_condition(value: str) -> str:
    parts = [part for part in value.split("_") if part]
    if len(parts) > 2:
        return "_".join(parts[-2:])
    return value


def truncate_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    maximum_width: int,
) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= maximum_width:
        return text

    suffix = "..."
    candidate = text
    while candidate:
        shortened = candidate + suffix
        if draw.textbbox((0, 0), shortened, font=font)[2] <= maximum_width:
            return shortened
        candidate = candidate[:-1]
    return suffix


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def collect_cells(root: Path) -> list[CellRecord]:
    if not root.exists():
        raise FileNotFoundError(f"Cell root does not exist: {root}")

    records: list[CellRecord] = []
    for raw_path in root.rglob("raw.tif"):
        folder = raw_path.parent
        skeleton_path = folder / "skeleton.tif"
        soma_path = folder / "soma.tif"
        try:
            relative = folder.relative_to(root)
        except ValueError:
            relative = folder

        parts = relative.parts
        condition = parts[-2] if len(parts) >= 2 else root.name
        cell_id = parts[-1] if parts else folder.name

        records.append(
            CellRecord(
                folder=folder,
                condition=condition,
                cell_id=cell_id,
                raw_path=raw_path,
                skeleton_path=skeleton_path,
                soma_path=soma_path,
            )
        )

    records.sort(
        key=lambda record: (
            natural_key(record.condition),
            natural_key(record.cell_id),
        )
    )
    return records


def read_2d(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(
            f"{label} must be 2D, got shape {array.shape}: {path}"
        )
    return array


def normalize_raw(
    raw: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    raw_float = raw.astype(np.float32, copy=False)
    finite = raw_float[np.isfinite(raw_float)]
    if finite.size == 0:
        return np.zeros(raw.shape, dtype=np.uint8)

    low, high = np.percentile(
        finite,
        [lower_percentile, upper_percentile],
    )
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(raw.shape, dtype=np.uint8)

    scaled = np.clip((raw_float - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def resize_contained(
    image: Image.Image,
    width: int,
    height: int,
    resample: Image.Resampling,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    source_width, source_height = image.size
    scale = min(width / source_width, height / source_height)
    resized_size = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )
    resized = image.resize(resized_size, resample=resample)
    canvas = Image.new(image.mode, (width, height), fill)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def make_overlay(
    record: CellRecord,
    thumbnail_size: int,
    skeleton_thickness: int,
    skeleton_alpha: float,
    soma_alpha: float,
    lower_percentile: float,
    upper_percentile: float,
) -> Image.Image:
    raw = read_2d(record.raw_path, "raw.tif")
    skeleton = read_2d(record.skeleton_path, "skeleton.tif") > 0
    soma = read_2d(record.soma_path, "soma.tif") > 0

    if raw.shape != skeleton.shape or raw.shape != soma.shape:
        raise RuntimeError(
            f"Shape mismatch in {record.folder}:\n"
            f"  raw={raw.shape}, skeleton={skeleton.shape}, soma={soma.shape}"
        )

    raw_8bit = normalize_raw(
        raw,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    raw_image = resize_contained(
        Image.fromarray(raw_8bit, mode="L"),
        width=thumbnail_size,
        height=thumbnail_size,
        resample=Image.Resampling.LANCZOS,
        fill=0,
    )
    skeleton_image = resize_contained(
        Image.fromarray((skeleton.astype(np.uint8) * 255), mode="L"),
        width=thumbnail_size,
        height=thumbnail_size,
        resample=Image.Resampling.NEAREST,
        fill=0,
    )
    soma_image = resize_contained(
        Image.fromarray((soma.astype(np.uint8) * 255), mode="L"),
        width=thumbnail_size,
        height=thumbnail_size,
        resample=Image.Resampling.NEAREST,
        fill=0,
    )

    if skeleton_thickness > 0:
        kernel_size = 2 * skeleton_thickness + 1
        skeleton_image = skeleton_image.filter(
            ImageFilter.MaxFilter(kernel_size)
        )

    raw_rgb = np.repeat(
        np.asarray(raw_image, dtype=np.uint8)[..., None],
        3,
        axis=2,
    ).astype(np.float32)
    skeleton_mask = np.asarray(skeleton_image) > 0
    soma_mask = np.asarray(soma_image) > 0

    overlay = raw_rgb.copy()
    overlay[soma_mask] = (
        (1.0 - soma_alpha) * overlay[soma_mask]
        + soma_alpha * SOMA_COLOR
    )
    overlay[skeleton_mask] = (
        (1.0 - skeleton_alpha) * overlay[skeleton_mask]
        + skeleton_alpha * SKELETON_COLOR
    )

    return Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def draw_header(
    sheet: Image.Image,
    title: str,
    subtitle: str,
    width: int,
    header_height: int,
) -> None:
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(max(20, header_height // 4), bold=True)
    subtitle_font = load_font(max(13, header_height // 7))
    legend_font = load_font(max(12, header_height // 8))

    draw.text(
        (24, 14),
        title,
        fill=(20, 35, 60),
        font=title_font,
    )
    draw.text(
        (24, 18 + title_font.size),
        subtitle,
        fill=(85, 95, 110),
        font=subtitle_font,
    )

    legend_y = header_height - 27
    legend_x = max(24, width - 355)
    draw.line(
        (legend_x, legend_y + 7, legend_x + 30, legend_y + 7),
        fill=tuple(SKELETON_COLOR.astype(np.uint8)),
        width=5,
    )
    draw.text(
        (legend_x + 38, legend_y),
        "Skeleton",
        fill=(35, 40, 48),
        font=legend_font,
    )
    draw.rectangle(
        (legend_x + 138, legend_y + 1, legend_x + 162, legend_y + 14),
        fill=tuple(SOMA_COLOR.astype(np.uint8)),
    )
    draw.text(
        (legend_x + 170, legend_y),
        "Soma",
        fill=(35, 40, 48),
        font=legend_font,
    )


def create_sheet(
    records: list[CellRecord],
    output_path: Path,
    columns: int,
    thumbnail_size: int,
    gap: int,
    skeleton_thickness: int,
    skeleton_alpha: float,
    soma_alpha: float,
    lower_percentile: float,
    upper_percentile: float,
    title: str,
) -> list[dict[str, object]]:
    if not records:
        raise RuntimeError("No cells were provided to create_sheet.")

    label_height = max(38, thumbnail_size // 4)
    tile_width = thumbnail_size
    tile_height = thumbnail_size + label_height
    rows = math.ceil(len(records) / columns)
    header_height = max(92, thumbnail_size // 2)
    margin = 20

    width = 2 * margin + columns * tile_width + (columns - 1) * gap
    height = (
        header_height
        + margin
        + rows * tile_height
        + (rows - 1) * gap
        + margin
    )

    sheet = Image.new("RGB", (width, height), (248, 249, 251))
    draw_header(
        sheet,
        title=title,
        subtitle=(
            f"{len(records)} Zellen | Raw in Graustufen | "
            "Skeleton cyan | Soma magenta"
        ),
        width=width,
        header_height=header_height,
    )

    condition_font = load_font(max(10, label_height // 4))
    cell_font = load_font(max(11, label_height // 3), bold=True)
    draw = ImageDraw.Draw(sheet)
    index_rows: list[dict[str, object]] = []

    for index, record in enumerate(records):
        row = index // columns
        column = index % columns
        x = margin + column * (tile_width + gap)
        y = header_height + row * (tile_height + gap)

        try:
            overlay = make_overlay(
                record=record,
                thumbnail_size=thumbnail_size,
                skeleton_thickness=skeleton_thickness,
                skeleton_alpha=skeleton_alpha,
                soma_alpha=soma_alpha,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
            status = "ok"
        except Exception as error:
            overlay = Image.new(
                "RGB",
                (thumbnail_size, thumbnail_size),
                (72, 24, 28),
            )
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.text(
                (8, 8),
                "ERROR",
                fill=(255, 225, 225),
                font=load_font(max(12, thumbnail_size // 10), bold=True),
            )
            overlay_draw.text(
                (8, 32),
                str(error)[:90],
                fill=(255, 225, 225),
                font=load_font(max(10, thumbnail_size // 14)),
            )
            status = f"error: {error}"

        sheet.paste(overlay, (x, y))
        draw.rectangle(
            (x, y, x + thumbnail_size - 1, y + thumbnail_size - 1),
            outline=(205, 210, 218),
            width=1,
        )

        condition_label = truncate_to_width(
            draw,
            compact_condition(record.condition),
            condition_font,
            thumbnail_size - 8,
        )
        cell_label = truncate_to_width(
            draw,
            record.cell_id,
            cell_font,
            thumbnail_size - 8,
        )
        draw.text(
            (x + 4, y + thumbnail_size + 4),
            cell_label,
            fill=(25, 30, 40),
            font=cell_font,
        )
        draw.text(
            (x + 4, y + thumbnail_size + 5 + cell_font.size),
            condition_label,
            fill=(90, 98, 110),
            font=condition_font,
        )

        index_rows.append(
            {
                "sheet": output_path.name,
                "position": index + 1,
                "row": row + 1,
                "column": column + 1,
                "condition": record.condition,
                "cell_id": record.cell_id,
                "cell_folder": str(record.folder),
                "status": status,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, compress_level=6)
    return index_rows


def chunks(
    records: list[CellRecord],
    chunk_size: int,
) -> list[list[CellRecord]]:
    return [
        records[start : start + chunk_size]
        for start in range(0, len(records), chunk_size)
    ]


def validate_args(args: argparse.Namespace) -> None:
    if args.thumbnail_size < 48:
        raise ValueError("--thumbnail-size must be at least 48.")
    if args.columns < 0:
        raise ValueError("--columns must be 0 or greater.")
    if args.gap < 0:
        raise ValueError("--gap must not be negative.")
    if args.skeleton_thickness < 0:
        raise ValueError("--skeleton-thickness must not be negative.")
    if not 0.0 <= args.skeleton_alpha <= 1.0:
        raise ValueError("--skeleton-alpha must be between 0 and 1.")
    if not 0.0 <= args.soma_alpha <= 1.0:
        raise ValueError("--soma-alpha must be between 0 and 1.")
    if not 0.0 <= args.lower_percentile < args.upper_percentile <= 100.0:
        raise ValueError(
            "Percentiles must satisfy 0 <= lower < upper <= 100."
        )
    if args.page_columns < 1 or args.page_rows < 1:
        raise ValueError("--page-columns and --page-rows must be positive.")
    if args.page_thumbnail_size < 64:
        raise ValueError("--page-thumbnail-size must be at least 64.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"CELL CROP OVERVIEW: {SCRIPT_VERSION}")
    print(f"Cells root: {args.cells_root}")

    records = collect_cells(args.cells_root)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(
            f"No cell folders containing raw.tif found in {args.cells_root}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = args.columns
    if columns == 0:
        columns = max(1, math.ceil(math.sqrt(len(records) * 1.6)))

    all_index_rows = create_sheet(
        records=records,
        output_path=args.output_dir / "all_cells_overview.png",
        columns=columns,
        thumbnail_size=args.thumbnail_size,
        gap=args.gap,
        skeleton_thickness=args.skeleton_thickness,
        skeleton_alpha=args.skeleton_alpha,
        soma_alpha=args.soma_alpha,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        title="Übersicht aller Zell-Crops",
    )

    if args.make_pages:
        page_size = args.page_columns * args.page_rows
        pages_dir = args.output_dir / "detail_pages"
        for page_number, page_records in enumerate(
            chunks(records, page_size),
            start=1,
        ):
            page_path = pages_dir / f"cells_page_{page_number:03d}.png"
            all_index_rows.extend(
                create_sheet(
                    records=page_records,
                    output_path=page_path,
                    columns=args.page_columns,
                    thumbnail_size=args.page_thumbnail_size,
                    gap=max(args.gap, 10),
                    skeleton_thickness=max(args.skeleton_thickness, 1),
                    skeleton_alpha=args.skeleton_alpha,
                    soma_alpha=args.soma_alpha,
                    lower_percentile=args.lower_percentile,
                    upper_percentile=args.upper_percentile,
                    title=f"Zell-Crops - Detailseite {page_number}",
                )
            )

    index_path = args.output_dir / "overview_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sheet",
                "position",
                "row",
                "column",
                "condition",
                "cell_id",
                "cell_folder",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(all_index_rows)

    failed = sum(
        not str(row["status"]).startswith("ok")
        for row in all_index_rows[: len(records)]
    )

    print()
    print("=" * 72)
    print("OVERVIEW COMPLETE")
    print("=" * 72)
    print(f"Cells:       {len(records)}")
    print(f"Errors:      {failed}")
    print(f"Overview:    {args.output_dir / 'all_cells_overview.png'}")
    print(f"Index:       {index_path}")
    if args.make_pages:
        print(f"Detail pages:{args.output_dir / 'detail_pages'}")
    print()
    print("Cyan = Skeleton | Magenta = Soma")
    print("The source TIFF files were not modified.")


if __name__ == "__main__":
    main()
