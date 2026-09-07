from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


DEFAULT_OUTPUT = Path(
    r"X:\Niro\04_Raw_Data\InVitro\mica\250317_MCS24GFP\C3-C10_training_expansion"
)
FILENAME_PATTERN = re.compile(
    r"^(?P<provenance>\d{6}_MCS24GFP_div\d+_plate\d+_C(?:0[3-9]|10))_"
    r"FOV(?P<fov>0[1-3])_x(?P<x>\d{5})_y(?P<y>\d{5})_"
    r"(?P<width>\d+)x(?P<height>\d+)\.tif$",
    re.IGNORECASE,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scaled_preview(image: np.ndarray, size: int = 512) -> Image.Image:
    low, high = np.percentile(image, [0.5, 99.8])
    if high <= low:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        scaled = np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
        scaled = scaled.astype(np.uint8)
    return Image.fromarray(scaled, mode="L").resize((size, size), Image.Resampling.LANCZOS)


def create_montage(samples: list[tuple[str, np.ndarray]], path: Path) -> None:
    tile_size = 512
    label_height = 34
    columns = 4
    rows = (len(samples) + columns - 1) // columns
    montage = Image.new("L", (columns * tile_size, rows * (tile_size + label_height)), 0)
    draw = ImageDraw.Draw(montage)
    for index, (label, image) in enumerate(samples):
        column = index % columns
        row = index // columns
        left = column * tile_size
        top = row * (tile_size + label_height)
        montage.paste(scaled_preview(image, tile_size), (left, top))
        draw.rectangle((left, top + tile_size, left + tile_size, top + tile_size + label_height), fill=0)
        draw.text((left + 8, top + tile_size + 9), label, fill=255)
    montage.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the prepared MICA C3-C10 FOV data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    output = parse_args().output
    overview_rows = read_csv(output / "overview_manifest.csv")
    fov_rows = read_csv(output / "fov_manifest.csv")
    errors: list[str] = []

    if len(overview_rows) != 64:
        errors.append(f"Expected 64 overview rows, found {len(overview_rows)}")
    if len(fov_rows) != 192:
        errors.append(f"Expected 192 FOV rows, found {len(fov_rows)}")

    per_provenance = Counter(row["provenance"] for row in fov_rows)
    wrong_counts = {name: count for name, count in per_provenance.items() if count != 3}
    if wrong_counts:
        errors.append(f"Expected three FOVs per overview: {wrong_counts}")

    crop_paths = [Path(row["crop_tiff"]) for row in fov_rows]
    if len(crop_paths) != len(set(crop_paths)):
        errors.append("Duplicate crop paths detected")

    for row in overview_rows:
        source = Path(row["source_lof"])
        copied = Path(row["copied_lof"])
        if not source.is_file() or not copied.is_file():
            errors.append(f"Missing source or copied LOF for {row['provenance']}")
        elif source.stat().st_size != copied.stat().st_size:
            errors.append(f"Copied LOF size mismatch for {row['provenance']}")
        for metadata_field in ("copied_xlif", "copied_xlcf"):
            if not Path(row[metadata_field]).is_file():
                errors.append(f"Missing {metadata_field} for {row['provenance']}")

    metrics: list[dict[str, object]] = []
    montage_samples: dict[tuple[int, int], tuple[str, np.ndarray]] = {}
    for index, (row, crop_path) in enumerate(zip(fov_rows, crop_paths), start=1):
        match = FILENAME_PATTERN.fullmatch(crop_path.name)
        if match is None:
            errors.append(f"Unexpected crop filename: {crop_path.name}")
            continue
        expected = {
            "provenance": row["provenance"],
            "fov": f"{int(row['fov']):02d}",
            "x": f"{int(row['x']):05d}",
            "y": f"{int(row['y']):05d}",
            "width": row["width"],
            "height": row["height"],
        }
        for field, expected_value in expected.items():
            if match.group(field) != expected_value:
                errors.append(f"Filename/manifest mismatch in {crop_path.name}: {field}")

        if not crop_path.is_file():
            errors.append(f"Missing crop: {crop_path}")
            continue
        image = tifffile.imread(crop_path)
        expected_shape = (int(row["height"]), int(row["width"]))
        if image.shape != expected_shape:
            errors.append(f"Wrong shape for {crop_path.name}: {image.shape}")
        if image.dtype != np.uint16:
            errors.append(f"Wrong dtype for {crop_path.name}: {image.dtype}")

        percentile_1, median, percentile_99 = np.percentile(image, [1, 50, 99])
        standard_deviation = float(np.std(image))
        if standard_deviation == 0 or percentile_99 <= percentile_1:
            errors.append(f"Constant or empty-looking crop: {crop_path.name}")
        metrics.append(
            {
                "filename": crop_path.name,
                "provenance": row["provenance"],
                "div": int(row["div"]),
                "plate": int(row["plate"]),
                "well": row["well"],
                "fov": int(row["fov"]),
                "minimum": int(image.min()),
                "maximum": int(image.max()),
                "mean": round(float(np.mean(image)), 4),
                "std": round(standard_deviation, 4),
                "p01": round(float(percentile_1), 4),
                "p50": round(float(median), 4),
                "p99": round(float(percentile_99), 4),
                "zero_fraction": round(float(np.mean(image == 0)), 8),
                "saturated_fraction": round(float(np.mean(image == np.iinfo(np.uint16).max)), 8),
            }
        )
        sample_key = (int(row["div"]), int(row["plate"]))
        if sample_key not in montage_samples:
            montage_samples[sample_key] = (
                f"DIV{sample_key[0]:02d} plate{sample_key[1]} {row['well']} FOV{int(row['fov']):02d}",
                image,
            )
        if index % 24 == 0:
            print(f"Checked {index}/{len(fov_rows)} TIFFs", flush=True)

    if metrics:
        write_csv(output / "fov_quality_metrics.csv", metrics)
    if montage_samples:
        create_montage(
            [montage_samples[key] for key in sorted(montage_samples)],
            output / "quality_montage.png",
        )

    div_plate_counts: dict[str, int] = defaultdict(int)
    for row in fov_rows:
        div_plate_counts[f"DIV{int(row['div']):02d}_plate{int(row['plate'])}"] += 1
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "overview_count": len(overview_rows),
        "fov_count": len(fov_rows),
        "div_plate_fov_counts": dict(sorted(div_plate_counts.items())),
        "minimum_std": min((float(row["std"]) for row in metrics), default=None),
        "maximum_saturated_fraction": max(
            (float(row["saturated_fraction"]) for row in metrics), default=None
        ),
        "errors": errors,
    }
    with (output / "quality_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
