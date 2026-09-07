from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


REPORT_VERSION = "skeleton-recall-ab-report-v1-2026-08-12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a side-by-side HTML comparison of the baseline Dataset136 "
            "model and nnUNetTrainerSkeletonRecallCells."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--skeleton-recall-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-max-size", type=int, default=1800)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--num-crops", type=int, default=8)
    return parser.parse_args()


def case_id_from_input(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_0000") else stem


def read_2d(path: Path) -> np.ndarray:
    try:
        arr = tifffile.memmap(path)
    except Exception:
        arr = tifffile.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected a 2D TIFF: {path} has shape {arr.shape}")
    return arr


def preview_stride(shape: tuple[int, int], max_size: int) -> int:
    return max(1, math.ceil(max(shape) / max_size))


def robust_limits(image: np.ndarray, stride: int) -> tuple[float, float]:
    sample = np.asarray(image[::stride, ::stride], dtype=np.float32)
    lo, hi = np.percentile(sample, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(sample)), float(np.max(sample))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalize_gray(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (np.asarray(image, dtype=np.float32) - lo) / (hi - lo)
    return np.uint8(np.clip(out, 0.0, 1.0) * 255.0)


def semantic_rgb(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    rgb[label == 1] = (0, 225, 255)       # skeleton: cyan
    rgb[label == 2] = (255, 0, 190)       # soma: magenta
    rgb[label > 2] = (255, 140, 0)
    return rgb


def overlay_rgb(gray: np.ndarray, label: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    base = np.repeat(np.asarray(gray)[..., None], 3, axis=2).astype(np.float32)
    colors = semantic_rgb(label).astype(np.float32)
    foreground = np.asarray(label) > 0
    base[foreground] = (
        (1.0 - alpha) * base[foreground] + alpha * colors[foreground]
    )
    return np.uint8(np.clip(base, 0, 255))


def difference_rgb(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    old = np.asarray(old)
    new = np.asarray(new)
    rgb = np.zeros((*old.shape, 3), dtype=np.uint8)

    unchanged_skeleton = (old == 1) & (new == 1)
    unchanged_soma = (old == 2) & (new == 2)
    added_skeleton = (old != 1) & (new == 1)
    removed_skeleton = (old == 1) & (new != 1)
    other_change = (old != new) & ~added_skeleton & ~removed_skeleton

    rgb[unchanged_skeleton] = (0, 115, 130)
    rgb[unchanged_soma] = (125, 0, 95)
    rgb[added_skeleton] = (255, 220, 0)   # yellow
    rgb[removed_skeleton] = (255, 40, 40) # red
    rgb[other_change] = (170, 70, 255)
    return rgb


def save_image(path: Path, array: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(array))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=quality, optimize=True)
    else:
        image.save(path, optimize=True)


def collect_statistics(
    baseline: np.ndarray,
    recall: np.ndarray,
    tile_size: int,
) -> tuple[dict[str, int], np.ndarray]:
    if baseline.shape != recall.shape:
        raise RuntimeError(
            f"Prediction shape mismatch: {baseline.shape} versus {recall.shape}"
        )

    height, width = baseline.shape
    tile_rows = math.ceil(height / tile_size)
    tile_cols = math.ceil(width / tile_size)
    tile_changes = np.zeros((tile_rows, tile_cols), dtype=np.int64)
    stats = {
        "pixels": height * width,
        "baseline_skeleton": 0,
        "recall_skeleton": 0,
        "baseline_soma": 0,
        "recall_soma": 0,
        "changed": 0,
        "added_skeleton": 0,
        "removed_skeleton": 0,
    }

    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        old_rows = np.asarray(baseline[y0:y1])
        new_rows = np.asarray(recall[y0:y1])

        stats["baseline_skeleton"] += int(np.count_nonzero(old_rows == 1))
        stats["recall_skeleton"] += int(np.count_nonzero(new_rows == 1))
        stats["baseline_soma"] += int(np.count_nonzero(old_rows == 2))
        stats["recall_soma"] += int(np.count_nonzero(new_rows == 2))
        stats["changed"] += int(np.count_nonzero(old_rows != new_rows))
        stats["added_skeleton"] += int(
            np.count_nonzero((old_rows != 1) & (new_rows == 1))
        )
        stats["removed_skeleton"] += int(
            np.count_nonzero((old_rows == 1) & (new_rows != 1))
        )

        tile_y = y0 // tile_size
        for tile_x, x0 in enumerate(range(0, width, tile_size)):
            x1 = min(width, x0 + tile_size)
            tile_changes[tile_y, tile_x] = int(
                np.count_nonzero(old_rows[:, x0:x1] != new_rows[:, x0:x1])
            )

    return stats, tile_changes


def select_change_tiles(tile_changes: np.ndarray, count: int) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for row, col in np.argwhere(tile_changes > 0):
        candidates.append((int(tile_changes[row, col]), int(row), int(col)))
    candidates.sort(reverse=True)

    selected: list[tuple[int, int, int]] = []
    for score, row, col in candidates:
        if any(abs(row - r) <= 1 and abs(col - c) <= 1 for _, r, c in selected):
            continue
        selected.append((score, row, col))
        if len(selected) >= count:
            break
    return selected


def crop_bounds(
    center_y: int,
    center_x: int,
    crop_size: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = shape
    half = crop_size // 2
    y0 = max(0, min(height - crop_size, center_y - half))
    x0 = max(0, min(width - crop_size, center_x - half))
    y1 = min(height, y0 + crop_size)
    x1 = min(width, x0 + crop_size)
    return y0, y1, x0, x1


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def process_case(
    case_id: str,
    original_path: Path,
    baseline_path: Path,
    recall_path: Path,
    assets_dir: Path,
    report_root: Path,
    preview_max_size: int,
    crop_size: int,
    num_crops: int,
) -> dict:
    print(f"Processing {case_id}...")
    original = read_2d(original_path)
    baseline = read_2d(baseline_path)
    recall = read_2d(recall_path)

    if original.shape != baseline.shape or original.shape != recall.shape:
        raise RuntimeError(
            f"Shape mismatch for {case_id}: original={original.shape}, "
            f"baseline={baseline.shape}, skeleton_recall={recall.shape}"
        )

    stride = preview_stride(original.shape, preview_max_size)
    lo, hi = robust_limits(original, stride)
    original_small = normalize_gray(original[::stride, ::stride], lo, hi)
    baseline_small = np.asarray(baseline[::stride, ::stride])
    recall_small = np.asarray(recall[::stride, ::stride])

    case_assets = assets_dir / case_id
    overview_paths = {
        "original": case_assets / "overview_original.jpg",
        "baseline_mask": case_assets / "overview_baseline_mask.png",
        "recall_mask": case_assets / "overview_skeleton_recall_mask.png",
        "baseline_overlay": case_assets / "overview_baseline_overlay.jpg",
        "recall_overlay": case_assets / "overview_skeleton_recall_overlay.jpg",
        "difference": case_assets / "overview_difference.png",
    }
    save_image(overview_paths["original"], original_small)
    save_image(overview_paths["baseline_mask"], semantic_rgb(baseline_small))
    save_image(overview_paths["recall_mask"], semantic_rgb(recall_small))
    save_image(overview_paths["baseline_overlay"], overlay_rgb(original_small, baseline_small))
    save_image(overview_paths["recall_overlay"], overlay_rgb(original_small, recall_small))
    save_image(overview_paths["difference"], difference_rgb(baseline_small, recall_small))

    stats, tile_changes = collect_statistics(baseline, recall, crop_size)
    selected = select_change_tiles(tile_changes, num_crops)
    crops: list[dict] = []

    for rank, (change_count, tile_y, tile_x) in enumerate(selected, start=1):
        center_y = tile_y * crop_size + crop_size // 2
        center_x = tile_x * crop_size + crop_size // 2
        y0, y1, x0, x1 = crop_bounds(center_y, center_x, crop_size, original.shape)
        gray = normalize_gray(original[y0:y1, x0:x1], lo, hi)
        old_crop = np.asarray(baseline[y0:y1, x0:x1])
        new_crop = np.asarray(recall[y0:y1, x0:x1])
        crop_dir = case_assets / f"crop_{rank:02d}"
        paths = {
            "original": crop_dir / "original.jpg",
            "baseline": crop_dir / "baseline_overlay.jpg",
            "recall": crop_dir / "skeleton_recall_overlay.jpg",
            "difference": crop_dir / "difference.png",
        }
        save_image(paths["original"], gray)
        save_image(paths["baseline"], overlay_rgb(gray, old_crop))
        save_image(paths["recall"], overlay_rgb(gray, new_crop))
        save_image(paths["difference"], difference_rgb(old_crop, new_crop))
        crops.append(
            {
                "rank": rank,
                "bounds": [x0, y0, x1, y1],
                "changed_pixels": change_count,
                "paths": {key: rel(value, report_root) for key, value in paths.items()},
            }
        )

    return {
        "case_id": case_id,
        "shape": list(original.shape),
        "preview_stride": stride,
        "stats": stats,
        "overview_paths": {
            key: rel(value, report_root) for key, value in overview_paths.items()
        },
        "crops": crops,
        "source_paths": {
            "original": str(original_path),
            "baseline": str(baseline_path),
            "skeleton_recall": str(recall_path),
        },
    }


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def report_html(cases: list[dict]) -> str:
    sections: list[str] = []
    for case in cases:
        stats = case["stats"]
        delta = stats["recall_skeleton"] - stats["baseline_skeleton"]
        delta_sign = "+" if delta >= 0 else ""
        overview = case["overview_paths"]
        cards = [
            ("Original", overview["original"]),
            ("Baseline mask", overview["baseline_mask"]),
            ("Skeleton-Recall mask", overview["recall_mask"]),
            ("Baseline overlay", overview["baseline_overlay"]),
            ("Skeleton-Recall overlay", overview["recall_overlay"]),
            ("Changes", overview["difference"]),
        ]
        card_html = "".join(
            f'<figure><figcaption>{html.escape(title)}</figcaption>'
            f'<a href="{html.escape(path)}" target="_blank">'
            f'<img loading="lazy" src="{html.escape(path)}" alt="{html.escape(title)}"></a></figure>'
            for title, path in cards
        )

        crop_html = ""
        for crop in case["crops"]:
            labels = [
                ("Original", crop["paths"]["original"]),
                ("Baseline", crop["paths"]["baseline"]),
                ("Skeleton Recall", crop["paths"]["recall"]),
                ("Difference", crop["paths"]["difference"]),
            ]
            crop_cards = "".join(
                f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="{path}"></figure>'
                for title, path in labels
            )
            bounds = crop["bounds"]
            crop_html += (
                f'<article class="crop"><h4>Detail {crop["rank"]}: '
                f'x={bounds[0]}–{bounds[2]}, y={bounds[1]}–{bounds[3]} '
                f'({fmt_int(crop["changed_pixels"])} changed pixels)</h4>'
                f'<div class="crop-grid">{crop_cards}</div></article>'
            )

        sections.append(
            f'<section id="{html.escape(case["case_id"])}">'
            f'<h2>{html.escape(case["case_id"])}</h2>'
            f'<p class="shape">Image size: {case["shape"][1]} × {case["shape"][0]} pixels</p>'
            '<div class="stats">'
            f'<div><strong>{fmt_int(stats["baseline_skeleton"])}</strong><span>Baseline skeleton pixels</span></div>'
            f'<div><strong>{fmt_int(stats["recall_skeleton"])}</strong><span>Skeleton-Recall pixels</span></div>'
            f'<div><strong>{delta_sign}{fmt_int(delta)}</strong><span>Net skeleton change</span></div>'
            f'<div><strong>{fmt_int(stats["added_skeleton"])}</strong><span>Added skeleton pixels</span></div>'
            f'<div><strong>{fmt_int(stats["removed_skeleton"])}</strong><span>Removed skeleton pixels</span></div>'
            f'<div><strong>{fmt_int(stats["changed"])}</strong><span>All changed labels</span></div>'
            '</div>'
            '<h3>Full overview</h3>'
            f'<div class="overview-grid">{card_html}</div>'
            '<div class="legend"><span class="cyan">■</span> Skeleton '
            '<span class="magenta">■</span> Soma &nbsp; | &nbsp; Changes: '
            '<span class="yellow">■</span> added skeleton '
            '<span class="red">■</span> removed skeleton '
            '<span class="purple">■</span> other class change</div>'
            '<h3>Highest-change detail crops</h3>'
            f'{crop_html if crop_html else "<p>No differences found.</p>"}'
            '</section>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset136 baseline vs Skeleton Recall</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#151d31; --line:#2a3655; --text:#eef3ff; --muted:#aab6d0; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 Segoe UI,Arial,sans-serif; }}
header {{ position:sticky; top:0; z-index:5; padding:18px 26px; background:#0b1020ee; border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }}
header h1 {{ margin:0 0 6px; font-size:24px; }}
header p {{ margin:0; color:var(--muted); }}
main {{ max-width:1900px; margin:auto; padding:24px; }}
section {{ margin:0 0 42px; padding:22px; background:var(--panel); border:1px solid var(--line); border-radius:14px; }}
h2 {{ margin:0; font-size:28px; }} h3 {{ margin:28px 0 12px; }} h4 {{ color:#cad5ed; }}
.shape,.legend {{ color:var(--muted); }}
.stats {{ display:grid; grid-template-columns:repeat(6,minmax(140px,1fr)); gap:10px; margin-top:18px; }}
.stats div {{ padding:14px; background:#0d1425; border:1px solid var(--line); border-radius:10px; }}
.stats strong {{ display:block; font-size:22px; }} .stats span {{ color:var(--muted); font-size:12px; }}
.overview-grid {{ display:grid; grid-template-columns:repeat(3,minmax(260px,1fr)); gap:12px; }}
.crop-grid {{ display:grid; grid-template-columns:repeat(4,minmax(180px,1fr)); gap:10px; }}
figure {{ margin:0; overflow:hidden; background:#05070c; border:1px solid var(--line); border-radius:9px; }}
figcaption {{ padding:8px 10px; background:#10182a; font-weight:600; }}
figure img {{ display:block; width:100%; height:auto; image-rendering:auto; }}
.crop {{ margin:14px 0 22px; padding-top:5px; border-top:1px solid var(--line); }}
.cyan {{ color:#00e1ff; }} .magenta {{ color:#ff00be; }} .yellow {{ color:#ffdc00; }} .red {{ color:#ff2828; }} .purple {{ color:#aa46ff; }}
code {{ color:#d8e5ff; }}
@media(max-width:1100px) {{ .stats {{ grid-template-columns:repeat(3,1fr); }} .overview-grid {{ grid-template-columns:repeat(2,1fr); }} .crop-grid {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
<header><h1>Dataset136: baseline vs Skeleton Recall</h1>
<p>Same TestBleb-v4 inputs and Fold 0; both predictions use <code>checkpoint_best.pth</code>. This is a visual A/B comparison without manual ground truth.</p></header>
<main>{''.join(sections)}</main>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    for directory in (args.input_dir, args.baseline_dir, args.skeleton_recall_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output_dir / "assets"
    input_files = sorted(
        path for path in args.input_dir.glob("*.tif") if path.stem.endswith("_0000")
    )
    if not input_files:
        raise RuntimeError(f"No *_0000.tif files found in {args.input_dir}")

    cases: list[dict] = []
    for original_path in input_files:
        case_id = case_id_from_input(original_path)
        baseline_path = args.baseline_dir / f"{case_id}.tif"
        recall_path = args.skeleton_recall_dir / f"{case_id}.tif"
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        if not recall_path.is_file():
            raise FileNotFoundError(recall_path)
        cases.append(
            process_case(
                case_id=case_id,
                original_path=original_path,
                baseline_path=baseline_path,
                recall_path=recall_path,
                assets_dir=assets_dir,
                report_root=args.output_dir,
                preview_max_size=args.preview_max_size,
                crop_size=args.crop_size,
                num_crops=args.num_crops,
            )
        )

    report_path = args.output_dir / "comparison.html"
    report_path.write_text(report_html(cases), encoding="utf-8")
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps({"version": REPORT_VERSION, "cases": cases}, indent=2),
        encoding="utf-8",
    )
    print("=" * 72)
    print("COMPARISON REPORT COMPLETE")
    print("=" * 72)
    print(f"HTML: {report_path}")
    print(f"Metrics: {args.output_dir / 'comparison_metrics.json'}")


if __name__ == "__main__":
    main()
