from __future__ import annotations

"""Build a conservative neurite-evidence map from an overview image.

The map is deliberately *evidence only*: it contains no segmentation and never
changes class-1 skeleton material by itself. It can be supplied to
``postprocess_net129_no_loss.py --ridge-evidence`` to make candidate paths that
follow faint, line-like signal cheaper than paths through background.

The implementation uses tiled multi-scale Frangi/Hessian ridge filtering. Tiling
keeps the memory use practical for the 10k--20k pixel overview images used in
this project. The overlap must be at least a few times the largest scale.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import frangi


SCRIPT_VERSION = "neurite-ridge-evidence-v1-2026-08-12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a tiled multi-scale ridge/Hessian evidence map for "
            "conservative skeleton-gap routing."
        )
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--polarity", choices=("auto", "bright", "dark"), default="auto",
        help="Whether neurites are brighter or darker than their local background.",
    )
    parser.add_argument(
        "--sigma-min", type=float, default=0.8,
        help="Smallest ridge scale in pixels (default: 0.8).",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=4.0,
        help="Largest ridge scale in pixels (default: 4.0).",
    )
    parser.add_argument(
        "--sigma-steps", type=int, default=5,
        help="Number of scales between sigma-min and sigma-max (default: 5).",
    )
    parser.add_argument(
        "--background-sigma", type=float, default=18.0,
        help="Gaussian scale used for local background correction (default: 18).",
    )
    parser.add_argument(
        "--line-signal-weight", type=float, default=0.35,
        help="Weight of local bright/dark line signal in the final evidence (default: 0.35).",
    )
    parser.add_argument(
        "--ridge-weight", type=float, default=0.65,
        help="Weight of Frangi ridge evidence in the final evidence (default: 0.65).",
    )
    parser.add_argument(
        "--tile-size", type=int, default=2048,
        help="Core tile side length in pixels (default: 2048).",
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=96,
        help="Context overlap around every tile in pixels (default: 96).",
    )
    parser.add_argument(
        "--save-intermediates", action="store_true",
        help="Also write local-line-support and ridge-only TIFF files.",
    )
    parser.add_argument(
        "--keep-working-files", action="store_true",
        help="Keep temporary tiled arrays for debugging (normally removed).",
    )
    parser.add_argument("--qc-max-size", type=int, default=2200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_2d(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Original image not found: {path}")
    image = np.squeeze(np.asarray(tifffile.imread(path)))
    if image.ndim != 2:
        raise RuntimeError(f"Original image must be 2-D after squeeze, got {image.shape}")
    if not np.isfinite(image).any():
        raise RuntimeError("Original image contains no finite values.")
    return image


def prepare_output(directory: Path, overwrite: bool) -> None:
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {directory}\n"
            "Use a new test directory or pass --overwrite deliberately."
        )
    directory.mkdir(parents=True, exist_ok=True)


def sample_percentiles(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    stride = max(1, finite.size // 1_000_000)
    sample = finite[::stride]
    low, high = np.percentile(sample, [1.0, 99.5])
    if high <= low:
        low, high = float(np.min(sample)), float(np.max(sample))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def choose_polarity(image: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested
    # In this project cell material is normally bright. A robust global heuristic
    # protects against choosing the opposite mode only for genuinely inverted data.
    low, high = sample_percentiles(image)
    median = float(np.median(image[np.isfinite(image)][::max(1, image.size // 500_000)]))
    return "bright" if median <= (low + high) / 2.0 else "dark"


def normalize_patch(patch: np.ndarray, low: float, high: float, polarity: str) -> np.ndarray:
    normalized = np.clip((patch.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    if polarity == "dark":
        normalized = 1.0 - normalized
    return normalized.astype(np.float32, copy=False)


def write_tiff(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(path, array, bigtiff=array.nbytes >= 2**32 - 2**25)


def save_qc(path: Path, original: np.ndarray, evidence: np.ndarray, max_size: int) -> None:
    height, width = original.shape
    stride = max(1, int(np.ceil(max(height, width) / max(max_size, 1))))
    raw = original[::stride, ::stride].astype(np.float32)
    low, high = sample_percentiles(raw)
    gray = np.clip((raw - low) / max(high - low, 1e-6), 0.0, 1.0)
    rgb = np.repeat((gray * 255.0).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    overlay = np.clip(evidence[::stride, ::stride], 0.0, 1.0)[..., None]
    cyan = np.asarray([0.0, 235.0, 255.0], dtype=np.float32)
    alpha = 0.78 * overlay
    rgb = (1.0 - alpha) * rgb + alpha * cyan
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if args.sigma_min <= 0 or args.sigma_max < args.sigma_min:
        raise ValueError("Require 0 < --sigma-min <= --sigma-max")
    if args.sigma_steps < 1 or args.background_sigma <= 0:
        raise ValueError("--sigma-steps and --background-sigma must be positive")
    if args.tile_size < 128 or args.tile_overlap < 8:
        raise ValueError("--tile-size must be >= 128 and --tile-overlap must be >= 8")
    if args.line_signal_weight < 0 or args.ridge_weight < 0:
        raise ValueError("Evidence weights must be non-negative")
    if args.line_signal_weight + args.ridge_weight <= 0:
        raise ValueError("At least one evidence weight must be positive")

    prepare_output(args.output_dir, args.overwrite)
    original = read_2d(args.original)
    height, width = original.shape
    low, high = sample_percentiles(original)
    polarity = choose_polarity(original, args.polarity)
    sigmas = np.geomspace(args.sigma_min, args.sigma_max, args.sigma_steps).astype(float)

    work_dir = args.output_dir / "_working_tiled_arrays"
    work_dir.mkdir(exist_ok=True)
    line_path = work_dir / "line_support.float32.dat"
    ridge_path = work_dir / "ridge_evidence.float32.dat"
    combined_path = work_dir / "combined_evidence.float32.dat"
    # Intermediate maps are optional. On a 15k x 15k overview each float32 map
    # costs about 0.9 GB, so avoid allocating them unless the user requested
    # their separate TIFF outputs.
    line_map = (
        np.memmap(line_path, mode="w+", dtype=np.float32, shape=original.shape)
        if args.save_intermediates else None
    )
    ridge_map = (
        np.memmap(ridge_path, mode="w+", dtype=np.float32, shape=original.shape)
        if args.save_intermediates else None
    )
    combined = np.memmap(combined_path, mode="w+", dtype=np.float32, shape=original.shape)

    total_tiles = int(np.ceil(height / args.tile_size) * np.ceil(width / args.tile_size))
    tile_index = 0
    weight_total = args.line_signal_weight + args.ridge_weight
    line_weight = args.line_signal_weight / weight_total
    ridge_weight = args.ridge_weight / weight_total

    print(f"RIDGE EVIDENCE: {SCRIPT_VERSION}")
    print(f"Image shape: {original.shape}; polarity: {polarity}; tiles: {total_tiles}")
    print("This is evidence only; it does not edit any segmentation.")

    for y0 in range(0, height, args.tile_size):
        y1 = min(height, y0 + args.tile_size)
        yp0 = max(0, y0 - args.tile_overlap)
        yp1 = min(height, y1 + args.tile_overlap)
        for x0 in range(0, width, args.tile_size):
            x1 = min(width, x0 + args.tile_size)
            xp0 = max(0, x0 - args.tile_overlap)
            xp1 = min(width, x1 + args.tile_overlap)
            patch = normalize_patch(original[yp0:yp1, xp0:xp1], low, high, polarity)

            background = ndi.gaussian_filter(patch, sigma=args.background_sigma, mode="reflect")
            # Positive local contrast is deliberately conservative: uniformly
            # bright regions (such as a soma interior) have low line evidence.
            local_line = np.clip((patch - background) / 0.12, 0.0, 1.0)
            ridge = np.asarray(
                frangi(patch, sigmas=sigmas, black_ridges=False), dtype=np.float32
            )
            ridge = np.clip(ridge, 0.0, 1.0)
            evidence = np.clip(line_weight * local_line + ridge_weight * ridge, 0.0, 1.0)

            cy0, cy1 = y0 - yp0, y1 - yp0
            cx0, cx1 = x0 - xp0, x1 - xp0
            if line_map is not None:
                line_map[y0:y1, x0:x1] = local_line[cy0:cy1, cx0:cx1]
            if ridge_map is not None:
                ridge_map[y0:y1, x0:x1] = ridge[cy0:cy1, cx0:cx1]
            combined[y0:y1, x0:x1] = evidence[cy0:cy1, cx0:cx1]
            tile_index += 1
            print(f"  tile {tile_index}/{total_tiles}: y={y0}:{y1}, x={x0}:{x1}", flush=True)

    if line_map is not None:
        line_map.flush()
    if ridge_map is not None:
        ridge_map.flush()
    combined.flush()
    # Evidence is bounded to [0, 1]; float16 halves disk use for large overview
    # images while retaining far more precision than is meaningful for routing.
    write_tiff(
        args.output_dir / "04_combined_neurite_evidence.tif",
        np.asarray(combined, dtype=np.float16),
    )
    if args.save_intermediates:
        write_tiff(args.output_dir / "02_local_line_support.tif", np.asarray(line_map, dtype=np.float16))
        write_tiff(args.output_dir / "03_multiscale_ridge_evidence.tif", np.asarray(ridge_map, dtype=np.float16))
    save_qc(args.output_dir / "05_ridge_evidence_qc.png", original, combined, args.qc_max_size)

    summary = {
        "script_version": SCRIPT_VERSION,
        "original": str(args.original.resolve()),
        "shape": [int(height), int(width)],
        "selected_polarity": polarity,
        "intensity_percentiles": {"p1": low, "p99_5": high},
        "sigmas": sigmas.tolist(),
        "tiled": {"tile_size": args.tile_size, "tile_overlap": args.tile_overlap, "tile_count": total_tiles},
        "output": str((args.output_dir / "04_combined_neurite_evidence.tif").resolve()),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "_script_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n", encoding="utf-8"
    )

    del combined
    if line_map is not None:
        del line_map
    if ridge_map is not None:
        del ridge_map
    if not args.keep_working_files:
        for temporary in (line_path, ridge_path, combined_path):
            if temporary.exists():
                temporary.unlink()
        try:
            work_dir.rmdir()
        except OSError:
            pass

    print("\nRIDGE EVIDENCE COMPLETE")
    print(f"Use for route scoring: {args.output_dir / '04_combined_neurite_evidence.tif'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
