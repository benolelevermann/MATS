from __future__ import annotations

"""Adaptive hysteresis thresholding of nnU-Net probabilities, after Du et al. 2023.

Replaces the plain ``argmax`` that nnU-Net applies to its softmax output. ``argmax``
treats every class symmetrically, which systematically disadvantages the 1-px skeleton
class: at every weak pixel it loses against background or soma, and each such loss can
sever a neurite. Hysteresis instead seeds on confident skeleton pixels and grows into
adjacent less-confident ones, so a weak stretch in the middle of an otherwise confident
neurite is kept rather than cut.

Method, following Du, Zhang, Song, Bao, Zhang, Wu, Liu: "Retinal blood vessel
segmentation by using the MS-LSDNet network and geometric skeleton reconnection
method", Computers in Biology and Medicine 153:106416, 2023, section 3.3:

  Eq. 1-4  Otsu on the probability map gives threshold T and, as a by-product, the
           between-class variance curve g(t). g(t) is roughly bell-shaped around T.
  Eq. 5-6  T+ and T- are placed so that they cut off a fraction alpha of the variance
           mass to the right of / left of T. The paper uses alpha = 1/3. The rationale
           is that values near T are ambiguous while values towards either tail are
           certain, so T+/T- bracket the ambiguous band.
  Eq. 7-8  s0 is the band between T- and T+. Every connected component of s0 that is
           8-adjacent to the confident set s+ is retained; the rest is dropped.

The thresholds are therefore derived from the data, not hand-tuned. alpha is the only
free parameter.

DELIBERATE DEVIATIONS from the paper, and why:

* The paper is binary (vessel / background). Here there are three classes. Hysteresis
  is applied to the SKELETON class only, because that is the thin class that argmax
  mistreats. Soma is taken from argmax unchanged - it reaches Dice 0.936 and has no
  connectivity problem - and soma always wins a conflict, so no soma pixel is lost.
* The paper's beta/gamma step (drop components below beta px that are further than
  gamma from the main structure) is NOT implemented. On retinal vessels that removes
  noise; here 40-90 percent of skeleton is detached from any soma, so the same filter
  would delete a large part of the real data and would improve the "attached to soma"
  ratio cosmetically without making a single cell more complete.
* The paper's thickness estimate (Eq. 11, h = area / skeleton length) is meaningless
  for a 1-px skeleton, where it always evaluates to about 1. Not implemented.

Nothing here needs a retrained network: it operates on the .npz probabilities that
nnUNetv2_predict already wrote with --save_probabilities.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from canonical_skeleton import canonicalize_skeleton

SCRIPT_VERSION = "adaptive-or-fixed-hysteresis-v3-canonical-1px-2026-09-09"
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
SKELETON, SOMA = 1, 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probabilities", type=Path, nargs="+", required=True,
                        help="One or more nnU-Net .npz probability files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0 / 3.0,
                        help="Variance fraction that separates T+/T- from Otsu's T (paper: 1/3).")
    parser.add_argument("--t-high", type=float,
                        help="Optional fixed confident seed threshold in [0,1]. Use together with --t-low.")
    parser.add_argument("--t-low", type=float,
                        help="Optional fixed weak growth threshold in [0,T_high). Use together with --t-high.")
    parser.add_argument("--min-skeleton-px", type=int, default=0,
                        help="Optional: drop skeleton components below this size AFTER hysteresis "
                             "(0 = keep everything, which is the default and what the analysis assumes).")
    parser.add_argument("--write-semantic", action="store_true",
                        help="Write the resulting 0/1/2 map per case (adds ~90 MB per 90 Mpx image).")
    parser.add_argument(
        "--canonical-1px",
        action="store_true",
        help=(
            "Topology-preserving thinning after hysteresis. The thick hysteresis band "
            "is retained as a diagnostic TIFF."
        ),
    )
    parser.add_argument(
        "--max-soma-gap-px",
        type=float,
        default=3.0,
        help=(
            "With --canonical-1px, explicitly bridge only skeleton-to-soma gaps up "
            "to this Euclidean distance before final thinning (default: 3 px)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_thresholds(
    gray: np.ndarray,
    alpha: float,
    t_high: float | None,
    t_low: float | None,
) -> tuple[int, np.ndarray, int, int, str]:
    """Return Otsu diagnostics plus either adaptive or explicitly fixed thresholds."""
    threshold, variance = otsu_with_variance_curve(gray)
    if (t_high is None) != (t_low is None):
        raise ValueError("--t-high and --t-low must be provided together.")
    if t_high is None or t_low is None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"--alpha must be in (0,1), got {alpha}.")
        high, low = bracket_thresholds(threshold, variance, alpha)
        return threshold, variance, high, low, "adaptive"
    if not 0.0 < t_high <= 1.0:
        raise ValueError(f"--t-high must be in (0,1], got {t_high}.")
    if not 0.0 <= t_low < t_high:
        raise ValueError(f"--t-low must be in [0,T_high), got {t_low} with T_high={t_high}.")
    high = int(round(t_high * 255.0))
    low = int(round(t_low * 255.0))
    if low >= high:
        raise ValueError("The effective 8-bit T_low must remain below T_high.")
    return threshold, variance, high, low, "fixed"


def otsu_with_variance_curve(gray: np.ndarray) -> tuple[int, np.ndarray]:
    """Otsu threshold T and the between-class variance curve g(t), Eq. 1-4."""
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probability = histogram / histogram.sum()
    levels = np.arange(256)

    weight_background = np.cumsum(probability)              # sum_0^t P(t)
    weight_foreground = 1.0 - weight_background             # sum_t^255 P(t)
    cumulative_mean = np.cumsum(probability * levels)
    total_mean = cumulative_mean[-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_background = cumulative_mean / weight_background                    # u0, Eq. 1
        mean_foreground = (total_mean - cumulative_mean) / weight_foreground     # u1, Eq. 2
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2  # g, Eq. 3
    variance = np.nan_to_num(variance)
    return int(np.argmax(variance)), variance                                    # T, Eq. 4


def bracket_thresholds(threshold: int, variance: np.ndarray, alpha: float) -> tuple[int, int]:
    """T+ and T- as the levels cutting off fraction alpha of variance mass, Eq. 5-6."""
    right = np.cumsum(variance[threshold:])
    high = threshold + int(np.searchsorted(right, alpha * right[-1])) if right[-1] > 0 else threshold
    left = np.cumsum(variance[: threshold + 1][::-1])
    low = threshold - int(np.searchsorted(left, alpha * left[-1])) if left[-1] > 0 else threshold
    return int(min(high, 255)), int(max(low, 0))


def hysteresis(gray: np.ndarray, high: int, low: int, forbidden: np.ndarray) -> np.ndarray:
    """Seed on >= high, grow into the [low, high) band where 8-connected, Eq. 7-8."""
    seed = (gray >= high) & ~forbidden
    weak = (gray >= low) & ~forbidden
    band = weak & ~seed                                                # s0, Eq. 7
    labels, count = ndi.label(band, structure=CONNECTIVITY_8)
    if count == 0:
        return seed
    adjacent = np.unique(labels[ndi.binary_dilation(seed, CONNECTIVITY_8) & band])
    adjacent = adjacent[adjacent > 0]
    return seed | np.isin(labels, adjacent)                            # s, Eq. 8


def measure(skeleton: np.ndarray, soma: np.ndarray) -> dict:
    total = int(skeleton.sum())
    labels, count = ndi.label(skeleton, structure=CONNECTIVITY_8)
    sizes = np.bincount(labels.ravel())[1:] if count else np.array([0])
    attached = 0
    if total and soma.any():
        joint, _ = ndi.label(skeleton | soma, structure=CONNECTIVITY_8)
        soma_components = set(np.unique(joint[soma]).tolist()) - {0}
        if soma_components:
            attached = int((np.isin(joint, list(soma_components)) & skeleton).sum())
    return {
        "skeleton_px": total,
        "components": int(count),
        "largest_component": int(sizes.max()) if count else 0,
        "attached_to_soma_pct": round(100.0 * attached / total, 2) if total else 0.0,
    }


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "probabilities" if "probabilities" in data else list(data.keys())[0]
        array = np.squeeze(np.asarray(data[key]))
    if array.ndim != 3 or array.shape[0] < 3:
        raise RuntimeError(f"Expected (C,H,W) with C>=3, got {array.shape}: {path}")
    return array


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {out}\nPass --overwrite to replace it.")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, path in enumerate(args.probabilities, start=1):
        case = path.stem
        print(f"[{index}/{len(args.probabilities)}] {case}", flush=True)
        probabilities = load_probabilities(path)

        argmax = np.argmax(probabilities, axis=0)
        soma = argmax == SOMA
        baseline = measure(argmax == SKELETON, soma)

        gray = np.clip(probabilities[SKELETON] * 255.0, 0, 255).astype(np.uint8)
        threshold, variance, high, low, mode = resolve_thresholds(
            gray, args.alpha, args.t_high, args.t_low
        )
        skeleton = hysteresis(gray, high, low, forbidden=soma)

        dropped = 0
        if args.min_skeleton_px > 0:
            labels, count = ndi.label(skeleton, structure=CONNECTIVITY_8)
            if count:
                sizes = np.bincount(labels.ravel())
                keep = np.flatnonzero(sizes >= args.min_skeleton_px)
                keep = keep[keep > 0]
                pruned = np.isin(labels, keep)
                dropped = int(skeleton.sum() - pruned.sum())
                skeleton = pruned

        hysteresis_band = skeleton.copy()
        band_metrics = measure(hysteresis_band, soma)
        canonical_report = None
        if args.canonical_1px:
            if args.max_soma_gap_px < 0:
                raise ValueError("--max-soma-gap-px must be >= 0.")
            skeleton, canonical_report = canonicalize_skeleton(
                hysteresis_band,
                soma=soma,
                max_soma_gap_px=args.max_soma_gap_px,
            )

        adaptive = measure(skeleton, soma)

        if args.write_semantic:
            if args.canonical_1px:
                tifffile.imwrite(
                    out / f"{case}_hysteresis_band.tif",
                    hysteresis_band.astype(np.uint8),
                )
            semantic = np.zeros(argmax.shape, dtype=np.uint8)
            semantic[skeleton] = SKELETON
            semantic[soma] = SOMA
            tifffile.imwrite(out / f"{case}_adaptive_hysteresis_0-1-2.tif", semantic)

        row = {
            "case": case,
            "threshold_mode": mode,
            "otsu_T": round(threshold / 255.0, 4),
            "T_high": round(high / 255.0, 4),
            "T_low": round(low / 255.0, 4),
            "pruned_px": dropped,
            "canonical_1px": bool(args.canonical_1px),
            "hysteresis_band_skeleton_px": band_metrics["skeleton_px"],
            "hysteresis_band_components": band_metrics["components"],
            "canonical_pixels_removed": (
                canonical_report.pixels_removed_by_thinning
                if canonical_report is not None
                else 0
            ),
            "canonical_soma_gap_pixels_added": (
                canonical_report.soma_gap_pixels_added
                if canonical_report is not None
                else 0
            ),
            "canonical_detached_components_after": (
                canonical_report.detached_components_after
                if canonical_report is not None
                else ""
            ),
            **{f"argmax_{k}": v for k, v in baseline.items()},
            **{f"adaptive_{k}": v for k, v in adaptive.items()},
        }
        row["delta_components"] = adaptive["components"] - baseline["components"]
        row["delta_attached_pp"] = round(adaptive["attached_to_soma_pct"] - baseline["attached_to_soma_pct"], 2)
        row["delta_skeleton_px"] = adaptive["skeleton_px"] - baseline["skeleton_px"]
        rows.append(row)

        print(f"    T={row['otsu_T']:.3f}  T+={row['T_high']:.3f}  T-={row['T_low']:.3f}"
              f"  |  Komponenten {baseline['components']:,} -> {adaptive['components']:,}"
              f"  |  am Soma {baseline['attached_to_soma_pct']:.1f}% -> {adaptive['attached_to_soma_pct']:.1f}%",
              flush=True)
        del probabilities, argmax, gray

    fields = list(rows[0].keys())
    with (out / "adaptive_hysteresis_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "run_summary.json").write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "method": (
            "Du et al. 2023 adaptive thresholds"
            if args.t_high is None
            else "fixed T_high/T_low hysteresis sensitivity test"
        ),
        "threshold_mode": "adaptive" if args.t_high is None else "fixed",
        "alpha": args.alpha if args.t_high is None else None,
        "requested_t_high": args.t_high,
        "requested_t_low": args.t_low,
        "min_skeleton_px": args.min_skeleton_px,
        "canonical_1px": bool(args.canonical_1px),
        "max_soma_gap_px": args.max_soma_gap_px if args.canonical_1px else None,
        "cases": len(rows),
        "rows": rows,
    }, indent=2), encoding="utf-8")

    improved = sum(1 for r in rows if r["delta_components"] < 0)
    better_attached = sum(1 for r in rows if r["delta_attached_pp"] > 0)
    print("\n" + "=" * 78)
    print(f"Komponentenzahl gesunken bei : {improved}/{len(rows)} Bildern")
    print(f"Soma-Anbindung gestiegen bei : {better_attached}/{len(rows)} Bildern")
    print(f"Median Delta Anbindung       : {np.median([r['delta_attached_pp'] for r in rows]):+.1f} Prozentpunkte")
    print(f"geschrieben nach             : {out}")


if __name__ == "__main__":
    main()
