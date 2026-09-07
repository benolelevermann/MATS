from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


EPSILON = 1e-8
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
ONE_PIXEL_DISK = ndi.generate_binary_structure(2, 1)


def _safe_ratio(numerator: float, denominator: float, *, empty_value: float) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator / denominator)


def hard_cldice(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: int = 0,
) -> tuple[float, float, float]:
    """Compute hard clDice, optionally accepting a one-pixel localization error."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 2:
        raise ValueError("Only 2-D masks are supported")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not target.any():
        return float("nan"), float("nan"), float("nan")

    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    target_support = target
    prediction_support = prediction
    for _ in range(tolerance):
        target_support = ndi.binary_dilation(target_support, structure=ONE_PIXEL_DISK)
        prediction_support = ndi.binary_dilation(
            prediction_support, structure=ONE_PIXEL_DISK
        )

    topology_precision = _safe_ratio(
        np.count_nonzero(prediction_skeleton & target_support),
        np.count_nonzero(prediction_skeleton),
        empty_value=1.0,
    )
    topology_sensitivity = _safe_ratio(
        np.count_nonzero(target_skeleton & prediction_support),
        np.count_nonzero(target_skeleton),
        empty_value=0.0,
    )
    denominator = topology_precision + topology_sensitivity
    score = (
        2.0 * topology_precision * topology_sensitivity / denominator
        if denominator > 0
        else 0.0
    )
    return topology_precision, topology_sensitivity, float(score)


def _binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    total = int(prediction.sum() + target.sum())
    return 1.0 if total == 0 else float(2 * np.count_nonzero(prediction & target) / total)


def _component_count(mask: np.ndarray) -> int:
    return int(ndi.label(mask, structure=CONNECTIVITY_8)[1])


def _attached_skeleton_fraction(skeleton: np.ndarray, soma: np.ndarray) -> float:
    if not skeleton.any():
        return 0.0
    labels, count = ndi.label(skeleton, structure=CONNECTIVITY_8)
    soma_neighborhood = ndi.binary_dilation(soma, structure=CONNECTIVITY_8)
    touching_labels = np.unique(labels[soma_neighborhood & skeleton])
    touching_labels = touching_labels[touching_labels != 0]
    if count == 0 or touching_labels.size == 0:
        return 0.0
    attached = np.isin(labels, touching_labels)
    return float(np.count_nonzero(attached) / np.count_nonzero(skeleton))


def _skeleton_precision(prediction: np.ndarray, target: np.ndarray) -> float:
    return _safe_ratio(
        np.count_nonzero(prediction & target),
        np.count_nonzero(prediction),
        empty_value=1.0,
    )


def measure_case(prediction_labels: np.ndarray, target_labels: np.ndarray) -> dict[str, float]:
    prediction_skeleton = prediction_labels == 1
    target_skeleton = target_labels == 1
    prediction_soma = prediction_labels == 2
    target_soma = target_labels == 2

    strict_precision, strict_sensitivity, strict_score = hard_cldice(
        prediction_skeleton, target_skeleton
    )
    tolerant_precision, tolerant_sensitivity, tolerant_score = hard_cldice(
        prediction_skeleton, target_skeleton, tolerance=1
    )
    prediction_components = _component_count(prediction_skeleton)
    target_components = _component_count(target_skeleton)
    return {
        "strict_topology_precision": strict_precision,
        "strict_topology_sensitivity": strict_sensitivity,
        "strict_cldice": strict_score,
        "tolerant_topology_precision": tolerant_precision,
        "tolerant_topology_sensitivity": tolerant_sensitivity,
        "tolerant_cldice": tolerant_score,
        "skeleton_precision": _skeleton_precision(
            prediction_skeleton, target_skeleton
        ),
        "skeleton_dice": _binary_dice(prediction_skeleton, target_skeleton),
        "skeleton_components": float(prediction_components),
        "false_split_excess": float(max(0, prediction_components - target_components)),
        "attached_skeleton_fraction": _attached_skeleton_fraction(
            prediction_skeleton, prediction_soma
        ),
        "empty_skeleton": float(not prediction_skeleton.any()),
        "soma_dice": _binary_dice(prediction_soma, target_soma),
    }


def _read_labels(path: Path) -> np.ndarray:
    labels = np.squeeze(tifffile.imread(path))
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2-D label TIFF at {path}, got {labels.shape}")
    unexpected = set(np.unique(labels).tolist()) - {0, 1, 2}
    if unexpected:
        raise ValueError(f"Unexpected labels {sorted(unexpected)} in {path}")
    return labels


def _mean(rows: Iterable[dict[str, float]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def compare_directories(
    reference_dir: Path,
    baseline_dir: Path,
    candidate_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate_files = sorted(candidate_dir.glob("*.tif"))
    if not candidate_files:
        raise FileNotFoundError(f"No candidate TIFF predictions in {candidate_dir}")

    per_case: list[dict[str, object]] = []
    for candidate_path in candidate_files:
        reference_path = reference_dir / candidate_path.name
        baseline_path = baseline_dir / candidate_path.name
        if not reference_path.is_file() or not baseline_path.is_file():
            raise FileNotFoundError(
                f"Missing matching reference or baseline for {candidate_path.name}"
            )
        reference = _read_labels(reference_path)
        baseline = measure_case(_read_labels(baseline_path), reference)
        candidate = measure_case(_read_labels(candidate_path), reference)
        row: dict[str, object] = {"case": candidate_path.stem}
        for name, value in baseline.items():
            row[f"baseline_{name}"] = value
            row[f"candidate_{name}"] = candidate[name]
            row[f"delta_{name}"] = candidate[name] - value
        per_case.append(row)

    metric_names = list(measure_case(np.zeros((1, 1)), np.ones((1, 1))).keys())
    baseline_summary = {
        name: _mean(per_case, f"baseline_{name}") for name in metric_names
    }
    candidate_summary = {
        name: _mean(per_case, f"candidate_{name}") for name in metric_names
    }
    deltas = {
        name: candidate_summary[name] - baseline_summary[name] for name in metric_names
    }
    extra_empty = int(
        sum(row["candidate_empty_skeleton"] for row in per_case)
        - sum(row["baseline_empty_skeleton"] for row in per_case)
    )
    checks = {
        "strict_or_tolerant_cldice_improved": (
            deltas["strict_cldice"] > 0 or deltas["tolerant_cldice"] > 0
        ),
        "false_splits_reduced": deltas["false_split_excess"] < 0,
        "soma_attachment_not_worse": deltas["attached_skeleton_fraction"] >= 0,
        "no_additional_empty_skeletons": extra_empty <= 0,
        "skeleton_precision_drop_at_most_0.02": deltas["skeleton_precision"] >= -0.02,
        "soma_dice_drop_at_most_0.005": deltas["soma_dice"] >= -0.005,
    }
    summary: dict[str, object] = {
        "case_count": len(per_case),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta_candidate_minus_baseline": deltas,
        "additional_empty_skeleton_cases": extra_empty,
        "automatic_acceptance_checks": checks,
        "automatic_checks_pass": all(checks.values()),
        "manual_blinded_80_cell_check": (
            "pending: at least 8 net continuity wins and no increase in obvious false connections"
        ),
    }
    return per_case, summary


def _write_report(output_dir: Path, rows: list[dict[str, object]], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    baseline = summary["baseline"]
    candidate = summary["candidate"]
    delta = summary["delta_candidate_minus_baseline"]
    lines = [
        "# clDice validation comparison",
        "",
        f"Cases: {summary['case_count']}",
        "",
        "| Metric | Baseline | clDice | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in (
        "strict_cldice",
        "tolerant_cldice",
        "false_split_excess",
        "attached_skeleton_fraction",
        "skeleton_precision",
        "skeleton_dice",
        "soma_dice",
    ):
        lines.append(
            f"| {metric} | {baseline[metric]:.6f} | {candidate[metric]:.6f} | {delta[metric]:+.6f} |"
        )
    lines.extend(["", "## Automatic checks", ""])
    for name, passed in summary["automatic_acceptance_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend(["", f"Manual review: {summary['manual_blinded_80_cell_check']}", ""])
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _default_paths(project: Path) -> dict[str, Path]:
    dataset = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
    return {
        "reference": project / "nnUNet_raw" / dataset / "labelsTr",
        "baseline": project / "nnUNet_results" / dataset
        / "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x__nnUNetPlans__2d"
        / "fold_0" / "validation",
        "candidate": project / "nnUNet_results" / dataset
        / "nnUNetTrainerClDiceCellsAlpha01Debug50__nnUNetPlans__2d"
        / "fold_0" / "validation",
        "output": project / "skeleton_connectivity_runs" / "experiment06_cldice_comparison",
    }


def main() -> None:
    project = Path(__file__).resolve().parent
    defaults = _default_paths(project)
    parser = argparse.ArgumentParser(
        description="Compare raw Dataset138 baseline and clDice validation TIFFs."
    )
    parser.add_argument("--reference-dir", type=Path, default=defaults["reference"])
    parser.add_argument("--baseline-dir", type=Path, default=defaults["baseline"])
    parser.add_argument("--candidate-dir", type=Path, default=defaults["candidate"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output"])
    args = parser.parse_args()

    rows, summary = compare_directories(
        args.reference_dir, args.baseline_dir, args.candidate_dir
    )
    _write_report(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
