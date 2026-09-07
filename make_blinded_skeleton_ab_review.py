from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


REPORT_VERSION = "blinded-skeleton-ab-review-v1-2026-08-26"
DEFAULT_DATASET = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
FOREGROUND_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class ReviewPaths:
    project_root: Path
    images_dir: Path
    labels_dir: Path
    split_file: Path
    manifest: Path
    predictions_a: Path
    predictions_b: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed, blinded A/B HTML review for one-pixel skeleton "
            "predictions. Ratings are stored in the browser and can be exported "
            "as CSV or JSON."
        )
    )
    parser.add_argument(
        "--predictions-a",
        type=Path,
        required=True,
        help="Directory containing <case_id>.tif predictions for the first model.",
    )
    parser.add_argument(
        "--predictions-b",
        type=Path,
        required=True,
        help="Directory containing <case_id>.tif predictions for the second model.",
    )
    parser.add_argument("--name-a", default=None, help="Name revealed for model A.")
    parser.add_argument("--name-b", default=None, help="Name revealed for model B.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=80)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--blind-seed", type=int, default=20260826)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace report files in an existing output directory without deleting the directory.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ReviewPaths:
    project_root = args.project_root.resolve()
    raw_dataset = project_root / "nnUNet_raw" / DEFAULT_DATASET
    preprocessed_dataset = project_root / "nnUNet_preprocessed" / DEFAULT_DATASET
    manifest = args.manifest or (
        project_root / "one_px_skeleton_review" / "panel_manifest.csv"
    )
    return ReviewPaths(
        project_root=project_root,
        images_dir=(args.images_dir or raw_dataset / "imagesTr").resolve(),
        labels_dir=(args.labels_dir or raw_dataset / "labelsTr").resolve(),
        split_file=(args.split_file or preprocessed_dataset / "splits_final.json").resolve(),
        manifest=manifest.resolve(),
        predictions_a=args.predictions_a.resolve(),
        predictions_b=args.predictions_b.resolve(),
        output_dir=args.output_dir.resolve(),
    )


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")


def read_validation_ids(split_file: Path, fold: int) -> list[str]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or fold < 0 or fold >= len(payload):
        raise ValueError(f"Fold {fold} is not present in {split_file}")
    validation = payload[fold].get("val")
    if not isinstance(validation, list) or not validation:
        raise ValueError(f"Fold {fold} has no validation cases in {split_file}")
    case_ids = [str(case_id) for case_id in validation]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Fold {fold} contains duplicate validation case IDs")
    return case_ids


def load_or_create_manifest(
    manifest: Path,
    validation_ids: list[str],
    num_cases: int,
    selection_seed: int,
    fold: int,
) -> list[str]:
    validation_set = set(validation_ids)
    if manifest.is_file():
        with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        case_ids = [str(row.get("case_id", "")).strip() for row in rows]
        if not case_ids or any(not case_id for case_id in case_ids):
            raise ValueError(f"Manifest is empty or malformed: {manifest}")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"Manifest contains duplicate case IDs: {manifest}")
        outside = [case_id for case_id in case_ids if case_id not in validation_set]
        if outside:
            shown = ", ".join(outside[:10])
            raise ValueError(
                f"Manifest contains cases outside fold-{fold} validation: {shown}"
            )
        return case_ids

    if num_cases <= 0:
        raise ValueError("--num-cases must be positive")
    if num_cases > len(validation_ids):
        raise ValueError(
            f"Requested {num_cases} cases, but fold {fold} has only "
            f"{len(validation_ids)} validation cases"
        )
    rng = random.Random(selection_seed)
    selected = rng.sample(validation_ids, num_cases)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["order", "case_id", "fold", "selection_seed"],
        )
        writer.writeheader()
        for order, case_id in enumerate(selected, start=1):
            writer.writerow(
                {
                    "order": order,
                    "case_id": case_id,
                    "fold": fold,
                    "selection_seed": selection_seed,
                }
            )
    return selected


def find_tiff(directory: Path, stem: str) -> Path:
    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing TIFF for {stem} in {directory}")


def read_2d(path: Path) -> np.ndarray:
    array = np.squeeze(tifffile.imread(path))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF, got {array.shape}: {path}")
    return np.asarray(array)


def normalize_gray(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.5])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.uint8(normalized * 255.0)


def semantic_overlay(gray: np.ndarray, label: np.ndarray) -> np.ndarray:
    base = np.repeat(np.asarray(gray)[..., None], 3, axis=2).astype(np.float32)
    colors = np.zeros_like(base)
    colors[label == 1] = (0, 228, 255)
    colors[label == 2] = (255, 38, 190)
    colors[label > 2] = (255, 175, 0)
    foreground = label > 0
    base[foreground] = 0.10 * base[foreground] + 0.90 * colors[foreground]
    return np.uint8(np.clip(base, 0, 255))


def difference_image(display_a: np.ndarray, display_b: np.ndarray) -> np.ndarray:
    a = np.asarray(display_a)
    b = np.asarray(display_b)
    result = np.full((*a.shape, 3), 20, dtype=np.uint8)

    common_skeleton = (a == 1) & (b == 1)
    only_a_skeleton = (a == 1) & (b != 1)
    only_b_skeleton = (a != 1) & (b == 1)
    common_soma = (a == 2) & (b == 2)
    only_a_soma = (a == 2) & (b != 2)
    only_b_soma = (a != 2) & (b == 2)
    other_difference = (a != b) & ~(
        only_a_skeleton
        | only_b_skeleton
        | only_a_soma
        | only_b_soma
    )

    result[common_skeleton] = (130, 145, 150)
    result[only_a_skeleton] = (255, 184, 38)
    result[only_b_skeleton] = (72, 155, 255)
    result[common_soma] = (175, 70, 150)
    result[only_a_soma] = (255, 118, 44)
    result[only_b_soma] = (94, 94, 255)
    result[other_difference] = (230, 230, 230)
    return result


def skeleton_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred = prediction == 1
    target = ground_truth == 1
    denominator = int(pred.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(pred & target) / denominator)


def union_components(label: np.ndarray) -> int:
    return int(ndi.label(label > 0, structure=FOREGROUND_STRUCTURE)[1])


def soma_attached_skeleton_fraction(label: np.ndarray) -> float:
    skeleton = label == 1
    skeleton_count = int(skeleton.sum())
    if skeleton_count == 0:
        return 0.0
    components, _ = ndi.label(label > 0, structure=FOREGROUND_STRUCTURE)
    soma_component_ids = np.unique(components[label == 2])
    soma_component_ids = soma_component_ids[soma_component_ids > 0]
    if soma_component_ids.size == 0:
        return 0.0
    connected = skeleton & np.isin(components, soma_component_ids)
    return float(connected.sum() / skeleton_count)


def prediction_statistics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, object]:
    return {
        "skeleton_dice": round(skeleton_dice(prediction, ground_truth), 5),
        "skeleton_pixels": int(np.count_nonzero(prediction == 1)),
        "soma_pixels": int(np.count_nonzero(prediction == 2)),
        "union_components_8": union_components(prediction),
        "soma_attached_skeleton_fraction": round(
            soma_attached_skeleton_fraction(prediction), 5
        ),
    }


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array)).save(path, optimize=True)


def safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned or "comparison"


def should_flip(case_id: str, blind_seed: int, model_a: str, model_b: str) -> bool:
    payload = f"{blind_seed}|{model_a}|{model_b}|{case_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return bool(digest[0] & 1)


def validate_output(output_dir: Path, overwrite: bool) -> None:
    report_path = output_dir / "review.html"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"Report already exists: {report_path}. Use a new --output-dir or --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def relative_url(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_report(
    paths: ReviewPaths,
    case_ids: list[str],
    model_a: str,
    model_b: str,
    blind_seed: int,
    fold: int,
    selection_seed: int,
) -> dict[str, object]:
    assets_dir = paths.output_dir / "assets"
    cases: list[dict[str, object]] = []
    blind_cases: list[dict[str, str]] = []

    missing: list[str] = []
    resolved_files: list[tuple[str, Path, Path, Path, Path]] = []
    for case_id in case_ids:
        try:
            image_path = find_tiff(paths.images_dir, f"{case_id}_0000")
            label_path = find_tiff(paths.labels_dir, case_id)
            prediction_a_path = find_tiff(paths.predictions_a, case_id)
            prediction_b_path = find_tiff(paths.predictions_b, case_id)
            resolved_files.append(
                (case_id, image_path, label_path, prediction_a_path, prediction_b_path)
            )
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        preview = "\n".join(f"- {message}" for message in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n- ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            "The fixed panel is incomplete. No partial report was generated:\n"
            f"{preview}{extra}"
        )

    for order, (case_id, image_path, label_path, pred_a_path, pred_b_path) in enumerate(
        resolved_files, start=1
    ):
        image = read_2d(image_path)
        ground_truth = read_2d(label_path)
        prediction_a = read_2d(pred_a_path)
        prediction_b = read_2d(pred_b_path)
        if not (
            image.shape
            == ground_truth.shape
            == prediction_a.shape
            == prediction_b.shape
        ):
            raise ValueError(
                f"Shape mismatch for {case_id}: image={image.shape}, "
                f"GT={ground_truth.shape}, model A={prediction_a.shape}, "
                f"model B={prediction_b.shape}"
            )

        flip = should_flip(case_id, blind_seed, model_a, model_b)
        display_a = prediction_b if flip else prediction_a
        display_b = prediction_a if flip else prediction_b
        display_a_source = "model_b" if flip else "model_a"
        display_b_source = "model_a" if flip else "model_b"

        gray = normalize_gray(image)
        case_assets = assets_dir / safe_identifier(case_id)
        asset_paths = {
            "original": case_assets / "original.png",
            "ground_truth": case_assets / "ground_truth_overlay.png",
            "display_a": case_assets / "prediction_a_overlay.png",
            "display_b": case_assets / "prediction_b_overlay.png",
            "difference": case_assets / "difference.png",
        }
        save_png(asset_paths["original"], gray)
        save_png(asset_paths["ground_truth"], semantic_overlay(gray, ground_truth))
        save_png(asset_paths["display_a"], semantic_overlay(gray, display_a))
        save_png(asset_paths["display_b"], semantic_overlay(gray, display_b))
        save_png(asset_paths["difference"], difference_image(display_a, display_b))

        metrics_a = prediction_statistics(prediction_a, ground_truth)
        metrics_b = prediction_statistics(prediction_b, ground_truth)
        cases.append(
            {
                "order": order,
                "case_id": case_id,
                "shape": list(image.shape),
                "assets": {
                    key: relative_url(path, paths.output_dir)
                    for key, path in asset_paths.items()
                },
                "display_a_source": display_a_source,
                "display_b_source": display_b_source,
                "metrics": {"model_a": metrics_a, "model_b": metrics_b},
            }
        )
        blind_cases.append(
            {
                "case_id": case_id,
                "display_a": model_b if flip else model_a,
                "display_b": model_a if flip else model_b,
            }
        )

    identity_payload = "|".join(
        (
            model_a,
            model_b,
            str(paths.predictions_a),
            str(paths.predictions_b),
            str(paths.manifest),
            str(blind_seed),
        )
    ).encode("utf-8")
    identity_digest = hashlib.sha256(identity_payload).hexdigest()[:12]
    report_id = safe_identifier(f"{model_a}-vs-{model_b}-{identity_digest}")
    return {
        "version": REPORT_VERSION,
        "report_id": report_id,
        "dataset": DEFAULT_DATASET,
        "fold": fold,
        "selection_seed": selection_seed,
        "blind_seed": blind_seed,
        "manifest": str(paths.manifest),
        "models": {"model_a": model_a, "model_b": model_b},
        "prediction_directories": {
            "model_a": str(paths.predictions_a),
            "model_b": str(paths.predictions_b),
        },
        "cases": cases,
        "blind_key": blind_cases,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geblendeter Skeleton-A/B-Vergleich</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0d1117;
  --panel: #151b24;
  --panel-2: #1b2330;
  --line: #2c3747;
  --text: #eef3f8;
  --muted: #a8b4c3;
  --cyan: #35d7f2;
  --magenta: #ff50bd;
  --green: #47d18c;
  --amber: #ffbf47;
  --red: #ff6b6b;
  --blue: #579dff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, select, textarea { font: inherit; }
button { cursor: pointer; }
.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  border-bottom: 1px solid var(--line);
  background: rgba(13, 17, 23, .96);
  backdrop-filter: blur(12px);
}
.topbar-inner, main { width: min(1700px, calc(100% - 32px)); margin: 0 auto; }
.topbar-inner { padding: 18px 0 14px; }
.title-row, .toolbar, .progress-row, .card-head, .rating-row, .summary-head {
  display: flex;
  align-items: center;
  gap: 12px;
}
.title-row { justify-content: space-between; align-items: flex-start; }
h1 { margin: 0; font-size: clamp(22px, 3vw, 34px); letter-spacing: -.03em; }
.subtitle { margin: 5px 0 0; color: var(--muted); }
.blind-badge {
  flex: 0 0 auto;
  padding: 7px 11px;
  border: 1px solid #58461f;
  border-radius: 999px;
  background: #2a2315;
  color: #ffd782;
  font-weight: 700;
}
.blind-badge.revealed { border-color: #22593e; background: #143524; color: #8ff2bb; }
.progress-row { margin-top: 14px; }
.progress-track { flex: 1; height: 9px; overflow: hidden; border-radius: 999px; background: #252f3d; }
.progress-fill { width: 0; height: 100%; background: linear-gradient(90deg, var(--cyan), var(--green)); transition: width .2s ease; }
#progress-text { min-width: 120px; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.toolbar { margin-top: 12px; flex-wrap: wrap; }
.toolbar button, .toolbar select, .file-button {
  min-height: 38px;
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-2);
  color: var(--text);
}
.toolbar button:hover, .file-button:hover { border-color: #53657c; }
.toolbar .primary { border-color: #277699; background: #123c4e; color: #bdefff; font-weight: 700; }
.toolbar .finish { margin-left: auto; border-color: #347758; background: #163b2a; color: #a9f2ca; font-weight: 700; }
.toolbar button:disabled { cursor: not-allowed; opacity: .5; }
.file-button input { display: none; }
main { padding: 24px 0 60px; }
.instructions, .summary {
  margin-bottom: 20px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
}
.instructions p { margin: 0 0 8px; }
.instructions p:last-child { margin-bottom: 0; }
.legend { color: var(--muted); }
.legend-a { color: var(--amber); font-weight: 700; }
.legend-b { color: var(--blue); font-weight: 700; }
.cards { display: grid; gap: 20px; }
.case-card {
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
  background: var(--panel);
  scroll-margin-top: 190px;
}
.case-card.complete { border-color: #2d7956; }
.case-card.hidden { display: none; }
.card-head { justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--line); }
.case-title { font-size: 18px; font-weight: 800; }
.case-order { color: var(--muted); font-variant-numeric: tabular-nums; }
.state-pill { padding: 5px 9px; border-radius: 999px; background: #30291c; color: #ffd782; font-size: 13px; font-weight: 700; }
.complete .state-pill { background: #173e2b; color: #9bf1c1; }
.image-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(160px, 1fr));
  gap: 1px;
  background: var(--line);
}
figure { margin: 0; min-width: 0; background: #090c10; }
figcaption { padding: 8px 10px; background: #111721; color: var(--muted); font-weight: 700; }
figure a { display: grid; place-items: center; min-height: 240px; padding: 12px; }
figure img {
  display: block;
  width: 100%;
  max-height: 390px;
  object-fit: contain;
  image-rendering: pixelated;
}
.model-name { display: block; margin-top: 2px; color: var(--text); font-size: 12px; }
.review-body { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 18px; padding: 16px; }
.ratings { display: grid; gap: 10px; }
.rating-row { justify-content: space-between; gap: 18px; }
.rating-title { min-width: 190px; font-weight: 700; }
.choice-group { display: grid; grid-template-columns: repeat(3, minmax(92px, 1fr)); gap: 7px; width: min(100%, 470px); }
.choice-group button {
  min-height: 40px;
  padding: 7px 9px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #111721;
  color: var(--muted);
}
.choice-group button:hover { border-color: #62748c; color: var(--text); }
.choice-group button.selected-a { border-color: #c68820; background: #3e2d10; color: #ffda91; }
.choice-group button.selected-tie { border-color: #64748b; background: #273140; color: #e4eaf1; }
.choice-group button.selected-b { border-color: #397ece; background: #122f52; color: #acd3ff; }
.notes label { display: block; margin-bottom: 6px; font-weight: 700; }
.notes textarea { width: 100%; min-height: 116px; resize: vertical; padding: 10px; border: 1px solid var(--line); border-radius: 8px; background: #0f141c; color: var(--text); }
.metrics { display: none; margin-top: 12px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #10161f; color: var(--muted); font-size: 13px; }
.revealed-page .metrics { display: block; }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
.metric-grid strong { color: var(--text); }
.summary { display: none; }
.revealed-page .summary { display: block; }
.summary-head { justify-content: space-between; }
.summary h2 { margin: 0; }
.summary-grid { display: grid; grid-template-columns: repeat(5, minmax(180px, 1fr)); gap: 10px; margin-top: 14px; }
.summary-item { padding: 12px; border-radius: 9px; background: #10161f; }
.summary-item strong { display: block; margin-bottom: 6px; }
.summary-item span { display: block; color: var(--muted); }
.empty { padding: 40px; text-align: center; color: var(--muted); }
@media (max-width: 1180px) {
  .image-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
  .image-grid figure:last-child { grid-column: 1 / -1; }
  .summary-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 760px) {
  .topbar-inner, main { width: min(100% - 20px, 1700px); }
  .title-row, .rating-row { align-items: stretch; flex-direction: column; }
  .toolbar .finish { margin-left: 0; }
  .image-grid, .review-body, .summary-grid { grid-template-columns: 1fr; }
  .image-grid figure:last-child { grid-column: auto; }
  .choice-group { width: 100%; }
  figure a { min-height: 190px; }
}
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-inner">
    <div class="title-row">
      <div>
        <h1>Geblendeter Skeleton-A/B-Vergleich</h1>
        <p class="subtitle" id="report-subtitle"></p>
      </div>
      <span class="blind-badge" id="blind-badge">Modelle verborgen</span>
    </div>
    <div class="progress-row">
      <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
      <span id="progress-text">0 / 0 vollständig</span>
    </div>
    <div class="toolbar">
      <select id="filter" aria-label="Fälle filtern">
        <option value="all">Alle Fälle</option>
        <option value="open">Nur offene</option>
        <option value="complete">Nur vollständige</option>
      </select>
      <button type="button" id="next-open">Nächster offener Fall</button>
      <button type="button" id="export-csv">CSV exportieren</button>
      <button type="button" id="export-json">Sicherung exportieren</button>
      <label class="file-button">Sicherung importieren<input type="file" id="import-json" accept="application/json,.json"></label>
      <button type="button" class="finish" id="finish-review">Bewertung abschließen &amp; entblinden</button>
    </div>
  </div>
</div>

<main>
  <section class="instructions">
    <p><strong>Vorgehen:</strong> Beurteile A und B nur anhand der Bilder. Für einen vollständigen Fall müssen alle fünf Zeilen bewertet sein. Die Auswahl wird sofort lokal in diesem Browser gespeichert.</p>
    <p class="legend">Skeleton = cyan, Soma = magenta. In der Differenz: gemeinsam = grau, <span class="legend-a">nur A = orange</span>, <span class="legend-b">nur B = blau</span>. Modellnamen und technische Werte erscheinen erst nach dem Abschluss.</p>
  </section>

  <section class="summary" id="summary">
    <div class="summary-head"><h2>Entblindete Zusammenfassung</h2><span id="summary-status"></span></div>
    <div class="summary-grid" id="summary-grid"></div>
  </section>

  <section class="cards" id="cards"></section>
</main>

<script>
const REPORT = __REPORT_JSON__;
const STORAGE_KEY = `skeleton-ab-review:${REPORT.report_id}`;
const FIELDS = [
  { key: "continuity", title: "Kontinuität", labels: ["A besser", "Gleich", "B besser"] },
  { key: "false_connections", title: "Weniger Fehlverbindungen", labels: ["A weniger", "Gleich", "B weniger"] },
  { key: "soma_attachment", title: "Soma-Anschluss", labels: ["A besser", "Gleich", "B besser"] },
  { key: "one_pixel_width", title: "Saubere 1-px-Breite", labels: ["A besser", "Gleich", "B besser"] },
  { key: "overall", title: "Gesamturteil", labels: ["A besser", "Gleich", "B besser"] },
];

function freshState() {
  return { version: 1, report_id: REPORT.report_id, finalized: false, ratings: {} };
}

function loadState() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (parsed?.report_id === REPORT.report_id && parsed?.ratings) return parsed;
  } catch (_) {}
  return freshState();
}

let state = loadState();

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function ratingFor(caseId) {
  if (!state.ratings[caseId]) state.ratings[caseId] = {};
  return state.ratings[caseId];
}

function isComplete(caseId) {
  const rating = state.ratings[caseId] || {};
  return FIELDS.every((field) => ["a", "tie", "b"].includes(rating[field.key]));
}

function completedCount() {
  return REPORT.cases.filter((item) => isComplete(item.case_id)).length;
}

function modelName(source) {
  return REPORT.models[source];
}

function modelForChoice(item, choice) {
  if (choice === "tie") return "tie";
  if (choice === "a") return item.display_a_source;
  if (choice === "b") return item.display_b_source;
  return "";
}

function escapeCsv(value) {
  const text = String(value ?? "");
  return /[;"\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function download(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportJson() {
  const payload = {
    version: 1,
    report: {
      report_id: REPORT.report_id,
      dataset: REPORT.dataset,
      fold: REPORT.fold,
      manifest: REPORT.manifest,
      models: state.finalized ? REPORT.models : undefined,
    },
    state,
  };
  download(`${REPORT.report_id}-ratings.json`, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
}

function exportCsv() {
  const columns = ["order", "case_id", ...FIELDS.map((field) => field.key), "comment", "complete"];
  if (state.finalized) columns.push("display_a_model", "display_b_model", ...FIELDS.map((field) => `${field.key}_actual_winner`));
  const rows = [columns];
  for (const item of REPORT.cases) {
    const rating = state.ratings[item.case_id] || {};
    const row = [item.order, item.case_id, ...FIELDS.map((field) => rating[field.key] || ""), rating.comment || "", isComplete(item.case_id) ? "yes" : "no"];
    if (state.finalized) {
      row.push(
        modelName(item.display_a_source),
        modelName(item.display_b_source),
        ...FIELDS.map((field) => {
          const winner = modelForChoice(item, rating[field.key]);
          return winner === "tie" ? "tie" : winner ? modelName(winner) : "";
        }),
      );
    }
    rows.push(row);
  }
  const csv = "\ufeff" + rows.map((row) => row.map(escapeCsv).join(";")).join("\r\n");
  download(`${REPORT.report_id}-ratings.csv`, csv, "text/csv;charset=utf-8");
}

function imageTile(title, url, modelSource) {
  const figure = document.createElement("figure");
  const caption = document.createElement("figcaption");
  caption.textContent = title;
  if (modelSource) {
    const name = document.createElement("span");
    name.className = "model-name";
    name.textContent = state.finalized ? modelName(modelSource) : "Modell verborgen";
    caption.append(name);
  }
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  const image = document.createElement("img");
  image.src = url;
  image.alt = title;
  image.loading = "lazy";
  link.append(image);
  figure.append(caption, link);
  return figure;
}

function metricBlock(item) {
  const block = document.createElement("div");
  block.className = "metrics";
  const a = item.metrics[item.display_a_source];
  const b = item.metrics[item.display_b_source];
  const format = (value, digits = 3) => typeof value === "number" ? value.toLocaleString("de-DE", { maximumFractionDigits: digits }) : value;
  block.innerHTML = `<div class="metric-grid">
    <div><strong>A · ${modelName(item.display_a_source)}</strong><br>Dice ${format(a.skeleton_dice)} · ${format(a.skeleton_pixels, 0)} px · β₀ᴬ ${format(a.union_components_8, 0)} · Soma ${(100 * a.soma_attached_skeleton_fraction).toLocaleString("de-DE", {maximumFractionDigits: 1})}%</div>
    <div><strong>B · ${modelName(item.display_b_source)}</strong><br>Dice ${format(b.skeleton_dice)} · ${format(b.skeleton_pixels, 0)} px · β₀ᴬ ${format(b.union_components_8, 0)} · Soma ${(100 * b.soma_attached_skeleton_fraction).toLocaleString("de-DE", {maximumFractionDigits: 1})}%</div>
  </div>`;
  return block;
}

function choiceGroup(item, field) {
  const group = document.createElement("div");
  group.className = "choice-group";
  const rating = ratingFor(item.case_id);
  [["a", field.labels[0]], ["tie", field.labels[1]], ["b", field.labels[2]]].forEach(([value, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.disabled = state.finalized;
    if (rating[field.key] === value) button.classList.add(`selected-${value}`);
    button.setAttribute("aria-pressed", rating[field.key] === value ? "true" : "false");
    button.addEventListener("click", () => {
      if (state.finalized) return;
      ratingFor(item.case_id)[field.key] = value;
      ratingFor(item.case_id).updated_at = new Date().toISOString();
      persist();
      render();
    });
    group.append(button);
  });
  return group;
}

function caseCard(item) {
  const card = document.createElement("article");
  card.className = `case-card ${isComplete(item.case_id) ? "complete" : ""}`;
  card.id = `case-${item.order}`;
  card.dataset.state = isComplete(item.case_id) ? "complete" : "open";

  const head = document.createElement("div");
  head.className = "card-head";
  const title = document.createElement("div");
  title.innerHTML = `<span class="case-title">Zelle ${item.case_id}</span> <span class="case-order">${item.order} / ${REPORT.cases.length}</span>`;
  const pill = document.createElement("span");
  pill.className = "state-pill";
  pill.textContent = isComplete(item.case_id) ? "vollständig" : "offen";
  head.append(title, pill);

  const images = document.createElement("div");
  images.className = "image-grid";
  images.append(
    imageTile("Original", item.assets.original),
    imageTile("Ground Truth", item.assets.ground_truth),
    imageTile("Prediction A", item.assets.display_a, item.display_a_source),
    imageTile("Prediction B", item.assets.display_b, item.display_b_source),
    imageTile("Differenz A/B", item.assets.difference),
  );

  const reviewBody = document.createElement("div");
  reviewBody.className = "review-body";
  const ratings = document.createElement("div");
  ratings.className = "ratings";
  for (const field of FIELDS) {
    const row = document.createElement("div");
    row.className = "rating-row";
    const rowTitle = document.createElement("span");
    rowTitle.className = "rating-title";
    rowTitle.textContent = field.title;
    row.append(rowTitle, choiceGroup(item, field));
    ratings.append(row);
  }

  const notes = document.createElement("div");
  notes.className = "notes";
  const label = document.createElement("label");
  label.htmlFor = `comment-${item.order}`;
  label.textContent = "Kommentar (optional)";
  const textarea = document.createElement("textarea");
  textarea.id = `comment-${item.order}`;
  textarea.placeholder = "Zum Beispiel: Ast rechts fehlt; B verbindet fälschlich zum Nachbarn …";
  textarea.value = ratingFor(item.case_id).comment || "";
  textarea.disabled = state.finalized;
  textarea.addEventListener("input", () => {
    ratingFor(item.case_id).comment = textarea.value;
    ratingFor(item.case_id).updated_at = new Date().toISOString();
    persist();
  });
  notes.append(label, textarea, metricBlock(item));
  reviewBody.append(ratings, notes);
  card.append(head, images, reviewBody);
  return card;
}

function renderSummary() {
  const grid = document.querySelector("#summary-grid");
  grid.replaceChildren();
  if (!state.finalized) return;
  const names = REPORT.models;
  for (const field of FIELDS) {
    const counts = { model_a: 0, tie: 0, model_b: 0, open: 0 };
    for (const item of REPORT.cases) {
      const choice = state.ratings[item.case_id]?.[field.key];
      const actual = modelForChoice(item, choice);
      if (actual) counts[actual] += 1;
      else counts.open += 1;
    }
    const box = document.createElement("div");
    box.className = "summary-item";
    box.innerHTML = `<strong>${field.title}</strong>
      <span>${names.model_a}: ${counts.model_a}</span>
      <span>Gleich: ${counts.tie}</span>
      <span>${names.model_b}: ${counts.model_b}</span>
      ${counts.open ? `<span>Offen: ${counts.open}</span>` : ""}`;
    grid.append(box);
  }
  document.querySelector("#summary-status").textContent = `${completedCount()} von ${REPORT.cases.length} vollständig`;
}

function applyFilter() {
  const filter = document.querySelector("#filter").value;
  document.querySelectorAll(".case-card").forEach((card) => {
    card.classList.toggle("hidden", filter !== "all" && card.dataset.state !== filter);
  });
}

function render() {
  document.body.classList.toggle("revealed-page", state.finalized);
  const completed = completedCount();
  document.querySelector("#progress-fill").style.width = `${100 * completed / REPORT.cases.length}%`;
  document.querySelector("#progress-text").textContent = `${completed} / ${REPORT.cases.length} vollständig`;
  document.querySelector("#report-subtitle").textContent = `${REPORT.dataset} · Fold ${REPORT.fold} · feste Auswahl mit ${REPORT.cases.length} Zellen`;
  const badge = document.querySelector("#blind-badge");
  badge.textContent = state.finalized ? "Modelle entblindet" : "Modelle verborgen";
  badge.classList.toggle("revealed", state.finalized);
  const finish = document.querySelector("#finish-review");
  finish.disabled = state.finalized;
  finish.textContent = state.finalized ? "Bewertung abgeschlossen" : "Bewertung abschließen & entblinden";

  const cards = document.querySelector("#cards");
  cards.replaceChildren(...REPORT.cases.map(caseCard));
  renderSummary();
  applyFilter();
}

document.querySelector("#filter").addEventListener("change", applyFilter);
document.querySelector("#next-open").addEventListener("click", () => {
  const next = REPORT.cases.find((item) => !isComplete(item.case_id));
  if (!next) return alert("Alle Fälle sind vollständig bewertet.");
  document.querySelector("#filter").value = "all";
  applyFilter();
  document.querySelector(`#case-${next.order}`).scrollIntoView({ behavior: "smooth", block: "start" });
});
document.querySelector("#export-csv").addEventListener("click", exportCsv);
document.querySelector("#export-json").addEventListener("click", exportJson);
document.querySelector("#import-json").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    const imported = payload.state || payload;
    if (imported.report_id !== REPORT.report_id || !imported.ratings) throw new Error("Die Sicherung gehört nicht zu diesem Vergleich.");
    if (!confirm("Die lokal gespeicherten Bewertungen durch die importierte Sicherung ersetzen?")) return;
    state = imported;
    persist();
    render();
  } catch (error) {
    alert(`Import fehlgeschlagen: ${error.message}`);
  } finally {
    event.target.value = "";
  }
});
document.querySelector("#finish-review").addEventListener("click", () => {
  const completed = completedCount();
  if (completed !== REPORT.cases.length) {
    alert(`Noch ${REPORT.cases.length - completed} Fälle sind unvollständig. Nutze „Nächster offener Fall“.`);
    return;
  }
  if (!confirm("Jetzt abschließen und die Modellnamen anzeigen? Die Bewertung wird danach gesperrt, damit die Entblindung sie nicht beeinflusst.")) return;
  state.finalized = true;
  state.finalized_at = new Date().toISOString();
  persist();
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

render();
</script>
</body>
</html>
'''


def write_outputs(paths: ReviewPaths, report: dict[str, object]) -> None:
    report_for_html = dict(report)
    report_for_html.pop("blind_key")
    report_json = json.dumps(report_for_html, ensure_ascii=False).replace("</", "<\\/")
    html_text = HTML_TEMPLATE.replace("__REPORT_JSON__", report_json)
    (paths.output_dir / "review.html").write_text(html_text, encoding="utf-8")

    metadata = dict(report_for_html)
    metadata["cases"] = [
        {
            "order": case["order"],
            "case_id": case["case_id"],
            "shape": case["shape"],
            "metrics": case["metrics"],
        }
        for case in report_for_html["cases"]
    ]
    (paths.output_dir / "report_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (paths.output_dir / "blind_key.json").write_text(
        json.dumps(
            {
                "version": REPORT_VERSION,
                "report_id": report["report_id"],
                "models": report["models"],
                "cases": report["blind_key"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    for directory, label in (
        (paths.images_dir, "Image"),
        (paths.labels_dir, "Label"),
        (paths.predictions_a, "Model A prediction"),
        (paths.predictions_b, "Model B prediction"),
    ):
        require_directory(directory, label)
    validate_output(paths.output_dir, args.overwrite)

    validation_ids = read_validation_ids(paths.split_file, args.fold)
    case_ids = load_or_create_manifest(
        manifest=paths.manifest,
        validation_ids=validation_ids,
        num_cases=args.num_cases,
        selection_seed=args.selection_seed,
        fold=args.fold,
    )
    model_a = args.name_a or paths.predictions_a.parent.name or paths.predictions_a.name
    model_b = args.name_b or paths.predictions_b.parent.name or paths.predictions_b.name
    if model_a == model_b and paths.predictions_a != paths.predictions_b:
        raise ValueError("--name-a and --name-b must be distinct for different prediction folders")

    report = build_report(
        paths=paths,
        case_ids=case_ids,
        model_a=model_a,
        model_b=model_b,
        blind_seed=args.blind_seed,
        fold=args.fold,
        selection_seed=args.selection_seed,
    )
    write_outputs(paths, report)
    print(f"Fixed panel: {paths.manifest}")
    print(f"Cases: {len(case_ids)} (fold {args.fold} validation only)")
    print(f"Review website: {paths.output_dir / 'review.html'}")
    print(f"Blind key: {paths.output_dir / 'blind_key.json'}")


if __name__ == "__main__":
    main()
