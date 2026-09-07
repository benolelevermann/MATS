from __future__ import annotations

"""Create an HTML and CSV overview of all nnU-Net training runs.

The script scans an ``nnUNet_results`` directory. It groups all training logs
in one ``fold_*`` result directory into one run, so interrupted/resumed training
does not appear as many unrelated networks.

Outputs
-------
* ``nnunet_training_overview.html`` - readable overview with curves per run
* ``nnunet_training_runs.csv``      - one summary row per run
* ``nnunet_training_epochs.csv``    - parsed metrics per epoch
* ``nnunet_training_overview.json`` - structured version of the summary

No nnU-Net installation or GPU is needed. The result folders are read only.
"""

import argparse
import ast
import csv
import html
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "nnunet-training-overview-v1-2026-08-15"
LOG_NAME_PATTERN = "training_log_*.txt"
TIMESTAMP_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?):\s*(?P<message>.*)$"
)
EPOCH_PATTERN = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s*$")
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LEARNING_RATE_PATTERN = re.compile(r"Current learning rate:\s*(%s)" % FLOAT_PATTERN)
TRAIN_LOSS_PATTERN = re.compile(r"train_loss\s+(%s)" % FLOAT_PATTERN)
VAL_LOSS_PATTERN = re.compile(r"val_loss\s+(%s)" % FLOAT_PATTERN)
EPOCH_TIME_PATTERN = re.compile(r"Epoch time:\s*(%s)\s*s" % FLOAT_PATTERN)
EMA_PATTERN = re.compile(r"New best EMA pseudo Dice:\s*(%s)" % FLOAT_PATTERN)
MEAN_VALIDATION_DICE_PATTERN = re.compile(r"Mean Validation Dice:\s*(%s)" % FLOAT_PATTERN)
SPLIT_PATTERN = re.compile(
    r"This split has\s+(?P<train>\d+)\s+training and\s+(?P<val>\d+)\s+validation cases"
)
PATCH_SIZE_PATTERN = re.compile(r"['\"]patch_size['\"]:\s*\[([^\]]+)\]")
DEFAULT_EPOCHS = 1000


@dataclass
class EpochMetric:
    epoch: int
    order: int
    timestamp: str | None = None
    learning_rate: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    dice: list[float] = field(default_factory=list)
    epoch_seconds: float | None = None
    new_best_ema: float | None = None


@dataclass
class RunRecord:
    run_id: str
    dataset: str
    trainer: str
    plans: str
    configuration: str
    fold: str
    run_dir: str
    log_files: list[str]
    status: str
    has_checkpoint_best: bool
    has_checkpoint_final: bool
    has_checkpoint_latest: bool
    planned_epochs: int | None
    last_epoch: int | None
    epochs_with_metrics: int
    training_cases: int | None
    validation_cases: int | None
    dataset_cases: int | None
    batch_size: int | None
    patch_size: str
    gpu: str
    loss: str
    skeleton_recall_weight: float | None
    skeleton_tube_radius: float | None
    skeleton_recall_class_weights: str
    best_ema_pseudo_dice: float | None
    final_mean_validation_dice: float | None
    last_learning_rate: float | None
    last_train_loss: float | None
    last_val_loss: float | None
    last_skeleton_dice: float | None
    last_soma_dice: float | None
    best_skeleton_dice: float | None
    best_soma_dice: float | None
    last_epoch_seconds: float | None
    mean_last_20_epoch_seconds: float | None
    classes: str
    warning: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Report directory. Existing report files with the same names are replaced.",
    )
    parser.add_argument(
        "--title",
        default="nnU-Net Training Overview",
        help="Title shown in the HTML report.",
    )
    parser.add_argument(
        "--recent-epochs",
        type=int,
        default=12,
        help="Rows shown in each run's recent-epoch table.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    numeric = safe_float(value)
    return int(numeric) if numeric is not None else None


def clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value).replace("\r", " ").replace("\n", " ").strip() or fallback


def load_debug_json(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "debug.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parse_python_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return value if isinstance(value, dict) else {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def parse_dice(message: str) -> list[float]:
    if "Pseudo dice" not in message:
        return []
    float32_values = re.findall(r"np\.float(?:16|32|64)\(\s*(%s)\s*\)" % FLOAT_PATTERN, message)
    if float32_values:
        return [float(value) for value in float32_values]
    bracket = re.search(r"\[(.*)\]", message)
    source = bracket.group(1) if bracket else message
    numbers = re.findall(r"(?<![A-Za-z0-9_])(%s)" % FLOAT_PATTERN, source)
    return [float(value) for value in numbers]


def parse_log(log_path: Path, metrics: dict[int, EpochMetric], order_start: int) -> tuple[int, dict[str, Any]]:
    """Update epoch metrics and return the next chronological order counter."""
    current_epoch: int | None = None
    order = order_start
    metadata: dict[str, Any] = {
        "training_cases": None,
        "validation_cases": None,
        "mean_validation_dice": None,
        "best_ema": None,
    }
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return order, metadata

    for line in lines:
        timestamp_match = TIMESTAMP_PATTERN.match(line.strip())
        timestamp = timestamp_match.group("timestamp") if timestamp_match else None
        message = timestamp_match.group("message").strip() if timestamp_match else line.strip()
        epoch_match = EPOCH_PATTERN.match(message)
        if epoch_match:
            current_epoch = int(epoch_match.group("epoch"))
            order += 1
            metrics[current_epoch] = EpochMetric(
                epoch=current_epoch,
                order=order,
                timestamp=timestamp,
            )
            continue

        split_match = SPLIT_PATTERN.search(message)
        if split_match:
            metadata["training_cases"] = int(split_match.group("train"))
            metadata["validation_cases"] = int(split_match.group("val"))

        mean_dice_match = MEAN_VALIDATION_DICE_PATTERN.search(message)
        if mean_dice_match:
            metadata["mean_validation_dice"] = float(mean_dice_match.group(1))

        if current_epoch is None or current_epoch not in metrics:
            continue
        metric = metrics[current_epoch]
        if timestamp:
            metric.timestamp = timestamp
        learning_rate_match = LEARNING_RATE_PATTERN.search(message)
        if learning_rate_match:
            metric.learning_rate = float(learning_rate_match.group(1))
        train_loss_match = TRAIN_LOSS_PATTERN.search(message)
        if train_loss_match:
            metric.train_loss = float(train_loss_match.group(1))
        val_loss_match = VAL_LOSS_PATTERN.search(message)
        if val_loss_match:
            metric.val_loss = float(val_loss_match.group(1))
        epoch_time_match = EPOCH_TIME_PATTERN.search(message)
        if epoch_time_match:
            metric.epoch_seconds = float(epoch_time_match.group(1))
        dice = parse_dice(message)
        if dice:
            metric.dice = dice
        ema_match = EMA_PATTERN.search(message)
        if ema_match:
            metric.new_best_ema = float(ema_match.group(1))
            metadata["best_ema"] = max(
                value
                for value in (metadata.get("best_ema"), metric.new_best_ema)
                if value is not None
            )
    return order, metadata


def parse_parent_name(run_dir: Path) -> tuple[str, str, str]:
    """Return trainer, plans and configuration from a standard nnU-Net folder."""
    parts = run_dir.parent.name.split("__")
    trainer = parts[0] if parts else run_dir.parent.name
    plans = parts[1] if len(parts) > 1 else ""
    configuration = "__".join(parts[2:]) if len(parts) > 2 else ""
    return trainer, plans, configuration


def extract_debug_fields(debug: dict[str, Any]) -> dict[str, Any]:
    config_text = clean_text(debug.get("configuration_manager"))
    patch_match = PATCH_SIZE_PATTERN.search(config_text)
    patch_size = ""
    if patch_match:
        patch_size = " × ".join(part.strip() for part in patch_match.group(1).split(","))
    dataset_json = parse_python_dict(debug.get("dataset_json"))
    labels = dataset_json.get("labels", {}) if isinstance(dataset_json, dict) else {}
    class_entries: list[tuple[int, str]] = []
    if isinstance(labels, dict):
        for name, value in labels.items():
            if isinstance(value, int):
                class_entries.append((value, str(name)))
    class_entries.sort()
    classes = ", ".join(f"{value}={name}" for value, name in class_entries)
    class_weights = debug.get("skeleton_recall_class_weights")
    if class_weights is None:
        class_weights_text = ""
    elif isinstance(class_weights, (list, tuple)):
        class_weights_text = ":".join(str(value) for value in class_weights)
    else:
        class_weights_text = clean_text(class_weights)
    return {
        "planned_epochs": safe_int(debug.get("num_epochs")),
        "dataset_cases": safe_int(dataset_json.get("numTraining")) if dataset_json else None,
        "batch_size": safe_int(debug.get("batch_size")),
        "patch_size": patch_size,
        "gpu": clean_text(debug.get("gpu_name")),
        "loss": clean_text(debug.get("loss")),
        "skeleton_recall_weight": safe_float(debug.get("skeleton_recall_weight")),
        "skeleton_tube_radius": safe_float(debug.get("skeleton_tube_radius")),
        "skeleton_recall_class_weights": class_weights_text,
        "classes": classes,
        "debug_best_ema": safe_float(debug.get("_best_ema")),
    }


def values(metrics: Iterable[EpochMetric], attribute: str) -> list[float]:
    result = []
    for metric in metrics:
        value = getattr(metric, attribute)
        if value is not None and math.isfinite(value):
            result.append(float(value))
    return result


def dice_values(metrics: Iterable[EpochMetric], index: int) -> list[float]:
    return [
        metric.dice[index]
        for metric in metrics
        if len(metric.dice) > index and math.isfinite(metric.dice[index])
    ]


def last_available(metrics: list[EpochMetric], attribute: str) -> float | None:
    for metric in reversed(metrics):
        value = getattr(metric, attribute)
        if value is not None:
            return float(value)
    return None


def last_dice(metrics: list[EpochMetric], index: int) -> float | None:
    for metric in reversed(metrics):
        if len(metric.dice) > index:
            return float(metric.dice[index])
    return None


def latest_complete_metric(metrics: list[EpochMetric]) -> EpochMetric | None:
    candidates = [
        metric
        for metric in metrics
        if metric.train_loss is not None or metric.val_loss is not None or metric.dice
    ]
    return candidates[-1] if candidates else (metrics[-1] if metrics else None)


def status_for(run_dir: Path, planned_epochs: int | None, last_epoch: int | None) -> str:
    if (run_dir / "checkpoint_final.pth").is_file():
        return "completed"
    if last_epoch is None:
        return "no epoch metrics"
    if planned_epochs is not None and last_epoch >= planned_epochs - 1:
        return "completed log/no final checkpoint"
    if (run_dir / "checkpoint_latest.pth").is_file() or (run_dir / "checkpoint_best.pth").is_file():
        return "interrupted or in progress"
    return "partial log"


def warning_for(record: RunRecord) -> str:
    notes: list[str] = []
    if record.epochs_with_metrics == 0:
        notes.append("No epoch metrics found")
    if not record.has_checkpoint_best:
        notes.append("No checkpoint_best.pth")
    if record.last_epoch is not None and record.planned_epochs and record.last_epoch < record.planned_epochs - 1:
        notes.append("Run ended before planned epoch count")
    if record.best_skeleton_dice is None:
        notes.append("No pseudo Dice parsed")
    return "; ".join(notes)


def scan_run(run_dir: Path, results_root: Path) -> tuple[RunRecord, list[EpochMetric]]:
    dataset = run_dir.relative_to(results_root).parts[0]
    trainer, plans, configuration = parse_parent_name(run_dir)
    fold = run_dir.name
    log_paths = sorted(run_dir.glob(LOG_NAME_PATTERN), key=lambda path: (path.stat().st_mtime, path.name))
    metrics_by_epoch: dict[int, EpochMetric] = {}
    order = 0
    parsed_meta: dict[str, Any] = {}
    for log_path in log_paths:
        order, metadata = parse_log(log_path, metrics_by_epoch, order)
        for key, value in metadata.items():
            if value is not None:
                if key == "best_ema":
                    previous = parsed_meta.get(key)
                    parsed_meta[key] = max(value, previous) if previous is not None else value
                else:
                    parsed_meta[key] = value
    metrics = sorted(metrics_by_epoch.values(), key=lambda item: (item.epoch, item.order))
    debug_fields = extract_debug_fields(load_debug_json(run_dir))
    if not debug_fields["skeleton_recall_class_weights"] and "Skeleton2xSoma1x" in trainer:
        debug_fields["skeleton_recall_class_weights"] = "2.0:1.0 (inferred from trainer name)"

    last_epoch = metrics[-1].epoch if metrics else None
    latest = latest_complete_metric(metrics)
    best_ema_candidates = [
        value
        for value in (parsed_meta.get("best_ema"), debug_fields["debug_best_ema"])
        if value is not None
    ]
    record = RunRecord(
        run_id=" | ".join((dataset, trainer, configuration or "unknown-config", fold)),
        dataset=dataset,
        trainer=trainer,
        plans=plans,
        configuration=configuration,
        fold=fold,
        run_dir=str(run_dir),
        log_files=[path.name for path in log_paths],
        status=status_for(run_dir, debug_fields["planned_epochs"], last_epoch),
        has_checkpoint_best=(run_dir / "checkpoint_best.pth").is_file(),
        has_checkpoint_final=(run_dir / "checkpoint_final.pth").is_file(),
        has_checkpoint_latest=(run_dir / "checkpoint_latest.pth").is_file(),
        planned_epochs=debug_fields["planned_epochs"] or DEFAULT_EPOCHS,
        last_epoch=last_epoch,
        epochs_with_metrics=len(metrics),
        training_cases=parsed_meta.get("training_cases"),
        validation_cases=parsed_meta.get("validation_cases"),
        dataset_cases=debug_fields["dataset_cases"],
        batch_size=debug_fields["batch_size"],
        patch_size=debug_fields["patch_size"],
        gpu=debug_fields["gpu"],
        loss=debug_fields["loss"],
        skeleton_recall_weight=debug_fields["skeleton_recall_weight"],
        skeleton_tube_radius=debug_fields["skeleton_tube_radius"],
        skeleton_recall_class_weights=debug_fields["skeleton_recall_class_weights"],
        best_ema_pseudo_dice=max(best_ema_candidates) if best_ema_candidates else None,
        final_mean_validation_dice=parsed_meta.get("mean_validation_dice"),
        last_learning_rate=last_available(metrics, "learning_rate"),
        last_train_loss=last_available(metrics, "train_loss"),
        last_val_loss=last_available(metrics, "val_loss"),
        last_skeleton_dice=last_dice(metrics, 0),
        last_soma_dice=last_dice(metrics, 1),
        best_skeleton_dice=max(dice_values(metrics, 0), default=None),
        best_soma_dice=max(dice_values(metrics, 1), default=None),
        last_epoch_seconds=latest.epoch_seconds if latest else None,
        mean_last_20_epoch_seconds=(
            sum(values(metrics[-20:], "epoch_seconds")) / len(values(metrics[-20:], "epoch_seconds"))
            if values(metrics[-20:], "epoch_seconds")
            else None
        ),
        classes=debug_fields["classes"],
        warning="",
    )
    record.warning = warning_for(record)
    return record, metrics


def find_run_directories(results_root: Path) -> list[Path]:
    folders = {path.parent for path in results_root.rglob(LOG_NAME_PATTERN)}
    folders.update(path.parent for path in results_root.rglob("debug.json"))
    return sorted(
        (folder for folder in folders if folder.name.startswith("fold_")),
        key=lambda path: str(path).casefold(),
    )


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    match = re.match(r"Dataset(\d+)", dataset)
    return (int(match.group(1)) if match else 10**9, dataset.casefold())


def format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def svg_line(points: list[tuple[float, float]], width: int, height: int, padding: int) -> str:
    if len(points) < 2:
        return ""
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if min_x == max_x:
        max_x += 1
    if math.isclose(min_y, max_y):
        min_y -= 0.01
        max_y += 0.01
    coordinates = []
    for x, y in points:
        px = padding + (x - min_x) / (max_x - min_x) * (width - 2 * padding)
        py = height - padding - (y - min_y) / (max_y - min_y) * (height - 2 * padding)
        coordinates.append(f"{px:.1f},{py:.1f}")
    return " ".join(coordinates)


def plot_svg(
    metrics: list[EpochMetric],
    series: list[tuple[str, str, list[float | None]]],
    title: str,
    fixed_y_range: tuple[float, float] | None = None,
) -> str:
    width, height, padding = 490, 190, 30
    rows: list[tuple[str, str, list[tuple[float, float]]]] = []
    for label, color, y_values in series:
        points = [
            (float(metric.epoch), float(value))
            for metric, value in zip(metrics, y_values)
            if value is not None and math.isfinite(float(value))
        ]
        if points:
            rows.append((label, color, points))
    if not rows:
        return "<p class='empty'>No parsed values available.</p>"
    all_points = [point for _label, _color, points in rows for point in points]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    if fixed_y_range is not None:
        min_y, max_y = fixed_y_range
    elif math.isclose(min_y, max_y):
        min_y -= 0.01
        max_y += 0.01
    if math.isclose(min_x, max_x):
        max_x += 1
    def coordinates(points: list[tuple[float, float]]) -> str:
        result = []
        for x, y in points:
            px = padding + (x - min_x) / (max_x - min_x) * (width - 2 * padding)
            py = height - padding - (y - min_y) / (max_y - min_y) * (height - 2 * padding)
            result.append(f"{px:.1f},{py:.1f}")
        return " ".join(result)
    paths = "".join(
        f"<polyline points='{coordinates(points)}' fill='none' stroke='{color}' stroke-width='2' />"
        for _label, color, points in rows
    )
    legend = " ".join(
        f"<span><i style='background:{color}'></i>{html.escape(label)}</span>"
        for label, color, _points in rows
    )
    return (
        f"<div class='plot'><h4>{html.escape(title)}</h4>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='{padding}' y='{padding}' width='{width - 2 * padding}' height='{height - 2 * padding}' class='plot-bg'/>"
        f"<line x1='{padding}' x2='{width - padding}' y1='{height - padding}' y2='{height - padding}' class='axis'/>"
        f"<line x1='{padding}' x2='{padding}' y1='{padding}' y2='{height - padding}' class='axis'/>"
        f"<text x='{padding}' y='{padding - 8}' class='axis-label'>{min_y:.2f}–{max_y:.2f}</text>"
        f"<text x='{padding}' y='{height - 7}' class='axis-label'>Epoch {int(min_x)}–{int(max_x)}</text>"
        f"{paths}</svg><div class='legend'>{legend}</div></div>"
    )


def html_table(rows: list[RunRecord]) -> str:
    headers = [
        "Dataset", "Trainer", "Fold", "Status", "Epoch", "Best EMA",
        "Best Skeleton Dice", "Best Soma Dice", "Last train loss", "Last val loss",
        "Final validation Dice", "Checkpoints",
    ]
    output = ["<table class='summary'><thead><tr>"]
    output.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    output.append("</tr></thead><tbody>")
    for index, record in enumerate(rows, start=1):
        checkpoints = " ".join(
            name
            for name, exists in (
                ("best", record.has_checkpoint_best),
                ("final", record.has_checkpoint_final),
                ("latest", record.has_checkpoint_latest),
            )
            if exists
        ) or "—"
        output.append("<tr>")
        cells = [
            record.dataset,
            record.trainer,
            record.fold,
            f"<span class='status {html.escape(record.status).replace(' ', '-')}'>{html.escape(record.status)}</span>",
            f"{format_number(record.last_epoch)} / {format_number(record.planned_epochs)}",
            format_number(record.best_ema_pseudo_dice),
            format_number(record.best_skeleton_dice),
            format_number(record.best_soma_dice),
            format_number(record.last_train_loss),
            format_number(record.last_val_loss),
            format_number(record.final_mean_validation_dice),
            checkpoints,
        ]
        for cell in cells:
            output.append(f"<td>{cell}</td>")
        output.append("</tr>")
    output.append("</tbody></table>")
    return "".join(output)


def metric_detail_table(metrics: list[EpochMetric], count: int) -> str:
    selected = metrics[-count:]
    if not selected:
        return "<p class='empty'>No epoch metrics parsed.</p>"
    rows = ["<table class='epochs'><thead><tr><th>Epoch</th><th>LR</th><th>Train loss</th><th>Val loss</th><th>Skeleton Dice</th><th>Soma Dice</th><th>Time</th></tr></thead><tbody>"]
    for metric in selected:
        rows.append("<tr>")
        cells = (
            format_number(metric.epoch, 0),
            format_number(metric.learning_rate, 5),
            format_number(metric.train_loss),
            format_number(metric.val_loss),
            format_number(metric.dice[0] if len(metric.dice) > 0 else None),
            format_number(metric.dice[1] if len(metric.dice) > 1 else None),
            f"{format_number(metric.epoch_seconds, 2)} s" if metric.epoch_seconds is not None else "—",
        )
        rows.extend(f"<td>{html.escape(cell)}</td>" for cell in cells)
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def detail_card(record: RunRecord, metrics: list[EpochMetric], recent_epochs: int) -> str:
    dice_plot = plot_svg(
        metrics,
        [
            ("Skeleton Dice", "#19b5fe", [metric.dice[0] if len(metric.dice) > 0 else None for metric in metrics]),
            ("Soma Dice", "#e946a5", [metric.dice[1] if len(metric.dice) > 1 else None for metric in metrics]),
        ],
        "Pseudo Dice over epochs",
        fixed_y_range=(0.0, 1.0),
    )
    loss_plot = plot_svg(
        metrics,
        [
            ("Train loss", "#f59e0b", [metric.train_loss for metric in metrics]),
            ("Validation loss", "#7c3aed", [metric.val_loss for metric in metrics]),
        ],
        "Loss over epochs",
    )
    parameter_rows = [
        ("Run directory", record.run_dir),
        ("Dataset cases", format_number(record.dataset_cases)),
        ("Train / validation", f"{format_number(record.training_cases)} / {format_number(record.validation_cases)}"),
        ("Input / patch / batch", f"{record.configuration or '—'} / {record.patch_size or '—'} / {format_number(record.batch_size)}"),
        ("Loss", record.loss or "—"),
        ("Skeleton recall", (
            f"global {format_number(record.skeleton_recall_weight)}; "
            f"tube radius {format_number(record.skeleton_tube_radius)} px; "
            f"class weights {record.skeleton_recall_class_weights or 'standard/none'}"
        )),
        ("GPU", record.gpu or "—"),
        ("Classes", record.classes or "—"),
        ("Log files", ", ".join(record.log_files) or "—"),
    ]
    parameter_html = "".join(
        f"<dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd>"
        for key, value in parameter_rows
    )
    warning = (
        f"<p class='warning'>{html.escape(record.warning)}</p>"
        if record.warning
        else ""
    )
    return (
        "<details class='run'>"
        f"<summary><strong>{html.escape(record.dataset)}</strong> · {html.escape(record.trainer)} · "
        f"{html.escape(record.fold)} — Best skeleton Dice {format_number(record.best_skeleton_dice)}</summary>"
        f"{warning}<dl class='params'>{parameter_html}</dl>"
        f"<div class='plots'>{dice_plot}{loss_plot}</div>"
        f"<h4>Last {recent_epochs} parsed epochs</h4>{metric_detail_table(metrics, recent_epochs)}"
        "</details>"
    )


def make_html(title: str, records: list[RunRecord], metric_map: dict[str, list[EpochMetric]], recent_epochs: int) -> str:
    completed = sum(record.status.startswith("completed") for record in records)
    best_skeleton = max(
        (record.best_skeleton_dice for record in records if record.best_skeleton_dice is not None),
        default=None,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    cards = "".join(detail_card(record, metric_map[record.run_id], recent_epochs) for record in records)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
:root {{ --ink:#14213d; --muted:#61708a; --line:#dce4ee; --paper:#f6f8fb; --cyan:#19b5fe; --pink:#e946a5; --amber:#f59e0b; --purple:#7c3aed; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.45 Arial, sans-serif; }}
header {{ padding:32px max(32px, calc((100vw - 1480px)/2)); color:white; background:linear-gradient(125deg,#14213d,#185f86); }}
h1 {{ margin:0 0 7px; font-size:30px; }} h2 {{ margin-top:32px; }} h4 {{ margin:10px 0 8px; }} .subtitle {{ color:#d8edf8; margin:0; }}
main {{ width:min(1480px, calc(100% - 40px)); margin:24px auto 80px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:20px 0; }} .card {{ background:white; border:1px solid var(--line); border-radius:10px; padding:15px; }} .card b {{ display:block; font-size:24px; margin-top:4px; }} .card span {{ color:var(--muted); font-size:12px; }}
.note {{ background:#fffbe8; border:1px solid #f0dd91; border-radius:8px; padding:12px 15px; }}
table {{ border-collapse:collapse; width:100%; background:white; }} th,td {{ padding:9px 10px; border:1px solid var(--line); vertical-align:top; text-align:left; }} th {{ background:#ebf2f8; white-space:nowrap; }} .summary {{ font-size:12px; overflow:auto; display:block; }}
.status {{ display:inline-block; padding:3px 7px; border-radius:999px; white-space:nowrap; font-size:11px; background:#e6ecf3; }} .status.completed {{ background:#d9f5e4; color:#116534; }} .status.interrupted-or-in-progress {{ background:#fff0c8; color:#8b5b00; }}
details.run {{ background:white; border:1px solid var(--line); border-radius:10px; padding:14px; margin:14px 0; }} details summary {{ cursor:pointer; font-size:16px; }} .warning {{ color:#8b5b00; background:#fff7df; padding:8px; border-radius:6px; }}
.params {{ display:grid; grid-template-columns:170px 1fr; gap:5px 14px; margin:16px 0; }} .params dt {{ font-weight:bold; }} .params dd {{ margin:0; word-break:break-word; color:#34445f; }}
.plots {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:15px; }} .plot {{ border:1px solid var(--line); border-radius:8px; padding:8px; }} .plot svg {{ width:100%; height:auto; }} .plot-bg {{ fill:#fbfdff; stroke:#dce4ee; }} .axis {{ stroke:#aab6c5; stroke-width:1; }} .axis-label {{ font-size:10px; fill:#61708a; }} .legend span {{ margin-right:13px; font-size:12px; }} .legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }} .epochs {{ font-size:12px; }} .empty {{ color:var(--muted); }}
footer {{ color:var(--muted); margin-top:28px; font-size:12px; }}
@media(max-width:900px) {{ .cards,.plots {{ grid-template-columns:1fr; }} .params {{ grid-template-columns:1fr; gap:2px; }} .params dd {{ margin-bottom:8px; }} }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p class="subtitle">Generated {timestamp} · parsed from nnU-Net result folders · {html.escape(SCRIPT_VERSION)}</p></header>
<main>
<section class="cards"><div class="card"><span>Training runs</span><b>{len(records)}</b></div><div class="card"><span>Completed runs</span><b>{completed}</b></div><div class="card"><span>Datasets</span><b>{len(set(record.dataset for record in records))}</b></div><div class="card"><span>Highest observed Skeleton Dice</span><b>{format_number(best_skeleton)}</b></div></section>
<p class="note"><strong>Interpretation:</strong> Dice and loss are only directly comparable when task, split, labels, configuration and training duration are equivalent. “Pseudo Dice” is nnU-Net's per-epoch validation metric; “Mean Validation Dice” appears only after a completed validation export.</p>
<h2>All runs</h2>{html_table(records)}
<h2>Details and curves</h2>{cards}
<footer>Files generated alongside this report: <code>nnunet_training_runs.csv</code>, <code>nnunet_training_epochs.csv</code>, and <code>nnunet_training_overview.json</code>.</footer>
</main></body></html>"""


def csv_ready(value: Any) -> Any:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return value


def write_reports(
    output_dir: Path,
    title: str,
    records: list[RunRecord],
    metric_map: dict[str, list[EpochMetric]],
    recent_epochs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [{key: csv_ready(value) for key, value in asdict(record).items()} for record in records]
    if run_rows:
        with (output_dir / "nnunet_training_runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)
    epoch_rows: list[dict[str, Any]] = []
    for record in records:
        for metric in metric_map[record.run_id]:
            epoch_rows.append(
                {
                    "run_id": record.run_id,
                    "dataset": record.dataset,
                    "trainer": record.trainer,
                    "configuration": record.configuration,
                    "fold": record.fold,
                    "epoch": metric.epoch,
                    "timestamp": metric.timestamp or "",
                    "learning_rate": metric.learning_rate,
                    "train_loss": metric.train_loss,
                    "val_loss": metric.val_loss,
                    "skeleton_dice": metric.dice[0] if len(metric.dice) > 0 else None,
                    "soma_dice": metric.dice[1] if len(metric.dice) > 1 else None,
                    "epoch_seconds": metric.epoch_seconds,
                    "new_best_ema": metric.new_best_ema,
                }
            )
    epoch_fields = [
        "run_id", "dataset", "trainer", "configuration", "fold", "epoch", "timestamp",
        "learning_rate", "train_loss", "val_loss", "skeleton_dice", "soma_dice",
        "epoch_seconds", "new_best_ema",
    ]
    with (output_dir / "nnunet_training_epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=epoch_fields)
        writer.writeheader()
        writer.writerows(epoch_rows)
    payload = {
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": run_rows,
    }
    (output_dir / "nnunet_training_overview.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "nnunet_training_overview.html").write_text(
        make_html(title, records, metric_map, recent_epochs), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")
    if args.recent_epochs <= 0:
        raise ValueError("--recent-epochs must be positive.")
    run_dirs = find_run_directories(args.results_root)
    if not run_dirs:
        raise RuntimeError(f"No fold result folders/logs found below: {args.results_root}")
    records: list[RunRecord] = []
    metric_map: dict[str, list[EpochMetric]] = {}
    for index, run_dir in enumerate(run_dirs, start=1):
        record, metrics = scan_run(run_dir, args.results_root)
        records.append(record)
        metric_map[record.run_id] = metrics
        print(
            f"[{index}/{len(run_dirs)}] {record.dataset} | {record.trainer} | {record.fold}: "
            f"epoch={record.last_epoch}, best skeleton Dice={format_number(record.best_skeleton_dice)}, "
            f"status={record.status}"
        )
    records.sort(key=lambda item: (dataset_sort_key(item.dataset), item.trainer.casefold(), item.fold))
    write_reports(args.output_dir, args.title, records, metric_map, args.recent_epochs)
    print("\n" + "=" * 72)
    print("NNUNET TRAINING OVERVIEW COMPLETE")
    print("=" * 72)
    print(f"Runs:   {len(records)}")
    print(f"HTML:   {args.output_dir / 'nnunet_training_overview.html'}")
    print(f"CSV:    {args.output_dir / 'nnunet_training_runs.csv'}")


if __name__ == "__main__":
    main()
