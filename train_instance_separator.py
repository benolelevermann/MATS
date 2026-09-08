from __future__ import annotations

"""Train only cell separation; network 141 remains unchanged."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from instance_separator import (
    MODEL_VERSION,
    SomaSeededSeparationLoss,
    SomaSeededSeparationUNet,
    SyntheticInstanceDataset,
    membership_metrics,
)


METRIC_NAMES = (
    "membership_dice",
    "membership_precision",
    "membership_recall",
    "skeleton_recall",
    "other_skeleton_false_positive_rate",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a soma-seeded separator on Dataset143 instance labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "nnUNet_raw" / "Dataset143_dataset141_plus_synthetic_multicell",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "instance_separator_results" / "seeded_debug5",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--continue", dest="resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)  # type: ignore[union-attr]
        for key in ("input", "membership", "valid", "semantic")
    }


def run_epoch(
    *,
    model: SomaSeededSeparationUNet,
    loader: DataLoader,
    criterion: SomaSeededSeparationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_totals = {name: 0.0 for name in ("loss", "positive_bce", "negative_bce", "dice_loss")}
    metric_totals = {name: 0.0 for name in METRIC_NAMES}
    samples = metric_batches = 0
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(batch["input"])
                losses = criterion(
                    output, batch["membership"], batch["valid"], batch["semantic"]
                )
            if training:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            batch_size = int(batch["input"].shape[0])
            samples += batch_size
            for name in loss_totals:
                loss_totals[name] += float(losses[name].item()) * batch_size
            if not training or batch_index % 20 == 0:
                metrics = membership_metrics(
                    output, batch["membership"], batch["valid"], batch["semantic"]
                )
                for name in METRIC_NAMES:
                    metric_totals[name] += metrics[name]
                metric_batches += 1
            if training and (batch_index == 1 or batch_index % 50 == 0):
                print(
                    f"epoch {epoch:03d} batch {batch_index:03d}/{len(loader):03d} "
                    f"loss={losses['loss'].item():.4f}",
                    flush=True,
                )
    result = {name: value / max(1, samples) for name, value in loss_totals.items()}
    result.update(
        {name: value / max(1, metric_batches) for name, value in metric_totals.items()}
    )
    result["seconds"] = time.perf_counter() - started
    return result


def make_checkpoint(
    model: SomaSeededSeparationUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_loss: float,
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "model_version": MODEL_VERSION,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.patch_size <= 0 or args.patch_size % 32:
        raise ValueError("--patch-size must be a positive multiple of 32")
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "checkpoint_latest.pth"
    best_path = output_dir / "checkpoint_best.pth"
    final_path = output_dir / "checkpoint_final.pth"
    history_path = output_dir / "training_history.csv"
    if not args.resume and (latest_path.exists() or final_path.exists()):
        raise FileExistsError(f"Training output already exists: {output_dir}")

    config: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "dataset_root": str(dataset_root),
        "epochs": args.epochs,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "base_channels": args.base_channels,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "input_channels": ["raw_zscore", "skeleton_141_hysteresis", "soma_141", "queried_soma"],
        "output": "membership_probability_for_queried_soma",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    training_dataset = SyntheticInstanceDataset(
        dataset_root, "training", patch_size=args.patch_size, augment=True, seed=args.seed
    )
    validation_dataset = SyntheticInstanceDataset(
        dataset_root, "validation", patch_size=args.patch_size, augment=False, seed=args.seed
    )
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SomaSeededSeparationUNet(args.base_channels).to(device)
    criterion = SomaSeededSeparationLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-5
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    start_epoch = 1
    best_validation_loss = float("inf")
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"No checkpoint to continue: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError("Checkpoint belongs to another separator architecture")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])

    print(
        f"Separation only: {training_dataset.scene_count} scenes / {len(training_dataset)} soma queries "
        f"for training; {validation_dataset.scene_count} / {len(validation_dataset)} for validation on {device}",
        flush=True,
    )
    fieldnames = [
        "epoch", "learning_rate", "train_loss", "val_loss",
        *[f"val_{name}" for name in METRIC_NAMES], "epoch_seconds",
    ]
    append = args.resume and history_path.is_file()
    with history_path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                model=model, loader=training_loader, criterion=criterion, device=device,
                optimizer=optimizer, scaler=scaler, epoch=epoch,
            )
            val_metrics = run_epoch(
                model=model, loader=validation_loader, criterion=criterion, device=device,
                optimizer=None, scaler=scaler, epoch=epoch,
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                **{f"val_{name}": val_metrics[name] for name in METRIC_NAMES},
                "epoch_seconds": train_metrics["seconds"] + val_metrics["seconds"],
            }
            writer.writerow(row)
            handle.flush()
            improved = val_metrics["loss"] < best_validation_loss
            if improved:
                best_validation_loss = val_metrics["loss"]
            payload = make_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_validation_loss, config
            )
            torch.save(payload, latest_path)
            if improved:
                torch.save(payload, best_path)
            print(
                f"epoch {epoch:03d}: train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} "
                f"Dice={val_metrics['membership_dice']:.1%} "
                f"Skeleton recall={val_metrics['skeleton_recall']:.1%} "
                f"wrong skeleton={val_metrics['other_skeleton_false_positive_rate']:.1%} "
                f"({row['epoch_seconds']:.1f}s)",
                flush=True,
            )
    torch.save(
        make_checkpoint(
            model, optimizer, scheduler, scaler, args.epochs, best_validation_loss, config
        ),
        final_path,
    )
    print(f"Training complete: {final_path}", flush=True)


if __name__ == "__main__":
    main()
