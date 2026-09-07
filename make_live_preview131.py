from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile


PROJECT = Path(r"C:\Ole\20260721_CellClassification_v2")

NNUNET_RAW = PROJECT / "nnUNet_raw"
NNUNET_PREPROCESSED = PROJECT / "nnUNet_preprocessed"
NNUNET_RESULTS = PROJECT / "nnUNet_results"

PREVIEW_ROOT = PROJECT / "preview131"
PREVIEW_INPUT = PREVIEW_ROOT / "input"
PREVIEW_PREDICTION = PREVIEW_ROOT / "prediction"
PREVIEW_LABELS = PREVIEW_ROOT / "manual_labels"

OUTPUT_IMAGE = PREVIEW_ROOT / "preview_comparison.png"
OUTPUT_METRICS = PREVIEW_ROOT / "preview_metrics.csv"

NUMBER_OF_CASES = 100


def find_single_dataset(root: Path, dataset_id: int) -> Path:
    matches = sorted(
        path
        for path in root.glob(f"Dataset{dataset_id:03d}_*")
        if path.is_dir()
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Erwartet wurde genau ein Dataset{dataset_id:03d}, "
            f"gefunden wurden:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def find_fold_directory(dataset_name: str) -> Path:
    dataset_results = NNUNET_RESULTS / dataset_name

    candidates = [
        path
        for path in dataset_results.glob("*__*__2d/fold_0")
        if (path / "checkpoint_best.pth").exists()
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Der Modellordner für Dataset131 konnte nicht "
            "eindeutig gefunden werden:\n"
            + "\n".join(str(path) for path in candidates)
        )

    return candidates[0]


def colorize(label: np.ndarray) -> np.ndarray:
    """
    Hintergrund = schwarz
    Skeleton    = grün
    Soma        = magenta
    """
    label = np.squeeze(label)

    rgb = np.zeros(
        (label.shape[0], label.shape[1], 3),
        dtype=np.float32,
    )

    rgb[label == 1] = (0.0, 1.0, 0.0)
    rgb[label == 2] = (1.0, 0.0, 1.0)

    return rgb


def dice_for_class(
    prediction: np.ndarray,
    target: np.ndarray,
    class_id: int,
) -> float:
    prediction_mask = prediction == class_id
    target_mask = target == class_id

    denominator = (
        int(prediction_mask.sum())
        + int(target_mask.sum())
    )

    if denominator == 0:
        return 1.0

    intersection = int(
        np.logical_and(
            prediction_mask,
            target_mask,
        ).sum()
    )

    return 2.0 * intersection / denominator


def main() -> None:
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(
        NNUNET_PREPROCESSED
    )
    os.environ["nnUNet_results"] = str(
        NNUNET_RESULTS
    )

    raw_dataset = find_single_dataset(
        NNUNET_RAW,
        131,
    )

    preprocessed_dataset = find_single_dataset(
        NNUNET_PREPROCESSED,
        131,
    )

    fold_directory = find_fold_directory(
        raw_dataset.name
    )

    with open(
        raw_dataset / "dataset.json",
        "r",
        encoding="utf-8",
    ) as file:
        dataset_json = json.load(file)

    file_ending = dataset_json.get(
        "file_ending",
        ".tif",
    )

    splits_path = (
        preprocessed_dataset
        / "splits_final.json"
    )

    with open(
        splits_path,
        "r",
        encoding="utf-8",
    ) as file:
        splits = json.load(file)

    validation_cases = splits[0]["val"][
        :NUMBER_OF_CASES
    ]

    print("Ausgewählte Validierungsfälle:")
    for case_id in validation_cases:
        print(f"  {case_id}")

    # Alten Preview-Ordner entfernen
    if PREVIEW_ROOT.exists():
        shutil.rmtree(PREVIEW_ROOT)

    PREVIEW_INPUT.mkdir(parents=True)
    PREVIEW_PREDICTION.mkdir(parents=True)
    PREVIEW_LABELS.mkdir(parents=True)

    # Validierungsbilder und Labels kopieren
    for case_id in validation_cases:
        input_source = (
            raw_dataset
            / "imagesTr"
            / f"{case_id}_0000{file_ending}"
        )

        label_source = (
            raw_dataset
            / "labelsTr"
            / f"{case_id}{file_ending}"
        )

        if not input_source.exists():
            raise FileNotFoundError(input_source)

        if not label_source.exists():
            raise FileNotFoundError(label_source)

        shutil.copy2(
            input_source,
            PREVIEW_INPUT / input_source.name,
        )

        shutil.copy2(
            label_source,
            PREVIEW_LABELS / label_source.name,
        )

    # Aktuellen Best-Checkpoint einfrieren.
    # So kann das Training checkpoint_best weiter aktualisieren,
    # ohne die laufende Preview zu beeinflussen.
    checkpoint_source = (
        fold_directory
        / "checkpoint_best.pth"
    )

    checkpoint_preview = (
        fold_directory
        / "checkpoint_preview.pth"
    )

    shutil.copy2(
        checkpoint_source,
        checkpoint_preview,
    )

    print()
    print("Checkpoint für Preview kopiert:")
    print(checkpoint_preview)
    print()
    print("Starte CPU-Vorhersage ...")

    # Predictor erst importieren, nachdem die Umgebungsvariablen
    # gesetzt wurden.
    from nnunetv2.inference.predict_from_raw_data import (
        predict_entry_point,
    )

    original_argv = sys.argv.copy()

    try:
        sys.argv = [
            "nnUNetv2_predict",
            "-i",
            str(PREVIEW_INPUT),
            "-o",
            str(PREVIEW_PREDICTION),
            "-d",
            "131",
            "-c",
            "2d",
            "-f",
            "0",
            "-chk",
            "checkpoint_preview.pth",
            "-device",
            "cpu",
            "-npp",
            "1",
            "-nps",
            "1",
        ]

        predict_entry_point()

    finally:
        sys.argv = original_argv

    print()
    print("Vorhersage fertig. Erzeuge Vergleichsbild ...")

    figure, axes = plt.subplots(
        len(validation_cases),
        5,
        figsize=(
            18,
            3.6 * len(validation_cases),
        ),
        squeeze=False,
    )

    metrics = []

    for row, case_id in enumerate(validation_cases):
        input129 = np.squeeze(
            tifffile.imread(
                PREVIEW_INPUT
                / f"{case_id}_0000{file_ending}"
            )
        ).astype(np.uint8)

        prediction131 = np.squeeze(
            tifffile.imread(
                PREVIEW_PREDICTION
                / f"{case_id}{file_ending}"
            )
        ).astype(np.uint8)

        manual_label = np.squeeze(
            tifffile.imread(
                PREVIEW_LABELS
                / f"{case_id}{file_ending}"
            )
        ).astype(np.uint8)

        error_before = (
            input129 != manual_label
        ).astype(np.uint8)

        error_after = (
            prediction131 != manual_label
        ).astype(np.uint8)

        skeleton_before = dice_for_class(
            input129,
            manual_label,
            1,
        )

        skeleton_after = dice_for_class(
            prediction131,
            manual_label,
            1,
        )

        soma_before = dice_for_class(
            input129,
            manual_label,
            2,
        )

        soma_after = dice_for_class(
            prediction131,
            manual_label,
            2,
        )

        metrics.append({
            "case_id": case_id,
            "skeleton_dice_net129": skeleton_before,
            "skeleton_dice_net131": skeleton_after,
            "skeleton_change": (
                skeleton_after - skeleton_before
            ),
            "soma_dice_net129": soma_before,
            "soma_dice_net131": soma_after,
            "soma_change": (
                soma_after - soma_before
            ),
            "wrong_pixels_net129": int(
                error_before.sum()
            ),
            "wrong_pixels_net131": int(
                error_after.sum()
            ),
        })

        images = [
            colorize(input129),
            colorize(prediction131),
            colorize(manual_label),
            error_before,
            error_after,
        ]

        titles = [
            "Input: Netz 129",
            "Output: Netz 131",
            "Manuelles Label",
            "Fehler Netz 129",
            "Fehler Netz 131",
        ]

        for column, image in enumerate(images):
            axis = axes[row, column]

            if image.ndim == 2:
                axis.imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
            else:
                axis.imshow(image)

            if row == 0:
                axis.set_title(
                    titles[column]
                )

            if column == 0:
                axis.set_ylabel(
                    case_id,
                    rotation=0,
                    labelpad=30,
                )

            axis.set_xticks([])
            axis.set_yticks([])

        axes[row, 0].set_xlabel(
            f"Skeleton Dice: {skeleton_before:.3f}\n"
            f"Soma Dice: {soma_before:.3f}"
        )

        axes[row, 1].set_xlabel(
            f"Skeleton Dice: {skeleton_after:.3f}\n"
            f"Soma Dice: {soma_after:.3f}"
        )

        axes[row, 3].set_xlabel(
            f"{int(error_before.sum())} falsche Pixel"
        )

        axes[row, 4].set_xlabel(
            f"{int(error_after.sum())} falsche Pixel"
        )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_IMAGE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    with open(
        OUTPUT_METRICS,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metrics[0].keys()),
        )

        writer.writeheader()
        writer.writerows(metrics)

    mean_skeleton_before = float(
        np.mean([
            row["skeleton_dice_net129"]
            for row in metrics
        ])
    )

    mean_skeleton_after = float(
        np.mean([
            row["skeleton_dice_net131"]
            for row in metrics
        ])
    )

    mean_soma_before = float(
        np.mean([
            row["soma_dice_net129"]
            for row in metrics
        ])
    )

    mean_soma_after = float(
        np.mean([
            row["soma_dice_net131"]
            for row in metrics
        ])
    )

    print()
    print("=" * 60)
    print("PREVIEW FERTIG")
    print("=" * 60)
    print(
        f"Skeleton Dice Netz 129: "
        f"{mean_skeleton_before:.4f}"
    )
    print(
        f"Skeleton Dice Netz 131: "
        f"{mean_skeleton_after:.4f}"
    )
    print(
        f"Soma Dice Netz 129:     "
        f"{mean_soma_before:.4f}"
    )
    print(
        f"Soma Dice Netz 131:     "
        f"{mean_soma_after:.4f}"
    )
    print()
    print(f"Bild:    {OUTPUT_IMAGE}")
    print(f"Metriken:{OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
