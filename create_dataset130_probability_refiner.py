from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile


# ============================================================
# EINSTELLUNGEN
# ============================================================

PROJECT_ROOT = Path(r"C:\Ole\20260721_CellClassification_v2")

NNUNET_RAW = PROJECT_ROOT / "nnUNet_raw"

SOURCE_DATASET_ID = 129
TARGET_DATASET_ID = 130

TARGET_DATASET_NAME = (
    f"Dataset{TARGET_DATASET_ID:03d}_SkeletonSomaProbabilityRefiner"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "refiner_generation"
    / "predictions129"
)

TARGET_DATASET_DIR = NNUNET_RAW / TARGET_DATASET_NAME
TARGET_IMAGES_TR = TARGET_DATASET_DIR / "imagesTr"
TARGET_LABELS_TR = TARGET_DATASET_DIR / "labelsTr"

REPORT_PATH = TARGET_DATASET_DIR / "dataset130_build_report.csv"

# Klassen des Modells 129
BACKGROUND_CLASS = 0
SKELETON_CLASS = 1
SOMA_CLASS = 2

# Bei True wird ein eventuell schon vorhandenes Dataset130 gelöscht.
# Nur auf True setzen, wenn du Dataset130 wirklich neu bauen willst.
RESET_TARGET_DATASET = False


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def find_source_dataset() -> Path:
    matches = sorted(
        NNUNET_RAW.glob(f"Dataset{SOURCE_DATASET_ID:03d}_*")
    )

    if not matches:
        raise FileNotFoundError(
            f"Kein Dataset{SOURCE_DATASET_ID:03d}_*-Ordner "
            f"in {NNUNET_RAW} gefunden."
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Mehrere passende Dataset129-Ordner gefunden:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def natural_case_key(case_id: str):
    if case_id.isdigit():
        return 0, int(case_id)

    return 1, case_id


def load_probabilities(
    npz_path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """
    Lädt Wahrscheinlichkeiten aus nnU-Net und bringt sie in
    die Form (Klassen, Höhe, Breite).

    Unterstützt beispielsweise:
      (3, H, W)
      (3, 1, H, W)
      (1, 3, H, W)
      (H, W, 3)
    """

    with np.load(npz_path) as archive:
        keys = list(archive.keys())

        if "probabilities" in archive:
            probabilities = np.asarray(
                archive["probabilities"]
            )
        elif "softmax" in archive:
            probabilities = np.asarray(
                archive["softmax"]
            )
        elif len(keys) == 1:
            probabilities = np.asarray(
                archive[keys[0]]
            )
        else:
            raise RuntimeError(
                f"{npz_path.name}: keine eindeutige "
                f"Wahrscheinlichkeitsmatrix gefunden. Keys: {keys}"
            )

    probabilities = np.squeeze(probabilities)

    if probabilities.ndim != 3:
        raise RuntimeError(
            f"{npz_path.name}: nach np.squeeze wurden "
            f"3 Dimensionen erwartet, gefunden: "
            f"{probabilities.shape}"
        )

    height, width = expected_shape

    # Standard: Klassen, Höhe, Breite
    if probabilities.shape[1:] == (height, width):
        probabilities_chw = probabilities

    # Alternative: Höhe, Breite, Klassen
    elif probabilities.shape[:2] == (height, width):
        probabilities_chw = np.moveaxis(
            probabilities,
            -1,
            0,
        )

    else:
        raise RuntimeError(
            f"{npz_path.name}: Probability-Shape passt nicht.\n"
            f"Probability: {probabilities.shape}\n"
            f"Label:       {expected_shape}"
        )

    if probabilities_chw.shape[0] < 3:
        raise RuntimeError(
            f"{npz_path.name}: drei Klassen erwartet "
            f"(Background, Skeleton, Soma), gefunden: "
            f"{probabilities_chw.shape[0]}"
        )

    if not np.all(np.isfinite(probabilities_chw)):
        raise RuntimeError(
            f"{npz_path.name}: enthält NaN oder unendliche Werte."
        )

    return probabilities_chw.astype(
        np.float32,
        copy=False,
    )


def prepare_target_directory() -> None:
    if TARGET_DATASET_DIR.exists():
        contains_files = any(
            TARGET_DATASET_DIR.rglob("*")
        )

        if contains_files and not RESET_TARGET_DATASET:
            raise RuntimeError(
                f"Das Ziel-Dataset existiert bereits:\n"
                f"{TARGET_DATASET_DIR}\n\n"
                f"Entferne es manuell oder setze im Script "
                f"RESET_TARGET_DATASET = True."
            )

        if RESET_TARGET_DATASET:
            print(
                f"Lösche vorhandenes Ziel-Dataset:\n"
                f"{TARGET_DATASET_DIR}"
            )
            shutil.rmtree(TARGET_DATASET_DIR)

    TARGET_IMAGES_TR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TARGET_LABELS_TR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main() -> None:
    source_dataset = find_source_dataset()

    source_images_tr = source_dataset / "imagesTr"
    source_labels_tr = source_dataset / "labelsTr"
    source_dataset_json_path = source_dataset / "dataset.json"

    if not source_images_tr.exists():
        raise FileNotFoundError(
            f"imagesTr fehlt: {source_images_tr}"
        )

    if not source_labels_tr.exists():
        raise FileNotFoundError(
            f"labelsTr fehlt: {source_labels_tr}"
        )

    if not source_dataset_json_path.exists():
        raise FileNotFoundError(
            f"dataset.json fehlt: {source_dataset_json_path}"
        )

    if not PREDICTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Vorhersageordner fehlt:\n{PREDICTIONS_DIR}"
        )

    with open(
        source_dataset_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        source_dataset_json = json.load(file)

    source_file_ending = source_dataset_json.get(
        "file_ending",
        ".tif",
    )

    source_labels_definition = source_dataset_json.get(
        "labels",
        {},
    )

    print("Source Dataset:")
    print(source_dataset)
    print()
    print("Source labels:")
    print(source_labels_definition)
    print()

    # Überprüfen, ob das Dataset die erwarteten Klassen besitzt
    expected_labels = {
        "background": BACKGROUND_CLASS,
        "skeleton": SKELETON_CLASS,
        "soma": SOMA_CLASS,
    }

    for name, value in expected_labels.items():
        if source_labels_definition.get(name) != value:
            raise RuntimeError(
                f"Unerwartete Labeldefinition für '{name}'.\n"
                f"Erwartet: {value}\n"
                f"Gefunden: {source_labels_definition.get(name)}"
            )

    image_suffix = f"_0000{source_file_ending}"

    source_image_files = sorted(
        source_images_tr.glob(
            f"*{image_suffix}"
        )
    )

    if not source_image_files:
        raise FileNotFoundError(
            f"Keine Eingabebilder in {source_images_tr} gefunden."
        )

    case_ids = [
        image_path.name[:-len(image_suffix)]
        for image_path in source_image_files
    ]

    case_ids = sorted(
        case_ids,
        key=natural_case_key,
    )

    print(f"Gefundene Dataset129-Fälle: {len(case_ids)}")

    missing_probabilities = []

    for case_id in case_ids:
        npz_path = PREDICTIONS_DIR / f"{case_id}.npz"

        if not npz_path.exists():
            missing_probabilities.append(case_id)

    if missing_probabilities:
        preview = "\n".join(
            missing_probabilities[:20]
        )

        raise FileNotFoundError(
            f"Für {len(missing_probabilities)} Fälle fehlen "
            f"die .npz-Wahrscheinlichkeiten.\n"
            f"Erste fehlende IDs:\n{preview}"
        )

    prepare_target_directory()

    report_rows = []

    for index, case_id in enumerate(case_ids, start=1):
        label_path = (
            source_labels_tr
            / f"{case_id}{source_file_ending}"
        )

        npz_path = (
            PREDICTIONS_DIR
            / f"{case_id}.npz"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Manuelles Label fehlt: {label_path}"
            )

        manual_label = np.squeeze(
            tifffile.imread(label_path)
        )

        if manual_label.ndim != 2:
            raise RuntimeError(
                f"{label_path.name}: Label ist nicht 2D: "
                f"{manual_label.shape}"
            )

        unique_values = set(
            np.unique(manual_label)
            .astype(int)
            .tolist()
        )

        invalid_values = unique_values - {0, 1, 2}

        if invalid_values:
            raise RuntimeError(
                f"{label_path.name}: unerwartete Labelwerte: "
                f"{sorted(invalid_values)}"
            )

        probabilities = load_probabilities(
            npz_path,
            expected_shape=manual_label.shape,
        )

        skeleton_probability = np.clip(
            probabilities[SKELETON_CLASS],
            0.0,
            1.0,
        ).astype(np.float32)

        soma_probability = np.clip(
            probabilities[SOMA_CLASS],
            0.0,
            1.0,
        ).astype(np.float32)

        skeleton_output_path = (
            TARGET_IMAGES_TR
            / f"{case_id}_0000.tif"
        )

        soma_output_path = (
            TARGET_IMAGES_TR
            / f"{case_id}_0001.tif"
        )

        label_output_path = (
            TARGET_LABELS_TR
            / f"{case_id}.tif"
        )

        # Kanal 0: Skeleton-Wahrscheinlichkeit
        tifffile.imwrite(
            skeleton_output_path,
            skeleton_probability,
            photometric="minisblack",
        )

        # Kanal 1: Soma-Wahrscheinlichkeit
        tifffile.imwrite(
            soma_output_path,
            soma_probability,
            photometric="minisblack",
        )

        # Ziel: vollständiges manuelles 0/1/2-Label
        tifffile.imwrite(
            label_output_path,
            manual_label.astype(np.uint8),
            photometric="minisblack",
        )

        report_rows.append({
            "case_id": case_id,
            "skeleton_probability_min": float(
                skeleton_probability.min()
            ),
            "skeleton_probability_max": float(
                skeleton_probability.max()
            ),
            "skeleton_probability_mean": float(
                skeleton_probability.mean()
            ),
            "soma_probability_min": float(
                soma_probability.min()
            ),
            "soma_probability_max": float(
                soma_probability.max()
            ),
            "soma_probability_mean": float(
                soma_probability.mean()
            ),
            "manual_skeleton_pixels": int(
                np.sum(manual_label == SKELETON_CLASS)
            ),
            "manual_soma_pixels": int(
                np.sum(manual_label == SOMA_CLASS)
            ),
        })

        print(
            f"[{index}/{len(case_ids)}] {case_id}: "
            f"Skeleton P "
            f"{skeleton_probability.min():.3f}-"
            f"{skeleton_probability.max():.3f}, "
            f"Soma P "
            f"{soma_probability.min():.3f}-"
            f"{soma_probability.max():.3f}"
        )

    target_dataset_json = {
        "channel_names": {
            "0": "nonorm",
            "1": "nonorm",
        },
        "labels": {
            "background": 0,
            "skeleton": 1,
            "soma": 2,
        },
        "numTraining": len(case_ids),
        "file_ending": ".tif",
    }

    with open(
        TARGET_DATASET_DIR / "dataset.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            target_dataset_json,
            file,
            indent=2,
        )

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = list(
            report_rows[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("=" * 70)
    print("Dataset130 wurde erfolgreich erzeugt")
    print("=" * 70)
    print(f"Fälle:       {len(case_ids)}")
    print(f"Dataset:     {TARGET_DATASET_DIR}")
    print(f"imagesTr:    {TARGET_IMAGES_TR}")
    print(f"labelsTr:    {TARGET_LABELS_TR}")
    print(f"Report:      {REPORT_PATH}")
    print()
    print("Kanal 0000 = Skeleton-Wahrscheinlichkeit")
    print("Kanal 0001 = Soma-Wahrscheinlichkeit")
    print("Label       = 0 Background, 1 Skeleton, 2 Soma")


if __name__ == "__main__":
    main()
