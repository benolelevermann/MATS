from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import tifffile


# ============================================================
# EINSTELLUNGEN
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Ole\20260721_CellClassification_v2"
)

NNUNET_RAW = PROJECT_ROOT / "nnUNet_raw"

PREDICTIONS_129 = (
    PROJECT_ROOT
    / "refiner_generation"
    / "predictions129"
)

TARGET_DATASET = (
    NNUNET_RAW
    / "Dataset131_HardOutputRefiner"
)

TARGET_IMAGES_TR = TARGET_DATASET / "imagesTr"
TARGET_LABELS_TR = TARGET_DATASET / "labelsTr"

# Nur auf True setzen, wenn Dataset131 vollständig
# gelöscht und neu erzeugt werden soll.
RESET_TARGET_DATASET = False

ALLOWED_LABELS = {0, 1, 2}


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def find_dataset129() -> Path:
    matches = sorted(
        path
        for path in NNUNET_RAW.glob("Dataset129_*")
        if path.is_dir()
    )

    if not matches:
        raise FileNotFoundError(
            f"Kein Dataset129-Ordner gefunden in:\n"
            f"{NNUNET_RAW}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Mehrere Dataset129-Ordner gefunden:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def prepare_target_directory() -> None:
    if TARGET_DATASET.exists():
        contains_files = any(TARGET_DATASET.rglob("*"))

        if contains_files and not RESET_TARGET_DATASET:
            raise RuntimeError(
                f"Dataset131 existiert bereits:\n"
                f"{TARGET_DATASET}\n\n"
                f"Lösche den Ordner manuell oder setze im "
                f"Skript RESET_TARGET_DATASET = True."
            )

        if RESET_TARGET_DATASET:
            print(
                f"Lösche vorhandenes Dataset131:\n"
                f"{TARGET_DATASET}"
            )
            shutil.rmtree(TARGET_DATASET)

    TARGET_IMAGES_TR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TARGET_LABELS_TR.mkdir(
        parents=True,
        exist_ok=True,
    )


def find_prediction(case_id: str) -> Path:
    possible_files = [
        PREDICTIONS_129 / f"{case_id}.tif",
        PREDICTIONS_129 / f"{case_id}.tiff",
        PREDICTIONS_129 / f"{case_id}.png",
    ]

    for path in possible_files:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Keine harte Vorhersage für Fall '{case_id}' "
        f"gefunden in:\n{PREDICTIONS_129}"
    )


def check_values(
    image: np.ndarray,
    name: str,
) -> None:
    unique_values = set(
        np.unique(image)
        .astype(int)
        .tolist()
    )

    invalid_values = unique_values - ALLOWED_LABELS

    if invalid_values:
        raise RuntimeError(
            f"{name} enthält unerwartete Werte: "
            f"{sorted(invalid_values)}\n"
            f"Erlaubt sind nur 0, 1 und 2."
        )


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main() -> None:
    dataset129 = find_dataset129()

    dataset_json_129_path = (
        dataset129 / "dataset.json"
    )

    labels129_dir = dataset129 / "labelsTr"

    if not dataset_json_129_path.exists():
        raise FileNotFoundError(
            f"dataset.json fehlt:\n"
            f"{dataset_json_129_path}"
        )

    if not labels129_dir.exists():
        raise FileNotFoundError(
            f"labelsTr fehlt:\n{labels129_dir}"
        )

    if not PREDICTIONS_129.exists():
        raise FileNotFoundError(
            f"Vorhersageordner fehlt:\n"
            f"{PREDICTIONS_129}"
        )

    with open(
        dataset_json_129_path,
        "r",
        encoding="utf-8",
    ) as file:
        dataset_json_129 = json.load(file)

    file_ending = dataset_json_129.get(
        "file_ending",
        ".tif",
    )

    label_files = sorted(
        labels129_dir.glob(f"*{file_ending}")
    )

    if not label_files:
        raise FileNotFoundError(
            f"Keine Labels gefunden in:\n"
            f"{labels129_dir}"
        )

    prepare_target_directory()

    number_of_cases = 0

    for index, label_path in enumerate(
        label_files,
        start=1,
    ):
        case_id = label_path.name[
            :-len(file_ending)
        ]

        prediction_path = find_prediction(case_id)

        prediction129 = np.squeeze(
            tifffile.imread(prediction_path)
        )

        manual_label = np.squeeze(
            tifffile.imread(label_path)
        )

        if prediction129.ndim != 2:
            raise RuntimeError(
                f"{prediction_path.name}: "
                f"Vorhersage ist nicht 2D, sondern "
                f"{prediction129.shape}"
            )

        if manual_label.ndim != 2:
            raise RuntimeError(
                f"{label_path.name}: "
                f"Label ist nicht 2D, sondern "
                f"{manual_label.shape}"
            )

        if prediction129.shape != manual_label.shape:
            raise RuntimeError(
                f"Shape-Unterschied bei {case_id}:\n"
                f"Vorhersage: {prediction129.shape}\n"
                f"Label:      {manual_label.shape}"
            )

        check_values(
            prediction129,
            prediction_path.name,
        )

        check_values(
            manual_label,
            label_path.name,
        )

        # Ein einzelner Input-Kanal:
        # 0 = Hintergrund
        # 1 = Skeleton
        # 2 = Soma
        input_output_path = (
            TARGET_IMAGES_TR
            / f"{case_id}_0000.tif"
        )

        label_output_path = (
            TARGET_LABELS_TR
            / f"{case_id}.tif"
        )

        tifffile.imwrite(
            input_output_path,
            prediction129.astype(np.uint8),
            photometric="minisblack",
        )

        tifffile.imwrite(
            label_output_path,
            manual_label.astype(np.uint8),
            photometric="minisblack",
        )

        skeleton_input_pixels = int(
            np.sum(prediction129 == 1)
        )

        soma_input_pixels = int(
            np.sum(prediction129 == 2)
        )

        skeleton_target_pixels = int(
            np.sum(manual_label == 1)
        )

        soma_target_pixels = int(
            np.sum(manual_label == 2)
        )

        number_of_cases += 1

        print(
            f"[{index}/{len(label_files)}] {case_id}: "
            f"Input Skeleton={skeleton_input_pixels}, "
            f"Input Soma={soma_input_pixels}, "
            f"Ziel Skeleton={skeleton_target_pixels}, "
            f"Ziel Soma={soma_target_pixels}"
        )

    dataset_json_131 = {
        "channel_names": {
            "0": "nonorm"
        },
        "labels": {
            "background": 0,
            "skeleton": 1,
            "soma": 2
        },
        "numTraining": number_of_cases,
        "file_ending": ".tif"
    }

    with open(
        TARGET_DATASET / "dataset.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset_json_131,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("Dataset131 wurde erfolgreich erzeugt")
    print("=" * 70)
    print(f"Fälle:   {number_of_cases}")
    print(f"Dataset: {TARGET_DATASET}")
    print()
    print("Input:")
    print("  0 = Hintergrund")
    print("  1 = Skeleton")
    print("  2 = Soma")
    print()
    print("Ziel:")
    print("  manuelles Label mit 0, 1 und 2")


if __name__ == "__main__":
    main()
