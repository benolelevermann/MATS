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

PROJECT = Path(r"C:\Ole\20260721_CellClassification_v2")
NNUNET_RAW = PROJECT / "nnUNet_raw"

# Enthält die Vorhersagen von Netz 129:
# case.tif = harte Vorhersage
# case.npz = Wahrscheinlichkeiten
PREDICTIONS_129 = (
    PROJECT
    / "refiner_generation"
    / "predictions129"
)

TARGET_DATASET = (
    NNUNET_RAW
    / "Dataset134_OriginalPlusProbabilitiesRefiner"
)

TARGET_IMAGES = TARGET_DATASET / "imagesTr"
TARGET_LABELS = TARGET_DATASET / "labelsTr"
REPORT_PATH = TARGET_DATASET / "dataset134_build_report.csv"

BACKGROUND_CLASS = 0
SKELETON_CLASS = 1
SOMA_CLASS = 2

# Auf True setzen, wenn ein vorhandenes Dataset134
# vollständig gelöscht und neu erstellt werden soll.
RESET_TARGET_DATASET = False


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def find_dataset129() -> Path:
    matches = sorted(
        path
        for path in NNUNET_RAW.glob("Dataset129_*")
        if path.is_dir()
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Kein Dataset129-Ordner gefunden in:\n{NNUNET_RAW}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Mehrere Dataset129-Ordner gefunden:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def prepare_target() -> None:
    if TARGET_DATASET.exists() and any(TARGET_DATASET.rglob("*")):
        if not RESET_TARGET_DATASET:
            raise RuntimeError(
                f"Dataset134 existiert bereits:\n"
                f"{TARGET_DATASET}\n\n"
                "Lösche den Ordner manuell oder setze im Skript "
                "RESET_TARGET_DATASET = True."
            )

        shutil.rmtree(TARGET_DATASET)

    TARGET_IMAGES.mkdir(parents=True, exist_ok=True)
    TARGET_LABELS.mkdir(parents=True, exist_ok=True)


def load_probabilities(
    npz_path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """
    Liefert die Wahrscheinlichkeiten in der Form:
        (Klassen, Höhe, Breite)
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
                f"{npz_path.name}: Wahrscheinlichkeiten nicht "
                f"eindeutig gefunden. Enthaltene Keys: {keys}"
            )

    probabilities = np.squeeze(probabilities)

    if probabilities.ndim != 3:
        raise RuntimeError(
            f"{npz_path.name}: Erwartet wurde ein 3D-Array, "
            f"gefunden wurde {probabilities.shape}"
        )

    # Normaler nnU-Net-Fall: (Klassen, Höhe, Breite)
    if probabilities.shape[1:] == expected_shape:
        probabilities_chw = probabilities

    # Alternative Speicherung: (Höhe, Breite, Klassen)
    elif probabilities.shape[:2] == expected_shape:
        probabilities_chw = np.moveaxis(
            probabilities,
            -1,
            0,
        )

    else:
        raise RuntimeError(
            f"{npz_path.name}: Shape passt nicht.\n"
            f"Probabilities: {probabilities.shape}\n"
            f"Erwartetes Bild: {expected_shape}"
        )

    if probabilities_chw.shape[0] < 3:
        raise RuntimeError(
            f"{npz_path.name}: Erwartet wurden drei Klassen "
            f"(Background, Skeleton, Soma), gefunden wurden "
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


def check_manual_label(
    label: np.ndarray,
    file_name: str,
) -> None:
    values = set(
        np.unique(label)
        .astype(int)
        .tolist()
    )

    invalid = values - {0, 1, 2}

    if invalid:
        raise RuntimeError(
            f"{file_name} enthält unerwartete Labelwerte: "
            f"{sorted(invalid)}"
        )


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main() -> None:
    dataset129 = find_dataset129()

    source_images = dataset129 / "imagesTr"
    source_labels = dataset129 / "labelsTr"
    source_json_path = dataset129 / "dataset.json"

    if not source_images.exists():
        raise FileNotFoundError(source_images)

    if not source_labels.exists():
        raise FileNotFoundError(source_labels)

    if not source_json_path.exists():
        raise FileNotFoundError(source_json_path)

    if not PREDICTIONS_129.exists():
        raise FileNotFoundError(
            f"Vorhersageordner fehlt:\n{PREDICTIONS_129}"
        )

    with open(
        source_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        source_json = json.load(file)

    file_ending = source_json.get(
        "file_ending",
        ".tif",
    )

    if file_ending.lower() not in {".tif", ".tiff"}:
        raise RuntimeError(
            f"Das Skript erwartet TIFF-Dateien. "
            f"Dataset129 verwendet: {file_ending}"
        )

    label_files = sorted(
        source_labels.glob(f"*{file_ending}")
    )

    if not label_files:
        raise FileNotFoundError(
            f"Keine Labels gefunden in:\n{source_labels}"
        )

    prepare_target()

    report_rows = []

    for index, label_path in enumerate(
        label_files,
        start=1,
    ):
        case_id = label_path.name[:-len(file_ending)]

        original_path = (
            source_images
            / f"{case_id}_0000{file_ending}"
        )

        probability_path = (
            PREDICTIONS_129
            / f"{case_id}.npz"
        )

        if not original_path.exists():
            raise FileNotFoundError(
                f"Originalbild fehlt:\n{original_path}"
            )

        if not probability_path.exists():
            raise FileNotFoundError(
                f"Wahrscheinlichkeiten fehlen:\n"
                f"{probability_path}\n\n"
                "Wurde Netz 129 mit --save_probabilities ausgeführt?"
            )

        original_image = np.squeeze(
            tifffile.imread(original_path)
        )

        manual_label = np.squeeze(
            tifffile.imread(label_path)
        )

        if original_image.ndim != 2:
            raise RuntimeError(
                f"{original_path.name}: Originalbild ist nicht 2D: "
                f"{original_image.shape}"
            )

        if manual_label.ndim != 2:
            raise RuntimeError(
                f"{label_path.name}: Label ist nicht 2D: "
                f"{manual_label.shape}"
            )

        if original_image.shape != manual_label.shape:
            raise RuntimeError(
                f"Shape-Unterschied bei {case_id}:\n"
                f"Originalbild: {original_image.shape}\n"
                f"Label:        {manual_label.shape}"
            )

        check_manual_label(
            manual_label,
            label_path.name,
        )

        probabilities = load_probabilities(
            probability_path,
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

        # Kanal 0000: Originalbild
        shutil.copy2(
            original_path,
            TARGET_IMAGES
            / f"{case_id}_0000{file_ending}",
        )

        # Kanal 0001: Skeleton-Wahrscheinlichkeit
        tifffile.imwrite(
            TARGET_IMAGES
            / f"{case_id}_0001{file_ending}",
            skeleton_probability,
            photometric="minisblack",
        )

        # Kanal 0002: Soma-Wahrscheinlichkeit
        tifffile.imwrite(
            TARGET_IMAGES
            / f"{case_id}_0002{file_ending}",
            soma_probability,
            photometric="minisblack",
        )

        # Ziel: manuelles Label
        tifffile.imwrite(
            TARGET_LABELS
            / f"{case_id}{file_ending}",
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
            f"[{index}/{len(label_files)}] {case_id}: "
            f"Skeleton P={skeleton_probability.min():.3f}-"
            f"{skeleton_probability.max():.3f}, "
            f"Soma P={soma_probability.min():.3f}-"
            f"{soma_probability.max():.3f}"
        )

    dataset_json = {
        "channel_names": {
            "0": "zscore",
            "1": "noNorm",
            "2": "noNorm",
        },
        "labels": {
            "background": 0,
            "skeleton": 1,
            "soma": 2,
        },
        "numTraining": len(report_rows),
        "file_ending": file_ending,
    }

    with open(
        TARGET_DATASET / "dataset.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset_json,
            file,
            indent=2,
        )

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(report_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("=" * 70)
    print("Dataset134 wurde erfolgreich erstellt")
    print("=" * 70)
    print(f"Fälle:   {len(report_rows)}")
    print(f"Dataset: {TARGET_DATASET}")
    print()
    print("0000 = Originalbild, Z-Score")
    print("0001 = Skeleton-Wahrscheinlichkeit, unverändert")
    print("0002 = Soma-Wahrscheinlichkeit, unverändert")
    print("Label = manuelles 0/1/2-Label")


if __name__ == "__main__":
    main()
