from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


MODEL_NAMES = {
    130: "Wahrscheinlichkeiten",
    131: "harte 0/1/2-Maske",
    132: "harte Maske, Z-Score",
    133: "Original + harte Masken",
    134: "Original + Wahrscheinlichkeiten",
}


def find_single_dataset(root: Path, dataset_id: int) -> Path | None:
    matches = sorted(
        path
        for path in root.glob(f"Dataset{dataset_id:03d}_*")
        if path.is_dir()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return None
    raise RuntimeError(
        f"Mehrere Dataset{dataset_id:03d}-Ordner gefunden in {root}:\n"
        + "\n".join(str(path) for path in matches)
    )


def model_has_best_checkpoint(
    results_root: Path,
    dataset_id: int,
    fold: int,
) -> bool:
    dataset_dir = find_single_dataset(results_root, dataset_id)
    if dataset_dir is None:
        return False

    matches = list(
        dataset_dir.glob(
            f"*__*__2d/fold_{fold}/checkpoint_best.pth"
        )
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"Mehrere checkpoint_best.pth fuer Dataset{dataset_id:03d}:\n"
            + "\n".join(str(path) for path in matches)
        )
    return len(matches) == 1


def load_original(path: Path, projection_axis: int) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    image = np.squeeze(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        print(
            f"Originalbild ist 3D {image.shape}. "
            f"Max-Projektion entlang Achse {projection_axis}."
        )
        return np.max(image, axis=projection_axis)

    raise RuntimeError(
        f"Originalbild muss 2D oder 3D sein. Gefunden: {image.shape}"
    )


def load_probabilities(
    npz_path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
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
                f"eindeutig gefunden. Keys: {keys}"
            )

    probabilities = np.squeeze(probabilities)

    if probabilities.ndim != 3:
        raise RuntimeError(
            f"{npz_path.name}: 3D-Array erwartet, "
            f"gefunden {probabilities.shape}"
        )

    if probabilities.shape[1:] == expected_shape:
        chw = probabilities
    elif probabilities.shape[:2] == expected_shape:
        chw = np.moveaxis(probabilities, -1, 0)
    else:
        raise RuntimeError(
            f"Probability-Shape passt nicht zum Originalbild.\n"
            f"Probabilities: {probabilities.shape}\n"
            f"Original:      {expected_shape}"
        )

    if chw.shape[0] < 3:
        raise RuntimeError(
            f"Drei Klassen erwartet, gefunden {chw.shape[0]}."
        )

    if not np.all(np.isfinite(chw)):
        raise RuntimeError(
            f"{npz_path.name} enthaelt NaN oder unendliche Werte."
        )

    return chw.astype(np.float32, copy=False)


def colorize(label: np.ndarray) -> np.ndarray:
    label = np.squeeze(label)
    rgb = np.zeros(
        (label.shape[0], label.shape[1], 3),
        dtype=np.float32,
    )
    rgb[label == 1] = (0.0, 1.0, 0.0)
    rgb[label == 2] = (1.0, 0.0, 1.0)
    return rgb


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low, high = np.percentile(finite, [1, 99])

    if high <= low:
        low = float(finite.min())
        high = float(finite.max())

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    return np.clip(
        (image - low) / (high - low),
        0.0,
        1.0,
    )


def save_channel(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        array,
        photometric="minisblack",
    )


def prepare_model_input(
    model_id: int,
    input_dir: Path,
    case_id: str,
    original: np.ndarray,
    hard129: np.ndarray,
    p_skeleton: np.ndarray,
    p_soma: np.ndarray,
) -> list[tuple[str, np.ndarray, bool]]:
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    hard_skeleton = (hard129 == 1).astype(np.uint8)
    hard_soma = (hard129 == 2).astype(np.uint8)

    if model_id == 130:
        channels = [
            ("P(Skeleton)", p_skeleton, False),
            ("P(Soma)", p_soma, False),
        ]
    elif model_id in (131, 132):
        channels = [
            ("harte 0/1/2-Maske", hard129.astype(np.uint8), True),
        ]
    elif model_id == 133:
        channels = [
            ("Originalbild", original, False),
            ("harte Skeleton-Maske", hard_skeleton, False),
            ("harte Soma-Maske", hard_soma, False),
        ]
    elif model_id == 134:
        channels = [
            ("Originalbild", original, False),
            ("P(Skeleton)", p_skeleton, False),
            ("P(Soma)", p_soma, False),
        ]
    else:
        raise ValueError(
            f"Keine Input-Definition fuer Dataset{model_id:03d}."
        )

    for channel_index, (_, channel, _) in enumerate(channels):
        save_channel(
            input_dir / f"{case_id}_{channel_index:04d}.tif",
            channel,
        )

    return channels


def run_nnunet_prediction(
    model_id: int,
    fold: int,
    input_dir: Path,
    output_dir: Path,
    device: str,
    save_probabilities: bool,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(model_id),
        "-c", "2d",
        "-f", str(fold),
        "-chk", "checkpoint_best.pth",
        "-device", device,
        "-npp", "1",
        "-nps", "1",
    ]

    if save_probabilities:
        argv.append("--save_probabilities")

    code = (
        "import sys; "
        "from nnunetv2.inference.predict_from_raw_data "
        "import predict_entry_point; "
        f"sys.argv={argv!r}; "
        "predict_entry_point()"
    )

    print()
    print("=" * 70)
    print(
        f"Starte Netz {model_id}: "
        f"{MODEL_NAMES.get(model_id, '')}"
    )
    print("=" * 70)

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Schickt dieselbe Netz-129-Overview-Vorhersage durch "
            "alle vorhandenen Refiner-Netze 130-134 und erstellt "
            "einen direkten Input-/Output-Vergleich."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            r"C:\Ole\20260721_CellClassification_v2"
        ),
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help=(
            "Dasselbe Original- oder Max-Projektionsbild, "
            "das Netz 129 als Input erhalten hat."
        ),
    )
    parser.add_argument(
        "--net129-output-dir",
        type=Path,
        required=True,
        help=(
            "Ordner mit overview.tif und overview.npz "
            "aus Netz 129."
        ),
    )
    parser.add_argument(
        "--case-id",
        default="overview",
    )
    parser.add_argument(
        "--models",
        type=int,
        nargs="+",
        default=[130, 131, 132, 133, 134],
        help=(
            "Zu testende Refiner. Nicht vorhandene Modelle "
            "werden automatisch uebersprungen."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--projection-axis",
        type=int,
        default=0,
        help=(
            "Achse fuer Max-Projektion, falls --original "
            "noch ein 3D-Stack ist."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help=(
            "Speichert auch die Wahrscheinlichkeiten jedes "
            "Refiner-Netzes."
        ),
    )
    args = parser.parse_args()

    project = args.project.resolve()
    original_path = args.original.resolve()
    net129_output_dir = args.net129_output_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project / "overview_all_refiners_comparison"
    )

    raw_root = project / "nnUNet_raw"
    preprocessed_root = project / "nnUNet_preprocessed"
    results_root = project / "nnUNet_results"

    os.environ["nnUNet_raw"] = str(raw_root)
    os.environ["nnUNet_preprocessed"] = str(
        preprocessed_root
    )
    os.environ["nnUNet_results"] = str(results_root)

    if not original_path.exists():
        raise FileNotFoundError(original_path)

    if not net129_output_dir.exists():
        raise FileNotFoundError(net129_output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = (
        net129_output_dir
        / f"{args.case_id}.npz"
    )
    hard129_path = (
        net129_output_dir
        / f"{args.case_id}.tif"
    )

    if not npz_path.exists():
        npz_files = sorted(
            net129_output_dir.glob("*.npz")
        )
        if len(npz_files) == 1:
            npz_path = npz_files[0]
            print(
                f"Nutze automatisch: {npz_path.name}"
            )
        else:
            raise FileNotFoundError(
                f"{npz_path} fehlt und der Ordner "
                f"enthaelt nicht genau eine NPZ-Datei."
            )

    if not hard129_path.exists():
        tif_files = sorted(
            net129_output_dir.glob("*.tif")
        )
        if len(tif_files) == 1:
            hard129_path = tif_files[0]
            print(
                f"Nutze automatisch: {hard129_path.name}"
            )

    original = load_original(
        original_path,
        args.projection_axis,
    )
    probabilities = load_probabilities(
        npz_path,
        expected_shape=original.shape,
    )

    p_skeleton = np.clip(
        probabilities[1],
        0.0,
        1.0,
    ).astype(np.float32)

    p_soma = np.clip(
        probabilities[2],
        0.0,
        1.0,
    ).astype(np.float32)

    if hard129_path.exists():
        hard129 = np.squeeze(
            tifffile.imread(hard129_path)
        ).astype(np.uint8)
    else:
        p_background = np.clip(
            1.0 - p_skeleton - p_soma,
            0.0,
            1.0,
        )
        hard129 = np.argmax(
            np.stack(
                [
                    p_background,
                    p_skeleton,
                    p_soma,
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.uint8)

    if hard129.shape != original.shape:
        raise RuntimeError(
            f"Shape-Unterschied:\n"
            f"Original: {original.shape}\n"
            f"Netz 129: {hard129.shape}"
        )

    unique_values = set(
        np.unique(hard129).astype(int).tolist()
    )
    if not unique_values.issubset({0, 1, 2}):
        raise RuntimeError(
            f"Netz-129-TIF enthaelt unerwartete Werte: "
            f"{sorted(unique_values)}"
        )

    available_models = []
    skipped_models = []

    for model_id in args.models:
        if model_has_best_checkpoint(
            results_root,
            model_id,
            args.fold,
        ):
            available_models.append(model_id)
        else:
            skipped_models.append(model_id)

    if not available_models:
        raise RuntimeError(
            "Keines der angeforderten Refiner-Netze besitzt "
            "einen checkpoint_best.pth."
        )

    print()
    print("Zu vergleichende Netze:")
    for model_id in available_models:
        print(
            f"  Netz {model_id}: "
            f"{MODEL_NAMES.get(model_id, '')}"
        )

    if skipped_models:
        print()
        print(
            "Uebersprungen, weil kein Best-Checkpoint "
            "gefunden wurde:"
        )
        for model_id in skipped_models:
            print(f"  Netz {model_id}")

    model_channels: dict[
        int,
        list[tuple[str, np.ndarray, bool]]
    ] = {}
    model_outputs: dict[int, np.ndarray] = {}

    inputs_root = output_dir / "inputs"
    outputs_root = output_dir / "outputs"

    for model_id in available_models:
        input_dir = (
            inputs_root
            / f"net_{model_id}"
        )
        model_output_dir = (
            outputs_root
            / f"net_{model_id}"
        )

        channels = prepare_model_input(
            model_id=model_id,
            input_dir=input_dir,
            case_id=args.case_id,
            original=original,
            hard129=hard129,
            p_skeleton=p_skeleton,
            p_soma=p_soma,
        )
        model_channels[model_id] = channels

        run_nnunet_prediction(
            model_id=model_id,
            fold=args.fold,
            input_dir=input_dir,
            output_dir=model_output_dir,
            device=args.device,
            save_probabilities=args.save_probabilities,
        )

        prediction_path = (
            model_output_dir
            / f"{args.case_id}.tif"
        )
        if not prediction_path.exists():
            raise FileNotFoundError(
                prediction_path
            )

        prediction = np.squeeze(
            tifffile.imread(prediction_path)
        ).astype(np.uint8)

        if prediction.shape != hard129.shape:
            raise RuntimeError(
                f"Netz {model_id}: Output-Shape "
                f"{prediction.shape} passt nicht zu "
                f"{hard129.shape}."
            )

        model_outputs[model_id] = prediction

    summary_rows = []

    baseline_skeleton = int(
        np.sum(hard129 == 1)
    )
    baseline_soma = int(
        np.sum(hard129 == 2)
    )

    summary_rows.append({
        "model": "129",
        "input": "Originalbild",
        "skeleton_pixels": baseline_skeleton,
        "soma_pixels": baseline_soma,
        "changed_pixels_vs_129": 0,
        "changed_to_background": 0,
        "changed_to_skeleton": 0,
        "changed_to_soma": 0,
    })

    for model_id, prediction in model_outputs.items():
        changed = prediction != hard129

        summary_rows.append({
            "model": str(model_id),
            "input": MODEL_NAMES.get(
                model_id,
                "",
            ),
            "skeleton_pixels": int(
                np.sum(prediction == 1)
            ),
            "soma_pixels": int(
                np.sum(prediction == 2)
            ),
            "changed_pixels_vs_129": int(
                np.sum(changed)
            ),
            "changed_to_background": int(
                np.sum(
                    np.logical_and(
                        changed,
                        prediction == 0,
                    )
                )
            ),
            "changed_to_skeleton": int(
                np.sum(
                    np.logical_and(
                        changed,
                        prediction == 1,
                    )
                )
            ),
            "changed_to_soma": int(
                np.sum(
                    np.logical_and(
                        changed,
                        prediction == 2,
                    )
                )
            ),
        })

        save_channel(
            output_dir
            / f"changed_pixels_129_vs_{model_id}.tif",
            changed.astype(np.uint8),
        )

    summary_path = (
        output_dir
        / "overview_refiner_summary.csv"
    )
    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------------
    # 1. Vergleich aller Outputs
    # --------------------------------------------------------
    output_titles = [
        "Originalbild",
        "Netz 129",
    ] + [
        f"Netz {model_id}"
        for model_id in available_models
    ]

    output_images = [
        normalize_for_display(original),
        colorize(hard129),
    ] + [
        colorize(model_outputs[model_id])
        for model_id in available_models
    ]

    figure, axes = plt.subplots(
        1,
        len(output_images),
        figsize=(4.2 * len(output_images), 5),
        squeeze=False,
    )

    for column, (title, image) in enumerate(
        zip(output_titles, output_images)
    ):
        axis = axes[0, column]

        if image.ndim == 2:
            axis.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        else:
            axis.imshow(image)

        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    outputs_figure_path = (
        output_dir
        / "all_outputs_comparison.png"
    )
    figure.savefig(
        outputs_figure_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    # --------------------------------------------------------
    # 2. Inputs und Output jedes Refiners
    # --------------------------------------------------------
    max_channels = 3
    number_of_columns = max_channels + 2

    figure, axes = plt.subplots(
        len(available_models),
        number_of_columns,
        figsize=(
            4.0 * number_of_columns,
            4.0 * len(available_models),
        ),
        squeeze=False,
    )

    for row, model_id in enumerate(
        available_models
    ):
        channels = model_channels[model_id]

        for column in range(max_channels):
            axis = axes[row, column]

            if column < len(channels):
                name, channel, categorical = (
                    channels[column]
                )

                if categorical:
                    axis.imshow(
                        colorize(channel)
                    )
                else:
                    axis.imshow(
                        normalize_for_display(channel)
                        if name == "Originalbild"
                        else channel,
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                    )

                axis.set_title(
                    f"Input {column}: {name}"
                )
            else:
                axis.axis("off")
                continue

            axis.set_xticks([])
            axis.set_yticks([])

        axes[row, max_channels].imshow(
            colorize(hard129)
        )
        axes[row, max_channels].set_title(
            "Ausgang: Netz 129"
        )
        axes[row, max_channels].set_xticks([])
        axes[row, max_channels].set_yticks([])

        axes[row, max_channels + 1].imshow(
            colorize(
                model_outputs[model_id]
            )
        )
        axes[row, max_channels + 1].set_title(
            f"Output Netz {model_id}"
        )
        axes[row, max_channels + 1].set_xticks([])
        axes[row, max_channels + 1].set_yticks([])

        axes[row, 0].set_ylabel(
            f"Netz {model_id}\n"
            f"{MODEL_NAMES.get(model_id, '')}",
            rotation=0,
            labelpad=65,
            va="center",
        )

    figure.tight_layout()
    inputs_figure_path = (
        output_dir
        / "all_inputs_and_outputs.png"
    )
    figure.savefig(
        inputs_figure_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    # --------------------------------------------------------
    # 3. Aenderungsmasken
    # --------------------------------------------------------
    figure, axes = plt.subplots(
        1,
        len(available_models),
        figsize=(
            4.2 * len(available_models),
            5,
        ),
        squeeze=False,
    )

    for column, model_id in enumerate(
        available_models
    ):
        changed = (
            model_outputs[model_id]
            != hard129
        )
        axes[0, column].imshow(
            changed,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, column].set_title(
            f"Netz {model_id}\n"
            f"{int(changed.sum())} geaenderte Pixel"
        )
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])

    figure.tight_layout()
    changes_figure_path = (
        output_dir
        / "all_changes_vs_129.png"
    )
    figure.savefig(
        changes_figure_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print()
    print("=" * 78)
    print("VERGLEICH FERTIG")
    print("=" * 78)
    print(f"Ordner:          {output_dir}")
    print(f"Outputs:         {outputs_figure_path}")
    print(f"Inputs/Outputs:  {inputs_figure_path}")
    print(f"Aenderungen:     {changes_figure_path}")
    print(f"Zusammenfassung: {summary_path}")
    print()
    print("Farben in den Segmentierungsbildern:")
    print("  Gruen    = Skeleton")
    print("  Magenta  = Soma")
    print("  Schwarz  = Hintergrund")


if __name__ == "__main__":
    main()
