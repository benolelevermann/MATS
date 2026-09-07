from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile


ROOT = Path(
    r"C:\Ole\20260721_CellClassification_v2\preview130"
)

INPUT_DIR = ROOT / "input"
PREDICTION_DIR = ROOT / "prediction"
LABEL_DIR = ROOT / "manual_labels"

OUTPUT_PATH = ROOT / "preview_comparison.png"


def colorize(label: np.ndarray) -> np.ndarray:
    """
    Hintergrund = schwarz
    Skeleton    = grün
    Soma        = magenta
    """
    label = np.squeeze(label)

    rgb = np.zeros(
        (*label.shape, 3),
        dtype=np.float32,
    )

    rgb[label == 1] = (0.0, 1.0, 0.0)
    rgb[label == 2] = (1.0, 0.0, 1.0)

    return rgb


def main():
    label_files = sorted(
        LABEL_DIR.glob("*.tif")
    )

    if not label_files:
        raise FileNotFoundError(
            f"Keine Labels in {LABEL_DIR}"
        )

    case_ids = [
        path.stem
        for path in label_files
    ]

    figure, axes = plt.subplots(
        len(case_ids),
        5,
        figsize=(18, 3.7 * len(case_ids)),
        squeeze=False,
    )

    for row, case_id in enumerate(case_ids):
        skeleton_probability = np.squeeze(
            tifffile.imread(
                INPUT_DIR
                / f"{case_id}_0000.tif"
            )
        ).astype(np.float32)

        soma_probability = np.squeeze(
            tifffile.imread(
                INPUT_DIR
                / f"{case_id}_0001.tif"
            )
        ).astype(np.float32)

        background_probability = np.clip(
            1.0
            - skeleton_probability
            - soma_probability,
            0.0,
            1.0,
        )

        # Harte Ausgangssegmentierung von Netz 129
        prediction129 = np.argmax(
            np.stack([
                background_probability,
                skeleton_probability,
                soma_probability,
            ]),
            axis=0,
        ).astype(np.uint8)

        prediction130 = np.squeeze(
            tifffile.imread(
                PREDICTION_DIR
                / f"{case_id}.tif"
            )
        )

        manual_label = np.squeeze(
            tifffile.imread(
                LABEL_DIR
                / f"{case_id}.tif"
            )
        )

        images = [
            skeleton_probability,
            soma_probability,
            colorize(prediction129),
            colorize(prediction130),
            colorize(manual_label),
        ]

        titles = [
            "P(Skeleton), Netz 129",
            "P(Soma), Netz 129",
            "Netz 129, hart",
            "Netz 130",
            "Manuelles Label",
        ]

        for column, (image, title) in enumerate(
            zip(images, titles)
        ):
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
                axis.set_title(title)

            if column == 0:
                axis.set_ylabel(case_id)

            axis.set_xticks([])
            axis.set_yticks([])

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Preview gespeichert:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
