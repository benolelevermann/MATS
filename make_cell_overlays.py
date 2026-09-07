from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi


SCRIPT_VERSION = "cell-overlays-v1-2026-07-24"


def read_image(path: Path, projection_axis: int) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    image = np.squeeze(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        print(
            f"{path.name}: 3D-Shape {image.shape}; "
            f"Max-Projektion entlang Achse {projection_axis}."
        )
        return np.max(image, axis=projection_axis)

    raise RuntimeError(
        f"{path}: 2D- oder 3D-Bild erwartet, gefunden {image.shape}."
    )


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low, high = np.percentile(finite, [1.0, 99.5])

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


def make_palette(max_label: int) -> np.ndarray:
    palette = np.zeros((max_label + 1, 3), dtype=np.float32)
    golden_ratio = 0.618033988749895

    for label_id in range(1, max_label + 1):
        hue = (label_id * golden_ratio) % 1.0
        palette[label_id] = colorsys.hsv_to_rgb(
            hue,
            0.82,
            1.0,
        )

    return palette


def save_two_channel_tiff(
    output_path: Path,
    original: np.ndarray,
    instances: np.ndarray,
) -> None:
    # Gemeinsamer Datentyp ist nötig, damit beide Kanäle in einem TIFF-Stack liegen.
    stack = np.stack(
        [
            original.astype(np.float32, copy=False),
            instances.astype(np.float32, copy=False),
        ],
        axis=0,
    )

    tifffile.imwrite(
        output_path,
        stack,
        imagej=True,
        metadata={
            "axes": "CYX",
            "Labels": [
                "Original",
                "Confident cell instances",
            ],
        },
    )


def save_colored_overlay(
    output_path: Path,
    original: np.ndarray,
    instances: np.ndarray,
    display_radius: float,
    alpha: float,
) -> None:
    labels = instances.astype(np.uint32, copy=False)
    max_label = int(labels.max())

    base_gray = normalize_for_display(original)
    base_rgb = np.repeat(base_gray[..., None], 3, axis=2)
    result = 0.55 * base_rgb

    if max_label > 0:
        if display_radius > 0:
            distance, nearest_indices = ndi.distance_transform_edt(
                labels == 0,
                return_indices=True,
            )
            display_labels = labels[
                nearest_indices[0],
                nearest_indices[1],
            ].copy()
            display_labels[distance > display_radius] = 0
        else:
            display_labels = labels

        palette = make_palette(max_label)
        colored = palette[display_labels]
        foreground = display_labels > 0

        result[foreground] = (
            (1.0 - alpha) * base_rgb[foreground]
            + alpha * colored[foreground]
        )

    figure, axis = plt.subplots(figsize=(14, 14))
    axis.imshow(np.clip(result, 0.0, 1.0))
    axis.set_title(
        f"Sicher zugeordnete Zellen: {max_label}\n"
        "Jede Zell-ID hat eine eigene Farbe"
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(figure)


def main() -> None:
    print("=" * 72)
    print(f"MAKE CELL OVERLAYS: {SCRIPT_VERSION}")
    print(f"AUSGEFUEHRTE DATEI: {Path(__file__).resolve()}")
    print("=" * 72)

    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt aus Originalbild und "
            "09_cell_instances_confident.tif ein 2-Kanal-TIFF "
            "sowie ein farbiges PNG."
        )
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--instances",
        type=Path,
        required=True,
        help="Pfad zu 09_cell_instances_confident.tif",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--projection-axis",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--display-radius",
        type=float,
        default=1.5,
        help="Nur optische Verbreiterung im PNG.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.82,
    )
    args = parser.parse_args()

    original_path = args.original.resolve()
    instances_path = args.instances.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        raise FileNotFoundError(original_path)
    if not instances_path.exists():
        raise FileNotFoundError(instances_path)

    original = read_image(
        original_path,
        projection_axis=args.projection_axis,
    )
    instances = np.squeeze(
        tifffile.imread(instances_path)
    )

    if instances.ndim != 2:
        raise RuntimeError(
            f"Instanzbild muss 2D sein, gefunden {instances.shape}."
        )

    if original.shape != instances.shape:
        raise RuntimeError(
            "Original und Zellinstanzen haben verschiedene Shapes:\n"
            f"Original:  {original.shape}\n"
            f"Instanzen: {instances.shape}"
        )

    two_channel_path = (
        output_dir / "original_plus_confident_cells_2ch.tif"
    )
    colored_png_path = (
        output_dir / "confident_cells_colored_overlay.png"
    )

    save_two_channel_tiff(
        two_channel_path,
        original,
        instances,
    )
    save_colored_overlay(
        colored_png_path,
        original,
        instances,
        display_radius=args.display_radius,
        alpha=args.alpha,
    )

    (output_dir / "_overlay_script_version.txt").write_text(
        f"{SCRIPT_VERSION}\n{Path(__file__).resolve()}\n",
        encoding="utf-8",
    )

    print()
    print("FERTIG")
    print(f"Zell-IDs:      {int(instances.max())}")
    print(f"2-Kanal-TIFF:  {two_channel_path}")
    print(f"Farbiges PNG:  {colored_png_path}")


if __name__ == "__main__":
    main()
