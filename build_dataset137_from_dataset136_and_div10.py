from __future__ import annotations

"""Create a fresh Dataset137 from Dataset136 plus rasterized div10 cells.

The script never changes Dataset136. It refuses to write into a non-empty
target directory and validates every new 2-D TIFF pair before copying the
2,013 existing Dataset136 cases plus the selected div10 cases.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile


SCRIPT_VERSION = "dataset137-assembler-v1-2026-08-15"
VALID_LABELS = {0, 1, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument(
        "--rasterized-new-dir",
        type=Path,
        required=True,
        help="Staging folder created by rasterize_div10_training_masks_fiji.py.",
    )
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Optional included_tracings_manifest.csv copied as provenance.",
    )
    return parser.parse_args()


def require_new_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output dataset directory is not empty: {path}\n"
            "Use a new Dataset ID/name; do not overwrite an existing dataset."
        )
    path.mkdir(parents=True, exist_ok=True)


def read_2d(path: Path, label: str) -> np.ndarray:
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{label} must be 2-D, got {array.shape}: {path}")
    return array


def case_id_from_image(path: Path) -> str:
    suffix = "_0000.tif"
    if not path.name.endswith(suffix):
        raise RuntimeError(f"Unexpected image filename: {path.name}")
    return path.name[: -len(suffix)]


def collect_pairs(images_dir: Path, labels_dir: Path) -> dict[str, tuple[Path, Path]]:
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTr or labelsTr: {images_dir}, {labels_dir}")
    pairs: dict[str, tuple[Path, Path]] = {}
    for image_path in sorted(images_dir.glob("*_0000.tif")):
        case_id = case_id_from_image(image_path)
        label_path = labels_dir / f"{case_id}.tif"
        if not label_path.is_file():
            raise RuntimeError(f"Missing label for {image_path.name}: {label_path}")
        if case_id in pairs:
            raise RuntimeError(f"Duplicate case ID: {case_id}")
        pairs[case_id] = (image_path, label_path)
    if not pairs:
        raise RuntimeError(f"No *_0000.tif image files in {images_dir}")
    label_ids = {path.stem for path in labels_dir.glob("*.tif")}
    unmatched_labels = label_ids - set(pairs)
    if unmatched_labels:
        example = ", ".join(sorted(unmatched_labels)[:5])
        raise RuntimeError(f"Labels without matching image (examples): {example}")
    return pairs


def validate_new_pair(case_id: str, image_path: Path, label_path: Path) -> dict[str, int]:
    image = read_2d(image_path, "Raw image")
    label = read_2d(label_path, "Label")
    if image.shape != label.shape:
        raise RuntimeError(
            f"Shape mismatch for {case_id}: image={image.shape}, label={label.shape}"
        )
    values = set(np.unique(label).astype(int).tolist())
    invalid = values - VALID_LABELS
    if invalid:
        raise RuntimeError(f"Invalid labels for {case_id}: {sorted(invalid)}")
    skeleton_pixels = int(np.count_nonzero(label == 1))
    soma_pixels = int(np.count_nonzero(label == 2))
    if skeleton_pixels == 0:
        raise RuntimeError(f"No skeleton pixels in new label: {case_id}")
    if soma_pixels == 0:
        raise RuntimeError(f"No soma pixels in new label: {case_id}")
    return {
        "height_px": int(image.shape[0]),
        "width_px": int(image.shape[1]),
        "skeleton_pixels": skeleton_pixels,
        "soma_pixels": soma_pixels,
    }


def copy_pair(image_path: Path, label_path: Path, images_out: Path, labels_out: Path) -> None:
    shutil.copy2(image_path, images_out / image_path.name)
    shutil.copy2(label_path, labels_out / label_path.name)


def main() -> None:
    args = parse_args()
    base_json_path = args.base_dataset / "dataset.json"
    if not base_json_path.is_file():
        raise FileNotFoundError(base_json_path)
    base_json = json.loads(base_json_path.read_text(encoding="utf-8"))
    if base_json.get("labels") != {"background": 0, "skeleton": 1, "soma": 2}:
        raise RuntimeError("Base dataset does not have the expected 0/1/2 labels.")
    if base_json.get("file_ending") != ".tif":
        raise RuntimeError("Base dataset must use .tif files.")

    base_pairs = collect_pairs(args.base_dataset / "imagesTr", args.base_dataset / "labelsTr")
    new_pairs = collect_pairs(args.rasterized_new_dir / "imagesTr", args.rasterized_new_dir / "labelsTr")
    collisions = set(base_pairs) & set(new_pairs)
    if collisions:
        raise RuntimeError(
            "New IDs collide with existing Dataset136 IDs: "
            + ", ".join(sorted(collisions)[:10])
        )

    new_stats: dict[str, dict[str, int]] = {}
    for case_id, (image_path, label_path) in new_pairs.items():
        new_stats[case_id] = validate_new_pair(case_id, image_path, label_path)

    require_new_empty_directory(args.output_dataset)
    images_out = args.output_dataset / "imagesTr"
    labels_out = args.output_dataset / "labelsTr"
    images_out.mkdir()
    labels_out.mkdir()

    for index, (case_id, pair) in enumerate(sorted(base_pairs.items()), start=1):
        copy_pair(pair[0], pair[1], images_out, labels_out)
        if index % 250 == 0 or index == len(base_pairs):
            print(f"Copied base Dataset136 cases: {index}/{len(base_pairs)}")
    for index, (case_id, pair) in enumerate(sorted(new_pairs.items()), start=1):
        copy_pair(pair[0], pair[1], images_out, labels_out)
        print(f"Copied new div10 case: {index}/{len(new_pairs)} ({case_id})")

    dataset_json = {
        "channel_names": {"0": "image"},
        "labels": {"background": 0, "skeleton": 1, "soma": 2},
        "numTraining": len(base_pairs) + len(new_pairs),
        "file_ending": ".tif",
    }
    (args.output_dataset / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2), encoding="utf-8"
    )

    report_rows = [
        {
            "case_id": case_id,
            "source": "Dataset136_existing",
            "height_px": "",
            "width_px": "",
            "skeleton_pixels": "",
            "soma_pixels": "",
        }
        for case_id in sorted(base_pairs)
    ]
    report_rows.extend(
        {
            "case_id": case_id,
            "source": "div10_rasterized",
            **new_stats[case_id],
        }
        for case_id in sorted(new_pairs)
    )
    with (args.output_dataset / "dataset137_build_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "case_id", "source", "height_px", "width_px", "skeleton_pixels", "soma_pixels"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    if args.selection_manifest is not None:
        if not args.selection_manifest.is_file():
            raise FileNotFoundError(args.selection_manifest)
        shutil.copy2(args.selection_manifest, args.output_dataset / "div10_selection_provenance.csv")

    summary = {
        "script_version": SCRIPT_VERSION,
        "base_dataset": str(args.base_dataset),
        "rasterized_new_dir": str(args.rasterized_new_dir),
        "base_case_count": len(base_pairs),
        "new_div10_case_count": len(new_pairs),
        "total_case_count": len(base_pairs) + len(new_pairs),
        "output_dataset": str(args.output_dataset),
    }
    (args.output_dataset / "dataset137_build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 72)
    print("DATASET137 BUILD COMPLETE")
    print("=" * 72)
    print(f"Existing Dataset136 cases: {len(base_pairs)}")
    print(f"New div10 cases:          {len(new_pairs)}")
    print(f"Total Dataset137 cases:   {len(base_pairs) + len(new_pairs)}")
    print(f"Dataset:                   {args.output_dataset}")


if __name__ == "__main__":
    main()
