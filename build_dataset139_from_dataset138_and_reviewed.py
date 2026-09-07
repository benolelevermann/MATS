from __future__ import annotations

"""Build a fresh Dataset139 from Dataset138 plus all approved review pairs."""

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import tifffile


SCRIPT_VERSION = "dataset139-reviewed-expansion-v1-2026-09-01"
EXPECTED_LABELS = {0, 1, 2}
DEFAULT_README = (
    "Dataset139 = unveraendertes Dataset138 plus alle im Web-Review freigegebenen "
    "Trainingspaare. Die Dataset138-Validierungsfaelle bleiben in der beigelegten "
    "Split-Datei unveraendert; neue Review-Zellen liegen nur im Training.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--approved-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--base-splits", type=Path)
    return parser.parse_args()


def collect_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    images = root / "imagesTr"
    labels = root / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Expected imagesTr and labelsTr in {root}")
    pairs: list[tuple[str, Path, Path]] = []
    for image in sorted(images.glob("*_0000.tif"), key=lambda path: path.name.lower()):
        case_id = image.name[: -len("_0000.tif")]
        label = labels / f"{case_id}.tif"
        if not label.is_file():
            raise RuntimeError(f"Missing label for {image.name}")
        pairs.append((case_id, image, label))
    if len(pairs) != len(list(labels.glob("*.tif"))):
        raise RuntimeError(f"Unpaired labels found in {labels}")
    return pairs


def read_pair(image_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.squeeze(np.asarray(tifffile.imread(image_path)))
    label = np.squeeze(np.asarray(tifffile.imread(label_path)))
    if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
        raise RuntimeError(
            f"Incompatible 2-D pair: {image_path.name} {image.shape}, {label_path.name} {label.shape}"
        )
    values = {int(value) for value in np.unique(label)}
    if not values <= EXPECTED_LABELS:
        raise RuntimeError(f"Unexpected labels {sorted(values)} in {label_path}")
    if not np.any(label == 1) or not np.any(label == 2):
        raise RuntimeError(f"Skeleton or soma is empty in {label_path}")
    return image, label.astype(np.uint8, copy=False)


def pair_digest(image: np.ndarray, label: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes(order="C"))
    digest.update(label.tobytes(order="C"))
    return digest.hexdigest()


def reviewed_case_id(source_case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", source_case_id).strip("_") or "reviewed"
    full = f"rv_{safe}"
    if len(full) <= 150:
        return full
    suffix = hashlib.sha1(full.encode("utf-8")).hexdigest()[:12]
    return f"{full[:137]}_{suffix}"


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "output_case_id",
        "origin",
        "source_case_id",
        "source_image",
        "source_label",
        "height",
        "width",
        "skeleton_pixels",
        "soma_pixels",
        "digest",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    base_dataset: Path,
    approved_dataset: Path,
    output_dataset: Path,
    base_splits: Path | None = None,
    *,
    base_origin: str = "dataset138",
    script_version: str = SCRIPT_VERSION,
    split_output_name: str = "splits_final_preserve_dataset138.json",
    readme_text: str = DEFAULT_README,
) -> dict[str, object]:
    if output_dataset.exists() and any(output_dataset.iterdir()):
        raise RuntimeError(
            f"Output dataset is not empty: {output_dataset}\nChoose a fresh Dataset ID."
        )
    base_json = json.loads((base_dataset / "dataset.json").read_text(encoding="utf-8"))
    if base_json.get("labels") != {"background": 0, "skeleton": 1, "soma": 2}:
        raise RuntimeError("Base dataset does not have the expected 0/1/2 labels.")
    base_pairs = collect_pairs(base_dataset)
    approved_pairs = collect_pairs(approved_dataset)

    prepared: list[tuple[str, str, Path, Path, np.ndarray, np.ndarray, str]] = []
    seen_digests: set[str] = set()
    output_ids: set[str] = set()
    skipped_duplicates: list[str] = []
    for origin, pairs in ((base_origin, base_pairs), ("approved_review", approved_pairs)):
        for source_case_id, image_path, label_path in pairs:
            image, label = read_pair(image_path, label_path)
            digest = pair_digest(image, label)
            # Dataset138 is the immutable baseline and must be copied exactly,
            # including any pre-existing duplicate case IDs with identical data.
            # Deduplication applies only to newly approved review pairs.
            if origin == "approved_review" and digest in seen_digests:
                skipped_duplicates.append(source_case_id)
                continue
            seen_digests.add(digest)
            output_id = source_case_id if origin == base_origin else reviewed_case_id(source_case_id)
            if output_id in output_ids:
                raise RuntimeError(f"Output case ID collision: {output_id}")
            output_ids.add(output_id)
            prepared.append(
                (output_id, origin, image_path, label_path, image, label, digest)
            )

    output_dataset.mkdir(parents=True, exist_ok=True)
    images_out = output_dataset / "imagesTr"
    labels_out = output_dataset / "labelsTr"
    images_out.mkdir()
    labels_out.mkdir()
    manifest: list[dict[str, object]] = []
    new_ids: list[str] = []
    for output_id, origin, image_path, label_path, image, label, digest in prepared:
        if origin == base_origin:
            shutil.copy2(image_path, images_out / f"{output_id}_0000.tif")
            shutil.copy2(label_path, labels_out / f"{output_id}.tif")
        else:
            tifffile.imwrite(images_out / f"{output_id}_0000.tif", image)
            tifffile.imwrite(labels_out / f"{output_id}.tif", label)
            new_ids.append(output_id)
        manifest.append(
            {
                "output_case_id": output_id,
                "origin": origin,
                "source_case_id": image_path.name[: -len("_0000.tif")],
                "source_image": str(image_path),
                "source_label": str(label_path),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "skeleton_pixels": int(np.count_nonzero(label == 1)),
                "soma_pixels": int(np.count_nonzero(label == 2)),
                "digest": digest,
            }
        )

    dataset_json = {
        "channel_names": {"0": "image"},
        "labels": {"background": 0, "skeleton": 1, "soma": 2},
        "numTraining": len(prepared),
        "file_ending": ".tif",
    }
    (output_dataset / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2), encoding="utf-8"
    )
    write_manifest(output_dataset / "build_manifest.csv", manifest)

    if base_splits is not None:
        splits = json.loads(base_splits.read_text(encoding="utf-8"))
        base_ids = {case_id for case_id, _image, _label in base_pairs}
        for index, split in enumerate(splits):
            train = [str(value) for value in split.get("train", [])]
            val = [str(value) for value in split.get("val", [])]
            unknown = (set(train) | set(val)) - base_ids
            if unknown:
                raise RuntimeError(f"Base split {index} contains unknown Dataset138 cases")
            if set(train) & set(val):
                raise RuntimeError(f"Base split {index} has train/validation overlap")
            split["train"] = sorted(set(train) | set(new_ids))
            split["val"] = val
        (output_dataset / split_output_name).write_text(
            json.dumps(splits, indent=2), encoding="utf-8"
        )

    summary = {
        "script_version": script_version,
        "base_dataset": str(base_dataset),
        "approved_dataset": str(approved_dataset),
        "output_dataset": str(output_dataset),
        "base_cases": len(base_pairs),
        "approved_pairs_found": len(approved_pairs),
        "approved_pairs_added": len(new_ids),
        "approved_duplicates_skipped": len(skipped_duplicates),
        "total_training": len(prepared),
        "base_splits": str(base_splits) if base_splits else None,
    }
    (output_dataset / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dataset / "README.txt").write_text(readme_text, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        args.base_dataset.resolve(),
        args.approved_dataset.resolve(),
        args.output_dataset.resolve(),
        args.base_splits.resolve() if args.base_splits else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
