from __future__ import annotations

"""Reassemble verified MATS tiles into full overview cases for nnU-Net fine-tuning."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from make_mats_annotation_overview import collect_latest_records, read_2d


SCRIPT_VERSION = "dataset140-mats-overview-finetune-v1-2026-09-03"
EXPECTED_LABELS = {0, 1, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-dir", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument(
        "--validation-image-id",
        default="Mosaic001_Merged-3",
        help="One entire annotated overview held out from fine-tuning.",
    )
    return parser.parse_args()


def output_case_id(image_id: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in image_id)
    return f"mats_overview_{safe.strip('_')}"


def assemble_case(image_id: str, records: list[Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    records = sorted(records, key=lambda record: (record.data["row_index"], record.data["column_index"]))
    raw_sources = {str(record.data.get("raw_source", "")) for record in records}
    if len(raw_sources) != 1:
        raise RuntimeError(f"{image_id}: uneinheitliche Rohbildquellen: {sorted(raw_sources)}")
    raw_source = Path(next(iter(raw_sources)))
    raw = read_2d(raw_source, f"Rohbild {image_id}")
    height = max(int(record.data["y1"]) for record in records)
    width = max(int(record.data["x1"]) for record in records)
    if raw.shape != (height, width):
        raise RuntimeError(
            f"{image_id}: Rohbildform {raw.shape} passt nicht zur Patchabdeckung {(height, width)}"
        )

    label = np.zeros((height, width), dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=np.uint8)
    patch_rows: list[dict[str, Any]] = []
    for record in records:
        y0, y1 = int(record.data["y0"]), int(record.data["y1"])
        x0, x1 = int(record.data["x0"]), int(record.data["x1"])
        expected_shape = (y1 - y0, x1 - x0)
        patch_image = read_2d(record.image_path, f"Patchbild {record.case_id}")
        patch_label = read_2d(record.label_path, f"Patchlabel {record.case_id}")
        if patch_image.shape != expected_shape or patch_label.shape != expected_shape:
            raise RuntimeError(
                f"{record.case_id}: Patchform passt nicht zu den Koordinaten: "
                f"Bild {patch_image.shape}, Label {patch_label.shape}, erwartet {expected_shape}"
            )
        if not np.array_equal(patch_image, raw[y0:y1, x0:x1]):
            raise RuntimeError(f"{record.case_id}: Exportbild entspricht nicht dem Rohbildausschnitt")
        values = {int(value) for value in np.unique(patch_label)}
        if not values <= EXPECTED_LABELS:
            raise RuntimeError(
                f"{record.case_id}: Klassen {sorted(values)}; für diesen Trainer sind nur 0/1/2 erlaubt"
            )
        if np.any(coverage[y0:y1, x0:x1]):
            raise RuntimeError(f"{record.case_id}: Patch überlappt eine bereits belegte Position")
        label[y0:y1, x0:x1] = patch_label.astype(np.uint8, copy=False)
        coverage[y0:y1, x0:x1] += 1
        patch_rows.append(
            {
                "case_id": record.case_id,
                "batch": record.batch_name,
                "verified_revision": int(record.data.get("verified_revision", -1)),
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "skeleton_pixels": int(np.count_nonzero(patch_label == 1)),
                "soma_pixels": int(np.count_nonzero(patch_label == 2)),
            }
        )
    uncovered = int(np.count_nonzero(coverage == 0))
    multiply_covered = int(np.count_nonzero(coverage > 1))
    if uncovered or multiply_covered:
        raise RuntimeError(
            f"{image_id}: Abdeckung nicht exakt (Lücken {uncovered}, mehrfach {multiply_covered})"
        )
    summary = {
        "image_id": image_id,
        "output_case_id": output_case_id(image_id),
        "raw_source": str(raw_source),
        "height": height,
        "width": width,
        "raw_dtype": str(raw.dtype),
        "patches": len(records),
        "skeleton_pixels": int(np.count_nonzero(label == 1)),
        "soma_pixels": int(np.count_nonzero(label == 2)),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "patch_records": patch_rows,
    }
    return raw, label, summary


def write_case_manifest(path: Path, summaries: list[dict[str, Any]], validation_image_id: str) -> None:
    fields = [
        "output_case_id", "image_id", "role", "raw_source", "height", "width",
        "raw_dtype", "patches", "skeleton_pixels", "soma_pixels", "coverage_min", "coverage_max",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    key: (
                        "validation"
                        if key == "role" and summary["image_id"] == validation_image_id
                        else "training"
                        if key == "role"
                        else summary[key]
                    )
                    for key in fields
                }
            )


def build_dataset(exports_dir: Path, output_dataset: Path, validation_image_id: str) -> dict[str, Any]:
    if output_dataset.exists():
        raise RuntimeError(f"Zieldatensatz existiert bereits und wird nicht überschrieben: {output_dataset}")
    staging = output_dataset.with_name(f".{output_dataset.name}.building")
    if staging.exists():
        raise RuntimeError(f"Unvollständiger früherer Aufbau vorhanden: {staging}")

    latest, export_metadata = collect_latest_records(exports_dir)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in latest:
        grouped[str(record.data["image_id"])].append(record)
    if validation_image_id not in grouped:
        raise RuntimeError(
            f"Validierungsbild {validation_image_id!r} fehlt. Vorhanden: {sorted(grouped)}"
        )
    if len(grouped) < 2:
        raise RuntimeError("Mindestens ein Trainings- und ein Validierungsbild werden benötigt")

    staging.mkdir(parents=True)
    images_dir = staging / "imagesTr"
    labels_dir = staging / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    summaries: list[dict[str, Any]] = []
    try:
        for image_id, records in sorted(grouped.items()):
            raw, label, summary = assemble_case(image_id, records)
            case_id = summary["output_case_id"]
            # Write the inspected array, rather than blindly copying the source,
            # so the stored training case is guaranteed to be the verified 2-D image.
            tifffile.imwrite(images_dir / f"{case_id}_0000.tif", raw)
            tifffile.imwrite(labels_dir / f"{case_id}.tif", label, photometric="minisblack")
            summaries.append(summary)

        train = [
            summary["output_case_id"]
            for summary in summaries
            if summary["image_id"] != validation_image_id
        ]
        validation = [
            summary["output_case_id"]
            for summary in summaries
            if summary["image_id"] == validation_image_id
        ]
        dataset_json = {
            "channel_names": {"0": "image"},
            "labels": {"background": 0, "skeleton": 1, "soma": 2},
            "numTraining": len(summaries),
            "file_ending": ".tif",
        }
        (staging / "dataset.json").write_text(
            json.dumps(dataset_json, indent=2), encoding="utf-8"
        )
        splits = [{"train": train, "val": validation}]
        (staging / "splits_final_overview.json").write_text(
            json.dumps(splits, indent=2), encoding="utf-8"
        )
        write_case_manifest(staging / "case_manifest.csv", summaries, validation_image_id)
        summary = {
            "script_version": SCRIPT_VERSION,
            "exports_dir": str(exports_dir),
            "output_dataset": str(output_dataset),
            "latest_patch_count": len(latest),
            "superseded_export_rows_ignored": export_metadata["superseded_rows"],
            "overview_cases": len(summaries),
            "train_image_ids": [item["image_id"] for item in summaries if item["image_id"] != validation_image_id],
            "validation_image_ids": [validation_image_id],
            "train_case_ids": train,
            "validation_case_ids": validation,
            "cases": [{key: value for key, value in item.items() if key != "patch_records"} for item in summaries],
        }
        (staging / "build_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (staging / "patch_provenance.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        (staging / "README.txt").write_text(
            "Dataset140 enthält die aus den neuesten MATS-Patches nahtlos rekonstruierten "
            "Übersichtsbilder. Mosaic 1 und 2 sind Training; Mosaic 3 ist komplett als "
            "Validierung zurückgehalten. Die Netzarchitektur und das 160x160-Patchformat "
            "werden später aus den Dataset139-Plans übernommen.\n",
            encoding="utf-8",
        )
        staging.rename(output_dataset)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        args.exports_dir.resolve(),
        args.output_dataset.resolve(),
        args.validation_image_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
