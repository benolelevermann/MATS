#!/usr/bin/env python3
"""Register and enforce the permanent manually traced div10_CC hold-out.

The registry uses content hashes and unique ``evotest_div10cc_cell<N>`` IDs.
Generic legacy names such as ``div10_cell300`` are deliberately not sufficient
evidence: unrelated tracing exports can reuse those numbers. ``check`` is
intended as a fail-closed preflight in every future training command. ``audit``
also inspects synthetic-donor manifests, because copying a hold-out cell into a
synthetic multi-cell scene is still data leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from roifile import ImagejRoi


SCRIPT_VERSION = "div10-cc-holdout-v1-2026-09-10"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_ROOT = PROJECT_ROOT / "EvoTest" / "div10_CC"
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "holdouts" / "div10_cc.csv"


def cell_number(value: str) -> int:
    match = re.search(r"(?:cell|Encrypted_)(\d+)(?:_0)?$", value)
    if not match:
        raise ValueError(f"Cannot extract a cell number from {value!r}")
    return int(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_point_roi(path: Path) -> tuple[float, float]:
    rois = ImagejRoi.fromfile(path)
    if len(rois) != 1:
        raise ValueError(f"Expected one ROI in {path}, got {len(rois)}")
    coordinates = rois[0].coordinates()
    if len(coordinates) != 1:
        raise ValueError(f"Expected one point in {path}, got {len(coordinates)}")
    return float(coordinates[0][0]), float(coordinates[0][1])


def build_registry(test_root: Path) -> list[dict[str, object]]:
    assignment_path = test_root / "Decrypted" / "treatment_assignment_full.csv"
    cell_root = test_root / "tracings_M237_totrace_Encrypted"
    if not assignment_path.is_file() or not cell_root.is_dir():
        raise FileNotFoundError(
            f"Incomplete div10_CC test set: {assignment_path} / {cell_root}"
        )

    rows: list[dict[str, object]] = []
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    for assignment in assignments:
        number = cell_number(assignment["cellid"])
        folder = cell_root / f"cell{number}"
        required = [folder / "raw.tif", folder / "seg.swc", folder / "location.zip"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing hold-out files: " + ", ".join(missing))
        global_x, global_y = read_point_roi(folder / "location.zip")
        source_fov_path = folder / "source_fov.txt"
        rows.append(
            {
                "case_id": f"evotest_div10cc_cell{number}",
                "legacy_case_id": f"div10_cell{number}",
                "manual_id": assignment["id"],
                "cell_number": number,
                "treatment": assignment.get("treatment", ""),
                "source_image": assignment.get("matched_original_name", ""),
                "source_fov_encrypted": (
                    source_fov_path.read_text(encoding="utf-8").strip()
                    if source_fov_path.is_file()
                    else assignment.get("raw_txt_content", "")
                ),
                "global_x": f"{global_x:.6f}",
                "global_y": f"{global_y:.6f}",
                "raw_sha256": sha256(folder / "raw.tif"),
                "manual_swc_sha256": sha256(folder / "seg.swc"),
            }
        )
    rows.sort(key=lambda row: int(row["cell_number"]))
    if len({str(row["case_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate case IDs in div10_CC hold-out registry")
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_registry(registry: Path) -> list[dict[str, str]]:
    with registry.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("case_id", "").strip() for row in rows):
        raise ValueError(f"Invalid or empty hold-out registry: {registry}")
    return rows


def case_id_from_training_image(path: Path) -> str | None:
    name = path.name
    for ending in ("_0000.tif", "_0000.tiff"):
        if name.lower().endswith(ending):
            return name[: -len(ending)]
    return None


def find_overlaps(
    dataset: Path, registry_rows: Iterable[dict[str, str]]
) -> list[dict[str, str]]:
    registry_rows = list(registry_rows)
    holdout_ids = {row["case_id"].strip() for row in registry_rows}
    hashes = {
        row["raw_sha256"].strip(): row["case_id"].strip()
        for row in registry_rows
        if row.get("raw_sha256", "").strip()
    }
    overlaps: list[dict[str, str]] = []
    images_dir = dataset / "imagesTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Training images directory is missing: {images_dir}")
    for image_path in sorted(images_dir.iterdir()):
        case_id = case_id_from_training_image(image_path)
        if case_id is None:
            continue
        image_hash = sha256(image_path)
        holdout_case_id = (
            case_id if case_id in holdout_ids else hashes.get(image_hash)
        )
        if holdout_case_id:
            overlaps.append(
                {
                    "dataset": dataset.name,
                    "leakage_kind": "direct_training_case",
                    "holdout_case_id": holdout_case_id,
                    "training_case_id": case_id,
                    "evidence": str(image_path.resolve()),
                }
            )

    for audit_path in sorted(dataset.rglob("*donor*audit*.csv")):
        with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                donor_id = (row.get("case_id") or row.get("donor_case_id") or "").strip()
                donor_path_text = (row.get("image_path") or "").strip()
                donor_path = Path(donor_path_text) if donor_path_text else None
                donor_hash = (
                    sha256(donor_path)
                    if donor_path is not None and donor_path.is_file()
                    else ""
                )
                holdout_case_id = (
                    donor_id if donor_id in holdout_ids else hashes.get(donor_hash)
                )
                if holdout_case_id:
                    overlaps.append(
                        {
                            "dataset": dataset.name,
                            "leakage_kind": "synthetic_donor",
                            "holdout_case_id": holdout_case_id,
                            "training_case_id": (
                                row.get("synthetic_case_id")
                                or row.get("scene_id")
                                or ""
                            ),
                            "evidence": str(audit_path.resolve()),
                        }
                    )
    return overlaps


def audit_datasets(raw_root: Path, registry: Path) -> list[dict[str, str]]:
    registry_rows = load_registry(registry)
    overlaps: list[dict[str, str]] = []
    for dataset in sorted(path for path in raw_root.glob("Dataset*") if path.is_dir()):
        if (dataset / "imagesTr").is_dir():
            overlaps.extend(find_overlaps(dataset, registry_rows))
    return overlaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create the canonical registry")
    create.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    create.add_argument("--output", type=Path, default=DEFAULT_REGISTRY)
    create.add_argument(
        "--local-copy",
        type=Path,
        default=DEFAULT_TEST_ROOT / "holdout_manifest.csv",
    )

    check = subparsers.add_parser("check", help="Fail if one dataset uses hold-out cells")
    check.add_argument("--dataset", type=Path, required=True)
    check.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)

    audit = subparsers.add_parser("audit", help="Audit every local nnU-Net dataset")
    audit.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "nnUNet_raw")
    audit.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    audit.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TEST_ROOT / "training_leakage_audit.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        rows = build_registry(args.test_root)
        write_csv(args.output, rows)
        if args.local_copy.resolve() != args.output.resolve():
            write_csv(args.local_copy, rows)
        print(json.dumps({"cells": len(rows), "registry": str(args.output)}, ensure_ascii=False))
        return 0

    if args.command == "check":
        overlaps = find_overlaps(args.dataset, load_registry(args.registry))
        if overlaps:
            kinds: dict[str, int] = defaultdict(int)
            for row in overlaps:
                kinds[row["leakage_kind"]] += 1
            print(
                "HOLD-OUT LEAKAGE: training is blocked. "
                f"{args.dataset.name} overlaps div10_CC: {dict(kinds)}",
                file=sys.stderr,
            )
            for row in overlaps[:10]:
                print(
                    f"  {row['leakage_kind']}: {row['holdout_case_id']}",
                    file=sys.stderr,
                )
            return 2
        print(f"Hold-out check passed: {args.dataset.name}")
        return 0

    overlaps = audit_datasets(args.raw_root, args.registry)
    if overlaps:
        write_csv(args.output, overlaps)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "dataset,leakage_kind,holdout_case_id,training_case_id,evidence\n",
            encoding="utf-8-sig",
        )
    unique_direct = {
        (row["dataset"], row["holdout_case_id"])
        for row in overlaps
        if row["leakage_kind"] == "direct_training_case"
    }
    unique_donors = {
        (row["dataset"], row["holdout_case_id"])
        for row in overlaps
        if row["leakage_kind"] == "synthetic_donor"
    }
    print(
        json.dumps(
            {
                "records": len(overlaps),
                "direct_dataset_cell_pairs": len(unique_direct),
                "synthetic_donor_dataset_cell_pairs": len(unique_donors),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
