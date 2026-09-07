from __future__ import annotations

"""Select div10 tracing cells for a new nnU-Net Dataset137 build.

This script does not copy, rename or modify any image data. It only creates
an auditable CSV manifest. The manifest is intentionally fail-closed: a cell
with a missing/ambiguous encryption-key mapping or a missing source file is
not included.

The expected source-cell layout is::

    cellXXXX/
        raw.tif
        seg.swc
        soma.zip
        source_fov.txt

The encryption CSV maps source_fov.txt (Encryption) to OriginalName. The
treatment is the final underscore-separated token in OriginalName. By default
only the *exact* treatments ``Blebbistatin`` and ``DMSO`` are excluded.
Consequently ``dilutedDMSO`` remains included.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_VERSION = "dataset137-div10-selection-v1-2026-08-15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracings-root", type=Path, required=True)
    parser.add_argument("--encryption-key", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty report directory. No source data are modified.",
    )
    parser.add_argument("--key-id-column", default="Encryption")
    parser.add_argument("--original-name-column", default="OriginalName")
    parser.add_argument(
        "--exclude-treatment",
        nargs="+",
        default=["Blebbistatin", "DMSO"],
        help=(
            "Exact treatment names to exclude, case-insensitive. "
            "Default preserves dilutedDMSO because it is not exactly DMSO."
        ),
    )
    parser.add_argument("--case-prefix", default="div10")
    return parser.parse_args()


def require_empty_or_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {path}\n"
            "Choose a new directory to avoid mixing reports from runs."
        )
    path.mkdir(parents=True, exist_ok=True)


def read_key(path: Path, id_column: str, original_name_column: str) -> dict[str, list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Encryption CSV has no header: {path}")
        missing = {id_column, original_name_column} - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"Encryption CSV is missing columns {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )
        mapping: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            encrypted = str(row[id_column]).strip()
            original_name = str(row[original_name_column]).strip()
            if encrypted:
                mapping[encrypted].append(
                    {"encrypted": encrypted, "original_name": original_name}
                )
    if not mapping:
        raise RuntimeError(f"Encryption CSV contains no usable key values: {path}")
    return mapping


def treatment_from_original_name(name: str) -> str:
    stem = Path(name).stem.strip()
    if not stem or "_" not in stem:
        return ""
    return stem.rsplit("_", 1)[-1].strip()


def safe_case_id(prefix: str, relative_cell_path: Path) -> str:
    raw = "_".join((prefix, *relative_cell_path.parts))
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
    if not cleaned:
        raise RuntimeError(f"Cannot make a safe case ID from {relative_cell_path}")
    return cleaned


def first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def main() -> None:
    args = parse_args()
    if not args.tracings_root.is_dir():
        raise FileNotFoundError(args.tracings_root)
    require_empty_or_new_directory(args.output_dir)

    encryption_map = read_key(
        args.encryption_key,
        id_column=args.key_id_column,
        original_name_column=args.original_name_column,
    )
    excluded_treatments = {value.casefold() for value in args.exclude_treatment}

    source_files = sorted(args.tracings_root.rglob("source_fov.txt"))
    if not source_files:
        raise RuntimeError(f"No source_fov.txt files found below {args.tracings_root}")

    rows: list[dict[str, str]] = []
    seen_case_ids: set[str] = set()
    for index, source_fov_path in enumerate(source_files, start=1):
        cell_dir = source_fov_path.parent
        relative_cell_dir = cell_dir.relative_to(args.tracings_root)
        case_id = safe_case_id(args.case_prefix, relative_cell_dir)
        encrypted_fov = source_fov_path.read_text(encoding="utf-8-sig").strip()

        raw_path = first_existing(cell_dir, ("raw.tif", "raw.tiff"))
        swc_path = first_existing(cell_dir, ("seg.swc", "seg-000.swc"))
        soma_path = cell_dir / "soma.zip"

        decision = "include"
        reason = ""
        original_name = ""
        treatment = ""
        matches = encryption_map.get(encrypted_fov, [])

        if case_id in seen_case_ids:
            decision, reason = "reject", "duplicate_case_id"
        elif not encrypted_fov:
            decision, reason = "reject", "empty_source_fov"
        elif len(matches) == 0:
            decision, reason = "reject", "source_fov_not_in_encryption_key"
        elif len(matches) > 1:
            decision, reason = "reject", "ambiguous_source_fov_in_encryption_key"
        else:
            original_name = matches[0]["original_name"]
            treatment = treatment_from_original_name(original_name)
            if not treatment:
                decision, reason = "reject", "treatment_not_parseable_from_original_name"
            elif treatment.casefold() in excluded_treatments:
                decision, reason = "exclude", "excluded_treatment_exact_match"
            elif raw_path is None:
                decision, reason = "reject", "missing_raw_tif"
            elif swc_path is None:
                decision, reason = "reject", "missing_seg_swc"
            elif not soma_path.is_file():
                decision, reason = "reject", "missing_soma_zip"

        seen_case_ids.add(case_id)
        row = {
            "case_id": case_id,
            "cell_dir": str(cell_dir),
            "relative_cell_dir": str(relative_cell_dir),
            "source_fov": encrypted_fov,
            "original_name": original_name,
            "treatment": treatment,
            "decision": decision,
            "reason": reason,
            "raw_path": str(raw_path) if raw_path else "",
            "swc_path": str(swc_path) if swc_path else "",
            "soma_zip_path": str(soma_path) if soma_path.is_file() else "",
        }
        rows.append(row)
        print(
            f"[{index}/{len(source_files)}] {case_id}: "
            f"{decision} {treatment or '(unknown)'} {reason}".rstrip()
        )

    fieldnames = list(rows[0].keys())
    all_manifest = args.output_dir / "all_tracings_manifest.csv"
    included_manifest = args.output_dir / "included_tracings_manifest.csv"
    for path, selected_rows in (
        (all_manifest, rows),
        (included_manifest, [row for row in rows if row["decision"] == "include"]),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)

    decision_counts = Counter(row["decision"] for row in rows)
    treatment_counts = Counter(row["treatment"] or "(unmapped)" for row in rows)
    summary = {
        "script_version": SCRIPT_VERSION,
        "tracings_root": str(args.tracings_root),
        "encryption_key": str(args.encryption_key),
        "excluded_treatments_exact": sorted(args.exclude_treatment),
        "total_cells": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "treatment_counts": dict(sorted(treatment_counts.items())),
        "all_manifest": str(all_manifest),
        "included_manifest": str(included_manifest),
    }
    summary_path = args.output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("DIV10 SELECTION COMPLETE")
    print("=" * 72)
    print(f"Total source cells: {len(rows)}")
    print(f"Included:           {decision_counts['include']}")
    print(f"Excluded:           {decision_counts['exclude']}")
    print(f"Rejected:           {decision_counts['reject']}")
    print(f"Manifest:           {included_manifest}")

    if decision_counts["include"] == 0:
        raise RuntimeError("No new cells were selected; stop before rasterization.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
