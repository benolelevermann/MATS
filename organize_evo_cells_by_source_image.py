from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from cell_pipeline_web.evo_export_layout import source_image_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group flat Evo cell folders by their original input image."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Optional pipeline-runs folder used to recover the exact uploaded filename.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def source_image_for_cell(cell: Path, runs_root: Path | None = None) -> str:
    review = load_json(cell / "review.json")
    metadata = load_json(cell / "metadata.json")
    job_id = str(review.get("job_id") or review.get("source_job") or "")
    if runs_root is not None and job_id:
        status = load_json(runs_root / job_id / "status.json")
        if str(status.get("filename") or "").strip():
            return str(status["filename"]).strip()
    candidates = (
        review.get("source_image"),
        review.get("case"),
        metadata.get("source_image"),
        metadata.get("source_case"),
        metadata.get("case"),
    )
    for value in candidates:
        if str(value or "").strip():
            return str(value).strip()
    raise ValueError(f"No source image recorded for {cell}")


def link_or_copy_file(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def group_cells(
    source_root: Path,
    output_root: Path,
    runs_root: Path | None = None,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    runs_root = runs_root.resolve() if runs_root is not None else None
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    cells = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and re.fullmatch(r"cell\d+", path.name)
    )
    if not cells:
        raise ValueError(f"No direct cell* folders found in {source_root}")

    rows: list[dict[str, object]] = []
    modes: Counter[str] = Counter()
    folder_sources: dict[str, str] = {}
    for cell in cells:
        source_image = source_image_for_cell(cell, runs_root)
        image_folder = source_image_folder(source_image)
        previous_source = folder_sources.setdefault(image_folder, source_image)
        if previous_source != source_image:
            raise ValueError(
                f"Two source names map to the same folder {image_folder!r}: "
                f"{previous_source!r} and {source_image!r}"
            )
        target = output_root / image_folder / cell.name
        if target.exists():
            raise FileExistsError(f"Cell target already exists: {target}")
        target.mkdir(parents=True)
        file_count = 0
        for source_file in sorted(cell.rglob("*")):
            if not source_file.is_file():
                continue
            relative = source_file.relative_to(cell)
            mode = link_or_copy_file(source_file, target / relative)
            modes[mode] += 1
            file_count += 1
        (target / "source_image_group.json").write_text(
            json.dumps(
                {
                    "source_image": source_image,
                    "image_folder": image_folder,
                    "source_cell": str(cell),
                    "grouped_cell": str(target),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "image_folder": image_folder,
                "source_image": source_image,
                "cell": cell.name,
                "source_cell": str(cell),
                "grouped_cell": str(target),
                "files": file_count,
            }
        )

    with (output_root / "grouping_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(str(row["image_folder"]) for row in rows)
    summary = {
        "source": str(source_root),
        "output": str(output_root),
        "runs_root": str(runs_root) if runs_root is not None else None,
        "cells": len(rows),
        "source_images": len(counts),
        "cells_per_image": dict(sorted(counts.items())),
        "file_transfer_modes": dict(modes),
    }
    (output_root / "grouping_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            group_cells(args.source, args.output, args.runs_root),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
