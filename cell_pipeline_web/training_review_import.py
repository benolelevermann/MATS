from __future__ import annotations

"""Import manually traced single-cell crops into the existing web review deck.

The source folders are treated as read-only. Every usable raw/SWC/soma triplet is
rasterized into a self-contained local review job. The web server can then use
the same left/right decision and non-destructive recrop workflow as prediction
jobs.
"""

import csv
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tifffile

from rasterize_div10_training_masks import (
    read_2d_raw,
    skeleton_from_swc,
    soma_from_roi_zip,
)

from .pipeline import _write_preview, _write_raw_preview, _write_semantic_preview


SCRIPT_VERSION = "rocki-training-review-import-v1-2026-09-01"
DEFAULT_JOB_ID = "training_review_rocki_20260901"
REQUIRED_FILES = ("raw.tif", "seg.swc", "soma.zip")


@dataclass(frozen=True)
class TrainingSource:
    key: str
    label: str
    root: Path


@dataclass(frozen=True)
class SourceCell:
    source: TrainingSource
    number: int
    root: Path
    source_fov: str


DEFAULT_SOURCES = (
    TrainingSource(
        "rocki_m239_cc_postrocki",
        "CC M239 · DIV10 · post ROCKi",
        Path(
            r"X:\Projects\scEvoView\ExperimentalData\InVitro\LiveCellImaging"
            r"\ROCKiValidation_FKER\CC_M239_seeded300826_S24tdTom_MFL26#02"
            r"\260809_cc_M239_S24tdTomMFL26#02_div10_postROCKi\TracingsFelix"
        ),
    ),
    TrainingSource(
        "rocki_m238_cc",
        "CC M238 · DIV10",
        Path(r"X:\Felix\Analysis\ROCKi\Validation\TracingsCC\div10_M238\TracingsFelix"),
    ),
    TrainingSource(
        "rocki_mfl26_mc",
        "MC tdTom MFL26#02 · DIV10",
        Path(
            r"X:\Felix\Analysis\ROCKi\Validation\TracingsMC"
            r"\TdTomMFL26#02_Tracing\TracingsFelix"
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cell_number(path: Path) -> int:
    match = re.fullmatch(r"cell(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def collect_cells(
    sources: Iterable[TrainingSource],
) -> tuple[list[SourceCell], list[dict[str, str]]]:
    cells: list[SourceCell] = []
    skipped: list[dict[str, str]] = []
    for source in sources:
        if not source.root.is_dir():
            raise FileNotFoundError(f"Training source not found: {source.root}")
        directories = sorted(
            (
                path
                for path in source.root.iterdir()
                if path.is_dir() and re.fullmatch(r"cell\d+", path.name, re.IGNORECASE)
            ),
            key=_cell_number,
        )
        for directory in directories:
            missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
            if missing:
                skipped.append(
                    {
                        "source_key": source.key,
                        "source_label": source.label,
                        "source_cell": directory.name,
                        "source_path": str(directory),
                        "reason": "missing: " + ", ".join(missing),
                    }
                )
                continue
            source_fov_path = directory / "source_fov.txt"
            source_fov = (
                source_fov_path.read_text(encoding="utf-8-sig").strip()
                if source_fov_path.is_file()
                else "unknown"
            )
            cells.append(
                SourceCell(source, _cell_number(directory), directory, source_fov or "unknown")
            )
    return cells, skipped


def default_label_loader(cell: SourceCell, shape: tuple[int, int]):
    skeleton, spacing_x, spacing_y = skeleton_from_swc(cell.root / "seg.swc", shape)
    soma, roi_count = soma_from_roi_zip(cell.root / "soma.zip", shape)
    return skeleton, soma, spacing_x, spacing_y, roi_count


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _case_id(cell: SourceCell) -> str:
    fov = re.sub(r"[^A-Za-z0-9_-]+", "_", cell.source_fov).strip("_") or "unknown"
    return f"{cell.source.key}_fov{fov}_cell{cell.number}"


def import_training_review(
    sources: Iterable[TrainingSource],
    runs_root: Path,
    job_id: str = DEFAULT_JOB_ID,
    *,
    overwrite: bool = False,
    label_loader: Callable[[SourceCell, tuple[int, int]], tuple] = default_label_loader,
) -> dict[str, object]:
    sources = tuple(sources)
    runs_root.mkdir(parents=True, exist_ok=True)
    destination = runs_root / job_id
    if destination.exists():
        if not overwrite:
            status_path = destination / "status.json"
            if status_path.is_file():
                return json.loads(status_path.read_text(encoding="utf-8"))
            raise RuntimeError(f"Review job already exists but has no status.json: {destination}")
        shutil.rmtree(destination)

    source_cells, skipped = collect_cells(sources)
    temporary = runs_root / f"{job_id}.tmp-{uuid.uuid4().hex[:8]}"
    cell_root = temporary / "04_evo_single_cells"
    cell_root.mkdir(parents=True)
    crops: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    failures: list[dict[str, str]] = list(skipped)

    try:
        for source_cell in source_cells:
            try:
                raw = read_2d_raw(source_cell.root / "raw.tif")
                skeleton, soma, spacing_x, spacing_y, roi_count = label_loader(
                    source_cell, raw.shape
                )
                skeleton = np.asarray(skeleton, dtype=bool)
                soma = np.asarray(soma, dtype=bool)
                if skeleton.shape != raw.shape or soma.shape != raw.shape:
                    raise RuntimeError("Rasterized label shape differs from raw image")
                semantic = np.zeros(raw.shape, dtype=np.uint8)
                semantic[skeleton] = 1
                semantic[soma] = 2
                skeleton_pixels = int(np.count_nonzero(semantic == 1))
                soma_pixels = int(np.count_nonzero(semantic == 2))
                if skeleton_pixels == 0 or soma_pixels == 0:
                    raise RuntimeError(
                        f"empty label class: skeleton={skeleton_pixels}, soma={soma_pixels}"
                    )

                folder = f"cell{len(crops) + 1:04d}"
                output = cell_root / folder
                output.mkdir()
                tifffile.imwrite(output / "raw.tif", raw)
                tifffile.imwrite(output / "prediction.tif", semantic)
                tifffile.imwrite(output / "postprocessed.tif", semantic)
                tifffile.imwrite(output / "skeleton.tif", (semantic == 1).astype(np.uint8))
                tifffile.imwrite(output / "soma.tif", (semantic == 2).astype(np.uint8))
                tifffile.imwrite(output / "seg.tif", semantic)
                tifffile.imwrite(output / "cell_mask.tif", (semantic > 0).astype(np.uint8))
                _write_raw_preview(output / "raw_preview.png", raw)
                _write_semantic_preview(output / "prediction_preview.png", raw, semantic)
                _write_semantic_preview(output / "postprocessing_preview.png", raw, semantic)
                _write_preview(output / "preview.png", raw, semantic == 1, semantic == 2)

                case_id = _case_id(source_cell)
                soma_coordinates = np.argwhere(semantic == 2)
                centroid_y, centroid_x = soma_coordinates.mean(axis=0)
                bounds = {
                    "x_min": 0,
                    "y_min": 0,
                    "x_max_exclusive": int(raw.shape[1]),
                    "y_max_exclusive": int(raw.shape[0]),
                    "width": int(raw.shape[1]),
                    "height": int(raw.shape[0]),
                }
                location = {
                    "local_x": float(centroid_x),
                    "local_y": float(centroid_y),
                    "global_x": float(centroid_x),
                    "global_y": float(centroid_y),
                }
                metadata = {
                    "import_version": SCRIPT_VERSION,
                    "case_id": case_id,
                    "classification_source": "manual_training_tracing",
                    "source_collection": source_cell.source.key,
                    "source_collection_label": source_cell.source.label,
                    "source_root": str(source_cell.source.root),
                    "source_cell": source_cell.root.name,
                    "source_cell_path": str(source_cell.root),
                    "source_fov": source_cell.source_fov,
                    "source_files": {
                        "raw": str(source_cell.root / "raw.tif"),
                        "swc": str(source_cell.root / "seg.swc"),
                        "soma": str(source_cell.root / "soma.zip"),
                    },
                    "spacing_x_um_per_px": float(spacing_x),
                    "spacing_y_um_per_px": float(spacing_y),
                    "soma_roi_count": int(roi_count),
                    "skeleton_pixels": skeleton_pixels,
                    "soma_pixels": soma_pixels,
                    "bounds": bounds,
                    "location": location,
                }
                (output / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
                (output / "location.json").write_text(
                    json.dumps(location, indent=2), encoding="utf-8"
                )
                (output / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                crop = {
                    "folder": folder,
                    "case_id": case_id,
                    "display_name": f"{source_cell.source.label} · FOV {source_cell.source_fov} · {source_cell.root.name}",
                    "source": "manual_training_tracing",
                    "source_collection": source_cell.source.key,
                    "source_collection_label": source_cell.source.label,
                    "source_cell": source_cell.root.name,
                    "source_fov": source_cell.source_fov,
                    "skeleton_pixels": skeleton_pixels,
                    "soma_pixels": soma_pixels,
                    "review": {"result": None, "approved_for_training": False},
                }
                crops.append(crop)
                manifest.append(
                    {
                        **crop,
                        "source_path": str(source_cell.root),
                        "width": int(raw.shape[1]),
                        "height": int(raw.shape[0]),
                        "soma_roi_count": int(roi_count),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "source_key": source_cell.source.key,
                        "source_label": source_cell.source.label,
                        "source_cell": source_cell.root.name,
                        "source_path": str(source_cell.root),
                        "reason": str(error),
                    }
                )

        if not crops:
            raise RuntimeError("No complete training cells could be imported.")
        _write_csv(
            cell_root / "manifest.csv",
            manifest,
            [
                "folder",
                "case_id",
                "display_name",
                "source_collection",
                "source_collection_label",
                "source_cell",
                "source_fov",
                "source_path",
                "width",
                "height",
                "skeleton_pixels",
                "soma_pixels",
                "soma_roi_count",
            ],
        )
        _write_csv(
            temporary / "import_failures.csv",
            failures,
            ["source_key", "source_label", "source_cell", "source_path", "reason"],
        )
        now = utc_now()
        status: dict[str, object] = {
            "id": job_id,
            "filename": "ROCKi-Tracings · manueller Trainingsdaten-Review",
            "case": "rocki_manual_tracings",
            "source": {
                "kind": "training_review_import",
                "collections": [
                    {"key": source.key, "label": source.label, "root": str(source.root)}
                    for source in sources
                ],
            },
            "review_kind": "manual_training_data",
            "created_at": now,
            "updated_at": now,
            "status": "completed",
            "phase": "complete",
            "progress": 100,
            "message": f"{len(crops)} manuell getracte Zellen sind für den Review bereit.",
            "details": {"skipped_or_failed": len(failures)},
            "events": [
                {
                    "time": now,
                    "phase": "complete",
                    "progress": 100,
                    "message": f"Lokaler Review-Import abgeschlossen: {len(crops)} Zellen.",
                }
            ],
            "summary": {
                "exported_cell_count": len(crops),
                "isolated_exported": 0,
                "neurotreetracer_cells_exported": len(crops),
                "neurotreetracer_groups_rejected": len(failures),
                "source_cell_directories": len(source_cells) + len(skipped),
                "import_failures": len(failures),
            },
            "crops": crops,
            "review_summary": {
                "total": len(crops),
                "reviewed": 0,
                "approved": 0,
                "rejected": 0,
                "pending": len(crops),
            },
            "download_url": None,
            "version": 1,
        }
        (temporary / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (temporary / "import_summary.json").write_text(
            json.dumps(
                {
                    "script_version": SCRIPT_VERSION,
                    "job_id": job_id,
                    "imported": len(crops),
                    "skipped_or_failed": len(failures),
                    "sources": [str(source.root) for source in sources],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.rename(destination)
        return status
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

