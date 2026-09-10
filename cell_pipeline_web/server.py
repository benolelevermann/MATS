from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import tifffile

from .fov_library import DEFAULT_FOV_ROOT, FovLibrary
from .dataset_inspector import TrainingDatasetInspector
from .evo_export_layout import (
    iter_evo_cell_folders,
    relative_evo_cell_folder,
    source_image_folder,
)
from .pipeline import (
    PipelineSettings,
    _write_preview,
    _write_raw_preview,
    _write_semantic_preview,
    default_settings,
    ensure_review_layers_for_run,
    run_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "web_pipeline_runs"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
TERMINAL_STATES = {"completed", "failed", "cancelled"}
REVIEW_VALUES = {"good", "bad"}
REVIEW_MODES = {"training", "evo"}
MIN_MANUAL_CROP_SIDE = 32
MANUAL_CROP_CLEARANCE = 3
REVIEW_ARTIFACTS = (
    "raw.tif",
    "prediction.tif",
    "postprocessed.tif",
    "skeleton.tif",
    "soma.tif",
    "seg.tif",
    "cell_mask.tif",
    "raw_preview.png",
    "prediction_preview.png",
    "postprocessing_preview.png",
    "preview.png",
    "bounds.json",
    "location.json",
    "metadata.json",
    "manual_crop.json",
)

HYSTERESIS_PROFILES: dict[str, dict[str, object]] = {
    "adaptive": {
        "id": "adaptive",
        "label": "Adaptiv (alpha 1/3)",
        "description": "Bildabhängige Standardschwellen; bei NewTest2 war T_low 0,326.",
        "mode": "adaptive",
    },
    "tlow-0300": {
        "id": "tlow-0300",
        "label": "Niedriger: T_low 0,30",
        "description": "Leicht empfindlicher; nimmt schwächere zusammenhängende Skelettpixel auf.",
        "mode": "fixed",
        "t_high": 0.647,
        "t_low": 0.300,
    },
    "tlow-0250": {
        "id": "tlow-0250",
        "label": "Niedriger: T_low 0,25",
        "description": "Deutlich empfindlicher; mehr Lückenschluss, aber höheres Fusionsrisiko.",
        "mode": "fixed",
        "t_high": 0.647,
        "t_low": 0.250,
    },
    "tlow-0200": {
        "id": "tlow-0200",
        "label": "Sehr niedrig: T_low 0,20",
        "description": "Experimentell; stärkster Lückenschluss und größtes Fehlverbindungsrisiko.",
        "mode": "fixed",
        "t_high": 0.647,
        "t_low": 0.200,
    },
}


def settings_for_hysteresis_profile(
    settings: PipelineSettings, profile_id: str
) -> PipelineSettings:
    try:
        profile = HYSTERESIS_PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"Unknown hysteresis profile: {profile_id}") from error
    if profile["mode"] == "adaptive":
        return replace(settings, hysteresis_mode="adaptive")
    return replace(
        settings,
        hysteresis_mode="fixed",
        hysteresis_t_high=float(profile["t_high"]),
        hysteresis_t_low=float(profile["t_low"]),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    name = Path(value).name
    stem = re.sub(r"_0000$", "", Path(name).stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (stem or "uploaded_image")[:100]


def _review_result(review: dict[str, object]) -> str | None:
    result = review.get("result")
    if result in REVIEW_VALUES:
        return str(result)
    training = review.get("training")
    postprocessing = review.get("postprocessing")
    if training in REVIEW_VALUES and postprocessing in REVIEW_VALUES:
        return "good" if training == "good" and postprocessing == "good" else "bad"
    return None


class JobStore:
    def __init__(
        self,
        runs_root: Path,
        settings: PipelineSettings,
        pipeline_runner=run_pipeline,
        fov_library: FovLibrary | None = None,
        default_hysteresis_profile: str = "adaptive",
        review_mode: str = "training",
        review_root: Path | None = None,
    ) -> None:
        self.runs_root = runs_root
        if review_mode not in REVIEW_MODES:
            raise ValueError(f"Unknown review mode: {review_mode}")
        self.review_mode = review_mode
        default_review_folder = (
            "review_dataset" if review_mode == "training" else "evo_pipeline_input_net141"
        )
        self.review_root = review_root or (runs_root.parent / default_review_folder)
        self.settings = settings
        self.pipeline_runner = pipeline_runner
        self.fov_library = fov_library
        if default_hysteresis_profile not in HYSTERESIS_PROFILES:
            raise ValueError(f"Unknown hysteresis profile: {default_hysteresis_profile}")
        self.default_hysteresis_profile = default_hysteresis_profile
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.review_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, object]] = {}
        self._locks: dict[str, threading.Condition] = {}
        self._queue: queue.PriorityQueue[tuple[int, str, str]] = queue.PriorityQueue()
        self._global_lock = threading.RLock()
        self._review_lock = threading.RLock()
        self._load_history()
        self._worker = threading.Thread(target=self._work_loop, name="cell-pipeline-worker", daemon=True)
        self._worker.start()

    def _load_history(self) -> None:
        queued_jobs: list[tuple[str, str]] = []
        for status_path in sorted(self.runs_root.glob("*/status.json"), reverse=True):
            try:
                job = json.loads(status_path.read_text(encoding="utf-8"))
                job_id = str(job["id"])
                job.setdefault("review_mode", self.review_mode)
                if job.get("status") == "running":
                    job["status"] = "queued"
                    job["phase"] = "queued"
                    job["progress"] = 0
                    job["message"] = "Interrupted job will restart cleanly from its saved input."
                    job["resume_clean"] = True
                    queued_jobs.append((str(job.get("created_at") or ""), job_id))
                elif job.get("status") == "queued":
                    queued_jobs.append((str(job.get("created_at") or ""), job_id))
                if job.get("status") == "completed":
                    ensure_review_layers_for_run(self.runs_root / job_id)
                crops = list(job.get("crops") or [])
                reviews = self._load_review_cells(job_id)
                for crop in crops:
                    folder = str(crop.get("folder") or "")
                    self._decorate_crop(job_id, crop)
                    if folder in reviews:
                        crop["review"] = reviews[folder]
                    elif "review" not in crop:
                        crop["review"] = {
                            "result": None,
                            "approved_for_training": False,
                            "approved_for_evo": False,
                        }
                job["crops"] = crops
                job["review_summary"] = self._review_summary(crops)
                self._jobs[job_id] = job
                self._locks[job_id] = threading.Condition(threading.RLock())
                if job.get("resume_clean"):
                    self._persist(job)
            except Exception:
                continue
        for _created_at, job_id in sorted(queued_jobs):
            self._enqueue(job_id)

    @staticmethod
    def _job_priority(job: dict[str, object]) -> int:
        # A manually uploaded image is an explicit one-off request and must not
        # wait behind a large background FOV batch.
        return 0 if dict(job.get("source") or {}).get("kind") == "upload" else 10

    def _enqueue(self, job_id: str) -> None:
        job = self._jobs[job_id]
        self._queue.put(
            (
                self._job_priority(job),
                str(job.get("created_at") or ""),
                job_id,
            )
        )

    def _persist(self, job: dict[str, object]) -> None:
        run_root = self.runs_root / str(job["id"])
        run_root.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(run_root / "status.json", job)

    def _cell_root(self, job_id: str, folder: str) -> Path:
        if not re.fullmatch(r"cell\d{4}", folder):
            raise KeyError(folder)
        root = self.runs_root / job_id / "04_evo_single_cells" / folder
        if not root.is_dir():
            raise KeyError(folder)
        return root

    @staticmethod
    def _active_cell_root(cell_root: Path) -> Path:
        curated = cell_root / "curated"
        return curated if (curated / "manual_crop.json").is_file() else cell_root

    def _decorate_crop(self, job_id: str, crop: dict[str, object]) -> dict[str, object]:
        folder = str(crop.get("folder") or "")
        cell_root = self._cell_root(job_id, folder)
        relative_root = f"/runs/{job_id}/04_evo_single_cells/{folder}"
        fallback_preview = str(crop.get("preview_url") or f"{relative_root}/preview.png")
        curated = cell_root / "curated"
        manual_crop: dict[str, object] | None = None
        if (curated / "manual_crop.json").is_file():
            try:
                manual_crop = json.loads((curated / "manual_crop.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                manual_crop = None
        active_relative = f"{relative_root}/curated" if manual_crop is not None else relative_root
        cache_token = str((manual_crop or {}).get("updated_at") or "auto").replace(":", "")
        cache_suffix = f"?v={cache_token}"
        active_root = curated if manual_crop is not None else cell_root
        crop["preview_url"] = (
            f"{active_relative}/preview.png{cache_suffix}"
            if (active_root / "preview.png").is_file()
            else fallback_preview
        )
        for key, filename in (
            ("raw_preview_url", "raw_preview.png"),
            ("prediction_preview_url", "prediction_preview.png"),
            ("postprocessing_preview_url", "postprocessing_preview.png"),
        ):
            crop[key] = (
                f"{active_relative}/{filename}{cache_suffix}"
                if (active_root / filename).is_file()
                else crop["preview_url"]
            )
        crop["edit_preview_url"] = f"{relative_root}/preview.png"
        crop["manual_crop"] = manual_crop
        bounds_path = cell_root / "bounds.json"
        if bounds_path.is_file():
            try:
                bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
                crop["source_width"] = int(bounds["width"])
                crop["source_height"] = int(bounds["height"])
            except (OSError, ValueError, TypeError, KeyError):
                pass
        return crop

    def _review_id(self, job_id: str, folder: str) -> str:
        job = self._jobs.get(job_id, {})
        crop = next(
            (
                item
                for item in list(job.get("crops") or [])
                if str(item.get("folder") or "") == folder
            ),
            {},
        )
        case = _safe_name(
            str(crop.get("case_id") or job.get("case") or job.get("filename") or "cell")
        )
        return f"{case}__{job_id}__{folder}"

    @staticmethod
    def _write_json_atomic(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            for attempt in range(10):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError:
                    if attempt == 9:
                        raise
                    # Antivirus/indexing and a browser request can briefly keep
                    # status.json open on Windows. Retrying preserves the atomic
                    # write without turning a completed pipeline into a failed job.
                    time.sleep(min(0.05 * (2**attempt), 0.75))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _review_state_path(self, job_id: str) -> Path:
        return self.runs_root / job_id / "05_review" / "reviews.json"

    def _load_review_cells(self, job_id: str) -> dict[str, dict[str, object]]:
        path = self._review_state_path(job_id)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cells = payload.get("cells") or {}
            if isinstance(cells, dict):
                result = {
                    str(key): dict(value) for key, value in cells.items() if isinstance(value, dict)
                }
                for review in result.values():
                    review["result"] = _review_result(review)
                    accepted = review["result"] == "good"
                    review["approved_for_training"] = (
                        accepted and self.review_mode == "training"
                    )
                    review["approved_for_evo"] = (
                        accepted and self.review_mode == "evo"
                    )
                return result
        except (OSError, ValueError, TypeError):
            pass
        return {}

    @staticmethod
    def _review_summary(crops: list[dict[str, object]]) -> dict[str, int]:
        total = len(crops)
        reviewed = 0
        approved = 0
        rejected = 0
        for crop in crops:
            review = dict(crop.get("review") or {})
            result = _review_result(review)
            if result is not None:
                reviewed += 1
                if result == "good":
                    approved += 1
                else:
                    rejected += 1
        return {
            "total": total,
            "reviewed": reviewed,
            "approved": approved,
            "rejected": rejected,
            "pending": total - reviewed,
        }

    @staticmethod
    def _remove_generated_directory(path: Path, allowed_roots: tuple[Path, ...]) -> None:
        resolved = path.resolve()
        if not any(resolved == root.resolve() or root.resolve() in resolved.parents for root in allowed_roots):
            raise RuntimeError(f"Refusing to remove review path outside generated roots: {path}")
        if path.is_dir():
            shutil.rmtree(path)

    @staticmethod
    def _copy_review_artifacts(source: Path, target: Path, review: dict[str, object]) -> None:
        target.mkdir(parents=True, exist_ok=False)
        for name in REVIEW_ARTIFACTS:
            source_path = source / name
            if source_path.is_file():
                shutil.copy2(source_path, target / name)
        (target / "review.json").write_text(
            json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _next_evo_cell_folder(self) -> str:
        cells_root = self.review_root / "cells"
        identifiers = {
            int(match.group(1))
            for path in iter_evo_cell_folders(cells_root)
            if path.is_dir() and (match := re.fullmatch(r"cell(\d+)", path.name))
        }
        for review_path in self.runs_root.glob("*/05_review/reviews.json"):
            try:
                payload = json.loads(review_path.read_text(encoding="utf-8"))
                for review in dict(payload.get("cells") or {}).values():
                    match = re.fullmatch(
                        r"cell(\d+)", str(dict(review).get("evo_cell_folder") or "")
                    )
                    if match:
                        identifiers.add(int(match.group(1)))
            except (OSError, ValueError, TypeError):
                continue
        return f"cell{max(identifiers, default=0) + 1:06d}"

    def _sync_evo_review_exports(
        self,
        job_id: str,
        folder: str,
        review: dict[str, object],
        cell_root: Path,
    ) -> None:
        local_review_root = self.runs_root / job_id / "05_review"
        allowed_roots = (local_review_root, self.review_root)
        result = _review_result(review)
        for rating in REVIEW_VALUES:
            local_target = local_review_root / "evo" / rating / folder
            self._remove_generated_directory(local_target, allowed_roots)
        if result in REVIEW_VALUES:
            self._copy_review_artifacts(
                cell_root,
                local_review_root / "evo" / str(result) / folder,
                review,
            )

        output_folder = str(review.get("evo_cell_folder") or "")
        if not re.fullmatch(r"cell\d+", output_folder):
            if result == "good":
                raise RuntimeError("Approved Evo cell has no valid output folder.")
            return
        image_folder = str(review.get("evo_image_folder") or "")
        if not image_folder or Path(image_folder).name != image_folder:
            if result == "good":
                raise RuntimeError("Approved Evo cell has no valid source-image folder.")
            return
        target = self.review_root / "cells" / image_folder / output_folder
        self._remove_generated_directory(target, allowed_roots)
        legacy_target = self.review_root / "cells" / output_folder
        if legacy_target != target:
            self._remove_generated_directory(legacy_target, allowed_roots)
        if result != "good":
            return

        required = (
            "raw.tif",
            "skeleton.tif",
            "soma.tif",
            "seg.tif",
            "seg-000.swc",
            "seg.traces",
            "soma.zip",
            "bounds.zip",
        )
        missing = [name for name in required if not (cell_root / name).is_file()]
        if missing:
            raise RuntimeError(
                "Evo cell is not fully finalized: " + ", ".join(missing)
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cell_root, target, copy_function=shutil.copy2)
        self._write_json_atomic(
            target / "review.json",
            review
            | {
                "source_job": job_id,
                "source_cell": folder,
                "source_image": self._jobs[job_id].get("filename"),
                "evo_image_folder": image_folder,
                "review_id": self._review_id(job_id, folder),
            },
        )

    def _sync_review_exports(self, job_id: str, folder: str, review: dict[str, object]) -> None:
        run_root = self.runs_root / job_id
        cell_root = run_root / "04_evo_single_cells" / folder
        if not cell_root.is_dir():
            raise FileNotFoundError(f"Cell folder not found: {cell_root}")
        if self.review_mode == "evo":
            self._sync_evo_review_exports(job_id, folder, review, cell_root)
            return
        source = self._active_cell_root(cell_root)
        review_id = self._review_id(job_id, folder)
        legacy_review_id = f"{job_id}__{folder}"
        local_review_root = run_root / "05_review"
        allowed_roots = (local_review_root, self.review_root)

        result = _review_result(review)
        for axis in ("training", "postprocessing"):
            value = result
            for rating in REVIEW_VALUES:
                local_target = local_review_root / axis / rating / folder
                global_target = self.review_root / axis / rating / review_id
                self._remove_generated_directory(local_target, allowed_roots)
                self._remove_generated_directory(global_target, allowed_roots)
                if legacy_review_id != review_id:
                    self._remove_generated_directory(
                        self.review_root / axis / rating / legacy_review_id, allowed_roots
                    )
            if value in REVIEW_VALUES:
                self._copy_review_artifacts(source, local_review_root / axis / str(value) / folder, review)
                self._copy_review_artifacts(source, self.review_root / axis / str(value) / review_id, review)

        approved_root = self.review_root / "approved"
        approved_paths = {
            "image": approved_root / "imagesTr" / f"{review_id}_0000.tif",
            "label": approved_root / "labelsTr" / f"{review_id}.tif",
            "skeleton": approved_root / "skeletons" / f"{review_id}.tif",
            "soma": approved_root / "somas" / f"{review_id}.tif",
            "metadata": approved_root / "metadata" / f"{review_id}.json",
        }
        for path in approved_paths.values():
            path.unlink(missing_ok=True)
        if legacy_review_id != review_id:
            legacy_paths = (
                approved_root / "imagesTr" / f"{legacy_review_id}_0000.tif",
                approved_root / "labelsTr" / f"{legacy_review_id}.tif",
                approved_root / "skeletons" / f"{legacy_review_id}.tif",
                approved_root / "somas" / f"{legacy_review_id}.tif",
                approved_root / "metadata" / f"{legacy_review_id}.json",
            )
            for path in legacy_paths:
                path.unlink(missing_ok=True)
        if result == "good":
            for directory in {path.parent for path in approved_paths.values()}:
                directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "raw.tif", approved_paths["image"])
            shutil.copy2(source / "seg.tif", approved_paths["label"])
            shutil.copy2(source / "skeleton.tif", approved_paths["skeleton"])
            shutil.copy2(source / "soma.tif", approved_paths["soma"])
            source_metadata = {}
            metadata_path = source / "metadata.json"
            if metadata_path.is_file():
                source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._write_json_atomic(
                approved_paths["metadata"],
                {
                    "review": review,
                    "job": {
                        "id": job_id,
                        "case": self._jobs[job_id].get("case"),
                        "filename": self._jobs[job_id].get("filename"),
                        "source": self._jobs[job_id].get("source"),
                    },
                    "source_metadata": source_metadata,
                },
            )

    def _write_review_state(self, job_id: str, cells: dict[str, dict[str, object]]) -> None:
        path = self._review_state_path(job_id)
        payload = {
            "version": 1,
            "job_id": job_id,
            "review_mode": self.review_mode,
            "updated_at": _utc_now(),
            "cells": cells,
        }
        self._write_json_atomic(path, payload)
        csv_path = path.with_name("reviews.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "job_id",
                    "case",
                    "cell",
                    "result",
                    "training",
                    "postprocessing",
                    "approved_for_training",
                    "approved_for_evo",
                    "source_image",
                    "evo_image_folder",
                    "evo_cell_folder",
                    "updated_at",
                ],
            )
            writer.writeheader()
            for folder in sorted(cells):
                review = cells[folder]
                writer.writerow(
                    {
                        "job_id": job_id,
                        "case": self._jobs.get(job_id, {}).get("case", ""),
                        "cell": folder,
                        "result": _review_result(review) or "",
                        "training": review.get("training") or "",
                        "postprocessing": review.get("postprocessing") or "",
                        "approved_for_training": int(
                            _review_result(review) == "good"
                            and self.review_mode == "training"
                        ),
                        "approved_for_evo": int(
                            _review_result(review) == "good"
                            and self.review_mode == "evo"
                        ),
                        "source_image": self._jobs.get(job_id, {}).get("filename", ""),
                        "evo_image_folder": review.get("evo_image_folder") or "",
                        "evo_cell_folder": review.get("evo_cell_folder") or "",
                        "updated_at": review.get("updated_at") or "",
                    }
                )

    def _rebuild_global_review_index(self) -> None:
        rows: list[dict[str, object]] = []
        for path in sorted(self.runs_root.glob("*/05_review/reviews.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                job_id = str(payload["job_id"])
                for folder, review in dict(payload.get("cells") or {}).items():
                    rows.append(
                        {
                            "job_id": job_id,
                            "case": review.get("case") or "",
                            "cell": folder,
                            "result": _review_result(review) or "",
                            "training": review.get("training") or "",
                            "postprocessing": review.get("postprocessing") or "",
                            "approved_for_training": int(
                                _review_result(review) == "good"
                                and self.review_mode == "training"
                            ),
                            "approved_for_evo": int(
                                _review_result(review) == "good"
                                and self.review_mode == "evo"
                            ),
                            "source_image": review.get("source_image") or "",
                            "evo_image_folder": review.get("evo_image_folder") or "",
                            "evo_cell_folder": review.get("evo_cell_folder") or "",
                            "updated_at": review.get("updated_at") or "",
                        }
                    )
            except (OSError, ValueError, TypeError, KeyError):
                continue
        index_path = self.review_root / "review_index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "job_id",
                    "case",
                    "cell",
                    "result",
                    "training",
                    "postprocessing",
                    "approved_for_training",
                    "approved_for_evo",
                    "source_image",
                    "evo_image_folder",
                    "evo_cell_folder",
                    "updated_at",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        if self.review_mode == "training":
            approved_root = self.review_root / "approved"
            approved_count = len(list((approved_root / "labelsTr").glob("*.tif")))
            self._write_json_atomic(
                approved_root / "dataset.json",
                {
                    "channel_names": {"0": "image"},
                    "labels": {"background": 0, "skeleton": 1, "soma": 2},
                    "numTraining": approved_count,
                    "file_ending": ".tif",
                },
            )
        else:
            selected_rows = []
            for row in rows:
                if not row.get("approved_for_evo") or not row.get("evo_cell_folder"):
                    continue
                image_folder = str(row.get("evo_image_folder") or "")
                cell_folder = str(row["evo_cell_folder"])
                relative_folder = (
                    relative_evo_cell_folder(image_folder, cell_folder)
                    if image_folder
                    else cell_folder
                )
                selected_rows.append(
                    {
                        "cell_folder": relative_folder,
                        "image_folder": image_folder,
                        "cell_name": cell_folder,
                        "source_image": row.get("source_image") or "",
                        "job_id": row["job_id"],
                        "case": row["case"],
                        "source_cell": row["cell"],
                    }
                )
            manifest_path = self.review_root / "selected_cells.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "cell_folder",
                        "image_folder",
                        "cell_name",
                        "source_image",
                        "job_id",
                        "case",
                        "source_cell",
                    ],
                )
                writer.writeheader()
                writer.writerows(selected_rows)
            self._write_json_atomic(
                self.review_root / "selection_summary.json",
                {
                    "mode": "evo",
                    "selected_cells": len(selected_rows),
                    "source_images": len(
                        {row["image_folder"] for row in selected_rows if row["image_folder"]}
                    ),
                    "cells_root": str((self.review_root / "cells").resolve()),
                    "updated_at": _utc_now(),
                },
            )

    def crop_info(self, job_id: str, folder: str) -> dict[str, object]:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        cell_root = self._cell_root(job_id, folder)
        raw = np.squeeze(np.asarray(tifffile.imread(cell_root / "raw.tif")))
        semantic = np.squeeze(np.asarray(tifffile.imread(cell_root / "seg.tif")))
        if raw.ndim != 2 or semantic.shape != raw.shape:
            raise ValueError("Cell source layers have incompatible dimensions.")
        coordinates = np.argwhere(semantic > 0)
        if coordinates.size == 0:
            raise ValueError("The cell label is empty.")
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0) + 1
        manual_crop = None
        manual_path = cell_root / "curated" / "manual_crop.json"
        if manual_path.is_file():
            manual_crop = json.loads(manual_path.read_text(encoding="utf-8"))
        selection = dict(
            manual_crop
            or {"x": 0, "y": 0, "width": int(raw.shape[1]), "height": int(raw.shape[0])}
        )
        return {
            "job_id": job_id,
            "cell": folder,
            "source_width": int(raw.shape[1]),
            "source_height": int(raw.shape[0]),
            "minimum_side": MIN_MANUAL_CROP_SIDE,
            "clearance": MANUAL_CROP_CLEARANCE,
            "label_bounds": {
                "x_min": int(minimum[1]),
                "y_min": int(minimum[0]),
                "x_max_exclusive": int(maximum[1]),
                "y_max_exclusive": int(maximum[0]),
            },
            "selection": {
                "x": int(selection["x"]),
                "y": int(selection["y"]),
                "width": int(selection["width"]),
                "height": int(selection["height"]),
            },
            "preview_url": f"/runs/{job_id}/04_evo_single_cells/{folder}/preview.png",
        }

    def recrop_cell(
        self,
        job_id: str,
        folder: str,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> dict[str, object]:
        if self.review_mode == "evo":
            raise ValueError(
                "Crop resizing is disabled in Evo mode because it would invalidate the finalized traces."
            )
        if job_id not in self._locks:
            raise KeyError(job_id)
        condition = self._locks[job_id]
        with condition:
            job = self._jobs[job_id]
            if job.get("status") != "completed":
                raise ValueError("Only completed jobs can be cropped.")
            crops = list(job.get("crops") or [])
            crop = next((item for item in crops if item.get("folder") == folder), None)
            if crop is None:
                raise KeyError(folder)
            cell_root = self._cell_root(job_id, folder)
            layer_names = (
                "raw.tif",
                "prediction.tif",
                "postprocessed.tif",
                "skeleton.tif",
                "soma.tif",
                "seg.tif",
                "cell_mask.tif",
            )
            layers = {
                name: np.squeeze(np.asarray(tifffile.imread(cell_root / name)))
                for name in layer_names
                if (cell_root / name).is_file()
            }
            if "raw.tif" not in layers or "seg.tif" not in layers:
                raise FileNotFoundError("Cell raw image or semantic label is missing.")
            shape = layers["raw.tif"].shape
            if len(shape) != 2 or any(array.shape != shape for array in layers.values()):
                raise ValueError("Cell source layers have incompatible dimensions.")
            if width < MIN_MANUAL_CROP_SIDE or height < MIN_MANUAL_CROP_SIDE:
                raise ValueError(f"Manual crops must be at least {MIN_MANUAL_CROP_SIDE} px wide and high.")
            x1, y1 = x + width, y + height
            if x < 0 or y < 0 or x1 > shape[1] or y1 > shape[0]:
                raise ValueError("The manual crop lies outside the automatic cell crop.")
            coordinates = np.argwhere(layers["seg.tif"] > 0)
            if coordinates.size == 0:
                raise ValueError("The cell label is empty.")
            minimum = coordinates.min(axis=0)
            maximum = coordinates.max(axis=0)
            if (
                int(minimum[1]) < x + MANUAL_CROP_CLEARANCE
                or int(minimum[0]) < y + MANUAL_CROP_CLEARANCE
                or int(maximum[1]) >= x1 - MANUAL_CROP_CLEARANCE
                or int(maximum[0]) >= y1 - MANUAL_CROP_CLEARANCE
            ):
                raise ValueError(
                    f"Keep at least {MANUAL_CROP_CLEARANCE} px around every skeleton and soma pixel."
                )

            updated_at = _utc_now()
            manual_crop = {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "source_width": int(shape[1]),
                "source_height": int(shape[0]),
                "updated_at": updated_at,
            }
            temporary = cell_root / f"curated.tmp-{uuid.uuid4().hex[:8]}"
            temporary.mkdir(parents=False, exist_ok=False)
            try:
                cropped = {
                    name: array[y:y1, x:x1]
                    for name, array in layers.items()
                }
                for name, array in cropped.items():
                    tifffile.imwrite(temporary / name, array)
                raw_crop = cropped["raw.tif"]
                prediction = cropped.get("prediction.tif", cropped["seg.tif"])
                postprocessed = cropped.get("postprocessed.tif", cropped["seg.tif"])
                semantic = cropped["seg.tif"]
                skeleton = semantic == 1
                soma = semantic == 2
                _write_raw_preview(temporary / "raw_preview.png", raw_crop)
                _write_semantic_preview(temporary / "prediction_preview.png", raw_crop, prediction)
                _write_semantic_preview(
                    temporary / "postprocessing_preview.png", raw_crop, postprocessed
                )
                _write_preview(temporary / "preview.png", raw_crop, skeleton, soma)

                source_metadata = {}
                metadata_path = cell_root / "metadata.json"
                if metadata_path.is_file():
                    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                original_bounds = dict(source_metadata.get("bounds") or {})
                original_x = int(original_bounds.get("x_min", 0))
                original_y = int(original_bounds.get("y_min", 0))
                soma_coordinates = np.argwhere(soma)
                centroid_y, centroid_x = soma_coordinates.mean(axis=0)
                metadata = dict(source_metadata)
                metadata["manual_recrop"] = manual_crop
                metadata["bounds"] = {
                    "x_min": original_x + x,
                    "y_min": original_y + y,
                    "x_max_exclusive": original_x + x1,
                    "y_max_exclusive": original_y + y1,
                    "width": int(width),
                    "height": int(height),
                }
                metadata["location"] = {
                    "local_x": float(centroid_x),
                    "local_y": float(centroid_y),
                    "global_x": float(original_x + x + centroid_x),
                    "global_y": float(original_y + y + centroid_y),
                }
                self._write_json_atomic(temporary / "metadata.json", metadata)
                self._write_json_atomic(temporary / "manual_crop.json", manual_crop)
                curated = cell_root / "curated"
                if curated.is_dir():
                    history_path = cell_root / "manual_crop_history.jsonl"
                    old_crop_path = curated / "manual_crop.json"
                    if old_crop_path.is_file():
                        old_crop = json.loads(old_crop_path.read_text(encoding="utf-8"))
                        with history_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(old_crop, ensure_ascii=False) + "\n")
                    shutil.rmtree(curated)
                temporary.rename(curated)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

            review = dict(crop.get("review") or {})
            if _review_result(review) in REVIEW_VALUES:
                with self._review_lock:
                    self._sync_review_exports(job_id, folder, review)
                    self._rebuild_global_review_index()
            self._decorate_crop(job_id, crop)
            job["crops"] = crops
            job["updated_at"] = updated_at
            job["version"] = int(job.get("version", 0)) + 1
            self._persist(job)
            condition.notify_all()
            return {"crop": crop, "review_summary": self._review_summary(crops)}

    def rate_cell(
        self,
        job_id: str,
        folder: str,
        training: str | None = None,
        postprocessing: str | None = None,
        result: str | None = None,
    ) -> dict[str, object]:
        if training is not None and training not in REVIEW_VALUES:
            raise ValueError("training must be 'good', 'bad' or null.")
        if postprocessing is not None and postprocessing not in REVIEW_VALUES:
            raise ValueError("postprocessing must be 'good', 'bad' or null.")
        if result is not None and result not in REVIEW_VALUES:
            raise ValueError("result must be 'good', 'bad' or null.")
        single_decision = result in REVIEW_VALUES
        if result is None and training in REVIEW_VALUES and postprocessing in REVIEW_VALUES:
            result = "good" if training == "good" and postprocessing == "good" else "bad"
        if single_decision:
            # Keep the legacy columns synchronized for old exports and analysis scripts.
            training = result
            postprocessing = result
        if job_id not in self._locks:
            raise KeyError(job_id)
        condition = self._locks[job_id]
        with condition:
            job = self._jobs[job_id]
            if job.get("status") != "completed":
                raise ValueError("Only completed jobs can be reviewed.")
            crops = list(job.get("crops") or [])
            crop = next((item for item in crops if item.get("folder") == folder), None)
            if crop is None or not re.fullmatch(r"cell\d{4}", folder):
                raise KeyError(folder)
            review = {
                "job_id": job_id,
                "case": job.get("case"),
                "cell": folder,
                "result": result,
                "training": training,
                "postprocessing": postprocessing,
                "approved_for_training": (
                    result == "good" and self.review_mode == "training"
                ),
                "approved_for_evo": (
                    result == "good" and self.review_mode == "evo"
                ),
                "updated_at": _utc_now(),
            }
            review_cells = self._load_review_cells(job_id)
            with self._review_lock:
                previous = dict(review_cells.get(folder) or {})
                if self.review_mode == "evo":
                    output_folder = str(previous.get("evo_cell_folder") or "")
                    image_folder = str(
                        previous.get("evo_image_folder")
                        or source_image_folder(str(job.get("filename") or job.get("case") or "image"))
                    )
                    if (
                        result == "good"
                        and not re.fullmatch(r"cell\d+", output_folder)
                    ):
                        output_folder = self._next_evo_cell_folder()
                    if re.fullmatch(r"cell\d+", output_folder):
                        review["evo_cell_folder"] = output_folder
                    review["source_image"] = str(job.get("filename") or "")
                    review["evo_image_folder"] = image_folder
                self._sync_review_exports(job_id, folder, review)
                review_cells[folder] = review
                self._write_review_state(job_id, review_cells)
                self._rebuild_global_review_index()
            crop["review"] = review
            job["crops"] = crops
            job["review_summary"] = self._review_summary(crops)
            job["updated_at"] = _utc_now()
            job["version"] = int(job.get("version", 0)) + 1
            self._persist(job)
            condition.notify_all()
            return {
                "review": review,
                "review_summary": job["review_summary"],
                "approved_download_url": "/api/review-dataset/download",
            }

    def approved_archive(self) -> Path:
        with self._review_lock:
            self._rebuild_global_review_index()
            if self.review_mode == "training":
                approved_root = self.review_root / "approved"
                archive_path = self.review_root / "approved_cells.zip"
                archive_prefix = Path("approved")
            else:
                approved_root = self.review_root / "cells"
                archive_path = self.review_root / "evo_selected_cells.zip"
                archive_prefix = Path("cells")
            temporary = archive_path.with_suffix(".zip.tmp")
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for path in sorted(approved_root.rglob("*")):
                    if path.is_file():
                        archive.write(path, archive_prefix / path.relative_to(approved_root))
            os.replace(temporary, archive_path)
            return archive_path

    def review_dataset_summary(self) -> dict[str, object]:
        if self.review_mode == "training":
            approved = len(
                list((self.review_root / "approved" / "labelsTr").glob("*.tif"))
            )
            cells_root = self.review_root / "approved"
        else:
            approved = len(
                list(iter_evo_cell_folders(self.review_root / "cells"))
            )
            cells_root = self.review_root / "cells"
        return {
            "mode": self.review_mode,
            "approved": approved,
            "root": str(self.review_root),
            "cells_root": str(cells_root),
            "download_url": "/api/review-dataset/download",
        }

    def fov_library_public(self) -> dict[str, object]:
        if self.fov_library is None:
            return {"available": False, "count": 0, "fovs": []}
        return self.fov_library.public()

    def expand_fovs(
        self, identifiers: list[str], count_per_overview: int = 6
    ) -> dict[str, object]:
        if self.fov_library is None:
            raise ValueError("No FOV library is configured.")
        created = self.fov_library.expand_from_examples(identifiers, count_per_overview)
        return {
            "created": len(created),
            "overview_count": len({record.provenance for record in created}),
            "fovs": [record.public() for record in created],
            "library": self.fov_library.public(),
        }

    def _allocate_job(self, filename: str) -> tuple[str, str, Path, Path]:
        case_name = _safe_name(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        run_root = self.runs_root / job_id
        input_root = run_root / "01_input"
        input_root.mkdir(parents=True, exist_ok=False)
        input_path = input_root / f"{case_name}_0000.tif"
        return job_id, case_name, run_root, input_path

    def _register_job(
        self,
        job_id: str,
        case_name: str,
        filename: str,
        input_path: Path,
        source: dict[str, object] | None = None,
        hysteresis_profile: str | None = None,
    ) -> dict[str, object]:
        profile_id = hysteresis_profile or self.default_hysteresis_profile
        if profile_id not in HYSTERESIS_PROFILES:
            raise ValueError(f"Unknown hysteresis profile: {profile_id}")
        job: dict[str, object] = {
            "id": job_id,
            "filename": Path(filename).name,
            "case": case_name,
            "input_path": str(input_path),
            "source": source or {"kind": "upload"},
            "model": {
                "dataset_id": self.settings.dataset_id,
                "trainer": self.settings.trainer,
                "plans": self.settings.plans,
                "checkpoint": self.settings.checkpoint,
            },
            "hysteresis": dict(HYSTERESIS_PROFILES[profile_id]),
            "review_mode": self.review_mode,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "message": "Input ready. The job is waiting for the processing worker.",
            "details": {},
            "events": [
                {
                    "time": _utc_now(),
                    "phase": "queued",
                    "progress": 0,
                    "message": "Input ready. Job queued.",
                }
            ],
            "summary": None,
            "crops": [],
            "review_summary": {
                "total": 0,
                "reviewed": 0,
                "approved": 0,
                "rejected": 0,
                "pending": 0,
            },
            "download_url": None,
            "version": 1,
        }
        with self._global_lock:
            self._jobs[job_id] = job
            self._locks[job_id] = threading.Condition(threading.RLock())
            self._persist(job)
        self._enqueue(job_id)
        return self.public(job_id)

    def create_fov_jobs(
        self,
        identifiers: list[str],
        hysteresis_profile: str | None = None,
    ) -> list[dict[str, object]]:
        if self.fov_library is None:
            raise ValueError("No FOV library is configured.")
        unique = list(dict.fromkeys(str(value) for value in identifiers))
        if not unique:
            raise ValueError("Select at least one FOV.")
        if len(unique) > 192:
            raise ValueError("At most 192 FOVs can be queued at once.")
        records = [self.fov_library.get(identifier) for identifier in unique]
        jobs: list[dict[str, object]] = []
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        for batch_index, record in enumerate(records, start=1):
            if not record.source.is_file():
                raise FileNotFoundError(f"FOV not found: {record.source}")
            job_id, case_name, _run_root, input_path = self._allocate_job(record.filename)
            try:
                shutil.copy2(record.source, input_path)
            except Exception:
                shutil.rmtree(input_path.parents[1], ignore_errors=True)
                raise
            jobs.append(
                self._register_job(
                    job_id,
                    case_name,
                    record.filename,
                    input_path,
                    source={
                        "kind": "fov_library",
                        "batch_id": batch_id,
                        "batch_index": batch_index,
                        "batch_total": len(records),
                        "fov_id": record.identifier,
                        "acquisition": record.acquisition,
                        "div": record.div,
                        "plate": record.plate,
                        "well": record.well,
                        "fov": record.fov,
                        "x": record.x,
                        "y": record.y,
                        "source_path": str(record.source),
                    },
                    hysteresis_profile=hysteresis_profile,
                )
            )
        return jobs

    def create(
        self,
        filename: str,
        content_length: int,
        stream,
        hysteresis_profile: str | None = None,
    ) -> dict[str, object]:
        if content_length <= 0:
            raise ValueError("The upload is empty.")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError("The upload is larger than 512 MB.")
        job_id, case_name, run_root, input_path = self._allocate_job(filename)
        remaining = content_length
        try:
            with input_path.open("wb") as output:
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("The upload ended before Content-Length bytes were received.")
                    output.write(block)
                    remaining -= len(block)
        except Exception:
            shutil.rmtree(run_root, ignore_errors=True)
            raise
        return self._register_job(
            job_id,
            case_name,
            filename,
            input_path,
            hysteresis_profile=hysteresis_profile,
        )

    def _update(
        self,
        job_id: str,
        phase: str,
        progress: int,
        message: str,
        details: dict[str, object] | None = None,
        status: str | None = None,
    ) -> None:
        condition = self._locks[job_id]
        with condition:
            job = self._jobs[job_id]
            if status is not None:
                job["status"] = status
            job["phase"] = phase
            job["progress"] = int(progress)
            job["message"] = message
            job["updated_at"] = _utc_now()
            if details:
                current_details = dict(job.get("details") or {})
                current_details.update(details)
                job["details"] = current_details
            events = list(job.get("events") or [])
            if not events or events[-1].get("message") != message:
                events.append(
                    {
                        "time": _utc_now(),
                        "phase": phase,
                        "progress": int(progress),
                        "message": message,
                    }
                )
                job["events"] = events[-160:]
            job["version"] = int(job.get("version", 0)) + 1
            self._persist(job)
            condition.notify_all()

    def _work_loop(self) -> None:
        while True:
            _priority, _created_at, job_id = self._queue.get()
            try:
                self._execute(job_id)
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        job = self._jobs[job_id]
        run_root = self.runs_root / job_id
        if job.pop("resume_clean", False):
            cleanup_paths = list(run_root.glob("02_prediction_dataset*"))
            cleanup_paths.extend(
                run_root / name
                for name in ("03_adaptive_hysteresis", "04_evo_single_cells", "work_neurotreetracer")
            )
            for path in cleanup_paths:
                if path.is_dir():
                    shutil.rmtree(path)
            for name in ("evo_single_cells.zip", "pipeline_summary.json", "error_traceback.txt"):
                (run_root / name).unlink(missing_ok=True)
            self._persist(job)
        self._update(job_id, "starting", 2, "Pipeline is starting.", status="running")

        def progress(
            phase: str,
            percent: int,
            message: str,
            details: dict[str, object] | None = None,
        ) -> None:
            self._update(job_id, phase, percent, message, details=details, status="running")

        try:
            profile_id = str(
                dict(job.get("hysteresis") or {}).get(
                    "id", self.default_hysteresis_profile
                )
            )
            job_settings = settings_for_hysteresis_profile(self.settings, profile_id)
            result = self.pipeline_runner(
                Path(str(job["input_path"])),
                run_root,
                progress,
                job_settings,
            )
            manifest_path = run_root / "04_evo_single_cells" / "manifest.csv"
            crops: list[dict[str, object]] = []
            existing_reviews = self._load_review_cells(job_id)
            if manifest_path.is_file():
                with manifest_path.open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        folder = row["folder"]
                        relative_root = f"/runs/{job_id}/04_evo_single_cells/{folder}"
                        crop = {
                            "folder": folder,
                            "source": row["source"],
                            "skeleton_pixels": int(row["skeleton_pixels"]),
                            "soma_pixels": int(row["soma_pixels"]),
                            "preview_url": f"{relative_root}/preview.png",
                            "metadata_url": f"{relative_root}/metadata.json",
                            "source_width": int(row["x_max_exclusive"]) - int(row["x_min"]),
                            "source_height": int(row["y_max_exclusive"]) - int(row["y_min"]),
                            "review": existing_reviews.get(
                                folder,
                                {
                                    "result": None,
                                    "approved_for_training": False,
                                    "approved_for_evo": False,
                                },
                            ),
                        }
                        crops.append(
                            self._decorate_crop(job_id, crop)
                        )
            with self._locks[job_id]:
                stored = self._jobs[job_id]
                stored["summary"] = result
                stored["crops"] = crops
                stored["review_summary"] = self._review_summary(crops)
                stored["download_url"] = f"/runs/{job_id}/evo_single_cells.zip"
            self._update(
                job_id,
                "complete",
                100,
                f"Complete: {len(crops)} Evo-ready single-cell crops.",
                status="completed",
            )
        except Exception as error:
            (run_root / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self._update(
                job_id,
                "failed",
                int(self._jobs[job_id].get("progress", 0)),
                str(error),
                status="failed",
            )

    def public(self, job_id: str) -> dict[str, object]:
        with self._global_lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = dict(self._jobs[job_id])
        job.pop("input_path", None)
        return job

    def list(self) -> list[dict[str, object]]:
        with self._global_lock:
            identifiers = sorted(self._jobs, reverse=True)[:250]
        result = []
        for job_id in identifiers:
            job = self.public(job_id)
            job["events"] = list(job.get("events") or [])[-3:]
            job["crops"] = []
            result.append(job)
        return result

    def wait_for_change(self, job_id: str, version: int, timeout: float = 20.0) -> dict[str, object]:
        if job_id not in self._locks:
            raise KeyError(job_id)
        condition = self._locks[job_id]
        deadline = time.monotonic() + timeout
        with condition:
            while int(self._jobs[job_id].get("version", 0)) <= version:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                condition.wait(timeout=remaining)
        return self.public(job_id)


class CellPipelineHandler(BaseHTTPRequestHandler):
    server_version = "CellPipeline/1.0"

    @property
    def store(self) -> JobStore:
        return self.server.job_store  # type: ignore[attr-defined]

    @property
    def dataset_inspector(self) -> TrainingDatasetInspector:
        return self.server.dataset_inspector  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        sys_message = format % args
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {sys_message}")

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    def _serve_bytes(self, body: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, cache: bool = False, download: bool = False) -> None:
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "File not found.")
            return
        mime, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                self.wfile.write(block)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        recrop_match = re.fullmatch(
            r"/api/jobs/([A-Za-z0-9_-]+)/crops/(cell\d{4})/recrop", parsed.path
        )
        if recrop_match:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 16 * 1024:
                    raise ValueError("Invalid crop payload size.")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Crop payload must be a JSON object.")
                values = [payload.get(key) for key in ("x", "y", "width", "height")]
                if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                    raise ValueError("x, y, width and height must be integer pixels.")
                result = self.store.recrop_cell(
                    recrop_match.group(1), recrop_match.group(2), *values
                )
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Job or cell not found.")
                return
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
                return
            self._json(result)
            return
        review_match = re.fullmatch(
            r"/api/jobs/([A-Za-z0-9_-]+)/reviews/(cell\d{4})", parsed.path
        )
        if review_match:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 16 * 1024:
                    raise ValueError("Invalid review payload size.")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Review payload must be a JSON object.")
                result = self.store.rate_cell(
                    review_match.group(1),
                    review_match.group(2),
                    payload.get("training"),
                    payload.get("postprocessing"),
                    payload.get("result"),
                )
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Job or cell not found.")
                return
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
                return
            self._json(result)
            return
        if parsed.path == "/api/fovs/expand":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 128 * 1024:
                    raise ValueError("Invalid FOV expansion payload size.")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                identifiers = payload.get("fov_ids") if isinstance(payload, dict) else None
                count = payload.get("count_per_overview", 6) if isinstance(payload, dict) else 6
                if not isinstance(identifiers, list) or not all(
                    isinstance(value, str) for value in identifiers
                ):
                    raise ValueError("fov_ids must be a list of FOV identifiers.")
                if isinstance(count, bool) or not isinstance(count, int):
                    raise ValueError("count_per_overview must be an integer.")
                result = self.store.expand_fovs(identifiers, count)
            except KeyError as error:
                self._error(HTTPStatus.BAD_REQUEST, f"Unknown FOV: {error.args[0]}")
                return
            except (
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                FileNotFoundError,
                FileExistsError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
                return
            self._json(result, status=HTTPStatus.CREATED)
            return
        if parsed.path == "/api/fov-jobs":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 128 * 1024:
                    raise ValueError("Invalid FOV selection payload size.")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                identifiers = payload.get("fov_ids") if isinstance(payload, dict) else None
                if not isinstance(identifiers, list) or not all(
                    isinstance(value, str) for value in identifiers
                ):
                    raise ValueError("fov_ids must be a list of FOV identifiers.")
                hysteresis_profile = (
                    payload.get("hysteresis_profile") if isinstance(payload, dict) else None
                )
                if hysteresis_profile is not None and not isinstance(hysteresis_profile, str):
                    raise ValueError("hysteresis_profile must be a string.")
                jobs = self.store.create_fov_jobs(identifiers, hysteresis_profile)
            except KeyError as error:
                self._error(HTTPStatus.BAD_REQUEST, f"Unknown FOV: {error.args[0]}")
                return
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError, FileNotFoundError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
                return
            self._json({"jobs": jobs}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path != "/api/jobs":
            self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
            return
        filename = unquote(self.headers.get("X-Filename", "uploaded_image.tif"))
        if Path(filename).suffix.lower() not in {".tif", ".tiff"}:
            self._error(HTTPStatus.BAD_REQUEST, "Please upload a .tif or .tiff image.")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            hysteresis_profile = unquote(
                self.headers.get("X-Hysteresis-Profile", self.store.default_hysteresis_profile)
            )
            job = self.store.create(
                filename,
                content_length,
                self.rfile,
                hysteresis_profile=hysteresis_profile,
            )
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        self._json(job, status=HTTPStatus.ACCEPTED)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/config":
            settings = self.store.settings
            self._json(
                {
                    "dataset_id": settings.dataset_id,
                    "trainer": settings.trainer,
                    "plans": settings.plans,
                    "checkpoint": settings.checkpoint,
                    "review_mode": self.store.review_mode,
                    "review_root": str(self.store.review_root),
                    "review_cells_root": str(
                        self.store.review_dataset_summary()["cells_root"]
                    ),
                    "default_hysteresis_profile": self.store.default_hysteresis_profile,
                    "hysteresis_profiles": list(HYSTERESIS_PROFILES.values()),
                }
            )
            return
        if path == "/api/training-dataset/139":
            try:
                self._json(self.dataset_inspector.index())
            except FileNotFoundError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        dataset_case_match = re.fullmatch(
            r"/api/training-dataset/139/cases/([A-Za-z0-9._-]+)", path
        )
        if dataset_case_match:
            try:
                seed = int(parse_qs(parsed.query).get("seed", ["1"])[0])
                self._json(self.dataset_inspector.detail(dataset_case_match.group(1), seed))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Trainingsfall nicht gefunden.")
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Ungültiger Augmentations-Seed.")
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        dataset_view_match = re.fullmatch(
            r"/api/training-dataset/139/cases/([A-Za-z0-9._-]+)/views/"
            r"(raw|raw-label|preprocessed|training)\.png",
            path,
        )
        if dataset_view_match:
            try:
                seed = int(parse_qs(parsed.query).get("seed", ["1"])[0])
                body = self.dataset_inspector.render_view(
                    dataset_view_match.group(1), dataset_view_match.group(2), seed
                )
                self._serve_bytes(body, "image/png", cache=dataset_view_match.group(2) != "training")
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Trainingsfall oder Ansicht nicht gefunden.")
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Ungültiger Augmentations-Seed.")
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        if path == "/api/fovs":
            self._json(self.store.fov_library_public())
            return
        fov_preview_match = re.fullmatch(r"/fov-previews/([A-Za-z0-9._-]+)\.png", path)
        if fov_preview_match:
            try:
                if self.store.fov_library is None:
                    raise KeyError(fov_preview_match.group(1))
                record = self.store.fov_library.get(fov_preview_match.group(1))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "FOV preview not found.")
                return
            self._serve_file(record.preview, cache=True)
            return
        crop_info_match = re.fullmatch(
            r"/api/jobs/([A-Za-z0-9_-]+)/crops/(cell\d{4})", path
        )
        if crop_info_match:
            try:
                self._json(self.store.crop_info(crop_info_match.group(1), crop_info_match.group(2)))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Job or cell not found.")
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        if path == "/api/review-dataset":
            self._json(self.store.review_dataset_summary())
            return
        if path == "/api/review-dataset/download":
            try:
                archive = self.store.approved_archive()
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
                return
            self._serve_file(archive, download=True)
            return
        if path == "/api/jobs":
            self._json({"jobs": self.store.list()})
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)", path)
        if match:
            try:
                self._json(self.store.public(match.group(1)))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Job not found.")
            return
        event_match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/events", path)
        if event_match:
            self._serve_events(event_match.group(1), parse_qs(parsed.query))
            return
        if path.startswith("/runs/"):
            relative = Path(unquote(path[len("/runs/") :]))
            if relative.is_absolute() or ".." in relative.parts:
                self._error(HTTPStatus.BAD_REQUEST, "Invalid path.")
                return
            root = self.store.runs_root.resolve()
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Invalid path.")
                return
            self._serve_file(candidate, cache=True, download=candidate.suffix.lower() == ".zip")
            return
        if path == "/":
            self._serve_file(STATIC_ROOT / "index.html")
            return
        if path == "/dataset-inspector":
            self._serve_file(STATIC_ROOT / "dataset-inspector.html")
            return
        static_path = STATIC_ROOT / path.lstrip("/")
        try:
            static_path.resolve().relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid path.")
            return
        self._serve_file(static_path, cache=True)

    def _serve_events(self, job_id: str, query: dict[str, list[str]]) -> None:
        try:
            version = int(query.get("version", ["0"])[0])
            job = self.store.wait_for_change(job_id, version)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Job not found.")
            return
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid version.")
            return
        body = f"data: {json.dumps(job, ensure_ascii=False)}\n\n".encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class CellPipelineServer(ThreadingHTTPServer):
    daemon_threads = True
    # HTTPServer enables SO_REUSEADDR. On Windows that can let two independent
    # processes listen on the same port and serve contradictory in-memory job
    # states. A restart must fail loudly instead of sharing the review port.
    allow_reuse_address = os.name != "nt"

    def __init__(
        self,
        address: tuple[str, int],
        job_store: JobStore,
        dataset_inspector: TrainingDatasetInspector,
    ):
        super().__init__(address, CellPipelineHandler)
        self.job_store = job_store
        self.dataset_inspector = dataset_inspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local configurable single-cell pipeline website.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dataset-id", type=int, default=138)
    parser.add_argument(
        "--trainer", default="nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
    )
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument(
        "--hysteresis-profile",
        choices=tuple(HYSTERESIS_PROFILES),
        default="adaptive",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--review-mode",
        choices=tuple(sorted(REVIEW_MODES)),
        default="training",
    )
    parser.add_argument("--review-root", type=Path)
    parser.add_argument(
        "--fov-root",
        type=Path,
        default=DEFAULT_FOV_ROOT,
        help="Prepared MICA FOV library containing fov_manifest.csv.",
    )
    parser.add_argument("--no-fiji", action="store_true", help="Skip Evo/SNT finalization.")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--open-path", default="/", choices=("/", "/dataset-inspector"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.review_mode == "evo" and args.no_fiji:
        raise ValueError(
            "Evo review mode requires Fiji finalization; remove --no-fiji."
        )
    settings = default_settings(
        device=args.device,
        dataset_id=args.dataset_id,
        trainer=args.trainer,
        plans=args.plans,
        checkpoint=args.checkpoint,
        finalize_fiji=not args.no_fiji,
    )
    fov_library = FovLibrary(args.fov_root) if (args.fov_root / "fov_manifest.csv").is_file() else None
    store = JobStore(
        args.runs_root.resolve(),
        settings,
        fov_library=fov_library,
        default_hysteresis_profile=args.hysteresis_profile,
        review_mode=args.review_mode,
        review_root=args.review_root.resolve() if args.review_root else None,
    )
    dataset_inspector = TrainingDatasetInspector.from_project(PROJECT_ROOT)
    server = CellPipelineServer((args.host, args.port), store, dataset_inspector)
    url = f"http://{args.host}:{args.port}"
    browser_url = url if args.open_path == "/" else f"{url}{args.open_path}"
    print(f"Cell Pipeline is ready at {url}")
    print(f"Jobs and outputs: {store.runs_root}")
    print(f"Review mode: {store.review_mode} | selected output: {store.review_root}")
    print(
        f"Network: Dataset {settings.dataset_id} | {settings.trainer} | "
        f"{settings.plans} | {settings.checkpoint}"
    )
    print(
        f"FOV library: {fov_library.root if fov_library is not None else 'not configured'}"
    )
    if args.open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(browser_url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
