from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import tifffile
from PIL import Image

from cell_pipeline_web.pipeline import _normalize_u8, default_settings
from cell_pipeline_web.evo_export_layout import (
    iter_evo_cell_folders,
    relative_evo_cell_folder,
    source_image_folder,
)
from .pipeline import (
    BlindSelectionSettings,
    PIPELINE_VERSION,
    run_blind_selection_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
TERMINAL_STATES = {"completed", "failed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    stem = re.sub(r"_0000$", "", Path(Path(value).name).stem)
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "overview")[:100]


class SelectionJobStore:
    def __init__(
        self,
        runs_root: Path,
        output_root: Path,
        learning_root: Path,
        settings: BlindSelectionSettings,
        pipeline_runner=run_blind_selection_pipeline,
    ) -> None:
        self.runs_root = runs_root
        self.output_root = output_root
        self.learning_root = learning_root
        self.settings = settings
        self.pipeline_runner = pipeline_runner
        for path in (runs_root, output_root / "cells", learning_root / "jobs"):
            path.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, object]] = {}
        self._locks: dict[str, threading.Condition] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._global_lock = threading.RLock()
        self._output_lock = threading.RLock()
        self._load_history()
        self._worker = threading.Thread(
            target=self._work_loop, name="blind-selection-worker", daemon=True
        )
        self._worker.start()

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
                    time.sleep(min(0.05 * (2**attempt), 0.75))
        finally:
            temporary.unlink(missing_ok=True)

    def _persist(self, job: dict[str, object]) -> None:
        self._write_json_atomic(
            self.runs_root / str(job["id"]) / "status.json", job
        )

    def _load_history(self) -> None:
        queued: list[tuple[str, str]] = []
        for path in sorted(self.runs_root.glob("*/status.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                job_id = str(job["id"])
                if job.get("status") == "running":
                    selection_path = self.runs_root / job_id / "manual_selection.json"
                    if selection_path.is_file():
                        job.update(
                            status="queued",
                            phase="queued",
                            progress=1,
                            message="Interrupted processing will restart from the fixed selection.",
                            resume_clean=True,
                        )
                        queued.append((str(job.get("created_at") or ""), job_id))
                    else:
                        job.update(
                            status="selecting",
                            phase="manual_selection",
                            progress=0,
                            message="Select cells in the raw image.",
                        )
                elif job.get("status") == "queued":
                    queued.append((str(job.get("created_at") or ""), job_id))
                self._jobs[job_id] = job
                self._locks[job_id] = threading.Condition(threading.RLock())
                self._persist(job)
            except Exception:
                continue
        for _created, job_id in sorted(queued):
            self._queue.put(job_id)

    def _allocate(self, filename: str) -> tuple[str, Path, Path]:
        case = _safe_name(filename)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"selection_{stamp}_{uuid.uuid4().hex[:8]}"
        run_root = self.runs_root / job_id
        input_path = run_root / "01_input" / f"{case}_0000.tif"
        input_path.parent.mkdir(parents=True)
        return job_id, run_root, input_path

    @staticmethod
    def _write_selection_preview(raw: np.ndarray, path: Path) -> tuple[int, int]:
        image = Image.fromarray(_normalize_u8(raw))
        image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        image.save(path, optimize=True)
        return image.size

    def create(self, filename: str, content_length: int, stream) -> dict[str, object]:
        if content_length <= 0:
            raise ValueError("The upload is empty.")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError("The upload is larger than 512 MB.")
        job_id, run_root, input_path = self._allocate(filename)
        try:
            remaining = content_length
            with input_path.open("wb") as output:
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Upload ended before all bytes were received.")
                    output.write(block)
                    remaining -= len(block)
            raw = np.squeeze(np.asarray(tifffile.imread(input_path)))
            if raw.ndim != 2 or min(raw.shape) < 32:
                raise ValueError(f"A two-dimensional TIFF is required; got {raw.shape}.")
            if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
                raise ValueError("The TIFF must contain finite numeric pixels.")
            preview_width, preview_height = self._write_selection_preview(
                raw, run_root / "raw_preview.png"
            )
        except Exception:
            shutil.rmtree(run_root, ignore_errors=True)
            raise
        now = _utc_now()
        job: dict[str, object] = {
            "id": job_id,
            "filename": Path(filename).name,
            "case": _safe_name(filename),
            "input_path": str(input_path),
            "created_at": now,
            "updated_at": now,
            "status": "selecting",
            "phase": "manual_selection",
            "progress": 0,
            "message": "Select the desired cells in the raw image. No model has run yet.",
            "image": {
                "width": int(raw.shape[1]),
                "height": int(raw.shape[0]),
                "preview_width": preview_width,
                "preview_height": preview_height,
                "preview_url": f"/runs/{job_id}/raw_preview.png",
            },
            "selection": {"locked": False, "count": 0, "points": []},
            "events": [],
            "summary": None,
            "crops": [],
            "review_summary": {"total": 0, "reviewed": 0, "approved": 0, "rejected": 0, "pending": 0},
            "version": 1,
        }
        with self._global_lock:
            self._jobs[job_id] = job
            self._locks[job_id] = threading.Condition(threading.RLock())
            self._persist(job)
        return self.public(job_id)

    def submit_selection(
        self, job_id: str, points: list[dict[str, object]]
    ) -> dict[str, object]:
        if job_id not in self._locks:
            raise KeyError(job_id)
        condition = self._locks[job_id]
        with condition:
            job = self._jobs[job_id]
            if job.get("status") != "selecting":
                raise ValueError("The selection is already locked and cannot be changed.")
            if not points:
                raise ValueError("Select at least one cell before continuing.")
            if len(points) > 5000:
                raise ValueError("A maximum of 5000 cells can be selected per image.")
            image = dict(job["image"])
            width, height = int(image["width"]), int(image["height"])
            clean: list[dict[str, object]] = []
            for index, point in enumerate(points, start=1):
                if not isinstance(point, dict):
                    raise ValueError("Every selection point must be an object.")
                x, y = point.get("x"), point.get("y")
                if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    raise ValueError("Every point needs numeric x and y coordinates.")
                x, y = int(round(x)), int(round(y))
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(f"Selection {index} lies outside the image.")
                clean.append(
                    {"selection_id": f"selection{index:04d}", "x": x, "y": y}
                )
            selection = {
                "schema_version": 1,
                "pipeline_version": PIPELINE_VERSION,
                "job_id": job_id,
                "case": job["case"],
                "source_filename": job["filename"],
                "image_width": width,
                "image_height": height,
                "created_at": _utc_now(),
                "locked": True,
                "blind_to_model_output": True,
                "points": clean,
            }
            selection_path = self.runs_root / job_id / "manual_selection.json"
            if selection_path.exists():
                raise ValueError("The immutable selection file already exists.")
            self._write_json_atomic(selection_path, selection)
            self._archive_selection_example(job, selection)
            job["selection"] = {
                "locked": True,
                "count": len(clean),
                "points": clean,
                "url": f"/runs/{job_id}/manual_selection.json",
            }
            job.update(
                status="queued",
                phase="queued",
                progress=1,
                message=f"Selection locked: {len(clean)} cells. Processing is queued.",
                updated_at=_utc_now(),
                version=int(job.get("version", 0)) + 1,
            )
            self._persist(job)
            condition.notify_all()
            self._queue.put(job_id)
            return self.public(job_id)

    def _archive_selection_example(
        self, job: dict[str, object], selection: dict[str, object]
    ) -> None:
        job_id = str(job["id"])
        destination = self.learning_root / "jobs" / job_id
        destination.mkdir(parents=True, exist_ok=False)
        source = Path(str(job["input_path"]))
        target = destination / "raw.tif"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        self._write_json_atomic(destination / "manual_selection.json", selection)
        self._rebuild_learning_index()

    def _update_learning_result(self, job_id: str) -> None:
        job = self._jobs[job_id]
        root = self.learning_root / "jobs" / job_id
        matching_path = self.runs_root / job_id / "model_matching.json"
        matching = (
            json.loads(matching_path.read_text(encoding="utf-8"))
            if matching_path.is_file()
            else None
        )
        reviews_path = self.runs_root / job_id / "05_review" / "reviews.json"
        reviews = (
            json.loads(reviews_path.read_text(encoding="utf-8"))
            if reviews_path.is_file()
            else {"cells": {}}
        )
        self._write_json_atomic(
            root / "result.json",
            {
                "job_id": job_id,
                "status": job.get("status"),
                "summary": job.get("summary"),
                "matching": matching,
                "reviews": reviews.get("cells", {}),
                "updated_at": _utc_now(),
            },
        )
        self._rebuild_learning_index()

    def _rebuild_learning_index(self) -> None:
        rows: list[dict[str, object]] = []
        for selection_path in sorted((self.learning_root / "jobs").glob("*/manual_selection.json")):
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                result_path = selection_path.with_name("result.json")
                result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
                rows.append(
                    {
                        "job_id": selection["job_id"],
                        "raw": str(selection_path.with_name("raw.tif").resolve()),
                        "selection": str(selection_path.resolve()),
                        "result": str(result_path.resolve()) if result_path.is_file() else None,
                        "selected_points": len(selection.get("points") or []),
                        "status": result.get("status", "selection_locked"),
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self._write_json_atomic(
            self.learning_root / "index.json",
            {"schema_version": 1, "examples": rows, "updated_at": _utc_now()},
        )
        temporary = self.learning_root / f"index.{uuid.uuid4().hex}.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, self.learning_root / "index.jsonl")

    def _update(
        self, job_id: str, phase: str, progress: int, message: str,
        details: dict[str, object] | None = None, status: str | None = None,
    ) -> None:
        condition = self._locks[job_id]
        with condition:
            job = self._jobs[job_id]
            if status:
                job["status"] = status
            job.update(phase=phase, progress=int(progress), message=message, updated_at=_utc_now())
            if details:
                job["details"] = dict(job.get("details") or {}) | details
            events = list(job.get("events") or [])
            if not events or events[-1].get("message") != message:
                events.append({"time": _utc_now(), "phase": phase, "progress": int(progress), "message": message})
            job["events"] = events[-120:]
            job["version"] = int(job.get("version", 0)) + 1
            self._persist(job)
            condition.notify_all()

    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute(job_id)
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        job = self._jobs[job_id]
        run_root = self.runs_root / job_id
        try:
            if job.pop("resume_clean", False):
                for path in list(run_root.glob("02_prediction_dataset*")) + [
                    run_root / "03_hysteresis_tlow0200", run_root / "04_evo_single_cells"
                ]:
                    if path.is_dir():
                        shutil.rmtree(path)
            selection = json.loads(
                (run_root / "manual_selection.json").read_text(encoding="utf-8")
            )
            self._update(job_id, "starting", 2, "Processing the locked selection.", status="running")

            def progress(phase: str, percent: int, message: str, details=None) -> None:
                self._update(job_id, phase, percent, message, details, "running")

            result = self.pipeline_runner(
                Path(str(job["input_path"])), run_root, selection, progress, self.settings
            )
            extra_manifest = {
                str(row["folder"]): row
                for row in list(result.get("manifest") or [])
            }
            crops: list[dict[str, object]] = []
            manifest_path = run_root / "04_evo_single_cells" / "manifest.csv"
            if manifest_path.is_file():
                with manifest_path.open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        folder = str(row["folder"])
                        relative = f"/runs/{job_id}/04_evo_single_cells/{folder}"
                        extra = dict(extra_manifest.get(folder) or {})
                        crops.append(
                            {
                                "folder": folder,
                                "selection_id": extra.get("selection_id"),
                                "selection_index": extra.get("selection_index"),
                                "source": row.get("source"),
                                "skeleton_pixels": int(row.get("skeleton_pixels") or 0),
                                "soma_pixels": int(row.get("soma_pixels") or 0),
                                "raw_preview_url": f"{relative}/raw_preview.png",
                                "prediction_preview_url": f"{relative}/prediction_preview.png",
                                "postprocessing_preview_url": f"{relative}/postprocessing_preview.png",
                                "preview_url": f"{relative}/preview.png",
                                "review": {"result": None, "approved_for_evo": False},
                            }
                        )
            with self._locks[job_id]:
                job = self._jobs[job_id]
                job["summary"] = result
                job["crops"] = crops
                job["review_summary"] = {
                    "total": len(crops), "reviewed": 0, "approved": 0,
                    "rejected": 0, "pending": len(crops),
                }
                job["download_url"] = f"/runs/{job_id}/evo_candidate_cells.zip"
            self._update(
                job_id, "complete", 100,
                f"Complete: {len(crops)} selected-cell candidates are ready for review.",
                status="completed",
            )
            self._update_learning_result(job_id)
        except Exception as error:
            (run_root / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self._update(job_id, "failed", int(job.get("progress", 0)), str(error), status="failed")
            if (self.learning_root / "jobs" / job_id).is_dir():
                self._update_learning_result(job_id)

    @staticmethod
    def _review_summary(crops: list[dict[str, object]]) -> dict[str, int]:
        values = [dict(crop.get("review") or {}).get("result") for crop in crops]
        approved = values.count("good")
        rejected = values.count("bad")
        return {
            "total": len(crops), "reviewed": approved + rejected,
            "approved": approved, "rejected": rejected,
            "pending": len(crops) - approved - rejected,
        }

    def _next_output_folder(self) -> str:
        identifiers: set[int] = set()
        for path in iter_evo_cell_folders(self.output_root / "cells"):
            match = re.fullmatch(r"cell(\d+)", path.name)
            if path.is_dir() and match:
                identifiers.add(int(match.group(1)))
        for path in self.runs_root.glob("*/05_review/reviews.json"):
            try:
                reviews = json.loads(path.read_text(encoding="utf-8"))
                for review in dict(reviews.get("cells") or {}).values():
                    match = re.fullmatch(r"cell(\d+)", str(dict(review).get("evo_cell_folder") or ""))
                    if match:
                        identifiers.add(int(match.group(1)))
            except Exception:
                continue
        return f"cell{max(identifiers, default=0) + 1:06d}"

    def rate_cell(self, job_id: str, folder: str, result: str) -> dict[str, object]:
        if result not in {"good", "bad"}:
            raise ValueError("result must be 'good' or 'bad'.")
        if job_id not in self._locks:
            raise KeyError(job_id)
        condition = self._locks[job_id]
        with condition, self._output_lock:
            job = self._jobs[job_id]
            if job.get("status") != "completed":
                raise ValueError("Only completed jobs can be reviewed.")
            crops = list(job.get("crops") or [])
            crop = next((value for value in crops if value.get("folder") == folder), None)
            if crop is None or not re.fullmatch(r"cell\d{4}", folder):
                raise KeyError(folder)
            review_path = self.runs_root / job_id / "05_review" / "reviews.json"
            reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {"cells": {}}
            cells = dict(reviews.get("cells") or {})
            previous = dict(cells.get(folder) or {})
            output_folder = str(previous.get("evo_cell_folder") or "")
            if result == "good" and not re.fullmatch(r"cell\d+", output_folder):
                output_folder = self._next_output_folder()
            image_folder = str(
                previous.get("evo_image_folder")
                or source_image_folder(str(job.get("filename") or job.get("case") or "image"))
            )
            review = {
                "job_id": job_id, "case": job["case"], "cell": folder,
                "selection_id": crop.get("selection_id"), "result": result,
                "approved_for_evo": result == "good", "updated_at": _utc_now(),
                "source_image": str(job.get("filename") or ""),
                "evo_image_folder": image_folder,
                "evo_cell_folder": output_folder or None,
            }
            if output_folder:
                target = self.output_root / "cells" / image_folder / output_folder
                if target.is_dir():
                    shutil.rmtree(target)
                legacy_target = self.output_root / "cells" / output_folder
                if legacy_target.is_dir():
                    shutil.rmtree(legacy_target)
                if result == "good":
                    source = self.runs_root / job_id / "04_evo_single_cells" / folder
                    required = ("raw.tif", "seg.tif", "seg-000.swc", "seg.traces", "soma.zip", "bounds.zip")
                    missing = [name for name in required if not (source / name).is_file()]
                    if missing:
                        raise RuntimeError("Evo finalization is incomplete: " + ", ".join(missing))
                    shutil.copytree(source, target, copy_function=shutil.copy2)
                    self._write_json_atomic(target / "review.json", review)
            cells[folder] = review
            self._write_json_atomic(
                review_path,
                {"schema_version": 1, "job_id": job_id, "updated_at": _utc_now(), "cells": cells},
            )
            crop["review"] = review
            job["crops"] = crops
            job["review_summary"] = self._review_summary(crops)
            job["updated_at"] = _utc_now()
            job["version"] = int(job.get("version", 0)) + 1
            self._persist(job)
            self._rebuild_output_index()
            self._update_learning_result(job_id)
            condition.notify_all()
            return {"review": review, "review_summary": job["review_summary"]}

    def _rebuild_output_index(self) -> None:
        rows: list[dict[str, object]] = []
        for path in sorted((self.output_root / "cells").rglob("cell*/review.json")):
            try:
                review = json.loads(path.read_text(encoding="utf-8"))
                image_folder = str(review.get("evo_image_folder") or "")
                cell_name = path.parent.name
                rows.append({
                    "cell_folder": (
                        relative_evo_cell_folder(image_folder, cell_name)
                        if image_folder else cell_name
                    ),
                    "image_folder": image_folder,
                    "cell_name": cell_name,
                    "source_image": review.get("source_image") or "",
                    "job_id": review["job_id"],
                    "case": review["case"],
                    "source_cell": review["cell"],
                    "selection_id": review.get("selection_id"),
                })
            except Exception:
                continue
        self._write_json_atomic(
            self.output_root / "selection_summary.json",
            {
                "selected_cells": len(rows),
                "source_images": len({row["image_folder"] for row in rows if row["image_folder"]}),
                "cells_root": str((self.output_root / 'cells').resolve()),
                "updated_at": _utc_now(),
            },
        )
        with (self.output_root / "selected_cells.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "cell_folder", "image_folder", "cell_name", "source_image",
                    "job_id", "case", "source_cell", "selection_id",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def selected_archive(self) -> Path:
        with self._output_lock:
            self._rebuild_output_index()
            archive = self.output_root / "evo_selected_cells.zip"
            temporary = archive.with_name(f"{archive.name}.{uuid.uuid4().hex}.tmp")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
                for path in sorted((self.output_root / "cells").rglob("*")):
                    if path.is_file():
                        output.write(path, Path("cells") / path.relative_to(self.output_root / "cells"))
            os.replace(temporary, archive)
            return archive

    def public(self, job_id: str) -> dict[str, object]:
        with self._global_lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = dict(self._jobs[job_id])
        job.pop("input_path", None)
        return job

    def list(self) -> list[dict[str, object]]:
        with self._global_lock:
            identifiers = sorted(self._jobs, reverse=True)[:100]
        result = []
        for job_id in identifiers:
            job = self.public(job_id)
            job["crops"] = []
            job["events"] = list(job.get("events") or [])[-2:]
            result.append(job)
        return result

    def wait_for_change(self, job_id: str, version: int, timeout: float = 20) -> dict[str, object]:
        if job_id not in self._locks:
            raise KeyError(job_id)
        condition = self._locks[job_id]
        deadline = time.monotonic() + timeout
        with condition:
            while int(self._jobs[job_id].get("version", 0)) <= version:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                condition.wait(remaining)
        return self.public(job_id)


class SelectionHandler(BaseHTTPRequestHandler):
    server_version = "BlindCellSelection/1.0"

    @property
    def store(self) -> SelectionJobStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {format % args}")

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _serve_file(self, path: Path, *, download: bool = False, cache: bool = False) -> None:
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

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1024 * 1024:
            raise ValueError("Invalid JSON payload size.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object.")
        return payload

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        selection_match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/selection", path)
        review_match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/reviews/(cell\d{4})", path)
        try:
            if selection_match:
                payload = self._read_json()
                points = payload.get("points")
                if not isinstance(points, list):
                    raise ValueError("points must be a list.")
                self._json(self.store.submit_selection(selection_match.group(1), points), HTTPStatus.ACCEPTED)
                return
            if review_match:
                payload = self._read_json()
                self._json(self.store.rate_cell(review_match.group(1), review_match.group(2), str(payload.get("result") or "")))
                return
            if path != "/api/jobs":
                self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
                return
            filename = unquote(self.headers.get("X-Filename", "overview.tif"))
            if Path(filename).suffix.lower() not in {".tif", ".tiff"}:
                raise ValueError("Please upload a .tif or .tiff image.")
            job = self.store.create(filename, int(self.headers.get("Content-Length", "0")), self.rfile)
            self._json(job, HTTPStatus.CREATED)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Job or cell not found.")
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/config":
            semantic = self.store.settings.semantic
            self._json(
                {"network": semantic.dataset_id, "checkpoint": semantic.checkpoint,
                 "hysteresis_low": semantic.hysteresis_t_low,
                 "separator_checkpoint": str(self.store.settings.separator_checkpoint),
                 "separator_epoch": 80, "output_root": str(self.store.output_root),
                 "learning_root": str(self.store.learning_root)}
            )
            return
        if path == "/api/jobs":
            self._json({"jobs": self.store.list()})
            return
        if path == "/api/output/download":
            try:
                self._serve_file(self.store.selected_archive(), download=True)
            except Exception as error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        event_match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/events", path)
        if event_match:
            try:
                version = int(parse_qs(parsed.query).get("version", ["0"])[0])
                self._json(self.store.wait_for_change(event_match.group(1), version))
            except (KeyError, ValueError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
            return
        job_match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)", path)
        if job_match:
            try:
                self._json(self.store.public(job_match.group(1)))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Job not found.")
            return
        if path.startswith("/runs/"):
            relative = Path(unquote(path[len("/runs/"):]))
            root = self.store.runs_root.resolve()
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Invalid path.")
                return
            self._serve_file(candidate, cache=True, download=candidate.suffix.lower() == ".zip")
            return
        target = STATIC_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
        try:
            target.resolve().relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid path.")
            return
        self._serve_file(target, cache=path != "/")


class SelectionServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def __init__(self, address: tuple[str, int], store: SelectionJobStore):
        super().__init__(address, SelectionHandler)
        self.store = store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind manual cell selection and Evo crop pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8783)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "blind_selection_runs")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "evo_manual_selection_output")
    parser.add_argument("--learning-root", type=Path, default=PROJECT_ROOT / "blind_selection_learning_dataset")
    parser.add_argument("--separator-checkpoint", type=Path, default=PROJECT_ROOT / "instance_separator_results" / "seeded_full80" / "checkpoint_final.pth")
    parser.add_argument("--no-fiji", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semantic = default_settings(
        PROJECT_ROOT,
        device=args.device,
        dataset_id=141,
        trainer="nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug",
        plans="nnUNetPlans139Exact",
        checkpoint="checkpoint_final.pth",
        hysteresis_mode="fixed",
        hysteresis_t_high=0.647,
        hysteresis_t_low=0.20,
        finalize_fiji=not args.no_fiji,
    )
    settings = BlindSelectionSettings(
        semantic=semantic, separator_checkpoint=args.separator_checkpoint.resolve()
    )
    store = SelectionJobStore(
        args.runs_root.resolve(), args.output_root.resolve(),
        args.learning_root.resolve(), settings,
    )
    server = SelectionServer((args.host, args.port), store)
    url = f"http://{args.host}:{args.port}/"
    print(f"Blind cell selection is ready at {url}")
    print(f"Approved Evo cells: {store.output_root / 'cells'}")
    print(f"Selection learning data: {store.learning_root}")
    if args.open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
