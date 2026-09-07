from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "web_pipeline_runs"
CANCELLABLE_STATES = {"queued", "running"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cancel_jobs(runs_root: Path, requested_ids: set[str] | None = None) -> list[str]:
    runs_root = runs_root.resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    cancelled: list[str] = []
    for status_path in sorted(runs_root.glob("*/status.json")):
        job_id = status_path.parent.name
        if requested_ids is not None and job_id not in requested_ids:
            continue
        job = json.loads(status_path.read_text(encoding="utf-8"))
        if job.get("status") not in CANCELLABLE_STATES:
            continue
        timestamp = utc_now()
        job["status"] = "cancelled"
        job["phase"] = "cancelled"
        job["message"] = "Analysis cancelled by the user. Completed reviews remain available."
        job["updated_at"] = timestamp
        job["version"] = int(job.get("version", 0)) + 1
        events = list(job.get("events") or [])
        events.append(
            {
                "time": timestamp,
                "phase": "cancelled",
                "progress": int(job.get("progress", 0)),
                "message": "Analysis cancelled by the user.",
            }
        )
        job["events"] = events[-160:]
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, status_path)
        cancelled.append(job_id)
    if requested_ids is not None:
        unknown = requested_ids.difference(cancelled).difference(
            path.parent.name for path in runs_root.glob("*/status.json")
        )
        if unknown:
            raise KeyError(f"Unknown job ids: {', '.join(sorted(unknown))}")
    return cancelled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark queued or running local cell-pipeline jobs as cancelled."
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("job_ids", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = set(args.job_ids) or None
    cancelled = cancel_jobs(args.runs_root, requested)
    print(json.dumps({"cancelled": cancelled, "count": len(cancelled)}, indent=2))


if __name__ == "__main__":
    main()
