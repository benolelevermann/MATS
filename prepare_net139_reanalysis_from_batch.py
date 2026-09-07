from __future__ import annotations

"""Create a fresh queued review batch from an earlier FOV batch's saved inputs."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runs", type=Path, required=True)
    parser.add_argument("--source-batch", required=True)
    parser.add_argument("--target-runs", type=Path, required=True)
    parser.add_argument("--target-batch", default="net139_reanalysis_20260903")
    return parser.parse_args()


def read_jobs(source_runs: Path, batch_id: str) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for status_path in source_runs.glob("*/status.json"):
        try:
            job = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        source = dict(job.get("source") or {})
        if source.get("kind") == "fov_library" and source.get("batch_id") == batch_id:
            job["_run_root"] = str(status_path.parent.resolve())
            jobs.append(job)
    jobs.sort(key=lambda item: int(dict(item.get("source") or {}).get("batch_index", 0)))
    if not jobs:
        raise RuntimeError(f"Kein FOV-Job fuer Batch {batch_id!r} in {source_runs}")
    indices = [int(dict(job.get("source") or {})["batch_index"]) for job in jobs]
    if indices != list(range(1, len(jobs) + 1)):
        raise RuntimeError(f"Der Quellbatch ist nicht lueckenlos: {indices[:5]} ... {indices[-5:]}")
    return jobs


def stable_job_id(source_job_id: str, index: int) -> str:
    digest = hashlib.sha1(source_job_id.encode("utf-8")).hexdigest()[:8]
    return f"net139_20260903_{index:03d}_{digest}"


def main() -> None:
    args = parse_args()
    source_runs = args.source_runs.resolve()
    target_runs = args.target_runs.resolve()
    jobs = read_jobs(source_runs, args.source_batch)
    target_runs.mkdir(parents=True, exist_ok=True)
    manifest_path = target_runs / "reanalysis_manifest.json"
    if any(target_runs.iterdir()) and not manifest_path.is_file():
        raise RuntimeError(
            f"Zielordner ist nicht leer und enthaelt kein passendes Manifest: {target_runs}"
        )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_batch") != args.source_batch
            or manifest.get("target_batch") != args.target_batch
        ):
            raise RuntimeError(f"Vorhandenes Manifest gehoert zu einem anderen Batch: {manifest_path}")

    base_time = datetime.now(timezone.utc).replace(microsecond=0)
    created = 0
    reused = 0
    manifest_jobs: list[dict[str, object]] = []
    for index, source_job in enumerate(jobs, start=1):
        source_job_id = str(source_job["id"])
        job_id = stable_job_id(source_job_id, index)
        run_root = target_runs / job_id
        status_path = run_root / "status.json"
        source_input = Path(str(source_job["input_path"]))
        if not source_input.is_file():
            raise FileNotFoundError(f"Gespeichertes Quellbild fehlt: {source_input}")
        case = str(source_job["case"])
        input_path = run_root / "01_input" / f"{case}_0000.tif"
        if status_path.is_file() and input_path.is_file():
            reused += 1
        else:
            if run_root.exists():
                raise RuntimeError(f"Unvollstaendiger Zieljob wird nicht ueberschrieben: {run_root}")
            input_path.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(source_input, input_path)
            source = dict(source_job.get("source") or {})
            source.update(
                {
                    "batch_id": args.target_batch,
                    "batch_index": index,
                    "batch_total": len(jobs),
                    "original_batch_id": args.source_batch,
                    "original_job_id": source_job_id,
                    "reanalysis_network": 139,
                }
            )
            created_at = (base_time + timedelta(microseconds=index)).isoformat(timespec="microseconds")
            status = {
                "id": job_id,
                "filename": str(source_job["filename"]),
                "case": case,
                "input_path": str(input_path),
                "source": source,
                "model": {
                    "dataset_id": 139,
                    "trainer": "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug",
                    "plans": "nnUNetPlans",
                    "checkpoint": "checkpoint_final.pth",
                },
                "created_at": created_at,
                "updated_at": created_at,
                "status": "queued",
                "phase": "queued",
                "progress": 0,
                "message": "Netz-139-Reanalyse wartet auf die Verarbeitung.",
                "details": {},
                "events": [
                    {
                        "time": created_at,
                        "phase": "queued",
                        "progress": 0,
                        "message": "Gespeichertes FOV fuer Netz 139 bereitgestellt.",
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
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
            created += 1
        manifest_jobs.append(
            {
                "index": index,
                "source_job_id": source_job_id,
                "target_job_id": job_id,
                "case": case,
                "source_input": str(source_input),
                "target_input": str(input_path),
            }
        )

    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_runs": str(source_runs),
                "source_batch": args.source_batch,
                "target_runs": str(target_runs),
                "target_batch": args.target_batch,
                "dataset_id": 139,
                "count": len(jobs),
                "jobs": manifest_jobs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Netz-139-Batch bereit: {len(jobs)} FOVs ({created} neu, {reused} vorhanden)")
    print(f"Ziel: {target_runs}")


if __name__ == "__main__":
    main()
