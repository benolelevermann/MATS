from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cancel_cell_pipeline_jobs import cancel_jobs


class CancelCellPipelineJobsTests(unittest.TestCase):
    def test_cancels_only_queued_and_running_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for job_id, status in (
                ("queued_job", "queued"),
                ("running_job", "running"),
                ("completed_job", "completed"),
            ):
                job_root = root / job_id
                job_root.mkdir()
                (job_root / "status.json").write_text(
                    json.dumps(
                        {
                            "id": job_id,
                            "status": status,
                            "phase": status,
                            "progress": 30 if status == "running" else 0,
                            "version": 1,
                            "events": [],
                        }
                    ),
                    encoding="utf-8",
                )

            cancelled = cancel_jobs(root)
            self.assertEqual(cancelled, ["queued_job", "running_job"])
            for job_id in cancelled:
                payload = json.loads((root / job_id / "status.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "cancelled")
                self.assertEqual(payload["phase"], "cancelled")
                self.assertEqual(payload["version"], 2)
                self.assertEqual(payload["events"][-1]["phase"], "cancelled")
            completed = json.loads(
                (root / "completed_job" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed["status"], "completed")


if __name__ == "__main__":
    unittest.main()
