from __future__ import annotations

import argparse
import json
from pathlib import Path

from cell_pipeline_web.training_review_import import (
    DEFAULT_JOB_ID,
    DEFAULT_SOURCES,
    import_training_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy and rasterize the three ROCKi TracingsFelix collections for web review."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("web_pipeline_runs"))
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    status = import_training_review(
        DEFAULT_SOURCES,
        args.runs_root.resolve(),
        args.job_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(status["summary"], indent=2, ensure_ascii=False))
    print(f"Review job: {status['id']}")


if __name__ == "__main__":
    main()

