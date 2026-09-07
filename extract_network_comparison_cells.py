from __future__ import annotations

"""Export topology-isolated cells from one or more network hysteresis maps."""

import argparse
import json
from pathlib import Path

import numpy as np

from cell_pipeline_web.pipeline import default_settings, extract_single_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--network",
        nargs=2,
        action="append",
        metavar=("ID", "PREDICTION_DIR"),
        help="Network id and prediction directory. Repeat for every network.",
    )
    # Kept for existing Dataset138/139 comparison commands.
    parser.add_argument("--pred-138", type=Path)
    parser.add_argument("--pred-139", type=Path)
    return parser.parse_args()


def find_prediction(folder: Path, case: str, network: str) -> Path:
    for name in (f"{case}.tif", f"{case}_{network}.tif"):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Keine Prediction fuer {case} in {folder}")


def reject_conflict_group(
    _group_dir: Path,
    _input_seg: np.ndarray,
    _group_soma: np.ndarray,
    _callback,
    _settings,
    _progress: int,
) -> np.ndarray:
    raise RuntimeError("comparison_excludes_multi_soma_conflicts")


def main() -> None:
    args = parse_args()
    network_roots: list[tuple[str, Path]] = []
    if args.network:
        network_roots.extend((str(network), Path(root)) for network, root in args.network)
    if args.pred_138 is not None:
        network_roots.append(("138", args.pred_138))
    if args.pred_139 is not None:
        network_roots.append(("139", args.pred_139))
    if not network_roots:
        raise RuntimeError("Mindestens ein --network ID PREDICTION_DIR ist erforderlich.")
    ids = [network for network, _ in network_roots]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Netz-IDs duerfen nicht doppelt vorkommen: {ids}")

    settings = default_settings(args.project_root, finalize_fiji=False)
    cases = sorted(path.name.removesuffix("_0000.tif") for path in args.inputs.glob("*_0000.tif"))
    if not cases:
        raise RuntimeError(f"Keine *_0000.tif-Dateien in {args.inputs}")

    summaries: list[dict[str, object]] = []
    for network, prediction_root in network_roots:
        hysteresis_root = args.comparison_root / f"hysteresis_net{network}"
        network_root = args.comparison_root / f"cells_net{network}"
        work_network_root = args.comparison_root / f"work_net{network}"
        network_root.mkdir(parents=True, exist_ok=True)
        work_network_root.mkdir(parents=True, exist_ok=True)

        for index, case in enumerate(cases, start=1):
            print(f"Netz {network} [{index}/{len(cases)}] {case}", flush=True)
            output_root = network_root / case
            work_root = work_network_root / case
            if output_root.exists() or work_root.exists():
                raise RuntimeError(f"Ausgabe existiert bereits: {output_root} oder {work_root}")

            def callback(phase: str, progress: int, message: str, _details=None) -> None:
                if phase in {"cell_classification", "cell_export"}:
                    print(f"  {progress:3d}% {message}", flush=True)

            summary = extract_single_cells(
                original_path=args.inputs / f"{case}_0000.tif",
                semantic_path=hysteresis_root / f"{case}_adaptive_hysteresis_0-1-2.tif",
                output_root=output_root,
                work_root=work_root,
                callback=callback,
                settings=settings,
                ntt_runner=reject_conflict_group,
                prediction_path=find_prediction(prediction_root, case, network),
            )
            summaries.append({"network": network, "case": case, **summary})

    (args.comparison_root / "cell_extraction_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"Geschrieben: {args.comparison_root / 'cell_extraction_summary.json'}")


if __name__ == "__main__":
    main()
