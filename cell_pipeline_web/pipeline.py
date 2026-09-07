from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tifffile
from PIL import Image
from scipy import io as sio
from scipy import ndimage as ndi


ProgressCallback = Callable[[str, int, str, dict[str, object] | None], None]
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
NTT_RECTANGLE_COUNT = 18 * 10 * 360
SCRIPT_VERSION = "configurable-network-hysteresis-ntt-evo-web-v3-2026-09-05"


@dataclass(frozen=True)
class PipelineSettings:
    project_root: Path
    device: str = "cuda"
    dataset_id: int = 138
    trainer: str = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
    plans: str = "nnUNetPlans"
    configuration: str = "2d"
    fold: int = 0
    checkpoint: str = "checkpoint_best.pth"
    hysteresis_mode: str = "adaptive"
    hysteresis_alpha: float = 1.0 / 3.0
    hysteresis_t_high: float = 0.647
    hysteresis_t_low: float = 0.326
    min_soma_area: int = 20
    min_skeleton_pixels: int = 8
    crop_margin: int = 48
    min_crop_size: int = 128
    edge_clearance: int = 3
    ntt_margin: int = 24
    ntt_min_crop_size: int = 128
    ntt_max_crop_side: int = 384
    ntt_dilate_radius: int = 3
    ntt_min_trace_pixels: int = 8
    ntt_min_coverage: float = 0.60
    ntt_min_hit_rate: float = 0.98
    finalize_fiji: bool = True

    @property
    def python(self) -> Path:
        return self.project_root / ".venv" / "Scripts" / "python.exe"

    @property
    def predictor(self) -> Path:
        return self.project_root / ".venv" / "Scripts" / "nnUNetv2_predict.exe"

    @property
    def hysteresis_script(self) -> Path:
        return self.project_root / "apply_adaptive_hysteresis.py"

    @property
    def octave_runner(self) -> Path:
        return self.project_root / "run_neurotreetracer_octave.ps1"

    @property
    def fiji(self) -> Path:
        return Path(r"C:\Program Files\Fiji.app\ImageJ-win64.exe")

    @property
    def fiji_finalizer(self) -> Path:
        return self.project_root / "r_pipeline" / "finalize_cells_with_fiji.py"

    @property
    def validator(self) -> Path:
        return self.project_root / "r_pipeline" / "validate_r_pipeline_cell_folders.py"


def default_settings(project_root: Path | None = None, **overrides: object) -> PipelineSettings:
    root = project_root or Path(__file__).resolve().parents[1]
    return PipelineSettings(project_root=root, **overrides)


def hysteresis_cli_arguments(settings: PipelineSettings) -> list[str]:
    if settings.hysteresis_mode == "adaptive":
        return ["--alpha", str(settings.hysteresis_alpha)]
    if settings.hysteresis_mode != "fixed":
        raise ValueError(f"Unknown hysteresis mode: {settings.hysteresis_mode}")
    if not 0.0 < settings.hysteresis_t_high <= 1.0:
        raise ValueError("Fixed T_high must be in (0,1].")
    if not 0.0 <= settings.hysteresis_t_low < settings.hysteresis_t_high:
        raise ValueError("Fixed T_low must be in [0,T_high).")
    return [
        "--t-high",
        str(settings.hysteresis_t_high),
        "--t-low",
        str(settings.hysteresis_t_low),
    ]


def _notify(
    callback: ProgressCallback,
    phase: str,
    progress: int,
    message: str,
    details: dict[str, object] | None = None,
) -> None:
    callback(phase, max(0, min(100, int(progress))), message, details)


def _assert_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required pipeline file(s) missing:\n" + "\n".join(missing))


def _run_command(
    command: list[str],
    callback: ProgressCallback,
    phase: str,
    progress: int,
    cwd: Path,
    env: dict[str, str] | None = None,
    notify_line: Callable[[str], bool] | None = None,
) -> list[str]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creation_flags,
    )
    output: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output.append(line)
        output = output[-200:]
        if notify_line is None or notify_line(line):
            _notify(callback, phase, progress, line)
    return_code = process.wait()
    if return_code:
        tail = "\n".join(output[-25:])
        raise RuntimeError(
            f"{phase} failed with exit code {return_code}."
            + (f"\n\nLast output:\n{tail}" if tail else "")
        )
    return output


def _read_2d(path: Path, label: str) -> np.ndarray:
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim != 2:
        raise RuntimeError(f"{label} must be a 2-D TIFF, got {array.shape}: {path}")
    return array


def _validate_input(path: Path) -> tuple[int, int]:
    image = _read_2d(path, "Uploaded image")
    if not np.issubdtype(image.dtype, np.number):
        raise RuntimeError(f"Uploaded image must be numeric, got {image.dtype}.")
    if not np.all(np.isfinite(image)):
        raise RuntimeError("Uploaded image contains NaN or infinity.")
    if min(image.shape) < 32:
        raise RuntimeError(f"Uploaded image is unexpectedly small: {image.shape}.")
    return int(image.shape[0]), int(image.shape[1])


def _prediction_environment(settings: PipelineSettings) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(settings.project_root / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(settings.project_root / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(settings.project_root / "nnUNet_results")
    external_trainers = settings.project_root / "custom_trainers" / "skeleton_recall"
    if external_trainers.is_dir():
        env["nnUNet_extTrainer"] = str(external_trainers)
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(external_trainers), old_pythonpath) if value
        )
    else:
        env.pop("nnUNet_extTrainer", None)
    # This variable is only allowed to bootstrap a fresh Dataset140 training,
    # never normal prediction from an already trained checkpoint.
    env.pop("NNUNET_MATS_FINETUNE_CHECKPOINT", None)
    return env


def _load_export_helpers(project_root: Path):
    helper_path = project_root / "r_pipeline" / "export_cells_for_r_pipeline.py"
    specification = importlib.util.spec_from_file_location("evo_cell_export_helpers", helper_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load Evo export helpers: {helper_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _touches_border_from_coords(coords: np.ndarray, shape: tuple[int, int], margin: int = 1) -> bool:
    if coords.size == 0:
        return False
    height, width = shape
    return bool(
        coords[:, 0].min() < margin
        or coords[:, 1].min() < margin
        or coords[:, 0].max() >= height - margin
        or coords[:, 1].max() >= width - margin
    )


def _axis_bounds(
    minimum: int,
    maximum_inclusive: int,
    limit: int,
    margin: int,
    minimum_size: int,
) -> tuple[int, int]:
    desired = max(maximum_inclusive - minimum + 1 + 2 * margin, minimum_size)
    desired = min(desired, limit)
    center = 0.5 * (minimum + maximum_inclusive)
    start = int(math.floor(center - desired / 2))
    start = max(0, min(start, limit - desired))
    return start, start + desired


def _square_bounds_from_coords(
    coords: np.ndarray,
    shape: tuple[int, int],
    margin: int,
    minimum_size: int,
) -> tuple[int, int, int, int]:
    if coords.size == 0:
        raise RuntimeError("Cannot crop empty coordinates.")
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    requested = max(
        int(max(maximum[0] - minimum[0] + 1, maximum[1] - minimum[1] + 1)) + 2 * margin,
        minimum_size,
    )
    requested = min(requested, shape[0], shape[1])
    y0, y1 = _axis_bounds(int(minimum[0]), int(maximum[0]), shape[0], margin, requested)
    x0, x1 = _axis_bounds(int(minimum[1]), int(maximum[1]), shape[1], margin, requested)
    return y0, y1, x0, x1


def _normalize_u8(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _write_preview(path: Path, raw: np.ndarray, skeleton: np.ndarray, soma: np.ndarray) -> None:
    gray = _normalize_u8(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    rgb[skeleton] = 0.18 * rgb[skeleton] + 0.82 * np.asarray([34, 211, 167])
    rgb[soma] = 0.16 * rgb[soma] + 0.84 * np.asarray([240, 91, 159])
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    image.thumbnail((720, 720), Image.Resampling.LANCZOS)
    image.save(path, optimize=True)


def _write_raw_preview(path: Path, raw: np.ndarray) -> None:
    image = Image.fromarray(_normalize_u8(raw))
    image.thumbnail((720, 720), Image.Resampling.LANCZOS)
    image.save(path, optimize=True)


def _write_semantic_preview(path: Path, raw: np.ndarray, semantic: np.ndarray) -> None:
    _write_preview(path, raw, semantic == 1, semantic == 2)


def _crop_clearance_ok(mask: np.ndarray, clearance: int) -> bool:
    if clearance <= 0:
        return True
    if mask.shape[0] <= 2 * clearance or mask.shape[1] <= 2 * clearance:
        return False
    border = np.ones(mask.shape, dtype=bool)
    border[clearance:-clearance, clearance:-clearance] = False
    return not bool(np.any(mask & border))


def _coords_in_bounds(coords: np.ndarray, y0: int, x0: int) -> np.ndarray:
    result = coords.astype(np.int64, copy=True)
    result[:, 0] -= y0
    result[:, 1] -= x0
    return result


def _export_cell(
    raw: np.ndarray,
    network_semantic: np.ndarray,
    postprocessed_semantic: np.ndarray,
    skeleton_coords: np.ndarray,
    soma_coords: np.ndarray,
    output_root: Path,
    output_index: int,
    helper,
    settings: PipelineSettings,
    source: str,
    source_component: int,
    conflict_group: int | None,
    qc: dict[str, object] | None = None,
) -> dict[str, object]:
    all_coords = np.vstack((skeleton_coords, soma_coords))
    if _touches_border_from_coords(all_coords, raw.shape, 1):
        raise RuntimeError("cell_touches_source_border")
    y0, y1, x0, x1 = _square_bounds_from_coords(
        all_coords, raw.shape, settings.crop_margin, settings.min_crop_size
    )
    local_skeleton_coords = _coords_in_bounds(skeleton_coords, y0, x0)
    local_soma_coords = _coords_in_bounds(soma_coords, y0, x0)
    height, width = y1 - y0, x1 - x0
    skeleton = np.zeros((height, width), dtype=bool)
    soma = np.zeros((height, width), dtype=bool)
    skeleton[local_skeleton_coords[:, 0], local_skeleton_coords[:, 1]] = True
    soma[local_soma_coords[:, 0], local_soma_coords[:, 1]] = True
    skeleton &= ~soma
    cell_mask = skeleton | soma
    if int(ndi.label(soma, structure=CONNECTIVITY_8)[1]) != 1:
        raise RuntimeError("exported_cell_does_not_contain_exactly_one_soma")
    if int(skeleton.sum()) < settings.min_skeleton_pixels:
        raise RuntimeError("not_enough_skeleton_pixels")
    if not _crop_clearance_ok(cell_mask, settings.edge_clearance):
        raise RuntimeError("insufficient_crop_clearance")

    folder_name = f"cell{output_index:04d}"
    cell_dir = output_root / folder_name
    cell_dir.mkdir(parents=True, exist_ok=False)
    try:
        raw_crop = raw[y0:y1, x0:x1]
        network_crop = network_semantic[y0:y1, x0:x1].astype(np.uint8, copy=False)
        postprocessed_crop = postprocessed_semantic[y0:y1, x0:x1].astype(np.uint8, copy=False)
        semantic = np.zeros((height, width), dtype=np.uint8)
        semantic[skeleton] = 1
        semantic[soma] = 2

        helper.save_tiff(cell_dir / "raw.tif", raw_crop)
        helper.save_tiff(cell_dir / "prediction.tif", network_crop)
        helper.save_tiff(cell_dir / "postprocessed.tif", postprocessed_crop)
        helper.save_tiff(cell_dir / "skeleton.tif", skeleton.astype(np.uint8))
        helper.save_tiff(cell_dir / "soma.tif", soma.astype(np.uint8))
        helper.save_tiff(cell_dir / "seg.tif", semantic)
        helper.save_tiff(cell_dir / "cell_mask.tif", cell_mask.astype(np.uint8))
        helper.write_pixel_csv(cell_dir / "seg.csv", skeleton, x0, y0)
        helper.write_swc(cell_dir / "seg-000.swc", skeleton, soma)
        _write_raw_preview(cell_dir / "raw_preview.png", raw_crop)
        _write_semantic_preview(cell_dir / "prediction_preview.png", raw_crop, network_crop)
        _write_semantic_preview(cell_dir / "postprocessing_preview.png", raw_crop, postprocessed_crop)
        _write_preview(cell_dir / "preview.png", raw_crop, skeleton, soma)

        centroid_y, centroid_x = np.argwhere(soma).mean(axis=0)
        bounds = {
            "x_min": int(x0),
            "y_min": int(y0),
            "x_max_exclusive": int(x1),
            "y_max_exclusive": int(y1),
            "width": int(width),
            "height": int(height),
        }
        location = {
            "local_x": float(centroid_x),
            "local_y": float(centroid_y),
            "global_x": float(centroid_x + x0),
            "global_y": float(centroid_y + y0),
        }
        metadata = {
            "script_version": SCRIPT_VERSION,
            "status": "accepted_single_cell",
            "folder": folder_name,
            "classification_source": source,
            "source_component": int(source_component),
            "conflict_group": conflict_group,
            "skeleton_pixels": int(skeleton.sum()),
            "soma_pixels": int(soma.sum()),
            "bounds": bounds,
            "location": location,
            "qc": qc or {},
            "review_layers": {
                "raw": "raw.tif",
                "prediction": "prediction.tif",
                "postprocessed": "postprocessed.tif",
                "isolated_cell": "seg.tif",
            },
        }
        (cell_dir / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
        (cell_dir / "location.json").write_text(json.dumps(location, indent=2), encoding="utf-8")
        (cell_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(cell_dir, ignore_errors=True)
        raise
    return {
        "folder": folder_name,
        "source": source,
        "source_component": int(source_component),
        "conflict_group": conflict_group or "",
        "skeleton_pixels": int(skeleton.sum()),
        "soma_pixels": int(soma.sum()),
        "x_min": int(x0),
        "y_min": int(y0),
        "x_max_exclusive": int(x1),
        "y_max_exclusive": int(y1),
        "preview": f"{folder_name}/preview.png",
    }


def _tracer_soma_instances(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask, structure=CONNECTIVITY_8)
    if count == 0:
        return labels.astype(np.uint16)
    height = mask.shape[0]
    statistics: list[tuple[int, int, int]] = []
    for component_id in range(1, count + 1):
        rows, cols = np.nonzero(labels == component_id)
        column_major = cols.astype(np.int64) * height + rows.astype(np.int64)
        statistics.append((component_id, len(rows), int(column_major.min())))
    statistics.sort(key=lambda item: (-item[1], item[2]))
    remap = np.zeros(count + 1, dtype=np.uint16)
    for rank, (component_id, _, _) in enumerate(statistics, start=1):
        remap[component_id] = rank
    return remap[labels]


def _conservative_soma_instances(
    soma_mask: np.ndarray, settings: PipelineSettings
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Split only oversized soma blobs with multiple clear distance maxima."""
    from soma_graph_cell_extraction import GraphSplitSettings, split_soma_mask

    if not np.any(soma_mask):
        return np.zeros(soma_mask.shape, dtype=np.uint32), []
    raw_labels, raw_count = ndi.label(soma_mask, structure=CONNECTIVITY_8)
    raw_areas = np.bincount(raw_labels.ravel(), minlength=raw_count + 1)
    if not any(
        int(raw_areas[raw_id]) >= settings.min_soma_area
        for raw_id in range(1, raw_count + 1)
    ):
        return np.zeros(soma_mask.shape, dtype=np.uint32), [
            {
                "raw_soma_id": raw_id,
                "area_px": int(raw_areas[raw_id]),
                "area_over_reference": 0.0,
                "candidate_for_split": 0,
                "detected_peaks": 0,
                "output_instances": "",
                "reason": "too_small",
            }
            for raw_id in range(1, raw_count + 1)
        ]
    instances, report, _instance_info, _reference_area = split_soma_mask(
        soma_mask,
        GraphSplitSettings(
            min_soma_area=settings.min_soma_area,
            large_soma_factor=1.75,
            h_maxima_radius_factor=0.16,
            max_soma_splits=4,
            min_split_part_factor=0.22,
        ),
    )
    return instances, report


def _separate_touching_soma_instances(instances: np.ndarray) -> np.ndarray:
    """Carve a narrow internal watershed line so the tracer sees separate seeds."""
    occupied = instances > 0
    boundary = np.zeros(instances.shape, dtype=bool)
    for row_delta, col_delta in ((0, 1), (1, -1), (1, 0), (1, 1)):
        source_rows = slice(max(0, -row_delta), instances.shape[0] - max(0, row_delta))
        source_cols = slice(max(0, -col_delta), instances.shape[1] - max(0, col_delta))
        target_rows = slice(max(0, row_delta), instances.shape[0] - max(0, -row_delta))
        target_cols = slice(max(0, col_delta), instances.shape[1] - max(0, -col_delta))
        source = instances[source_rows, source_cols]
        target = instances[target_rows, target_cols]
        different = (source > 0) & (target > 0) & (source != target)
        boundary[source_rows, source_cols] |= different
        boundary[target_rows, target_cols] |= different
    return occupied & ~boundary


def _validate_ntt_traces(
    traces: np.ndarray,
    input_seg: np.ndarray,
    input_soma: np.ndarray,
    original_skeleton: np.ndarray,
    settings: PipelineSettings,
) -> tuple[list[dict[str, object]] | None, dict[str, object]]:
    soma_instances = _tracer_soma_instances(input_soma)
    soma_count = int(soma_instances.max())
    qc: dict[str, object] = {
        "expected_somas": soma_count,
        "trace_rows": int(traces.shape[0]) if traces.ndim == 2 else 0,
    }
    if traces.size == 0 or traces.ndim != 2 or traces.shape[1] < 3:
        return None, qc | {"reason": "no_traces"}

    height, width = input_seg.shape
    soma_ids = traces[:, 0].astype(np.int64)
    linear = traces[:, 2].astype(np.int64)
    if linear.min() < 1 or linear.max() > height * width:
        return None, qc | {"reason": "trace_index_out_of_range"}
    rows = (linear - 1) % height
    cols = (linear - 1) // height
    traced_ids = sorted(set(soma_ids.tolist()))
    expected_ids = list(range(1, soma_count + 1))
    qc["traced_somas"] = traced_ids
    if traced_ids != expected_ids:
        return None, qc | {"reason": "not_every_soma_was_traced"}

    hit_rate = float(np.mean(input_seg[rows, cols] > 0))
    qc["hit_rate"] = round(hit_rate, 4)
    if hit_rate < settings.ntt_min_hit_rate:
        return None, qc | {"reason": "traces_left_the_segmented_structure"}

    pixel_owners: dict[int, int] = {}
    for soma_id, pixel in zip(soma_ids.tolist(), linear.tolist()):
        previous = pixel_owners.get(pixel)
        if previous is not None and previous != soma_id:
            return None, qc | {"reason": "same_trace_pixel_assigned_to_multiple_somas"}
        pixel_owners[pixel] = soma_id

    all_trace_mask = np.zeros(input_seg.shape, dtype=bool)
    cells: list[dict[str, object]] = []
    for soma_id in expected_ids:
        selection = soma_ids == soma_id
        cell_rows = rows[selection]
        cell_cols = cols[selection]
        trace_mask = np.zeros(input_seg.shape, dtype=bool)
        trace_mask[cell_rows, cell_cols] = True
        own_soma = soma_instances == soma_id
        other_soma = (soma_instances > 0) & ~own_soma
        trace_pixels = int(trace_mask.sum())
        if trace_pixels < settings.ntt_min_trace_pixels:
            return None, qc | {"reason": f"soma_{soma_id}_trace_too_short"}
        if not np.any(trace_mask & ndi.binary_dilation(own_soma, structure=_disk(3))):
            return None, qc | {"reason": f"soma_{soma_id}_trace_not_attached_to_own_soma"}
        if np.any(trace_mask & ndi.binary_dilation(other_soma, structure=_disk(1))):
            return None, qc | {"reason": f"soma_{soma_id}_trace_touches_other_soma"}
        trace_mask &= ~(soma_instances > 0)
        all_trace_mask |= trace_mask
        cells.append(
            {
                "soma_id": soma_id,
                "skeleton_mask": trace_mask,
                "soma_mask": own_soma,
                "trace_pixels": trace_pixels,
            }
        )

    covered = ndi.binary_dilation(all_trace_mask, structure=_disk(3)) & original_skeleton
    coverage = float(covered.sum() / max(1, int(original_skeleton.sum())))
    qc["skeleton_coverage_within_3px"] = round(coverage, 4)
    if coverage < settings.ntt_min_coverage:
        return None, qc | {"reason": "trace_coverage_too_low"}
    return cells, qc | {"reason": "accepted"}


def _run_ntt_group(
    group_dir: Path,
    input_seg: np.ndarray,
    input_soma: np.ndarray,
    callback: ProgressCallback,
    settings: PipelineSettings,
    progress: int,
) -> np.ndarray:
    export_dir = group_dir / "input"
    trace_dir = group_dir / "traces"
    export_dir.mkdir(parents=True, exist_ok=False)
    trace_dir.mkdir(parents=True, exist_ok=False)
    sio.savemat(export_dir / "inputSeg.mat", {"inputSeg": input_seg.astype(np.uint8)}, do_compression=True)
    sio.savemat(export_dir / "inputSoma.mat", {"inputSoma": input_soma.astype(np.uint8)}, do_compression=True)
    tifffile.imwrite(export_dir / "inputSeg_preview.tif", (input_seg * 255).astype(np.uint8))
    tifffile.imwrite(export_dir / "inputSoma_preview.tif", (input_soma * 255).astype(np.uint8))

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(settings.octave_runner),
        "-InputDir",
        str(export_dir),
        "-OutputDir",
        str(trace_dir),
        "-ProjectRoot",
        str(settings.project_root),
    ]
    _run_command(command, callback, "neurotreetracer", progress, settings.project_root)
    traces_path = trace_dir / "traces_flat.mat"
    if not traces_path.is_file():
        raise RuntimeError("NeuroTreeTracer did not write traces_flat.mat.")
    data = sio.loadmat(traces_path)
    if "traces" not in data:
        raise RuntimeError("NeuroTreeTracer result contains no 'traces' variable.")
    return np.asarray(data["traces"], dtype=np.int64)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "folder",
        "source",
        "source_component",
        "conflict_group",
        "skeleton_pixels",
        "soma_pixels",
        "x_min",
        "y_min",
        "x_max_exclusive",
        "y_max_exclusive",
        "preview",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_single_cells(
    original_path: Path,
    semantic_path: Path,
    output_root: Path,
    work_root: Path,
    callback: ProgressCallback,
    settings: PipelineSettings,
    ntt_runner: Callable[[Path, np.ndarray, np.ndarray, ProgressCallback, PipelineSettings, int], np.ndarray]
    | None = None,
    prediction_path: Path | None = None,
) -> dict[str, object]:
    raw = _read_2d(original_path, "Original image")
    semantic = _read_2d(semantic_path, "Adaptive semantic map").astype(np.uint8)
    if raw.shape != semantic.shape:
        raise RuntimeError(f"Shape mismatch: raw={raw.shape}, semantic={semantic.shape}")
    values = set(np.unique(semantic).astype(int).tolist())
    if not values <= {0, 1, 2}:
        raise RuntimeError(f"Adaptive semantic map contains invalid labels: {sorted(values)}")
    network_semantic = semantic
    if prediction_path is not None:
        network_semantic = _read_2d(prediction_path, "Network prediction").astype(np.uint8)
        if network_semantic.shape != raw.shape:
            raise RuntimeError(
                f"Shape mismatch: raw={raw.shape}, prediction={network_semantic.shape}"
            )
        prediction_values = set(np.unique(network_semantic).astype(int).tolist())
        if not prediction_values <= {0, 1, 2}:
            raise RuntimeError(
                f"Network prediction contains invalid labels: {sorted(prediction_values)}"
            )

    helper = _load_export_helpers(settings.project_root)
    output_root.mkdir(parents=True, exist_ok=False)
    work_root.mkdir(parents=True, exist_ok=False)
    skeleton = semantic == 1
    _connected_soma_labels, connected_soma_count = ndi.label(
        semantic == 2, structure=CONNECTIVITY_8
    )
    raw_soma_labels, soma_split_report = _conservative_soma_instances(
        semantic == 2, settings
    )
    raw_soma_count = int(raw_soma_labels.max())
    unsafe_unsplit_soma_ids: set[int] = set()
    for row in soma_split_report:
        if not row.get("candidate_for_split") or row.get("reason") == "watershed_split":
            continue
        unsafe_unsplit_soma_ids.update(
            int(value)
            for value in str(row.get("output_instances", "")).split(";")
            if value
        )
    soma_areas = np.bincount(raw_soma_labels.ravel(), minlength=raw_soma_count + 1)
    valid_soma_ids = [
        soma_id
        for soma_id in range(1, raw_soma_count + 1)
        if int(soma_areas[soma_id]) >= settings.min_soma_area
    ]
    valid_soma = np.isin(raw_soma_labels, valid_soma_ids)
    material = skeleton | valid_soma
    material_labels, material_count = ndi.label(material, structure=CONNECTIVITY_8)
    objects = ndi.find_objects(material_labels)

    direct_components: list[tuple[int, int, tuple[slice, slice]]] = []
    conflict_components: list[tuple[int, list[int], tuple[slice, slice]]] = []
    discarded_no_soma = 0
    discarded_direct = 0
    discarded_unsafe_soma_components: list[dict[str, object]] = []
    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        component = material_labels[slices] == component_id
        soma_ids = sorted(
            int(value)
            for value in np.unique(raw_soma_labels[slices][component])
            if int(value) in valid_soma_ids
        )
        skeleton_pixels = int(np.count_nonzero(component & skeleton[slices]))
        if not soma_ids:
            discarded_no_soma += 1
        elif any(soma_id in unsafe_unsplit_soma_ids for soma_id in soma_ids):
            discarded_unsafe_soma_components.append(
                {
                    "group": "",
                    "component": component_id,
                    "soma_count": len(soma_ids),
                    "reason": "oversized_soma_not_safely_split",
                }
            )
        elif len(soma_ids) == 1:
            if skeleton_pixels >= settings.min_skeleton_pixels:
                direct_components.append((component_id, soma_ids[0], slices))
            else:
                discarded_direct += 1
        else:
            conflict_components.append((component_id, soma_ids, slices))

    _notify(
        callback,
        "cell_classification",
        50,
        (
            f"{len(direct_components)} isolated cells and "
            f"{len(conflict_components)} touching/overlapping groups found."
        ),
        {
            "isolated_candidates": len(direct_components),
            "conflict_groups": len(conflict_components),
        },
    )

    manifest: list[dict[str, object]] = []
    rejected_groups: list[dict[str, object]] = list(discarded_unsafe_soma_components)
    output_index = 0
    isolated_exported = 0
    for index, (component_id, soma_id, slices) in enumerate(direct_components, start=1):
        local_component = material_labels[slices] == component_id
        local_skeleton = local_component & skeleton[slices]
        local_soma = raw_soma_labels[slices] == soma_id
        offset = np.asarray([slices[0].start, slices[1].start], dtype=np.int64)
        skeleton_coords = np.argwhere(local_skeleton) + offset
        soma_coords = np.argwhere(local_soma) + offset
        try:
            output_index += 1
            row = _export_cell(
                raw,
                network_semantic,
                semantic,
                skeleton_coords,
                soma_coords,
                output_root,
                output_index,
                helper,
                settings,
                source="isolated_topology",
                source_component=component_id,
                conflict_group=None,
            )
            manifest.append(row)
            isolated_exported += 1
        except Exception as error:
            output_index -= 1
            discarded_direct += 1
            rejected_groups.append(
                {
                    "group": "",
                    "component": component_id,
                    "soma_count": 1,
                    "reason": str(error),
                }
            )
        if index == len(direct_components) or index % 25 == 0:
            _notify(
                callback,
                "cell_classification",
                52,
                f"Exported {isolated_exported}/{len(direct_components)} isolated cells.",
            )

    runner = ntt_runner or _run_ntt_group
    ntt_groups_accepted = 0
    ntt_cells_exported = 0
    for group_number, (component_id, soma_ids, slices) in enumerate(conflict_components, start=1):
        percent = 55 + int(25 * (group_number - 1) / max(1, len(conflict_components)))
        _notify(
            callback,
            "neurotreetracer",
            percent,
            f"Tracing conflict group {group_number}/{len(conflict_components)} ({len(soma_ids)} cells).",
            {"group": group_number, "somas": len(soma_ids)},
        )
        local_component = material_labels[slices] == component_id
        offset = np.asarray([slices[0].start, slices[1].start], dtype=np.int64)
        component_coords = np.argwhere(local_component) + offset
        reason: str | None = None
        qc: dict[str, object] = {}
        if _touches_border_from_coords(component_coords, raw.shape, 1):
            reason = "conflict_group_touches_source_border"
        else:
            y0, y1, x0, x1 = _square_bounds_from_coords(
                component_coords, raw.shape, settings.ntt_margin, settings.ntt_min_crop_size
            )
            if max(y1 - y0, x1 - x0) > settings.ntt_max_crop_side:
                reason = "conflict_group_too_large_for_safe_neurotreetracer_memory"
            else:
                group_component = material_labels[y0:y1, x0:x1] == component_id
                group_skeleton = skeleton[y0:y1, x0:x1] & group_component
                group_soma_instances = raw_soma_labels[y0:y1, x0:x1].copy()
                group_soma_instances[~group_component] = 0
                group_soma = _separate_touching_soma_instances(group_soma_instances)
                if np.any(group_soma[[0, -1], :]) or np.any(group_soma[:, [0, -1]]):
                    reason = "soma_touches_neurotreetracer_crop_border"
                elif ndi.label(group_soma, structure=CONNECTIVITY_8)[1] != len(soma_ids):
                    reason = "split_soma_instances_not_safely_separable"
                else:
                    thick = ndi.binary_dilation(
                        group_skeleton, structure=_disk(settings.ntt_dilate_radius)
                    )
                    input_seg = thick | group_soma
                    memory_gb = NTT_RECTANGLE_COUNT * input_seg.size / 1e9
                    qc = {
                        "crop": [int(y0), int(y1), int(x0), int(x1)],
                        "estimated_memory_gb": round(float(memory_gb), 2),
                    }
                    group_dir = work_root / f"group_{group_number:04d}"
                    try:
                        traces = runner(
                            group_dir,
                            input_seg,
                            group_soma,
                            callback,
                            settings,
                            percent,
                        )
                        cells, trace_qc = _validate_ntt_traces(
                            traces,
                            input_seg,
                            group_soma,
                            group_skeleton,
                            settings,
                        )
                        qc |= trace_qc
                        (group_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
                        if cells is None:
                            reason = str(trace_qc["reason"])
                        else:
                            pending_rows: list[dict[str, object]] = []
                            pending_dirs: list[Path] = []
                            start_index = output_index
                            try:
                                tracer_soma_instances = _tracer_soma_instances(group_soma)
                                for cell in cells:
                                    local_skeleton_coords = np.argwhere(cell["skeleton_mask"])
                                    tracer_soma_id = int(cell["soma_id"])
                                    watershed_ids = np.unique(
                                        group_soma_instances[
                                            tracer_soma_instances == tracer_soma_id
                                        ]
                                    )
                                    watershed_ids = watershed_ids[watershed_ids > 0]
                                    if len(watershed_ids) != 1:
                                        raise RuntimeError(
                                            "tracer_soma_could_not_be_mapped_to_one_watershed_instance"
                                        )
                                    local_soma_coords = np.argwhere(
                                        group_soma_instances == int(watershed_ids[0])
                                    )
                                    skeleton_coords = local_skeleton_coords + np.asarray([y0, x0])
                                    soma_coords = local_soma_coords + np.asarray([y0, x0])
                                    output_index += 1
                                    row = _export_cell(
                                        raw,
                                        network_semantic,
                                        semantic,
                                        skeleton_coords,
                                        soma_coords,
                                        output_root,
                                        output_index,
                                        helper,
                                        settings,
                                        source="neurotreetracer",
                                        source_component=component_id,
                                        conflict_group=group_number,
                                        qc=qc | {"tracer_soma_id": tracer_soma_id},
                                    )
                                    pending_rows.append(row)
                                    pending_dirs.append(output_root / str(row["folder"]))
                            except Exception:
                                for directory in pending_dirs:
                                    shutil.rmtree(directory, ignore_errors=True)
                                output_index = start_index
                                raise
                            manifest.extend(pending_rows)
                            ntt_groups_accepted += 1
                            ntt_cells_exported += len(pending_rows)
                    except Exception as error:
                        reason = f"neurotreetracer_error: {error}"
        if reason is not None:
            rejected_groups.append(
                {
                    "group": group_number,
                    "component": component_id,
                    "soma_count": len(soma_ids),
                    "reason": reason,
                }
            )
            _notify(
                callback,
                "neurotreetracer",
                percent,
                f"Conflict group {group_number} rejected: {reason}",
            )

    _write_manifest(output_root / "manifest.csv", manifest)
    with (output_root / "rejected_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "component", "soma_count", "reason"])
        writer.writeheader()
        writer.writerows(rejected_groups)

    summary = {
        "script_version": SCRIPT_VERSION,
        "original": str(original_path.resolve()),
        "semantic": str(semantic_path.resolve()),
        "shape": [int(raw.shape[0]), int(raw.shape[1])],
        "raw_soma_components": int(connected_soma_count),
        "soma_instances_after_conservative_split": int(raw_soma_count),
        "soma_components_split": sum(
            row.get("reason") == "watershed_split" for row in soma_split_report
        ),
        "soma_split_candidates": sum(
            bool(row.get("candidate_for_split")) for row in soma_split_report
        ),
        "unsafe_unsplit_soma_instances": len(unsafe_unsplit_soma_ids),
        "components_rejected_for_unsafe_soma": len(
            discarded_unsafe_soma_components
        ),
        "valid_soma_components": len(valid_soma_ids),
        "ignored_small_soma_components": sum(
            row.get("reason") == "too_small" for row in soma_split_report
        ),
        "material_components": int(material_count),
        "isolated_candidates": len(direct_components),
        "isolated_exported": isolated_exported,
        "conflict_groups": len(conflict_components),
        "neurotreetracer_groups_accepted": ntt_groups_accepted,
        "neurotreetracer_groups_rejected": len(conflict_components) - ntt_groups_accepted,
        "neurotreetracer_cells_exported": ntt_cells_exported,
        "discarded_components_without_soma": discarded_no_soma,
        "discarded_direct_candidates": discarded_direct,
        "exported_cell_count": len(manifest),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(settings).items()
        },
    }
    (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _notify(
        callback,
        "cell_export",
        82,
        f"{len(manifest)} one-cell crops exported; {len(rejected_groups)} groups/candidates rejected.",
        summary,
    )
    return summary


def _finalize_for_evo(
    cell_root: Path,
    callback: ProgressCallback,
    settings: PipelineSettings,
) -> None:
    _assert_files([settings.fiji, settings.fiji_finalizer, settings.validator])
    env = os.environ.copy()
    env["CELL_EXPORT_ROOT"] = str(cell_root)
    command = [
        str(settings.fiji),
        "--allow-multiple",
        "--headless",
        "--console",
        "--run",
        str(settings.fiji_finalizer),
    ]
    _run_command(
        command,
        callback,
        "evo_finalization",
        88,
        settings.project_root,
        env=env,
        notify_line=lambda line: (
            "] OK " in line
            or line.startswith("GLOBAL OK")
            or line.startswith("Finalized")
        ),
    )
    _run_command(
        [str(settings.python), str(settings.validator), "--input-dir", str(cell_root)],
        callback,
        "evo_validation",
        94,
        settings.project_root,
    )


def _zip_cell_export(cell_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(cell_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(cell_root.parent))


def ensure_review_layers_for_run(run_root: Path) -> int:
    """Backfill review TIFFs/previews for runs created before review support existed."""
    cell_root = run_root / "04_evo_single_cells"
    if not cell_root.is_dir():
        return 0
    prediction_roots = sorted(run_root.glob("02_prediction_dataset*"))
    prediction_paths = sorted(
        path for root in prediction_roots if root.is_dir() for path in root.glob("*.tif")
    )
    postprocessed_paths = sorted(
        (run_root / "03_adaptive_hysteresis").glob("*_adaptive_hysteresis_0-1-2.tif")
    )
    prediction = _read_2d(prediction_paths[0], "Network prediction") if prediction_paths else None
    postprocessed = (
        _read_2d(postprocessed_paths[0], "Adaptive semantic map")
        if postprocessed_paths
        else None
    )
    if prediction is not None and postprocessed is not None and prediction.shape != postprocessed.shape:
        return 0

    updated = 0
    for cell_dir in sorted(path for path in cell_root.glob("cell*") if path.is_dir()):
        raw_path = cell_dir / "raw.tif"
        bounds_path = cell_dir / "bounds.json"
        if not raw_path.is_file():
            continue
        changed = False
        try:
            raw_crop = _read_2d(raw_path, "Cell raw crop")
            raw_preview = cell_dir / "raw_preview.png"
            if not raw_preview.is_file():
                _write_raw_preview(raw_preview, raw_crop)
                changed = True
            if prediction is None or postprocessed is None or not bounds_path.is_file():
                if changed:
                    updated += 1
                continue
            bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
            y0 = int(bounds["y_min"])
            y1 = int(bounds["y_max_exclusive"])
            x0 = int(bounds["x_min"])
            x1 = int(bounds["x_max_exclusive"])
            prediction_crop = prediction[y0:y1, x0:x1].astype(np.uint8, copy=False)
            postprocessed_crop = postprocessed[y0:y1, x0:x1].astype(np.uint8, copy=False)
            if prediction_crop.shape != raw_crop.shape or postprocessed_crop.shape != raw_crop.shape:
                continue
            targets = (
                (cell_dir / "prediction.tif", prediction_crop),
                (cell_dir / "postprocessed.tif", postprocessed_crop),
            )
            for path, array in targets:
                if not path.is_file():
                    tifffile.imwrite(path, array)
                    changed = True
            prediction_preview = cell_dir / "prediction_preview.png"
            if not prediction_preview.is_file():
                _write_semantic_preview(prediction_preview, raw_crop, prediction_crop)
                changed = True
            postprocessing_preview = cell_dir / "postprocessing_preview.png"
            if not postprocessing_preview.is_file():
                _write_semantic_preview(postprocessing_preview, raw_crop, postprocessed_crop)
                changed = True
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            continue
        if changed:
            updated += 1
    return updated


def run_pipeline(
    input_path: Path,
    run_root: Path,
    callback: ProgressCallback,
    settings: PipelineSettings,
) -> dict[str, object]:
    _assert_files(
        [
            settings.python,
            settings.predictor,
            settings.hysteresis_script,
            settings.octave_runner,
        ]
    )
    shape = _validate_input(input_path)
    case_id = re.sub(r"_0000$", "", input_path.stem)
    prediction_root = run_root / f"02_prediction_dataset{settings.dataset_id}"
    hysteresis_root = run_root / "03_adaptive_hysteresis"
    cell_root = run_root / "04_evo_single_cells"
    ntt_work_root = run_root / "work_neurotreetracer"
    prediction_root.mkdir(parents=True, exist_ok=True)

    _notify(
        callback,
        "prediction",
        8,
        f"Dataset {settings.dataset_id} prediction started for {shape[1]} x {shape[0]} px.",
    )
    prediction_command = [
        str(settings.predictor),
        "-i",
        str(input_path.parent),
        "-o",
        str(prediction_root),
        "-d",
        str(settings.dataset_id),
        "-c",
        settings.configuration,
        "-f",
        str(settings.fold),
        "-tr",
        settings.trainer,
        "-p",
        settings.plans,
        "-chk",
        settings.checkpoint,
        "-device",
        settings.device,
        "--save_probabilities",
    ]
    _run_command(
        prediction_command,
        callback,
        "prediction",
        18,
        settings.project_root,
        env=_prediction_environment(settings),
    )
    probability_path = prediction_root / f"{case_id}.npz"
    prediction_path = prediction_root / f"{case_id}.tif"
    _assert_files([probability_path, prediction_path])

    threshold_text = (
        f"adaptive alpha={settings.hysteresis_alpha:g}"
        if settings.hysteresis_mode == "adaptive"
        else f"fixed T_low={settings.hysteresis_t_low:g}, T_high={settings.hysteresis_t_high:g}"
    )
    _notify(
        callback,
        "hysteresis",
        32,
        f"Hysteresis is reconnecting weak skeleton pixels ({threshold_text}).",
    )
    hysteresis_root.mkdir(parents=True, exist_ok=True)
    hysteresis_command = [
        str(settings.python),
        str(settings.hysteresis_script),
        "--probabilities",
        str(probability_path),
        "--output-dir",
        str(hysteresis_root),
        *hysteresis_cli_arguments(settings),
        "--write-semantic",
        "--overwrite",
    ]
    _run_command(
        hysteresis_command,
        callback,
        "hysteresis",
        40,
        settings.project_root,
    )
    semantic_path = hysteresis_root / f"{case_id}_adaptive_hysteresis_0-1-2.tif"
    _assert_files([semantic_path])

    summary = extract_single_cells(
        original_path=input_path,
        semantic_path=semantic_path,
        output_root=cell_root,
        work_root=ntt_work_root,
        callback=callback,
        settings=settings,
        prediction_path=prediction_path,
    )

    fiji_finalized = False
    if settings.finalize_fiji and int(summary["exported_cell_count"]) > 0:
        _notify(callback, "evo_finalization", 86, "Creating SNT traces and Evo ROI archives.")
        _finalize_for_evo(cell_root, callback, settings)
        fiji_finalized = True
    elif settings.finalize_fiji:
        _notify(callback, "evo_finalization", 94, "No accepted cells; Fiji finalization was skipped.")
    archive_path = run_root / "evo_single_cells.zip"
    _notify(callback, "archive", 97, "Packing the Evo-ready cell folders.")
    _zip_cell_export(cell_root, archive_path)
    result = summary | {
        "cell_root": str(cell_root),
        "archive": str(archive_path),
        "fiji_finalized": fiji_finalized,
    }
    (run_root / "pipeline_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _notify(callback, "complete", 100, f"Complete: {summary['exported_cell_count']} crops are ready.", result)
    return result


__all__ = [
    "PipelineSettings",
    "default_settings",
    "hysteresis_cli_arguments",
    "extract_single_cells",
    "ensure_review_layers_for_run",
    "run_pipeline",
    "_validate_ntt_traces",
]
