from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
from scipy import ndimage as ndi

from apply_instance_separator import pad_to_multiple, padded_bounds
from cell_pipeline_web.pipeline import (
    CONNECTIVITY_8,
    PipelineSettings,
    _assert_files,
    _conservative_soma_instances,
    _export_cell,
    _finalize_for_evo,
    canonical_hysteresis_arguments,
    _load_export_helpers,
    _notify,
    _prediction_environment,
    _read_2d,
    _run_command,
    _validate_input,
    _write_manifest,
    _write_raw_preview,
    _write_semantic_preview,
    _zip_cell_export,
    hysteresis_cli_arguments,
)
from instance_separator import (
    MODEL_VERSION,
    SomaSeededSeparationUNet,
    assign_memberships_to_instances,
    predict_memberships,
)


ProgressCallback = Callable[[str, int, str, dict[str, object] | None], None]
PIPELINE_VERSION = "blind-selection-net141-separator80-v1-2026-09-08"


@dataclass(frozen=True)
class BlindSelectionSettings:
    semantic: PipelineSettings
    separator_checkpoint: Path
    soma_match_radius: float = 80.0
    membership_threshold: float = 0.5
    max_conflict_side: int = 1024


def match_points_to_somas(
    points: list[dict[str, object]],
    soma_instances: np.ndarray,
    *,
    maximum_distance: float,
) -> list[dict[str, object]]:
    """Match raw-image clicks to the nearest unique predicted soma."""

    if soma_instances.ndim != 2:
        raise ValueError("Soma instances must be two-dimensional.")
    has_soma = bool(np.any(soma_instances > 0))
    distances: np.ndarray | None = None
    nearest: np.ndarray | None = None
    if has_soma:
        distances, nearest = ndi.distance_transform_edt(
            soma_instances == 0, return_indices=True
        )
    claimed: dict[int, str] = {}
    matches: list[dict[str, object]] = []
    height, width = soma_instances.shape
    for index, point in enumerate(points, start=1):
        selection_id = str(point.get("selection_id") or f"selection{index:04d}")
        x = int(point["x"])
        y = int(point["y"])
        result: dict[str, object] = {
            "selection_id": selection_id,
            "selection_index": index,
            "x": x,
            "y": y,
            "status": "unmatched",
            "soma_id": None,
            "distance_px": None,
        }
        if not (0 <= x < width and 0 <= y < height):
            result["reason"] = "outside_image"
        elif not has_soma or distances is None or nearest is None:
            result["reason"] = "no_predicted_soma"
        else:
            nearest_y = int(nearest[0, y, x])
            nearest_x = int(nearest[1, y, x])
            distance = float(distances[y, x])
            soma_id = int(soma_instances[nearest_y, nearest_x])
            result["distance_px"] = round(distance, 3)
            result["nearest_x"] = nearest_x
            result["nearest_y"] = nearest_y
            if soma_id <= 0 or distance > maximum_distance:
                result["reason"] = "no_soma_within_radius"
            elif soma_id in claimed:
                result["status"] = "duplicate"
                result["soma_id"] = soma_id
                result["reason"] = "same_soma_as_previous_selection"
                result["duplicate_of"] = claimed[soma_id]
            else:
                result["status"] = "matched"
                result["soma_id"] = soma_id
                result["reason"] = "inside_soma" if distance == 0 else "nearest_soma"
                claimed[soma_id] = selection_id
        matches.append(result)
    return matches


def _unsafe_soma_ids(split_report: list[dict[str, object]]) -> set[int]:
    unsafe: set[int] = set()
    for row in split_report:
        if not row.get("candidate_for_split") or row.get("reason") == "watershed_split":
            continue
        unsafe.update(
            int(value)
            for value in str(row.get("output_instances", "")).split(";")
            if value
        )
    return unsafe


def _load_separator(settings: BlindSelectionSettings):
    import torch

    device = torch.device(settings.semantic.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    checkpoint = torch.load(
        settings.separator_checkpoint, map_location=device, weights_only=False
    )
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Separator checkpoint version {checkpoint.get('model_version')!r} "
            f"does not match {MODEL_VERSION!r}."
        )
    base_channels = int(dict(checkpoint.get("config") or {}).get("base_channels", 16))
    model = SomaSeededSeparationUNet(base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device, int(checkpoint.get("epoch", -1))


def _connected_membership(
    membership: np.ndarray, own_soma: np.ndarray
) -> tuple[np.ndarray, int]:
    labels, _ = ndi.label(membership, structure=CONNECTIVITY_8)
    seed_labels = [int(value) for value in np.unique(labels[own_soma]) if value > 0]
    connected = np.isin(labels, seed_labels) if seed_labels else np.zeros_like(membership)
    disconnected = int(np.count_nonzero(membership & ~connected))
    return connected, disconnected


def extract_selected_cells(
    *,
    original_path: Path,
    prediction_path: Path,
    semantic_path: Path,
    output_root: Path,
    selection: dict[str, object],
    callback: ProgressCallback,
    settings: BlindSelectionSettings,
) -> dict[str, object]:
    raw = _read_2d(original_path, "Original image")
    network_semantic = _read_2d(prediction_path, "Network prediction").astype(np.uint8)
    semantic = _read_2d(semantic_path, "Hysteresis semantic map").astype(np.uint8)
    if raw.shape != semantic.shape or raw.shape != network_semantic.shape:
        raise RuntimeError("Raw image, prediction and hysteresis output have different sizes.")
    if set(np.unique(semantic).astype(int).tolist()) - {0, 1, 2}:
        raise RuntimeError("Hysteresis output contains unsupported labels.")

    pipeline_settings = settings.semantic
    helper = _load_export_helpers(pipeline_settings.project_root)
    output_root.mkdir(parents=True, exist_ok=False)
    soma_instances, split_report = _conservative_soma_instances(
        semantic == 2, pipeline_settings
    )
    unsafe_ids = _unsafe_soma_ids(split_report)
    points = [dict(value) for value in list(selection.get("points") or [])]
    matches = match_points_to_somas(
        points, soma_instances, maximum_distance=settings.soma_match_radius
    )

    valid_soma_ids = {
        int(value) for value in np.unique(soma_instances) if int(value) > 0
    }
    material = (semantic == 1) | np.isin(soma_instances, list(valid_soma_ids))
    material_labels, material_count = ndi.label(material, structure=CONNECTIVITY_8)
    objects = ndi.find_objects(material_labels, max_label=material_count)
    component_by_soma: dict[int, int] = {}
    for soma_id in valid_soma_ids:
        component_ids = material_labels[soma_instances == soma_id]
        component_ids = component_ids[component_ids > 0]
        if component_ids.size:
            component_by_soma[soma_id] = int(np.bincount(component_ids).argmax())

    matches_by_component: dict[int, list[dict[str, object]]] = {}
    failures: list[dict[str, object]] = []
    for match in matches:
        if match["status"] != "matched":
            failures.append(dict(match))
            continue
        soma_id = int(match["soma_id"])
        component_id = component_by_soma.get(soma_id)
        if component_id is None:
            failures.append(dict(match) | {"reason": "soma_has_no_foreground_component"})
            continue
        matches_by_component.setdefault(component_id, []).append(match)

    _notify(
        callback,
        "selection_matching",
        50,
        f"{sum(len(v) for v in matches_by_component.values())} of {len(points)} selections matched to predicted somas.",
        {"matches": matches},
    )

    model = device = None
    separator_epoch: int | None = None
    manifest: list[dict[str, object]] = []
    output_index = 0
    for group_index, (component_id, selected_matches) in enumerate(
        sorted(matches_by_component.items()), start=1
    ):
        slices = objects[component_id - 1]
        if slices is None:
            for match in selected_matches:
                failures.append(dict(match) | {"reason": "missing_component_bounds"})
            continue
        component_view = material_labels[slices] == component_id
        soma_ids = sorted(
            int(value)
            for value in np.unique(soma_instances[slices][component_view])
            if int(value) > 0
        )
        selected_ids = {int(match["soma_id"]) for match in selected_matches}
        if any(soma_id in unsafe_ids for soma_id in soma_ids):
            for match in selected_matches:
                failures.append(dict(match) | {"reason": "unsafe_unsplit_soma"})
            continue

        # Memberships stay local to the component crop. A 5k overview image is
        # about 25 million pixels; keeping a full-size float map per selected
        # cell would otherwise consume hundreds of MB for one conflict group.
        memberships: dict[
            int, tuple[np.ndarray, np.ndarray, float, int, int]
        ] = {}
        conflict = len(soma_ids) > 1
        if not conflict:
            soma_id = soma_ids[0]
            local_membership = material_labels[slices] == component_id
            memberships[soma_id] = (
                local_membership,
                np.ones(local_membership.shape, dtype=np.float32),
                1.0,
                int(slices[0].start),
                int(slices[1].start),
            )
        else:
            y0, y1, x0, x1 = padded_bounds(
                slices, semantic.shape, pipeline_settings.crop_margin
            )
            if max(y1 - y0, x1 - x0) > settings.max_conflict_side:
                for match in selected_matches:
                    failures.append(dict(match) | {"reason": "conflict_group_too_large"})
                continue
            if model is None:
                _notify(
                    callback,
                    "instance_separation",
                    58,
                    "Loading checkpoint 80 for overlapping-cell separation.",
                )
                model, device, separator_epoch = _load_separator(settings)
            group_mask = material_labels[y0:y1, x0:x1] == component_id
            local_semantic = np.where(
                group_mask, semantic[y0:y1, x0:x1], 0
            ).astype(np.uint8)
            local_somas = np.where(
                np.isin(soma_instances[y0:y1, x0:x1], soma_ids),
                soma_instances[y0:y1, x0:x1],
                0,
            ).astype(np.uint16)
            original_shape = local_semantic.shape
            padded_raw, padded_semantic, padded_somas = pad_to_multiple(
                raw[y0:y1, x0:x1], local_semantic, local_somas
            )
            local_ids, probabilities = predict_memberships(
                model, padded_raw, padded_semantic, padded_somas, device
            )
            primary, secondary, confidence = assign_memberships_to_instances(
                probabilities,
                local_ids,
                padded_semantic,
                padded_somas,
                membership_threshold=settings.membership_threshold,
            )
            primary = primary[: original_shape[0], : original_shape[1]]
            secondary = secondary[: original_shape[0], : original_shape[1]]
            confidence = confidence[: original_shape[0], : original_shape[1]]
            for soma_id in selected_ids:
                local_membership = group_mask & (
                    (primary == soma_id) | (secondary == soma_id)
                )
                memberships[soma_id] = (
                    local_membership,
                    confidence,
                    float(confidence[group_mask].mean()) if group_mask.any() else 0.0,
                    y0,
                    x0,
                )

        for match in selected_matches:
            soma_id = int(match["soma_id"])
            if soma_id not in memberships:
                failures.append(dict(match) | {"reason": "separator_returned_no_membership"})
                continue
            membership, confidence, mean_confidence, offset_y, offset_x = memberships[soma_id]
            own_soma = (
                soma_instances[
                    offset_y : offset_y + membership.shape[0],
                    offset_x : offset_x + membership.shape[1],
                ]
                == soma_id
            )
            connected, disconnected_pixels = _connected_membership(membership, own_soma)
            local_semantic = semantic[
                offset_y : offset_y + membership.shape[0],
                offset_x : offset_x + membership.shape[1],
            ]
            skeleton = connected & (local_semantic == 1)
            disconnected_skeleton = int(
                np.count_nonzero(membership & ~connected & (local_semantic == 1))
            )
            low_confidence_skeleton = int(
                np.count_nonzero(skeleton & (confidence < 0.5))
            ) if conflict else 0
            reasons: list[str] = []
            if not own_soma.any():
                reasons.append("missing_soma")
            if int(skeleton.sum()) < pipeline_settings.min_skeleton_pixels:
                reasons.append("too_few_connected_skeleton_pixels")
            if conflict and disconnected_skeleton >= pipeline_settings.min_skeleton_pixels:
                reasons.append("disconnected_assignment_fragment")
            if conflict and low_confidence_skeleton >= pipeline_settings.min_skeleton_pixels:
                reasons.append("low_confidence_skeleton")
            if reasons:
                failures.append(
                    dict(match)
                    | {
                        "reason": ";".join(reasons),
                        "component_id": component_id,
                        "competing_soma_ids": soma_ids,
                    }
                )
                continue
            output_index += 1
            try:
                row = _export_cell(
                    raw,
                    network_semantic,
                    semantic,
                    np.argwhere(skeleton) + np.asarray([offset_y, offset_x]),
                    np.argwhere(own_soma) + np.asarray([offset_y, offset_x]),
                    output_root,
                    output_index,
                    helper,
                    pipeline_settings,
                    source="manual_blind_selection_checkpoint80" if conflict else "manual_blind_selection_direct",
                    source_component=component_id,
                    conflict_group=group_index if conflict else None,
                    qc={
                        "selection_id": match["selection_id"],
                        "selection_x": match["x"],
                        "selection_y": match["y"],
                        "matched_soma_id": soma_id,
                        "match_distance_px": match["distance_px"],
                        "competing_soma_ids": soma_ids,
                        "separator_checkpoint_epoch": separator_epoch,
                        "mean_assignment_confidence": round(mean_confidence, 4),
                        "low_confidence_skeleton_pixels": low_confidence_skeleton,
                        "disconnected_membership_pixels": disconnected_pixels,
                        "disconnected_skeleton_pixels": disconnected_skeleton,
                    },
                )
                row.update(
                    {
                        "selection_id": match["selection_id"],
                        "selection_index": match["selection_index"],
                        "selection_x": match["x"],
                        "selection_y": match["y"],
                        "matched_soma_id": soma_id,
                    }
                )
                manifest.append(row)
                match["status"] = "exported_candidate"
                match["cell_folder"] = row["folder"]
                match["component_id"] = component_id
                match["competing_soma_ids"] = soma_ids
            except RuntimeError as error:
                failures.append(
                    dict(match)
                    | {"reason": str(error), "component_id": component_id}
                )

        _notify(
            callback,
            "instance_separation",
            58 + int(22 * group_index / max(1, len(matches_by_component))),
            f"Processed selected cell group {group_index}/{len(matches_by_component)}.",
        )

    standard_fields = {
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
    }
    _write_manifest(
        output_root / "manifest.csv",
        [{key: value for key, value in row.items() if key in standard_fields} for row in manifest],
    )
    # A second machine-readable manifest preserves the selection provenance that
    # the legacy Evo manifest intentionally does not contain.
    (output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matching = {
        "pipeline_version": PIPELINE_VERSION,
        "maximum_match_distance_px": settings.soma_match_radius,
        "points": matches,
        "failures": failures,
        "soma_split_report": split_report,
    }
    (output_root.parent / "model_matching.json").write_text(
        json.dumps(matching, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "requested_selections": len(points),
        "matched_selections": sum(row["status"] in {"matched", "exported_candidate"} for row in matches),
        "exported_cell_count": len(manifest),
        "failed_selections": len(failures),
        "matches": matches,
        "failures": failures,
        "separator_checkpoint_epoch": separator_epoch,
        "manifest": manifest,
    }


def run_blind_selection_pipeline(
    input_path: Path,
    run_root: Path,
    selection: dict[str, object],
    callback: ProgressCallback,
    settings: BlindSelectionSettings,
) -> dict[str, object]:
    semantic_settings = settings.semantic
    required = [
        semantic_settings.python,
        semantic_settings.predictor,
        semantic_settings.hysteresis_script,
        settings.separator_checkpoint,
    ]
    if semantic_settings.finalize_fiji:
        required.extend(
            [semantic_settings.fiji, semantic_settings.fiji_finalizer, semantic_settings.validator]
        )
    _assert_files(required)
    shape = _validate_input(input_path)
    case_id = re.sub(r"_0000$", "", input_path.stem)
    prediction_root = run_root / f"02_prediction_dataset{semantic_settings.dataset_id}"
    hysteresis_root = run_root / "03_hysteresis_tlow0200"
    cell_root = run_root / "04_evo_single_cells"
    prediction_root.mkdir(parents=True, exist_ok=True)

    _notify(callback, "prediction", 8, f"Net 141 prediction started for {shape[1]} x {shape[0]} px.")
    _run_command(
        [
            str(semantic_settings.predictor), "-i", str(input_path.parent),
            "-o", str(prediction_root), "-d", str(semantic_settings.dataset_id),
            "-c", semantic_settings.configuration, "-f", str(semantic_settings.fold),
            "-tr", semantic_settings.trainer, "-p", semantic_settings.plans,
            "-chk", semantic_settings.checkpoint, "-device", semantic_settings.device,
            "--save_probabilities",
        ],
        callback, "prediction", 20, semantic_settings.project_root,
        env=_prediction_environment(semantic_settings),
    )
    probability_path = prediction_root / f"{case_id}.npz"
    prediction_path = prediction_root / f"{case_id}.tif"
    _assert_files([probability_path, prediction_path])

    _notify(callback, "hysteresis", 32, "Hysteresis is reconnecting skeleton pixels (T_low 0.20).")
    hysteresis_root.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            str(semantic_settings.python), str(semantic_settings.hysteresis_script),
            "--probabilities", str(probability_path), "--output-dir", str(hysteresis_root),
            *hysteresis_cli_arguments(semantic_settings), "--write-semantic", "--overwrite",
            *canonical_hysteresis_arguments(semantic_settings),
        ],
        callback, "hysteresis", 42, semantic_settings.project_root,
    )
    semantic_path = hysteresis_root / f"{case_id}_adaptive_hysteresis_0-1-2.tif"
    _assert_files([semantic_path])
    _write_raw_preview(run_root / "raw_after_selection.png", _read_2d(input_path, "raw"))
    _write_semantic_preview(
        run_root / "prediction_overview.png",
        _read_2d(input_path, "raw"),
        _read_2d(prediction_path, "prediction"),
    )
    _write_semantic_preview(
        run_root / "hysteresis_overview.png",
        _read_2d(input_path, "raw"),
        _read_2d(semantic_path, "hysteresis"),
    )

    summary = extract_selected_cells(
        original_path=input_path,
        prediction_path=prediction_path,
        semantic_path=semantic_path,
        output_root=cell_root,
        selection=selection,
        callback=callback,
        settings=settings,
    )
    finalized = False
    if semantic_settings.finalize_fiji and summary["exported_cell_count"]:
        _notify(callback, "evo_finalization", 84, "Creating Evo/SNT files for selected cells.")
        _finalize_for_evo(cell_root, callback, semantic_settings)
        finalized = True
    archive = run_root / "evo_candidate_cells.zip"
    _notify(callback, "archive", 97, "Packing candidate cell folders.")
    _zip_cell_export(cell_root, archive)
    result = summary | {
        "pipeline_version": PIPELINE_VERSION,
        "selection_is_blind": True,
        "separator_checkpoint": str(settings.separator_checkpoint),
        "fiji_finalized": finalized,
        "cell_root": str(cell_root),
        "archive": str(archive),
    }
    (run_root / "pipeline_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _notify(callback, "complete", 100, f"Complete: {summary['exported_cell_count']} selected-cell crops are ready.", result)
    return result


__all__ = [
    "BlindSelectionSettings",
    "extract_selected_cells",
    "match_points_to_somas",
    "run_blind_selection_pipeline",
]
