from __future__ import annotations

"""Apply a learned separator to fixed network-141 semantic output."""

import argparse
import html
import json
import math
import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi

from instance_separator import (
    CONNECTIVITY_8,
    MODEL_VERSION,
    SomaSeededSeparationUNet,
    assign_memberships_to_instances,
    predict_memberships,
    read_2d,
)
from soma_graph_cell_extraction import GraphSplitSettings, split_soma_mask


INSTANCE_COLORS = np.asarray(
    [
        (35, 213, 171),
        (255, 99, 155),
        (255, 190, 67),
        (101, 161, 255),
        (190, 114, 255),
        (78, 205, 255),
        (255, 127, 80),
        (141, 220, 100),
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate cell instances using a trained soma embedding model."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--membership-threshold", type=float, default=0.5)
    parser.add_argument("--max-conflict-side", type=int, default=1024)
    parser.add_argument("--crop-margin", type=int, default=48)
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--min-skeleton-pixels", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def padded_bounds(
    slices: tuple[slice, slice],
    shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    y0 = max(0, int(slices[0].start) - margin)
    y1 = min(shape[0], int(slices[0].stop) + margin)
    x0 = max(0, int(slices[1].start) - margin)
    x1 = min(shape[1], int(slices[1].stop) + margin)
    return y0, y1, x0, x1


def pad_to_multiple(
    raw: np.ndarray, semantic: np.ndarray, soma_instances: np.ndarray, multiple: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = semantic.shape
    padded_height = int(math.ceil(height / multiple) * multiple)
    padded_width = int(math.ceil(width / multiple) * multiple)
    padding = ((0, padded_height - height), (0, padded_width - width))
    raw_fill = float(np.median(raw[np.isfinite(raw)])) if np.isfinite(raw).any() else 0
    return (
        np.pad(raw, padding, mode="constant", constant_values=raw_fill),
        np.pad(semantic, padding, mode="constant"),
        np.pad(soma_instances, padding, mode="constant"),
    )


def normalize_u8(raw: np.ndarray) -> np.ndarray:
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.zeros(raw.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.7])
    if high <= low:
        return np.zeros(raw.shape, dtype=np.uint8)
    return np.rint(np.clip((raw - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def overlay_semantic(raw: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    gray = normalize_u8(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for mask, color in (
        (semantic == 1, np.asarray((0, 235, 255), dtype=np.float32)),
        (semantic == 2, np.asarray((255, 58, 163), dtype=np.float32)),
    ):
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_instances(
    raw: np.ndarray, primary: np.ndarray, secondary: np.ndarray
) -> np.ndarray:
    gray = normalize_u8(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for instance_id in (int(v) for v in np.unique(primary) if v > 0):
        color = INSTANCE_COLORS[(instance_id - 1) % len(INSTANCE_COLORS)]
        mask = primary == instance_id
        rgb[mask] = 0.22 * rgb[mask] + 0.78 * color
    overlap = secondary > 0
    if np.any(overlap):
        checker = (np.indices(primary.shape).sum(axis=0) % 2) == 0
        for instance_id in (int(v) for v in np.unique(secondary) if v > 0):
            mask = (secondary == instance_id) & checker
            color = INSTANCE_COLORS[(instance_id - 1) % len(INSTANCE_COLORS)]
            rgb[mask] = color
        rgb[overlap & ~checker] = 255
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def export_cells(
    *,
    raw: np.ndarray,
    semantic: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    soma_instances: np.ndarray,
    confidence: np.ndarray,
    conflicts: list[dict[str, object]],
    output_dir: Path,
    margin: int,
    minimum_size: int,
    min_skeleton_pixels: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_ids = sorted(
        set(int(v) for v in np.unique(primary) if v > 0)
        | set(int(v) for v in np.unique(secondary) if v > 0)
    )
    quality_by_id: dict[int, dict[str, object]] = {}
    cleaned_memberships: dict[int, np.ndarray] = {}
    for instance_id in instance_ids:
        membership = (primary == instance_id) | (secondary == instance_id)
        own_soma = soma_instances == instance_id
        labels, _count = ndi.label(membership, structure=CONNECTIVITY_8)
        seed_labels = set(int(v) for v in np.unique(labels[own_soma]) if v > 0)
        connected = np.isin(labels, list(seed_labels)) if seed_labels else np.zeros_like(membership)
        disconnected_skeleton = int(np.count_nonzero(membership & ~connected & (semantic == 1)))
        connected_skeleton = int(np.count_nonzero(connected & (semantic == 1)))
        low_confidence_skeleton = int(
            np.count_nonzero(connected & (semantic == 1) & (confidence < 0.5))
        )
        reasons: list[str] = []
        if not own_soma.any():
            reasons.append("missing_soma")
        if connected_skeleton < min_skeleton_pixels:
            reasons.append("too_few_connected_skeleton_pixels")
        if disconnected_skeleton >= min_skeleton_pixels:
            reasons.append("disconnected_skeleton_fragment")
        if low_confidence_skeleton >= min_skeleton_pixels:
            reasons.append("low_confidence_skeleton")
        quality_by_id[instance_id] = {
            "instance_id": instance_id,
            "connected_skeleton_pixels": connected_skeleton,
            "disconnected_skeleton_pixels": disconnected_skeleton,
            "low_confidence_skeleton_pixels": low_confidence_skeleton,
            "reasons": reasons,
        }
        cleaned_memberships[instance_id] = connected

    rejected_group_ids: set[int] = set()
    for conflict in conflicts:
        if conflict["status"] != "separated":
            continue
        soma_ids = [int(value) for value in conflict["soma_ids"]]
        group_reasons = {
            instance_id: quality_by_id.get(instance_id, {}).get("reasons", ["missing_assignment"])
            for instance_id in soma_ids
            if quality_by_id.get(instance_id, {}).get("reasons")
            or instance_id not in quality_by_id
        }
        if group_reasons:
            conflict["status"] = "rejected_uncertain"
            conflict["quality_reasons"] = group_reasons
            rejected_group_ids.update(soma_ids)
        else:
            conflict["status"] = "accepted_separation"

    for instance_id in instance_ids:
        quality = quality_by_id[instance_id]
        reasons = list(quality["reasons"])
        if instance_id in rejected_group_ids:
            reasons.append("conflict_group_rejected_as_a_whole")
        if reasons:
            rejected.append({**quality, "reasons": sorted(set(reasons))})
            continue
        membership = cleaned_memberships[instance_id]
        soma = membership & (semantic == 2) & (soma_instances == instance_id)
        skeleton = membership & (semantic == 1)
        pixels = np.argwhere(membership)
        y0, x0 = pixels.min(axis=0)
        y1, x1 = pixels.max(axis=0) + 1
        target_height = max(minimum_size, int(y1 - y0) + 2 * margin)
        target_width = max(minimum_size, int(x1 - x0) + 2 * margin)
        center_y = int(round((y0 + y1) / 2))
        center_x = int(round((x0 + x1) / 2))
        crop_y0 = int(np.clip(center_y - target_height // 2, 0, max(0, raw.shape[0] - target_height)))
        crop_x0 = int(np.clip(center_x - target_width // 2, 0, max(0, raw.shape[1] - target_width)))
        crop_y1 = min(raw.shape[0], crop_y0 + target_height)
        crop_x1 = min(raw.shape[1], crop_x0 + target_width)
        local_membership = membership[crop_y0:crop_y1, crop_x0:crop_x1]
        local_semantic = np.where(
            local_membership,
            semantic[crop_y0:crop_y1, crop_x0:crop_x1],
            0,
        ).astype(np.uint8)
        local_raw = raw[crop_y0:crop_y1, crop_x0:crop_x1]
        cell_dir = output_dir / f"cell{instance_id:04d}"
        cell_dir.mkdir()
        tifffile.imwrite(cell_dir / "raw.tif", local_raw)
        tifffile.imwrite(cell_dir / "seg.tif", local_semantic)
        save_png(cell_dir / "preview.png", overlay_semantic(local_raw, local_semantic))
        record = {
            "instance_id": instance_id,
            "bbox_yxyx": [crop_y0, crop_y1, crop_x0, crop_x1],
            "skeleton_pixels": int(np.count_nonzero(local_semantic == 1)),
            "soma_pixels": int(np.count_nonzero(local_semantic == 2)),
            "has_shared_pixels": bool(np.any((secondary == instance_id) & membership)),
            "quality": quality,
            "directory": str(cell_dir),
        }
        (cell_dir / "metadata.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        records.append(record)
    return records, rejected


def write_conflict_gallery(
    output_dir: Path,
    raw: np.ndarray,
    semantic: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    confidence: np.ndarray,
    conflicts: list[dict[str, object]],
) -> Path:
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for conflict in conflicts:
        component_id = int(conflict["component_id"])
        y0, y1, x0, x1 = (int(v) for v in conflict["crop_yxyx"])
        local_raw = raw[y0:y1, x0:x1]
        local_semantic = semantic[y0:y1, x0:x1]
        local_primary = primary[y0:y1, x0:x1]
        local_secondary = secondary[y0:y1, x0:x1]
        raw_name = f"conflict_{component_id:03d}_raw.png"
        semantic_name = f"conflict_{component_id:03d}_semantic.png"
        instances_name = f"conflict_{component_id:03d}_instances.png"
        confidence_name = f"conflict_{component_id:03d}_confidence.png"
        save_png(assets / raw_name, normalize_u8(local_raw))
        save_png(assets / semantic_name, overlay_semantic(local_raw, local_semantic))
        save_png(
            assets / instances_name,
            overlay_instances(local_raw, local_primary, local_secondary),
        )
        confidence_image = np.rint(
            np.clip(confidence[y0:y1, x0:x1], 0, 1) * 255
        ).astype(np.uint8)
        save_png(assets / confidence_name, confidence_image)
        soma_ids = ", ".join(str(v) for v in conflict["soma_ids"])
        status = html.escape(str(conflict["status"]))
        cards.append(
            f"""
            <section class="card {status}" id="conflict-{component_id}">
              <h2>Konflikt {component_id} · {len(conflict['soma_ids'])} Zellen · {status}</h2>
              <p>Soma-IDs: {html.escape(soma_ids)} · Ausschnitt {y1-y0}×{x1-x0}</p>
              <div class="panels">
                <figure><img src="assets/{raw_name}"><figcaption>Rohbild</figcaption></figure>
                <figure><img src="assets/{semantic_name}"><figcaption>Netz 141 + Hysterese</figcaption></figure>
                <figure><img src="assets/{instances_name}"><figcaption>Gelernte Trennung (Farbe = Zell-ID)</figcaption></figure>
                <figure><img src="assets/{confidence_name}"><figcaption>Zuordnungssicherheit (hell = sicher)</figcaption></figure>
              </div>
            </section>
            """
        )
    document = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Gelernte Zelltrennung</title>
<style>
body{{margin:0;background:#10161d;color:#e8eef5;font:15px system-ui,sans-serif}}
main{{max-width:1800px;margin:auto;padding:24px}} .card{{background:#17212b;margin:0 0 22px;padding:16px;border-radius:10px}}
.panels{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}} figure{{margin:0}}
.accepted_separation{{border-left:6px solid #23d5ab}} .rejected_uncertain{{border-left:6px solid #ffb341}}
.unresolved_too_large{{border-left:6px solid #ff637f}}
img{{width:100%;height:360px;object-fit:contain;background:#05080b}} figcaption{{padding:7px;color:#b9c8d8}}
@media(max-width:1100px){{.panels{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><h1>Netz 141 + gelernte Zelltrennung</h1>
<p>Gezeigt werden nur zusammenhängende Gruppen mit mindestens zwei Soma-Instanzen. Die Zell-ID ist die Farbe.</p>
{''.join(cards) if cards else '<p>Keine Mehrzell-Konflikte gefunden.</p>'}
</main></body></html>"""
    gallery = output_dir / "index.html"
    gallery.write_text(document, encoding="utf-8")
    return gallery


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    raw_path = args.raw.resolve()
    semantic_path = args.semantic.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    for required in (raw_path, semantic_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    raw = read_2d(raw_path)
    semantic = read_2d(semantic_path).astype(np.uint8)
    if raw.shape != semantic.shape:
        raise ValueError(f"Shape mismatch: raw={raw.shape}, semantic={semantic.shape}")
    invalid = set(int(v) for v in np.unique(semantic)) - {0, 1, 2}
    if invalid:
        raise ValueError(f"Semantic labels contain invalid values: {sorted(invalid)}")

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    base_channels = int(config.get("base_channels", 16))
    model = SomaSeededSeparationUNet(base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Checkpoint version {checkpoint.get('model_version')!r} does not match {MODEL_VERSION!r}"
        )

    soma_instances, soma_report, _soma_info, reference_area = split_soma_mask(
        semantic == 2, GraphSplitSettings()
    )
    material, material_count = ndi.label(semantic > 0, structure=CONNECTIVITY_8)
    objects = ndi.find_objects(material, max_label=material_count)
    primary = np.zeros(semantic.shape, dtype=np.uint16)
    secondary = np.zeros_like(primary)
    confidence = np.zeros(semantic.shape, dtype=np.float32)
    conflicts: list[dict[str, object]] = []
    direct_components = unresolved_components = 0

    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        component_mask = material[slices] == component_id
        local_soma_values = soma_instances[slices][component_mask]
        soma_ids = sorted(int(v) for v in np.unique(local_soma_values) if v > 0)
        if not soma_ids:
            unresolved_components += 1
            continue
        if len(soma_ids) == 1:
            view = primary[slices]
            view[component_mask] = soma_ids[0]
            confidence_view = confidence[slices]
            confidence_view[component_mask] = 1.0
            direct_components += 1
            continue

        y0, y1, x0, x1 = padded_bounds(slices, semantic.shape, args.crop_margin)
        conflict: dict[str, object] = {
            "component_id": component_id,
            "soma_ids": soma_ids,
            "crop_yxyx": [y0, y1, x0, x1],
            "component_pixels": int(np.count_nonzero(component_mask)),
        }
        if max(y1 - y0, x1 - x0) > args.max_conflict_side:
            conflict["status"] = "unresolved_too_large"
            unresolved_components += 1
            conflicts.append(conflict)
            continue

        group_mask = material[y0:y1, x0:x1] == component_id
        local_semantic = np.where(group_mask, semantic[y0:y1, x0:x1], 0).astype(
            np.uint8
        )
        local_soma = np.where(
            np.isin(soma_instances[y0:y1, x0:x1], soma_ids),
            soma_instances[y0:y1, x0:x1],
            0,
        ).astype(np.uint16)
        local_raw = raw[y0:y1, x0:x1]
        original_shape = local_semantic.shape
        padded_raw, padded_semantic, padded_soma = pad_to_multiple(
            local_raw, local_semantic, local_soma
        )
        local_ids, membership_probabilities = predict_memberships(
            model,
            padded_raw,
            padded_semantic,
            padded_soma,
            device,
        )
        local_primary, local_secondary, local_confidence = assign_memberships_to_instances(
            membership_probabilities,
            local_ids,
            padded_semantic,
            padded_soma,
            membership_threshold=args.membership_threshold,
        )
        local_primary = local_primary[: original_shape[0], : original_shape[1]]
        local_secondary = local_secondary[: original_shape[0], : original_shape[1]]
        local_confidence = local_confidence[: original_shape[0], : original_shape[1]]
        target_primary = primary[y0:y1, x0:x1]
        target_secondary = secondary[y0:y1, x0:x1]
        target_confidence = confidence[y0:y1, x0:x1]
        target_primary[group_mask] = local_primary[group_mask]
        target_secondary[group_mask] = local_secondary[group_mask]
        target_confidence[group_mask] = local_confidence[group_mask]
        conflict["status"] = "separated"
        conflict["mean_confidence"] = float(local_confidence[group_mask].mean())
        conflict["shared_pixels"] = int(np.count_nonzero(local_secondary[group_mask]))
        conflicts.append(conflict)

    tifffile.imwrite(output_dir / "instance_primary.tif", primary)
    tifffile.imwrite(output_dir / "instance_secondary.tif", secondary)
    tifffile.imwrite(output_dir / "assignment_confidence.tif", confidence)
    tifffile.imwrite(output_dir / "soma_instances.tif", soma_instances.astype(np.uint16))
    save_png(output_dir / "overview_instances.png", overlay_instances(raw, primary, secondary))
    cells, rejected_cells = export_cells(
        raw=raw,
        semantic=semantic,
        primary=primary,
        secondary=secondary,
        soma_instances=soma_instances,
        confidence=confidence,
        conflicts=conflicts,
        output_dir=output_dir / "cells",
        margin=args.crop_margin,
        minimum_size=args.min_crop_size,
        min_skeleton_pixels=args.min_skeleton_pixels,
    )
    (output_dir / "rejected_cells.json").write_text(
        json.dumps(rejected_cells, indent=2), encoding="utf-8"
    )
    gallery = write_conflict_gallery(
        output_dir, raw, semantic, primary, secondary, confidence, conflicts
    )
    summary = {
        "model_version": MODEL_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "raw": str(raw_path),
        "semantic": str(semantic_path),
        "shape": list(raw.shape),
        "material_components": int(material_count),
        "soma_instances": int(soma_instances.max()),
        "soma_reference_area": float(reference_area),
        "direct_single_soma_components": direct_components,
        "conflict_components": len(conflicts),
        "separated_conflict_components": sum(
            row["status"] in {"accepted_separation", "rejected_uncertain"}
            for row in conflicts
        ),
        "accepted_conflict_components": sum(
            row["status"] == "accepted_separation" for row in conflicts
        ),
        "rejected_uncertain_conflict_components": sum(
            row["status"] == "rejected_uncertain" for row in conflicts
        ),
        "unresolved_components": unresolved_components,
        "exported_cells": len(cells),
        "rejected_cells": len(rejected_cells),
        "conflicts": conflicts,
        "soma_split_report": soma_report,
        "cells": cells,
        "gallery": str(gallery),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "material_components",
        "soma_instances",
        "conflict_components",
        "separated_conflict_components",
        "accepted_conflict_components",
        "rejected_uncertain_conflict_components",
        "unresolved_components",
        "exported_cells",
        "rejected_cells",
        "gallery",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
