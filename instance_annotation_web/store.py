from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
from PIL import Image, ImageDraw


PALETTE = (
    "#69e3b2", "#ff9f68", "#6ea8fe", "#e98cff", "#ffd166", "#67d5e8",
    "#ff6f91", "#a8df65", "#b09cff", "#f49ac2", "#80cbc4", "#ffb347",
    "#7bdff2", "#f2b5d4", "#b8f2e6", "#ffa69e", "#cdb4db", "#90dbf4",
    "#f1c0e8", "#98f5e1", "#f6bd60", "#84a59d", "#f28482", "#a9def9",
)
ALLOWED_SEMANTIC_VALUES = {0, 1, 2}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.04 * (attempt + 1))
    temporary.unlink(missing_ok=True)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    name: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    role: str

    def public(self) -> dict[str, object]:
        return {
            "id": self.case_id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "role": self.role,
        }


def discover_cases(dataset_root: Path) -> list[CaseRecord]:
    roles: dict[str, str] = {}
    split_path = dataset_root / "splits_final_overview.json"
    if split_path.is_file():
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        if splits:
            roles.update({str(case_id): "training" for case_id in splits[0].get("train", [])})
            roles.update({str(case_id): "validation" for case_id in splits[0].get("val", [])})

    records: list[CaseRecord] = []
    for image_path in sorted((dataset_root / "imagesTr").glob("*_0000.tif")):
        case_id = image_path.stem.removesuffix("_0000")
        label_path = dataset_root / "labelsTr" / f"{case_id}.tif"
        if not label_path.is_file():
            continue
        with tifffile.TiffFile(image_path) as image_file:
            shape = tuple(int(value) for value in image_file.series[0].shape[-2:])
        records.append(
            CaseRecord(
                case_id=case_id,
                name=case_id.replace("mats_overview_", "").replace("_", " "),
                image_path=image_path,
                label_path=label_path,
                height=shape[0],
                width=shape[1],
                role=roles.get(case_id, "unassigned"),
            )
        )
    return records


class AnnotationStore:
    def __init__(self, dataset_root: Path, output_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.output_root = output_root.resolve()
        self.state_root = self.output_root / "working"
        self.export_root = self.output_root / "exports"
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)
        records = discover_cases(self.dataset_root)
        self.records = {record.case_id: record for record in records}
        self._locks = {case_id: threading.RLock() for case_id in self.records}
        self._image_cache: dict[str, np.ndarray] = {}
        self._label_cache: dict[str, np.ndarray] = {}
        for record in records:
            self._ensure_state(record)

    def cases(self) -> list[dict[str, object]]:
        return [self.case_details(case_id, include_cells=False) for case_id in self.records]

    def _record(self, case_id: str) -> CaseRecord:
        try:
            return self.records[case_id]
        except KeyError as error:
            raise KeyError(f"Unknown case: {case_id}") from error

    def _case_root(self, case_id: str) -> Path:
        return self.state_root / case_id

    def _metadata_path(self, case_id: str) -> Path:
        return self._case_root(case_id) / "state.json"

    def _load_metadata(self, case_id: str) -> dict[str, Any]:
        return json.loads(self._metadata_path(case_id).read_text(encoding="utf-8"))

    def _save_metadata(self, case_id: str, metadata: dict[str, Any]) -> None:
        metadata["updated_at"] = utc_now()
        atomic_json(self._metadata_path(case_id), metadata)

    @staticmethod
    def _new_memmap(path: Path, dtype: np.dtype[Any], shape: tuple[int, int]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npy")
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
        array[:] = 0
        array.flush()
        del array
        os.replace(temporary, path)

    def _ensure_state(self, record: CaseRecord) -> None:
        root = self._case_root(record.case_id)
        root.mkdir(parents=True, exist_ok=True)
        shape = (record.height, record.width)
        for name, dtype in (
            ("instance_primary.npy", np.uint16),
            ("instance_secondary.npy", np.uint16),
            ("ambiguous.npy", np.uint8),
        ):
            path = root / name
            if not path.is_file():
                self._new_memmap(path, dtype, shape)
        metadata_path = self._metadata_path(record.case_id)
        if not metadata_path.is_file():
            raw = self._read_image(record.case_id)
            sample = np.asarray(raw[::16, ::16], dtype=np.float32)
            finite = sample[np.isfinite(sample)]
            low, high = np.percentile(finite, (1.0, 99.7)) if finite.size else (0.0, 1.0)
            if high <= low:
                high = low + 1.0
            atomic_json(
                metadata_path,
                {
                    "case_id": record.case_id,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "revision": 0,
                    "next_cell_id": 1,
                    "completed": False,
                    "normalization": {"low": float(low), "high": float(high)},
                    "cells": {},
                    "last_export": None,
                },
            )

    def _maps(self, case_id: str) -> tuple[np.memmap, np.memmap, np.memmap]:
        root = self._case_root(case_id)
        return (
            np.load(root / "instance_primary.npy", mmap_mode="r+"),
            np.load(root / "instance_secondary.npy", mmap_mode="r+"),
            np.load(root / "ambiguous.npy", mmap_mode="r+"),
        )

    def _read_image(self, case_id: str) -> np.ndarray:
        if case_id not in self._image_cache:
            image = np.asarray(tifffile.imread(self._record(case_id).image_path))
            if image.ndim > 2:
                image = np.squeeze(image)
            if image.ndim != 2:
                raise ValueError(f"Raw image for {case_id} is not 2-D: {image.shape}")
            self._image_cache[case_id] = image
        return self._image_cache[case_id]

    def _read_label(self, case_id: str) -> np.ndarray:
        if case_id not in self._label_cache:
            label = np.asarray(tifffile.imread(self._record(case_id).label_path))
            if label.ndim > 2:
                label = np.squeeze(label)
            values = {int(value) for value in np.unique(label)}
            if label.ndim != 2 or not values <= ALLOWED_SEMANTIC_VALUES:
                raise ValueError(f"Invalid semantic label for {case_id}: shape={label.shape}, values={values}")
            self._label_cache[case_id] = label.astype(np.uint8, copy=False)
        return self._label_cache[case_id]

    def _bump(self, case_id: str, metadata: dict[str, Any], action: str) -> int:
        metadata["revision"] = int(metadata.get("revision", 0)) + 1
        metadata["last_action"] = action
        self._save_metadata(case_id, metadata)
        return int(metadata["revision"])

    def create_cell(self, case_id: str, label: str | None = None) -> dict[str, object]:
        with self._locks[case_id]:
            metadata = self._load_metadata(case_id)
            cell_id = int(metadata["next_cell_id"])
            if cell_id > np.iinfo(np.uint16).max:
                raise ValueError("Maximum number of cells reached")
            metadata["next_cell_id"] = cell_id + 1
            metadata["cells"][str(cell_id)] = {
                "id": cell_id,
                "label": (label or f"Zelle {cell_id}").strip()[:80],
                "color": PALETTE[(cell_id - 1) % len(PALETTE)],
                "created_at": utc_now(),
            }
            self._bump(case_id, metadata, f"create_cell:{cell_id}")
            return self.case_details(case_id)

    def rename_cell(self, case_id: str, cell_id: int, label: str) -> dict[str, object]:
        with self._locks[case_id]:
            metadata = self._load_metadata(case_id)
            cell = metadata["cells"].get(str(cell_id))
            if not cell:
                raise KeyError(f"Unknown cell: {cell_id}")
            cell["label"] = label.strip()[:80] or f"Zelle {cell_id}"
            self._bump(case_id, metadata, f"rename_cell:{cell_id}")
            return self.case_details(case_id)

    def delete_cell(self, case_id: str, cell_id: int) -> dict[str, object]:
        with self._locks[case_id]:
            metadata = self._load_metadata(case_id)
            if str(cell_id) not in metadata["cells"]:
                raise KeyError(f"Unknown cell: {cell_id}")
            primary, secondary, _ambiguous = self._maps(case_id)
            primary_mask = primary == cell_id
            secondary_mask = secondary == cell_id
            primary[primary_mask & (secondary > 0)] = secondary[primary_mask & (secondary > 0)]
            primary[primary_mask & (secondary == 0)] = 0
            secondary[primary_mask | secondary_mask] = 0
            primary.flush()
            secondary.flush()
            del metadata["cells"][str(cell_id)]
            self._bump(case_id, metadata, f"delete_cell:{cell_id}")
            return self.case_details(case_id)

    def case_details(self, case_id: str, include_cells: bool = True) -> dict[str, object]:
        record = self._record(case_id)
        with self._locks[case_id]:
            metadata = self._load_metadata(case_id)
            label = self._read_label(case_id)
            primary, secondary, ambiguous = self._maps(case_id)
            foreground = label > 0
            assigned = (primary > 0) | (secondary > 0)
            skeleton_total = int(np.count_nonzero(label == 1))
            soma_total = int(np.count_nonzero(label == 2))
            skeleton_assigned = int(np.count_nonzero((label == 1) & assigned))
            soma_assigned = int(np.count_nonzero((label == 2) & assigned))
            summary: dict[str, object] = {
                **record.public(),
                "revision": int(metadata["revision"]),
                "completed": bool(metadata.get("completed", False)),
                "cells_count": len(metadata["cells"]),
                "grid": {
                    "tile_size": 512,
                    "columns": int(np.ceil(record.width / 512)),
                    "rows": int(np.ceil(record.height / 512)),
                },
                "coverage": {
                    "skeleton_total": skeleton_total,
                    "skeleton_assigned": skeleton_assigned,
                    "skeleton_percent": round(100.0 * skeleton_assigned / max(1, skeleton_total), 1),
                    "soma_total": soma_total,
                    "soma_assigned": soma_assigned,
                    "soma_percent": round(100.0 * soma_assigned / max(1, soma_total), 1),
                    "unassigned_foreground": int(np.count_nonzero(foreground & ~assigned & (ambiguous == 0))),
                    "ambiguous": int(np.count_nonzero(ambiguous > 0)),
                    "overlap": int(np.count_nonzero(secondary > 0)),
                },
                "last_export": metadata.get("last_export"),
            }
            if include_cells:
                max_id = max([0, *(int(key) for key in metadata["cells"])])
                skeleton_counts = self._counts_by_id(primary, secondary, label == 1, max_id)
                soma_counts = self._counts_by_id(primary, secondary, label == 2, max_id)
                cells = []
                for key, cell in sorted(metadata["cells"].items(), key=lambda pair: int(pair[0])):
                    cell_id = int(key)
                    cells.append(
                        {
                            **cell,
                            "skeleton_pixels": int(skeleton_counts[cell_id]),
                            "soma_pixels": int(soma_counts[cell_id]),
                            "valid": bool(skeleton_counts[cell_id] and soma_counts[cell_id]),
                        }
                    )
                summary["cells"] = cells
                tiles = []
                for tile_row, y0 in enumerate(range(0, record.height, 512)):
                    for tile_column, x0 in enumerate(range(0, record.width, 512)):
                        y1, x1 = min(record.height, y0 + 512), min(record.width, x0 + 512)
                        tile_foreground = foreground[y0:y1, x0:x1]
                        tile_assigned = assigned[y0:y1, x0:x1]
                        tile_ambiguous = ambiguous[y0:y1, x0:x1] > 0
                        tiles.append(
                            {
                                "row": tile_row,
                                "column": tile_column,
                                "foreground": int(np.count_nonzero(tile_foreground)),
                                "unassigned": int(
                                    np.count_nonzero(
                                        tile_foreground & ~tile_assigned & ~tile_ambiguous
                                    )
                                ),
                                "ambiguous": int(np.count_nonzero(tile_ambiguous)),
                            }
                        )
                summary["tiles"] = tiles
            return summary

    @staticmethod
    def _counts_by_id(
        primary: np.ndarray,
        secondary: np.ndarray,
        class_mask: np.ndarray,
        max_id: int,
    ) -> np.ndarray:
        if max_id == 0:
            return np.zeros(1, dtype=np.int64)
        values = np.concatenate((primary[class_mask], secondary[class_mask]))
        values = values[values > 0]
        counts = np.bincount(values.astype(np.int64), minlength=max_id + 1)
        if counts.size < max_id + 1:
            counts = np.pad(counts, (0, max_id + 1 - counts.size))
        return counts

    @staticmethod
    def _stroke_region(
        points: Iterable[tuple[float, float]], radius: int, shape: tuple[int, int]
    ) -> tuple[int, int, int, int, np.ndarray]:
        clean = [(float(x), float(y)) for x, y in points]
        if not clean:
            raise ValueError("A stroke needs at least one point")
        radius = max(0, min(int(radius), 40))
        samples: list[tuple[int, int]] = []
        previous = clean[0]
        for current in clean:
            steps = max(1, int(max(abs(current[0] - previous[0]), abs(current[1] - previous[1]))) + 1)
            xs = np.linspace(previous[0], current[0], steps)
            ys = np.linspace(previous[1], current[1], steps)
            samples.extend((int(round(x)), int(round(y))) for x, y in zip(xs, ys))
            previous = current
        height, width = shape
        x0 = max(0, min(x for x, _y in samples) - radius)
        x1 = min(width, max(x for x, _y in samples) + radius + 1)
        y0 = max(0, min(y for _x, y in samples) - radius)
        y1 = min(height, max(y for _x, y in samples) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Stroke is outside the image")
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        for x, y in samples:
            if 0 <= x < width and 0 <= y < height:
                mask |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        return y0, y1, x0, x1, mask

    def apply_stroke(
        self,
        case_id: str,
        *,
        points: Iterable[tuple[float, float]],
        radius: int,
        operation: str,
        target: str,
        cell_id: int | None,
    ) -> dict[str, object]:
        record = self._record(case_id)
        y0, y1, x0, x1, mask = self._stroke_region(
            points, radius, (record.height, record.width)
        )
        with self._locks[case_id]:
            label = self._read_label(case_id)[y0:y1, x0:x1]
            mask &= self._target_mask(label, target)
            result = self._apply_region(case_id, y0, y1, x0, x1, mask, operation, cell_id)
            return {"change": result, "case": self.case_details(case_id)}

    @staticmethod
    def _target_mask(label: np.ndarray, target: str) -> np.ndarray:
        if target == "skeleton":
            return label == 1
        if target == "soma":
            return label == 2
        if target == "foreground":
            return label > 0
        raise ValueError(f"Unknown target: {target}")

    def apply_fill(
        self,
        case_id: str,
        *,
        x: int,
        y: int,
        operation: str,
        target: str,
        cell_id: int | None,
    ) -> dict[str, object]:
        record = self._record(case_id)
        if not (0 <= x < record.width and 0 <= y < record.height):
            raise ValueError("Point is outside the image")
        label = self._read_label(case_id)
        if target == "auto":
            value = int(label[y, x])
            if value == 1:
                target = "skeleton"
            elif value == 2:
                target = "soma"
            else:
                raise ValueError("Clicked pixel is background")
        target_mask = self._target_mask(label, target)
        if not target_mask[y, x]:
            raise ValueError(f"Clicked pixel is not {target}")
        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited = {(y, x)}
        points: list[tuple[int, int]] = []
        while queue:
            row, column = queue.popleft()
            points.append((row, column))
            if len(points) > 500_000:
                raise ValueError("Connected component is unexpectedly large")
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nr, nc = row + dy, column + dx
                    key = (nr, nc)
                    if (
                        0 <= nr < record.height
                        and 0 <= nc < record.width
                        and key not in visited
                        and target_mask[nr, nc]
                    ):
                        visited.add(key)
                        queue.append(key)
        rows = [point[0] for point in points]
        columns = [point[1] for point in points]
        y0, y1 = min(rows), max(rows) + 1
        x0, x1 = min(columns), max(columns) + 1
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        mask[np.asarray(rows) - y0, np.asarray(columns) - x0] = True
        with self._locks[case_id]:
            result = self._apply_region(case_id, y0, y1, x0, x1, mask, operation, cell_id)
            return {"change": result, "case": self.case_details(case_id)}

    def _apply_region(
        self,
        case_id: str,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        mask: np.ndarray,
        operation: str,
        cell_id: int | None,
    ) -> dict[str, int]:
        metadata = self._load_metadata(case_id)
        if operation in {"assign", "overlap", "erase"} and cell_id is not None:
            if str(cell_id) not in metadata["cells"]:
                raise KeyError(f"Unknown cell: {cell_id}")
        if operation in {"assign", "overlap"} and cell_id is None:
            raise ValueError("Select a cell first")
        primary, secondary, ambiguous = self._maps(case_id)
        p = primary[y0:y1, x0:x1]
        s = secondary[y0:y1, x0:x1]
        a = ambiguous[y0:y1, x0:x1]
        requested = int(np.count_nonzero(mask))
        changed = 0
        skipped = 0
        if operation == "assign":
            already = mask & ((p == cell_id) | (s == cell_id))
            available = mask & (p == 0)
            p[available] = cell_id
            changed = int(np.count_nonzero(available))
            skipped = requested - changed - int(np.count_nonzero(already))
        elif operation == "overlap":
            already = mask & ((p == cell_id) | (s == cell_id))
            empty_primary = mask & (p == 0)
            p[empty_primary] = cell_id
            free_secondary = mask & (p != 0) & (p != cell_id) & (s == 0)
            s[free_secondary] = cell_id
            overflow = mask & (p != 0) & (p != cell_id) & (s != 0) & (s != cell_id)
            a[overflow] = 1
            changed = int(np.count_nonzero(empty_primary | free_secondary | overflow))
            skipped = requested - changed - int(np.count_nonzero(already))
        elif operation == "erase":
            if cell_id is None:
                changed_mask = mask & ((p > 0) | (s > 0))
                p[mask] = 0
                s[mask] = 0
            else:
                primary_match = mask & (p == cell_id)
                secondary_match = mask & (s == cell_id)
                promote = primary_match & (s > 0)
                p[promote] = s[promote]
                p[primary_match & ~promote] = 0
                s[primary_match | secondary_match] = 0
                changed_mask = primary_match | secondary_match
            changed = int(np.count_nonzero(changed_mask))
        elif operation == "ambiguous":
            changed_mask = mask & (a == 0)
            a[mask] = 1
            changed = int(np.count_nonzero(changed_mask))
        elif operation == "clear_ambiguous":
            changed_mask = mask & (a > 0)
            a[mask] = 0
            changed = int(np.count_nonzero(changed_mask))
        else:
            raise ValueError(f"Unknown operation: {operation}")
        primary.flush()
        secondary.flush()
        ambiguous.flush()
        revision = self._bump(case_id, metadata, f"{operation}:{cell_id or 0}")
        return {"requested": requested, "changed": changed, "skipped": max(0, skipped), "revision": revision}

    def set_completed(self, case_id: str, completed: bool) -> dict[str, object]:
        with self._locks[case_id]:
            metadata = self._load_metadata(case_id)
            metadata["completed"] = bool(completed)
            self._bump(case_id, metadata, f"completed:{bool(completed)}")
            return self.case_details(case_id)

    def render_tile(
        self,
        case_id: str,
        *,
        x: int,
        y: int,
        size: int = 512,
        view: str = "combined",
        selected_cell: int | None = None,
    ) -> bytes:
        record = self._record(case_id)
        size = max(64, min(int(size), 1024))
        x = max(0, min(int(x), max(0, record.width - 1)))
        y = max(0, min(int(y), max(0, record.height - 1)))
        x1, y1 = min(record.width, x + size), min(record.height, y + size)
        raw = self._read_image(case_id)[y:y1, x:x1]
        label = self._read_label(case_id)[y:y1, x:x1]
        metadata = self._load_metadata(case_id)
        low = float(metadata["normalization"]["low"])
        high = float(metadata["normalization"]["high"])
        gray = np.clip((raw.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        primary, secondary, ambiguous = self._maps(case_id)
        p = np.asarray(primary[y:y1, x:x1])
        s = np.asarray(secondary[y:y1, x:x1])
        a = np.asarray(ambiguous[y:y1, x:x1])
        if view in {"semantic", "combined", "instances"}:
            if view == "instances":
                rgb = (rgb.astype(np.float32) * 0.28).astype(np.uint8)
            unassigned = (p == 0) & (s == 0)
            self._blend(rgb, (label == 1) & unassigned, (0, 235, 245), 0.68)
            self._blend(rgb, (label == 2) & unassigned, (242, 66, 166), 0.68)
        if view in {"combined", "instances"}:
            cells = metadata.get("cells", {})
            for key, cell in cells.items():
                cell_id = int(key)
                color = _hex_rgb(str(cell["color"]))
                alpha = 0.90 if selected_cell == cell_id else 0.72
                self._blend(rgb, p == cell_id, color, alpha)
                second_mask = s == cell_id
                if np.any(second_mask):
                    self._blend(rgb, second_mask, color, 0.82)
            overlap = s > 0
            if np.any(overlap):
                checker = (np.indices(overlap.shape).sum(axis=0) % 4) < 2
                self._blend(rgb, overlap & checker, (255, 255, 255), 0.40)
            self._blend(rgb, a > 0, (255, 194, 71), 0.90)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[: rgb.shape[0], : rgb.shape[1]] = rgb
        stream = BytesIO()
        Image.fromarray(canvas).save(stream, format="PNG", compress_level=2)
        return stream.getvalue()

    @staticmethod
    def _blend(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
        if not np.any(mask):
            return
        image[mask] = (
            image[mask].astype(np.float32) * (1.0 - alpha)
            + np.asarray(color, dtype=np.float32) * alpha
        ).astype(np.uint8)

    def render_thumbnail(self, case_id: str, max_side: int = 360) -> bytes:
        record = self._record(case_id)
        cache_path = self._case_root(case_id) / "thumbnail.png"
        if cache_path.is_file():
            return cache_path.read_bytes()
        raw = self._read_image(case_id)
        label = self._read_label(case_id)
        metadata = self._load_metadata(case_id)
        low = float(metadata["normalization"]["low"])
        high = float(metadata["normalization"]["high"])
        gray = np.clip((raw.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        self._blend(rgb, label == 1, (0, 235, 245), 0.62)
        self._blend(rgb, label == 2, (242, 66, 166), 0.62)
        image = Image.fromarray(rgb)
        scale = min(1.0, max_side / max(record.width, record.height))
        image = image.resize(
            (max(1, round(record.width * scale)), max(1, round(record.height * scale))),
            Image.Resampling.LANCZOS,
        )
        draw = ImageDraw.Draw(image)
        for column in range(1, int(np.ceil(record.width / 512))):
            x = round(column * 512 * scale)
            draw.line((x, 0, x, image.height), fill=(105, 227, 178), width=1)
        for row in range(1, int(np.ceil(record.height / 512))):
            y = round(row * 512 * scale)
            draw.line((0, y, image.width, y), fill=(105, 227, 178), width=1)
        stream = BytesIO()
        image.save(stream, format="PNG", compress_level=3)
        cache_path.write_bytes(stream.getvalue())
        return stream.getvalue()

    def validate(self, case_id: str) -> dict[str, object]:
        details = self.case_details(case_id)
        cells = list(details.get("cells", []))
        invalid = [cell for cell in cells if not cell["valid"]]
        coverage = dict(details["coverage"])
        warnings = []
        if coverage["unassigned_foreground"]:
            warnings.append(f"{coverage['unassigned_foreground']:,} Vordergrundpixel sind nicht zugeordnet.")
        if invalid:
            warnings.append(f"{len(invalid)} Zellen haben noch kein Soma oder kein Skeleton.")
        if coverage["ambiguous"]:
            warnings.append(f"{coverage['ambiguous']:,} Pixel sind als unsicher markiert und werden ignoriert.")
        return {
            "case_id": case_id,
            "ready": bool(cells) and not invalid and coverage["unassigned_foreground"] == 0,
            "valid_cells": sum(bool(cell["valid"]) for cell in cells),
            "invalid_cells": invalid,
            "warnings": warnings,
            "coverage": coverage,
        }

    def export(self, case_id: str) -> dict[str, object]:
        record = self._record(case_id)
        with self._locks[case_id]:
            metadata = self._load_metadata(case_id)
            revision = int(metadata["revision"])
            destination_root = self.export_root / case_id
            source_root = destination_root / "source"
            source_root.mkdir(parents=True, exist_ok=True)
            raw_destination = source_root / "raw.tif"
            semantic_destination = source_root / "semantic.tif"
            if not raw_destination.is_file():
                shutil.copy2(record.image_path, raw_destination)
            if not semantic_destination.is_file():
                shutil.copy2(record.label_path, semantic_destination)

            revision_name = f"rev_{revision:06d}"
            final_root = destination_root / "revisions" / revision_name
            if final_root.is_dir():
                return json.loads((final_root / "export_manifest.json").read_text(encoding="utf-8"))
            staging = final_root.with_name(f".{revision_name}.{uuid.uuid4().hex}.building")
            staging.mkdir(parents=True)
            try:
                primary, secondary, ambiguous = self._maps(case_id)
                p = np.asarray(primary)
                s = np.asarray(secondary)
                a = np.asarray(ambiguous)
                tifffile.imwrite(staging / "instance_primary.tif", p, compression="zlib")
                tifffile.imwrite(staging / "instance_secondary.tif", s, compression="zlib")
                tifffile.imwrite(staging / "ambiguous.tif", a, compression="zlib")
                cells_root = staging / "cells"
                cells_root.mkdir()
                positions = self._positions_by_cell(p, s, a)
                raw = self._read_image(case_id)
                semantic = self._read_label(case_id)
                exported_cells: list[dict[str, object]] = []
                for key, cell in sorted(metadata["cells"].items(), key=lambda item: int(item[0])):
                    cell_id = int(key)
                    flat = positions.get(cell_id, np.empty(0, dtype=np.int64))
                    if flat.size == 0:
                        exported_cells.append({**cell, "valid_for_training": False, "reason": "no_pixels"})
                        continue
                    rows, columns = np.unravel_index(flat, semantic.shape)
                    margin = 16
                    y0, y1 = max(0, int(rows.min()) - margin), min(record.height, int(rows.max()) + margin + 1)
                    x0, x1 = max(0, int(columns.min()) - margin), min(record.width, int(columns.max()) + margin + 1)
                    owned = ((p[y0:y1, x0:x1] == cell_id) | (s[y0:y1, x0:x1] == cell_id)) & (a[y0:y1, x0:x1] == 0)
                    label_crop = np.where(owned, semantic[y0:y1, x0:x1], 0).astype(np.uint8)
                    skeleton_pixels = int(np.count_nonzero(label_crop == 1))
                    soma_pixels = int(np.count_nonzero(label_crop == 2))
                    valid = bool(skeleton_pixels and soma_pixels)
                    cell_root = cells_root / f"cell{cell_id:04d}"
                    cell_root.mkdir()
                    tifffile.imwrite(cell_root / "raw.tif", raw[y0:y1, x0:x1])
                    tifffile.imwrite(cell_root / "label.tif", label_crop, compression="zlib")
                    tifffile.imwrite(cell_root / "skeleton.tif", (label_crop == 1).astype(np.uint8), compression="zlib")
                    tifffile.imwrite(cell_root / "soma.tif", (label_crop == 2).astype(np.uint8), compression="zlib")
                    tifffile.imwrite(cell_root / "ignore.tif", a[y0:y1, x0:x1].astype(np.uint8), compression="zlib")
                    overlap = ((p[y0:y1, x0:x1] == cell_id) & (s[y0:y1, x0:x1] > 0)) | (s[y0:y1, x0:x1] == cell_id)
                    tifffile.imwrite(cell_root / "overlap.tif", overlap.astype(np.uint8), compression="zlib")
                    cell_metadata = {
                        **cell,
                        "source_case": case_id,
                        "bounds_xyxy": [x0, y0, x1, y1],
                        "skeleton_pixels": skeleton_pixels,
                        "soma_pixels": soma_pixels,
                        "valid_for_training": valid,
                    }
                    atomic_json(cell_root / "metadata.json", cell_metadata)
                    exported_cells.append(cell_metadata)
                validation = self.validate(case_id)
                manifest = {
                    "format": "evo-instance-annotations-v1",
                    "case_id": case_id,
                    "source_role": record.role,
                    "revision": revision,
                    "exported_at": utc_now(),
                    "raw": str(record.image_path),
                    "semantic": str(record.label_path),
                    "primary_instance_file": "instance_primary.tif",
                    "secondary_instance_file": "instance_secondary.tif",
                    "ambiguous_file": "ambiguous.tif",
                    "cells": exported_cells,
                    "valid_cells": sum(bool(cell.get("valid_for_training")) for cell in exported_cells),
                    "validation": validation,
                }
                atomic_json(staging / "export_manifest.json", manifest)
                final_root.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(final_root)
                metadata["last_export"] = {
                    "revision": revision,
                    "path": str(final_root),
                    "exported_at": manifest["exported_at"],
                    "valid_cells": manifest["valid_cells"],
                }
                self._save_metadata(case_id, metadata)
                atomic_json(destination_root / "latest.json", metadata["last_export"])
                return manifest
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise

    @staticmethod
    def _positions_by_cell(
        primary: np.ndarray, secondary: np.ndarray, ambiguous: np.ndarray
    ) -> dict[int, np.ndarray]:
        positions: dict[int, list[np.ndarray]] = {}
        valid = ambiguous.reshape(-1) == 0
        for values in (primary.reshape(-1), secondary.reshape(-1)):
            flat = np.flatnonzero((values > 0) & valid)
            if flat.size == 0:
                continue
            ids = values[flat].astype(np.int64)
            order = np.argsort(ids, kind="stable")
            flat, ids = flat[order], ids[order]
            starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
            ends = np.r_[starts[1:], ids.size]
            for start, end in zip(starts, ends):
                positions.setdefault(int(ids[start]), []).append(flat[start:end])
        return {
            cell_id: np.unique(np.concatenate(chunks))
            for cell_id, chunks in positions.items()
        }
