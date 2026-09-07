from __future__ import annotations

import csv
import io
import json
import pickle
import random
import threading
from functools import lru_cache
from pathlib import Path

import blosc2
import numpy as np
import tifffile
import torch
from PIL import Image
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSkeletonRecallCells import (
    nnUNetDataLoaderSkeletonRecall,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from custom_trainers.skeleton_recall.nnUNetTrainerSkeletonRecallCells import (
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug,
)


DATASET_NAME = "Dataset139_dataset138_plus_reviewed_cells"


def padding_fraction(height: int, width: int, patch_size: int) -> float:
    real_pixels = min(int(height), patch_size) * min(int(width), patch_size)
    return 1.0 - real_pixels / float(patch_size * patch_size)


def _array_stats(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _display_gray(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def render_overlay_png(
    image: np.ndarray,
    label: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> bytes:
    gray = _display_gray(np.squeeze(image))
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)

    if valid_mask is not None:
        padding = ~np.asarray(valid_mask, dtype=bool)
        if padding.any():
            rgb[padding] = 0.48 * rgb[padding] + 0.52 * np.array(
                [242.0, 157.0, 45.0], dtype=np.float32
            )

    if label is not None:
        labels = np.squeeze(np.asarray(label))
        soma = labels == 2
        skeleton = labels == 1
        if soma.any():
            rgb[soma] = 0.30 * rgb[soma] + 0.70 * np.array(
                [238.0, 55.0, 151.0], dtype=np.float32
            )
        if skeleton.any():
            rgb[skeleton] = 0.08 * rgb[skeleton] + 0.92 * np.array(
                [28.0, 224.0, 238.0], dtype=np.float32
            )

    buffer = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


class _DatasetWithValidityMask:
    """Expose artificial loader padding as a second segmentation channel."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.identifiers = dataset.identifiers

    def load_case(self, identifier):
        data, segmentation, previous_stage, properties = self.dataset.load_case(identifier)
        data = np.asarray(data)
        segmentation = np.asarray(segmentation)
        valid = np.ones_like(segmentation, dtype=segmentation.dtype)
        return data, np.concatenate((segmentation, valid), axis=0), previous_stage, properties


class TrainingDatasetInspector:
    def __init__(
        self,
        raw_dataset: Path,
        preprocessed_dataset: Path,
        *,
        dataset_id: int = 139,
        configuration: str = "2d",
    ) -> None:
        self.raw_dataset = Path(raw_dataset)
        self.preprocessed_dataset = Path(preprocessed_dataset)
        self.dataset_id = int(dataset_id)
        self.configuration = configuration
        self.dataset_json_path = self.raw_dataset / "dataset.json"
        self.manifest_path = self.raw_dataset / "build_manifest.csv"
        self.plans_path = self.preprocessed_dataset / "nnUNetPlans.json"
        self.splits_path = self.preprocessed_dataset / "splits_final.json"
        self._index_lock = threading.RLock()
        self._training_lock = threading.RLock()
        self._index: dict[str, object] | None = None
        self._index_signature: tuple[tuple[int, int], ...] | None = None
        self._records: dict[str, dict[str, object]] = {}

    @classmethod
    def from_project(cls, project_root: Path) -> "TrainingDatasetInspector":
        project_root = Path(project_root)
        return cls(
            project_root / "nnUNet_raw" / DATASET_NAME,
            project_root / "nnUNet_preprocessed" / DATASET_NAME,
        )

    def available(self) -> bool:
        return all(
            path.is_file()
            for path in (
                self.dataset_json_path,
                self.manifest_path,
                self.plans_path,
                self.splits_path,
            )
        )

    def _source_signature(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (path.stat().st_mtime_ns, path.stat().st_size)
            for path in (
                self.dataset_json_path,
                self.manifest_path,
                self.plans_path,
                self.splits_path,
            )
        )

    @staticmethod
    def _collection(origin: str, source_case: str) -> str:
        if origin == "dataset138":
            return "Dataset138"
        if source_case.startswith("rocki_m238"):
            return "ROCKi · CC M238"
        if source_case.startswith("rocki_m239"):
            return "ROCKi · CC M239"
        if source_case.startswith("rocki_mc"):
            return "ROCKi · MC"
        if "MCS24GFP" in source_case:
            return "MICA · MCS24GFP"
        return "Frühere Uploads"

    @staticmethod
    def _distribution(values: np.ndarray, edges: list[float], labels: list[str]) -> list[dict[str, object]]:
        counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
        return [
            {"label": label, "count": int(count)}
            for label, count in zip(labels, counts, strict=True)
        ]

    def _build_index(self) -> dict[str, object]:
        if not self.available():
            raise FileNotFoundError("Dataset139 oder dessen Preprocessing ist noch nicht vollständig vorhanden.")

        dataset_json = json.loads(self.dataset_json_path.read_text(encoding="utf-8"))
        plans = json.loads(self.plans_path.read_text(encoding="utf-8"))
        splits = json.loads(self.splits_path.read_text(encoding="utf-8"))
        train = set(splits[0]["train"])
        validation = set(splits[0]["val"])
        configuration = plans["configurations"][self.configuration]
        patch_height, patch_width = map(int, configuration["patch_size"])
        if patch_height != patch_width:
            raise ValueError("Der Dataset-Inspector erwartet aktuell quadratische 2D-Patches.")

        records: dict[str, dict[str, object]] = {}
        cases: list[dict[str, object]] = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for raw in csv.DictReader(stream):
                case_id = str(raw["output_case_id"])
                height = int(raw["height"])
                width = int(raw["width"])
                origin = str(raw["origin"])
                source_case = str(raw["source_case_id"])
                pad = padding_fraction(height, width, patch_height)
                split = "training" if case_id in train else "validation" if case_id in validation else "unused"
                public = {
                    "case_id": case_id,
                    "origin": origin,
                    "collection": self._collection(origin, source_case),
                    "source_case_id": source_case,
                    "split": split,
                    "height": height,
                    "width": width,
                    "area": height * width,
                    "max_side": max(height, width),
                    "padding_fraction": pad,
                    "padding_percent": 100.0 * pad,
                    "skeleton_pixels": int(raw["skeleton_pixels"]),
                    "soma_pixels": int(raw["soma_pixels"]),
                }
                cases.append(public)
                records[case_id] = {**raw, **public}

        cases.sort(key=lambda item: str(item["case_id"]))
        heights = np.asarray([case["height"] for case in cases], dtype=float)
        widths = np.asarray([case["width"] for case in cases], dtype=float)
        max_sides = np.maximum(heights, widths)
        padding = np.asarray([case["padding_fraction"] for case in cases], dtype=float)
        skeleton = np.asarray([case["skeleton_pixels"] for case in cases], dtype=np.int64)
        soma = np.asarray([case["soma_pixels"] for case in cases], dtype=np.int64)

        summary = {
            "total_cases": len(cases),
            "training_cases": sum(case["split"] == "training" for case in cases),
            "validation_cases": sum(case["split"] == "validation" for case in cases),
            "patch_size": [patch_height, patch_width],
            "height": {
                "min": int(heights.min()),
                "median": float(np.median(heights)),
                "max": int(heights.max()),
            },
            "width": {
                "min": int(widths.min()),
                "median": float(np.median(widths)),
                "max": int(widths.max()),
            },
            "padding": {
                "mean_percent": float(100.0 * padding.mean()),
                "median_percent": float(100.0 * np.median(padding)),
                "cases_with_padding": int((padding > 0).sum()),
                "cases_over_50_percent": int((padding > 0.5).sum()),
            },
            "skeleton_pixels": {
                "total": int(skeleton.sum()),
                "median": float(np.median(skeleton)),
                "empty_cases": int((skeleton == 0).sum()),
            },
            "soma_pixels": {
                "total": int(soma.sum()),
                "median": float(np.median(soma)),
                "empty_cases": int((soma == 0).sum()),
            },
            "collections": [
                {"label": label, "count": sum(case["collection"] == label for case in cases)}
                for label in sorted({str(case["collection"]) for case in cases})
            ],
            "size_distribution": self._distribution(
                max_sides,
                [0, 64, 96, 128, 160, 256, 512, float("inf")],
                ["<64", "64–95", "96–127", "128–159", "160–255", "256–511", "≥512"],
            ),
            "padding_distribution": self._distribution(
                padding,
                [0, np.nextafter(0.0, 1.0), 0.25, 0.50, 0.75, 1.01],
                ["0 %", "0–25 %", "25–50 %", "50–75 %", ">75 %"],
            ),
        }
        preprocessing = {
            "configuration": self.configuration,
            "training_trainer": "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug",
            "augmentation": "LabelSafe: exakte 90°-Rotationen und Spiegelungen; keine freie Rotation oder Skalierung",
            "preprocessor": configuration["preprocessor_name"],
            "target_spacing": configuration["spacing"],
            "normalization": configuration["normalization_schemes"],
            "use_mask_for_normalization": configuration["use_mask_for_norm"],
            "patch_size": configuration["patch_size"],
            "batch_size": configuration["batch_size"],
            "resampling_data": configuration["resampling_fn_data_kwargs"],
            "resampling_label": configuration["resampling_fn_seg_kwargs"],
            "network": configuration["architecture"]["network_class_name"],
            "steps": [
                "Nicht belegte Nullränder entfernen",
                "Bildweise Z-Score-Normalisierung",
                "Auf Zielabstand 1 × 1 resamplen; bei Dataset139 bleibt die Pixelgröße unverändert",
                "Für das Training zunächst ca. 188 × 188 laden",
                "Pixelgenau um 90° rotieren oder spiegeln; keine interpolierende Rotation/Skalierung",
                "Als tatsächlichen 160 × 160-Netzinput ausgeben",
            ],
        }
        self._records = records
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": dataset_json.get("name") or DATASET_NAME,
            "labels": dataset_json["labels"],
            "summary": summary,
            "preprocessing": preprocessing,
            "cases": cases,
        }

    def index(self) -> dict[str, object]:
        with self._index_lock:
            if not self.available():
                raise FileNotFoundError(
                    "Dataset139 oder dessen Preprocessing ist noch nicht vollständig vorhanden."
                )
            signature = self._source_signature()
            if self._index is None or signature != self._index_signature:
                self._load_case.cache_clear()
                self.training_sample.cache_clear()
                self.render_view.cache_clear()
                self._index = self._build_index()
                self._index_signature = signature
            return self._index

    def _record(self, case_id: str) -> dict[str, object]:
        self.index()
        try:
            return self._records[case_id]
        except KeyError as error:
            raise KeyError(case_id) from error

    @lru_cache(maxsize=24)
    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        self._record(case_id)
        raw_image = np.squeeze(
            tifffile.imread(self.raw_dataset / "imagesTr" / f"{case_id}_0000.tif")
        )
        raw_label = np.squeeze(tifffile.imread(self.raw_dataset / "labelsTr" / f"{case_id}.tif"))
        data_folder = self.preprocessed_dataset / "nnUNetPlans_2d"
        preprocessed_image = np.asarray(
            blosc2.open(urlpath=str(data_folder / f"{case_id}.b2nd"), mode="r")
        )[0, 0].copy()
        preprocessed_label = np.asarray(
            blosc2.open(urlpath=str(data_folder / f"{case_id}_seg.b2nd"), mode="r")
        )[0, 0].copy()
        with (data_folder / f"{case_id}.pkl").open("rb") as stream:
            properties = pickle.load(stream)
        return raw_image, raw_label, preprocessed_image, preprocessed_label, properties

    @lru_cache(maxsize=32)
    def training_sample(
        self, case_id: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        self._record(case_id)
        seed = int(seed) & 0xFFFFFFFF
        with self._training_lock:
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            python_state = random.getstate()
            try:
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)

                plans = json.loads(self.plans_path.read_text(encoding="utf-8"))
                dataset_json = json.loads(self.dataset_json_path.read_text(encoding="utf-8"))
                plans_manager = PlansManager(plans)
                configuration = plans_manager.get_configuration(self.configuration)
                label_manager = plans_manager.get_label_manager(dataset_json)
                patch_size = np.asarray(configuration.patch_size, dtype=int)
                rotation = (-np.pi, np.pi)
                initial_patch = get_patch_size(
                    patch_size,
                    rotation,
                    rotation,
                    rotation,
                    (0.85, 1.25),
                )
                deep_supervision_scales = list(
                    list(item)
                    for item in 1
                    / np.cumprod(np.vstack(configuration.pool_op_kernel_sizes), axis=0)
                )[:-1]
                transforms = (
                    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug.get_training_transforms(
                        patch_size,
                        rotation,
                        deep_supervision_scales,
                        (0, 1),
                        False,
                        use_mask_for_norm=configuration.use_mask_for_norm,
                        is_cascaded=False,
                        foreground_labels=label_manager.foreground_labels,
                        regions=None,
                        ignore_label=label_manager.ignore_label,
                    )
                )
                dataset_class = infer_dataset_class(
                    str(self.preprocessed_dataset / configuration.data_identifier)
                )
                dataset = dataset_class(
                    str(self.preprocessed_dataset / configuration.data_identifier),
                    [case_id],
                )
                force_foreground = seed % 3 == 0
                loader = nnUNetDataLoaderSkeletonRecall(
                    _DatasetWithValidityMask(dataset),
                    1,
                    initial_patch,
                    patch_size,
                    label_manager=label_manager,
                    oversample_foreground_percent=1.0 if force_foreground else 0.0,
                    sampling_probabilities=None,
                    pad_sides=None,
                    transforms=transforms,
                    probabilistic_oversampling=False,
                )
                batch = loader.generate_train_batch()
                target = batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
                image = batch["data"][0, 0].numpy().copy()
                label = target[0, 0].numpy().copy()
                valid = target[0, 1].numpy().astype(bool, copy=True)
                return image, label, valid, force_foreground
            finally:
                np.random.set_state(numpy_state)
                torch.random.set_rng_state(torch_state)
                random.setstate(python_state)

    def detail(self, case_id: str, seed: int = 1) -> dict[str, object]:
        record = dict(self._record(case_id))
        raw_image, raw_label, pre_image, pre_label, properties = self._load_case(case_id)
        training_image, training_label, valid, force_foreground = self.training_sample(case_id, seed)
        bbox = properties.get("bbox_used_for_cropping")
        return {
            "case": {
                key: record[key]
                for key in (
                    "case_id",
                    "origin",
                    "collection",
                    "source_case_id",
                    "source_image",
                    "source_label",
                    "split",
                    "height",
                    "width",
                    "padding_fraction",
                    "padding_percent",
                    "skeleton_pixels",
                    "soma_pixels",
                )
            },
            "raw": {
                "shape": list(raw_image.shape),
                "dtype": str(raw_image.dtype),
                "stats": _array_stats(raw_image),
                "label_counts": {
                    "skeleton": int((raw_label == 1).sum()),
                    "soma": int((raw_label == 2).sum()),
                },
            },
            "preprocessed": {
                "shape": list(pre_image.shape),
                "dtype": str(pre_image.dtype),
                "stats": _array_stats(pre_image),
                "bbox_used_for_cropping": bbox,
                "label_counts": {
                    "skeleton": int((pre_label == 1).sum()),
                    "soma": int((pre_label == 2).sum()),
                },
            },
            "training_sample": {
                "seed": int(seed),
                "shape": list(training_image.shape),
                "stats": _array_stats(training_image),
                "force_foreground": force_foreground,
                "padding_percent": 100.0 * float((~valid).mean()),
                "label_counts": {
                    "skeleton": int((training_label == 1).sum()),
                    "soma": int((training_label == 2).sum()),
                },
            },
            "views": {
                "raw": f"/api/training-dataset/139/cases/{case_id}/views/raw.png",
                "raw_label": f"/api/training-dataset/139/cases/{case_id}/views/raw-label.png",
                "preprocessed": f"/api/training-dataset/139/cases/{case_id}/views/preprocessed.png",
                "training": f"/api/training-dataset/139/cases/{case_id}/views/training.png?seed={int(seed)}",
            },
        }

    @lru_cache(maxsize=128)
    def render_view(self, case_id: str, view: str, seed: int = 1) -> bytes:
        raw_image, raw_label, pre_image, pre_label, _ = self._load_case(case_id)
        if view == "raw":
            return render_overlay_png(raw_image)
        if view == "raw-label":
            return render_overlay_png(raw_image, raw_label)
        if view == "preprocessed":
            return render_overlay_png(pre_image, pre_label)
        if view == "training":
            training_image, training_label, valid, _ = self.training_sample(case_id, seed)
            return render_overlay_png(training_image, training_label, valid)
        raise KeyError(view)
