from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
from scipy import ndimage as ndi

from cell_pipeline_web.pipeline import (
    _conservative_soma_instances,
    _separate_touching_soma_instances,
    _tracer_soma_instances,
    _validate_ntt_traces,
    default_settings,
    extract_single_cells,
    hysteresis_cli_arguments,
)
from cell_pipeline_web.server import (
    CellPipelineServer,
    JobStore,
    settings_for_hysteresis_profile,
)
from apply_adaptive_hysteresis import resolve_thresholds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HysteresisProfileTests(unittest.TestCase):
    def test_adaptive_and_fixed_cli_arguments_are_distinct(self) -> None:
        adaptive = default_settings(PROJECT_ROOT)
        fixed = settings_for_hysteresis_profile(adaptive, "tlow-0250")
        self.assertEqual(hysteresis_cli_arguments(adaptive)[0], "--alpha")
        self.assertEqual(
            hysteresis_cli_arguments(fixed),
            ["--t-high", "0.647", "--t-low", "0.25"],
        )

    def test_fixed_thresholds_are_not_recomputed_from_the_image(self) -> None:
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        _otsu, _variance, high, low, mode = resolve_thresholds(
            gray, 1.0 / 3.0, 0.647, 0.25
        )
        self.assertEqual(mode, "fixed")
        self.assertEqual(high, round(0.647 * 255))
        self.assertEqual(low, round(0.25 * 255))

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            settings_for_hysteresis_profile(
                default_settings(PROJECT_ROOT), "not-a-profile"
            )


class AtomicJsonWriteTests(unittest.TestCase):
    def test_windows_server_does_not_reuse_an_active_review_port(self) -> None:
        if __import__("os").name == "nt":
            self.assertFalse(CellPipelineServer.allow_reuse_address)

    def test_retries_a_transient_windows_destination_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            real_replace = __import__("os").replace
            attempts = 0

            def flaky_replace(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError(5, "Access is denied", str(destination))
                real_replace(source, destination)

            with (
                mock.patch("cell_pipeline_web.server.os.replace", side_effect=flaky_replace),
                mock.patch("cell_pipeline_web.server.time.sleep") as sleep,
            ):
                JobStore._write_json_atomic(path, {"status": "completed"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "completed")
            self.assertEqual(attempts, 2)
            sleep.assert_called_once()
            self.assertEqual(list(path.parent.glob("status.json.*.tmp")), [])


def disk_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2


def traces_from_points(shape: tuple[int, int], records: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    height = shape[0]
    rows = []
    for soma_id, points in records:
        for row, col in points:
            rows.append((soma_id, 1, row + col * height + 1))
    return np.asarray(rows, dtype=np.int64)


class NeuroTreeTracerValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (64, 64)
        self.soma = np.zeros(self.shape, dtype=bool)
        self.soma |= disk_mask(self.shape, (20, 10), 3)
        self.soma |= disk_mask(self.shape, (42, 50), 3)
        self.skeleton = np.zeros(self.shape, dtype=bool)
        self.left_points = [(20, col) for col in range(10, 30)]
        self.right_points = [(42, col) for col in range(31, 51)]
        for row, col in self.left_points + self.right_points:
            self.skeleton[row, col] = True
        self.input_seg = self.soma | ndi.binary_dilation(self.skeleton, iterations=1)
        self.settings = default_settings(
            PROJECT_ROOT,
            finalize_fiji=False,
            ntt_min_trace_pixels=5,
            ntt_min_coverage=0.8,
        )

    def test_accepts_complete_noncolliding_traces(self) -> None:
        traces = traces_from_points(
            self.shape,
            [(1, self.left_points), (2, list(reversed(self.right_points)))],
        )
        cells, qc = _validate_ntt_traces(
            traces, self.input_seg, self.soma, self.skeleton, self.settings
        )
        self.assertIsNotNone(cells)
        self.assertEqual(len(cells or []), 2)
        self.assertEqual(qc["reason"], "accepted")

    def test_rejects_missing_soma_atomically(self) -> None:
        traces = traces_from_points(self.shape, [(1, self.left_points)])
        cells, qc = _validate_ntt_traces(
            traces, self.input_seg, self.soma, self.skeleton, self.settings
        )
        self.assertIsNone(cells)
        self.assertEqual(qc["reason"], "not_every_soma_was_traced")

    def test_rejects_pixel_owned_by_two_somas(self) -> None:
        common = (31, 31)
        input_seg = self.input_seg.copy()
        input_seg[common] = True
        traces = traces_from_points(
            self.shape,
            [(1, self.left_points + [common]), (2, self.right_points + [common])],
        )
        cells, qc = _validate_ntt_traces(
            traces, input_seg, self.soma, self.skeleton, self.settings
        )
        self.assertIsNone(cells)
        self.assertEqual(qc["reason"], "same_trace_pixel_assigned_to_multiple_somas")

    def test_tracer_soma_ids_follow_size_then_column_major_order(self) -> None:
        instances = _tracer_soma_instances(self.soma)
        self.assertEqual(int(instances[20, 10]), 1)
        self.assertEqual(int(instances[42, 50]), 2)


class CellExtractionTests(unittest.TestCase):
    def test_fused_soma_blob_is_split_into_separable_instances(self) -> None:
        shape = (120, 160)
        soma = np.zeros(shape, dtype=bool)
        for center in ((15, 15), (15, 50), (15, 85), (15, 120)):
            soma |= disk_mask(shape, center, 8)
        soma |= disk_mask(shape, (75, 65), 13)
        soma |= disk_mask(shape, (75, 86), 13)

        instances, report = _conservative_soma_instances(
            soma, default_settings(PROJECT_ROOT, finalize_fiji=False)
        )
        separated = _separate_touching_soma_instances(instances)

        self.assertEqual(ndi.label(soma, structure=np.ones((3, 3)))[1], 5)
        self.assertEqual(int(instances.max()), 6)
        self.assertEqual(
            sum(row["reason"] == "watershed_split" for row in report), 1
        )
        self.assertEqual(
            ndi.label(separated, structure=np.ones((3, 3)))[1], 6
        )

    def test_only_tiny_soma_components_are_ignored_without_crashing(self) -> None:
        soma = np.zeros((32, 32), dtype=bool)
        soma[4, 4] = True
        soma[20:22, 20:22] = True
        instances, report = _conservative_soma_instances(
            soma,
            default_settings(
                PROJECT_ROOT, finalize_fiji=False, min_soma_area=20
            ),
        )
        self.assertEqual(int(instances.max()), 0)
        self.assertEqual(len(report), 2)
        self.assertTrue(all(row["reason"] == "too_small" for row in report))

    def test_oversized_unsplit_soma_is_not_exported_as_a_safe_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (260, 300)
            raw = np.zeros(shape, dtype=np.uint16)
            semantic = np.zeros(shape, dtype=np.uint8)
            for row, col in ((40, 40), (40, 100), (40, 160), (40, 220)):
                semantic[row + 9 : row + 30, col] = 1
                semantic[disk_mask(shape, (row, col), 8)] = 2
            semantic[191:225, 140] = 1
            semantic[disk_mask(shape, (170, 140), 20)] = 2
            raw[semantic > 0] = 1200
            raw_path = root / "raw.tif"
            semantic_path = root / "semantic.tif"
            tifffile.imwrite(raw_path, raw)
            tifffile.imwrite(semantic_path, semantic)

            summary = extract_single_cells(
                raw_path,
                semantic_path,
                root / "cells",
                root / "ntt",
                lambda *_args: None,
                default_settings(
                    PROJECT_ROOT,
                    finalize_fiji=False,
                    min_skeleton_pixels=3,
                    crop_margin=8,
                    min_crop_size=48,
                    edge_clearance=1,
                ),
                ntt_runner=lambda *_args: self.fail(
                    "NeuroTreeTracer must not run for a rejected unsplit soma"
                ),
            )

            self.assertEqual(summary["soma_split_candidates"], 1)
            self.assertEqual(summary["unsafe_unsplit_soma_instances"], 1)
            self.assertEqual(summary["components_rejected_for_unsafe_soma"], 1)
            self.assertEqual(summary["exported_cell_count"], 4)
            rejected = (root / "cells" / "rejected_groups.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("oversized_soma_not_safely_split", rejected)

    def test_failed_overlap_split_exports_none_of_the_conflict_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (128, 128)
            raw = np.zeros(shape, dtype=np.uint16)
            semantic = np.zeros(shape, dtype=np.uint8)
            left = disk_mask(shape, (64, 31), 5)
            right = disk_mask(shape, (64, 97), 5)
            semantic[left | right] = 2
            semantic[64, 36:93] = 1
            raw[semantic > 0] = 1200
            raw_path = root / "raw.tif"
            semantic_path = root / "semantic.tif"
            tifffile.imwrite(raw_path, raw)
            tifffile.imwrite(semantic_path, semantic)

            def failed_split(group_dir, *_args):
                group_dir.mkdir(parents=True)
                return np.empty((0, 3), dtype=np.int64)

            summary = extract_single_cells(
                raw_path,
                semantic_path,
                root / "cells",
                root / "ntt",
                lambda *_args: None,
                default_settings(
                    PROJECT_ROOT,
                    finalize_fiji=False,
                    ntt_max_crop_side=512,
                ),
                ntt_runner=failed_split,
            )

            self.assertEqual(summary["conflict_groups"], 1)
            self.assertEqual(summary["neurotreetracer_groups_accepted"], 0)
            self.assertEqual(summary["neurotreetracer_groups_rejected"], 1)
            self.assertEqual(summary["exported_cell_count"], 0)
            self.assertEqual(list((root / "cells").glob("cell*")), [])

    def test_isolated_cell_exports_one_evo_crop_without_ntt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (128, 128)
            raw = np.zeros(shape, dtype=np.uint16)
            semantic = np.zeros(shape, dtype=np.uint8)
            soma = disk_mask(shape, (62, 58), 5)
            semantic[soma] = 2
            semantic[62, 63:97] = 1
            prediction = semantic.copy()
            prediction[62, 90:97] = 0
            raw[semantic > 0] = 1200
            raw_path = root / "raw.tif"
            semantic_path = root / "semantic.tif"
            prediction_path = root / "prediction.tif"
            tifffile.imwrite(raw_path, raw)
            tifffile.imwrite(semantic_path, semantic)
            tifffile.imwrite(prediction_path, prediction)
            ntt_called = False

            def ntt_runner(*_args, **_kwargs):
                nonlocal ntt_called
                ntt_called = True
                raise AssertionError("NTT must not run for an isolated cell")

            settings = default_settings(
                PROJECT_ROOT,
                finalize_fiji=False,
                min_soma_area=5,
                min_skeleton_pixels=3,
                crop_margin=8,
                min_crop_size=32,
                edge_clearance=1,
            )
            events = []
            summary = extract_single_cells(
                raw_path,
                semantic_path,
                root / "cells",
                root / "ntt",
                lambda *args: events.append(args),
                settings,
                ntt_runner=ntt_runner,
                prediction_path=prediction_path,
            )
            self.assertFalse(ntt_called)
            self.assertEqual(summary["exported_cell_count"], 1)
            self.assertEqual(summary["isolated_exported"], 1)
            cell = root / "cells" / "cell0001"
            for name in (
                "raw.tif",
                "prediction.tif",
                "postprocessed.tif",
                "skeleton.tif",
                "soma.tif",
                "seg.tif",
                "seg.csv",
                "seg-000.swc",
                "metadata.json",
                "raw_preview.png",
                "prediction_preview.png",
                "postprocessing_preview.png",
                "preview.png",
            ):
                self.assertTrue((cell / name).is_file(), name)
            exported_soma = np.squeeze(tifffile.imread(cell / "soma.tif")) > 0
            self.assertEqual(ndi.label(exported_soma, structure=np.ones((3, 3)))[1], 1)
            metadata = json.loads((cell / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["classification_source"], "isolated_topology")
            self.assertLess(
                int(np.count_nonzero(tifffile.imread(cell / "prediction.tif") == 1)),
                int(np.count_nonzero(tifffile.imread(cell / "postprocessed.tif") == 1)),
            )

    def test_all_rejected_is_a_valid_zero_crop_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = np.zeros((96, 96), dtype=np.uint16)
            semantic = np.zeros((96, 96), dtype=np.uint8)
            semantic[0:9, 47:50] = 1
            semantic[9:16, 44:53] = 2
            raw_path = root / "raw.tif"
            semantic_path = root / "semantic.tif"
            tifffile.imwrite(raw_path, raw)
            tifffile.imwrite(semantic_path, semantic)

            summary = extract_single_cells(
                raw_path,
                semantic_path,
                root / "cells",
                root / "ntt",
                lambda *_args: None,
                default_settings(PROJECT_ROOT, finalize_fiji=False),
            )

            self.assertEqual(summary["exported_cell_count"], 0)
            self.assertEqual(summary["discarded_direct_candidates"], 1)
            self.assertTrue((root / "cells" / "manifest.csv").is_file())
            rejected = (root / "cells" / "rejected_groups.csv").read_text(encoding="utf-8")
            self.assertIn("cell_touches_source_border", rejected)


class JobStoreTests(unittest.TestCase):
    def test_evo_output_numbers_are_not_reused_after_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            review_path = runs / "old_job" / "05_review" / "reviews.json"
            review_path.parent.mkdir(parents=True)
            review_path.write_text(
                json.dumps(
                    {
                        "job_id": "old_job",
                        "cells": {
                            "cell0001": {
                                "result": "bad",
                                "evo_cell_folder": "cell000007",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = JobStore(
                runs,
                default_settings(PROJECT_ROOT, finalize_fiji=True),
                review_mode="evo",
                review_root=root / "evo_selection",
            )
            self.assertEqual(store._next_evo_cell_folder(), "cell000008")

    def test_evo_review_copies_complete_cell_without_touching_training_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "evo_runs"
            evo_review = root / "evo_selection"

            def fake_pipeline(_input_path, run_root, callback, settings):
                cell = run_root / "04_evo_single_cells" / "cell0001"
                cell.mkdir(parents=True)
                raw = np.zeros((48, 48), dtype=np.uint16)
                label = np.zeros((48, 48), dtype=np.uint8)
                label[24, 10:38] = 1
                label[20:29, 8:17] = 2
                for name, array in (
                    ("raw.tif", raw),
                    ("prediction.tif", label),
                    ("postprocessed.tif", label),
                    ("skeleton.tif", (label == 1).astype(np.uint8)),
                    ("soma.tif", (label == 2).astype(np.uint8)),
                    ("seg.tif", label),
                    ("cell_mask.tif", (label > 0).astype(np.uint8)),
                ):
                    tifffile.imwrite(cell / name, array)
                (cell / "seg-000.swc").write_text(
                    "1 1 10 24 0 1 -1\n2 3 11 24 0 1 1\n", encoding="utf-8"
                )
                (cell / "seg.traces").write_text("<?xml version='1.0'?><tracings/>", encoding="utf-8")
                for name in ("soma.zip", "bounds.zip"):
                    with zipfile.ZipFile(cell / name, "w") as archive:
                        archive.writestr(name.replace(".zip", ".roi"), b"ROI")
                for name in ("bounds.json", "location.json", "metadata.json"):
                    (cell / name).write_text("{}", encoding="utf-8")
                for name in (
                    "raw_preview.png",
                    "prediction_preview.png",
                    "postprocessing_preview.png",
                    "preview.png",
                ):
                    (cell / name).write_bytes(b"png")
                manifest = run_root / "04_evo_single_cells" / "manifest.csv"
                manifest.write_text(
                    "folder,source,source_component,conflict_group,skeleton_pixels,soma_pixels,x_min,y_min,x_max_exclusive,y_max_exclusive,preview\n"
                    "cell0001,isolated_topology,1,,20,40,0,0,48,48,cell0001/preview.png\n",
                    encoding="utf-8",
                )
                with zipfile.ZipFile(run_root / "evo_single_cells.zip", "w"):
                    pass
                return {
                    "exported_cell_count": 1,
                    "isolated_exported": 1,
                    "neurotreetracer_cells_exported": 0,
                    "neurotreetracer_groups_rejected": 0,
                    "fiji_finalized": True,
                }

            store = JobStore(
                runs,
                default_settings(PROJECT_ROOT, finalize_fiji=True),
                pipeline_runner=fake_pipeline,
                review_mode="evo",
                review_root=evo_review,
            )
            job = store.create("overview.tif", 4, io.BytesIO(b"TIFF"))
            deadline = time.time() + 5
            while time.time() < deadline:
                current = store.public(str(job["id"]))
                if current["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual(current["status"], "completed")

            accepted = store.rate_cell(str(job["id"]), "cell0001", result="good")
            review = accepted["review"]
            self.assertTrue(review["approved_for_evo"])
            self.assertFalse(review["approved_for_training"])
            output_folder = str(review["evo_cell_folder"])
            target = evo_review / "cells" / output_folder
            for name in (
                "raw.tif",
                "seg.tif",
                "seg-000.swc",
                "seg.traces",
                "soma.zip",
                "bounds.zip",
                "review.json",
            ):
                self.assertTrue((target / name).is_file(), name)
            self.assertFalse((evo_review / "approved" / "imagesTr").exists())
            self.assertFalse((evo_review / "approved" / "labelsTr").exists())
            selection = json.loads(
                (evo_review / "selection_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["selected_cells"], 1)
            with self.assertRaisesRegex(ValueError, "disabled in Evo mode"):
                store.recrop_cell(str(job["id"]), "cell0001", 0, 0, 40, 40)
            archive_path = store.approved_archive()
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIn(f"cells/{output_folder}/seg-000.swc", archive.namelist())

            rejected = store.rate_cell(str(job["id"]), "cell0001", result="bad")
            self.assertFalse(rejected["review"]["approved_for_evo"])
            self.assertFalse(target.exists())

    def test_manual_upload_is_processed_before_older_fov_batch_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            order: list[str] = []

            def write_queued_job(job_id: str, case: str, created_at: str, source: dict[str, object]):
                input_path = runs / job_id / "01_input" / f"{case}_0000.tif"
                input_path.parent.mkdir(parents=True)
                input_path.write_bytes(b"TIFF")
                (runs / job_id / "status.json").write_text(
                    json.dumps(
                        {
                            "id": job_id,
                            "filename": f"{case}.tif",
                            "case": case,
                            "input_path": str(input_path),
                            "source": source,
                            "created_at": created_at,
                            "updated_at": created_at,
                            "status": "queued",
                            "phase": "queued",
                            "progress": 0,
                            "message": "queued",
                            "details": {},
                            "events": [],
                            "summary": None,
                            "crops": [],
                            "review_summary": {
                                "total": 0, "reviewed": 0, "approved": 0,
                                "rejected": 0, "pending": 0,
                            },
                            "download_url": None,
                            "version": 1,
                        }
                    ),
                    encoding="utf-8",
                )

            write_queued_job(
                "20260831_100000_fov00001",
                "background_fov",
                "2026-08-31T10:00:00+00:00",
                {"kind": "fov_library"},
            )
            write_queued_job(
                "20260831_110000_upload01",
                "manual_upload",
                "2026-08-31T11:00:00+00:00",
                {"kind": "upload"},
            )

            def fake_pipeline(input_path, run_root, callback, settings):
                order.append(input_path.stem)
                cells = run_root / "04_evo_single_cells"
                cells.mkdir(parents=True)
                (cells / "manifest.csv").write_text(
                    "folder,source,source_component,conflict_group,skeleton_pixels,soma_pixels,x_min,y_min,x_max_exclusive,y_max_exclusive,preview\n",
                    encoding="utf-8",
                )
                with zipfile.ZipFile(run_root / "evo_single_cells.zip", "w"):
                    pass
                return {
                    "exported_cell_count": 0,
                    "isolated_exported": 0,
                    "neurotreetracer_cells_exported": 0,
                    "neurotreetracer_groups_rejected": 0,
                }

            store = JobStore(
                runs,
                default_settings(PROJECT_ROOT, finalize_fiji=False),
                pipeline_runner=fake_pipeline,
            )
            deadline = time.time() + 5
            while time.time() < deadline and len(order) < 2:
                time.sleep(0.02)
            self.assertEqual(order, ["manual_upload_0000", "background_fov_0000"])

    def test_interrupted_running_job_restarts_from_input_and_cleans_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            job_id = "20260831_120000_resume01"
            run_root = runs / job_id
            input_path = run_root / "01_input" / "resume_0000.tif"
            input_path.parent.mkdir(parents=True)
            input_path.write_bytes(b"TIFF")
            partial = run_root / "02_prediction_dataset138"
            partial.mkdir()
            (partial / "partial.txt").write_text("partial", encoding="utf-8")
            (run_root / "status.json").write_text(
                json.dumps(
                    {
                        "id": job_id,
                        "filename": "resume.tif",
                        "case": "resume",
                        "input_path": str(input_path),
                        "created_at": "2026-08-31T12:00:00+00:00",
                        "updated_at": "2026-08-31T12:01:00+00:00",
                        "status": "running",
                        "phase": "prediction",
                        "progress": 18,
                        "message": "interrupted",
                        "details": {},
                        "events": [],
                        "summary": None,
                        "crops": [],
                        "review_summary": {"total": 0, "reviewed": 0, "approved": 0, "rejected": 0, "pending": 0},
                        "download_url": None,
                        "version": 1,
                    }
                ),
                encoding="utf-8",
            )

            def fake_pipeline(_input_path, target_root, callback, settings):
                self.assertFalse((target_root / "02_prediction_dataset138" / "partial.txt").exists())
                cells = target_root / "04_evo_single_cells"
                cells.mkdir(parents=True)
                (cells / "manifest.csv").write_text(
                    "folder,source,source_component,conflict_group,skeleton_pixels,soma_pixels,x_min,y_min,x_max_exclusive,y_max_exclusive,preview\n",
                    encoding="utf-8",
                )
                with zipfile.ZipFile(target_root / "evo_single_cells.zip", "w"):
                    pass
                return {
                    "exported_cell_count": 0,
                    "isolated_exported": 0,
                    "neurotreetracer_cells_exported": 0,
                    "neurotreetracer_groups_rejected": 0,
                }

            store = JobStore(
                runs,
                default_settings(PROJECT_ROOT, finalize_fiji=False),
                pipeline_runner=fake_pipeline,
            )
            deadline = time.time() + 5
            while time.time() < deadline:
                current = store.public(job_id)
                if current["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual(current["status"], "completed")

    def test_job_is_persisted_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"

            def fake_pipeline(input_path, run_root, callback, settings):
                self.assertTrue(input_path.is_file())
                callback("prediction", 30, "fake prediction", None)
                cell_root = run_root / "04_evo_single_cells" / "cell0001"
                cell_root.mkdir(parents=True)
                (cell_root / "preview.png").write_bytes(b"png")
                (cell_root / "metadata.json").write_text("{}", encoding="utf-8")
                raw = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
                semantic = np.zeros((64, 64), dtype=np.uint8)
                semantic[15, 8:24] = 1
                semantic[13:18, 13:18] = 2
                tifffile.imwrite(cell_root / "raw.tif", raw)
                tifffile.imwrite(cell_root / "prediction.tif", semantic)
                tifffile.imwrite(cell_root / "postprocessed.tif", semantic)
                tifffile.imwrite(cell_root / "seg.tif", semantic)
                tifffile.imwrite(cell_root / "skeleton.tif", (semantic == 1).astype(np.uint8))
                tifffile.imwrite(cell_root / "soma.tif", (semantic == 2).astype(np.uint8))
                manifest = run_root / "04_evo_single_cells" / "manifest.csv"
                manifest.write_text(
                    "folder,source,source_component,conflict_group,skeleton_pixels,soma_pixels,x_min,y_min,x_max_exclusive,y_max_exclusive,preview\n"
                    "cell0001,isolated_topology,1,,12,30,0,0,64,64,cell0001/preview.png\n",
                    encoding="utf-8",
                )
                archive = run_root / "evo_single_cells.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr("cell0001/test.txt", "ok")
                return {
                    "exported_cell_count": 1,
                    "isolated_exported": 1,
                    "neurotreetracer_cells_exported": 0,
                    "neurotreetracer_groups_rejected": 0,
                }

            store = JobStore(
                runs,
                default_settings(PROJECT_ROOT, finalize_fiji=False),
                pipeline_runner=fake_pipeline,
            )
            job = store.create("example.tif", 4, io.BytesIO(b"TIFF"))
            deadline = time.time() + 5
            while time.time() < deadline:
                current = store.public(str(job["id"]))
                if current["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual(current["status"], "completed")
            self.assertEqual(len(current["crops"]), 1)
            self.assertTrue((runs / str(job["id"]) / "status.json").is_file())

            job_id = str(job["id"])
            crop_info = store.crop_info(job_id, "cell0001")
            self.assertEqual(crop_info["source_width"], 64)
            with self.assertRaisesRegex(ValueError, "Keep at least"):
                store.recrop_cell(job_id, "cell0001", 8, 8, 32, 32)
            recropped = store.recrop_cell(job_id, "cell0001", 4, 4, 32, 32)
            self.assertEqual(recropped["crop"]["manual_crop"]["width"], 32)
            curated = runs / job_id / "04_evo_single_cells" / "cell0001" / "curated"
            self.assertEqual(np.squeeze(tifffile.imread(curated / "raw.tif")).shape, (32, 32))
            self.assertEqual(np.squeeze(tifffile.imread(curated / "seg.tif")).shape, (32, 32))

            reviewed = store.rate_cell(job_id, "cell0001", result="good")
            self.assertEqual(reviewed["review"]["result"], "good")
            self.assertEqual(reviewed["review"]["training"], "good")
            self.assertEqual(reviewed["review"]["postprocessing"], "good")
            self.assertTrue(reviewed["review"]["approved_for_training"])
            self.assertEqual(reviewed["review_summary"]["approved"], 1)
            review_id = store._review_id(job_id, "cell0001")
            approved = store.review_root / "approved"
            self.assertTrue((approved / "imagesTr" / f"{review_id}_0000.tif").is_file())
            self.assertTrue((approved / "labelsTr" / f"{review_id}.tif").is_file())
            self.assertEqual(
                np.squeeze(tifffile.imread(approved / "imagesTr" / f"{review_id}_0000.tif")).shape,
                (32, 32),
            )
            self.assertTrue((approved / "skeletons" / f"{review_id}.tif").is_file())
            dataset = json.loads((approved / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset["numTraining"], 1)
            archive = store.approved_archive()
            with zipfile.ZipFile(archive) as review_zip:
                self.assertIn("approved/dataset.json", review_zip.namelist())

            changed = store.rate_cell(job_id, "cell0001", "good", "bad")
            self.assertFalse(changed["review"]["approved_for_training"])
            self.assertFalse((approved / "labelsTr" / f"{review_id}.tif").exists())
            self.assertTrue(
                (store.review_root / "postprocessing" / "bad" / review_id / "skeleton.tif").is_file()
            )
            reloaded = JobStore(
                runs,
                default_settings(PROJECT_ROOT, finalize_fiji=False),
                pipeline_runner=fake_pipeline,
            )
            restored_review = reloaded.public(job_id)["crops"][0]["review"]
            self.assertEqual(restored_review["result"], "bad")
            self.assertEqual(restored_review["training"], "good")
            self.assertEqual(restored_review["postprocessing"], "bad")

    def test_review_id_uses_crop_specific_training_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            job_id = "training_review_test"
            cell = runs / job_id / "04_evo_single_cells" / "cell0001"
            cell.mkdir(parents=True)
            raw = np.zeros((40, 40), dtype=np.uint16)
            label = np.zeros((40, 40), dtype=np.uint8)
            label[20, 10:30] = 1
            label[17:24, 8:15] = 2
            for name, array in (
                ("raw.tif", raw),
                ("seg.tif", label),
                ("skeleton.tif", (label == 1).astype(np.uint8)),
                ("soma.tif", (label == 2).astype(np.uint8)),
            ):
                tifffile.imwrite(cell / name, array)
            (cell / "preview.png").write_bytes(b"png")
            status = {
                "id": job_id,
                "filename": "training review",
                "case": "generic",
                "created_at": "2026-09-01T12:00:00+00:00",
                "updated_at": "2026-09-01T12:00:00+00:00",
                "status": "completed",
                "phase": "complete",
                "progress": 100,
                "message": "ready",
                "events": [],
                "summary": {"exported_cell_count": 1},
                "crops": [
                    {
                        "folder": "cell0001",
                        "case_id": "rocki_m239_fov123_cell7",
                        "skeleton_pixels": 10,
                        "soma_pixels": 20,
                    }
                ],
                "review_summary": {"total": 1, "reviewed": 0, "approved": 0, "rejected": 0, "pending": 1},
                "version": 1,
            }
            (runs / job_id / "status.json").write_text(json.dumps(status), encoding="utf-8")
            store = JobStore(runs, default_settings(PROJECT_ROOT, finalize_fiji=False))
            self.assertTrue(store._review_id(job_id, "cell0001").startswith("rocki_m239_fov123_cell7__"))


if __name__ == "__main__":
    unittest.main()
