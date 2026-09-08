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

from blind_selection_web.pipeline import (
    BlindSelectionSettings,
    extract_selected_cells,
    match_points_to_somas,
)
from blind_selection_web.server import SelectionJobStore
from cell_pipeline_web.pipeline import default_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PointMatchingTests(unittest.TestCase):
    def test_inside_near_duplicate_and_unmatched_points_are_preserved(self) -> None:
        instances = np.zeros((80, 100), dtype=np.uint16)
        instances[10:20, 10:20] = 1
        instances[45:58, 64:77] = 2
        points = [
            {"selection_id": "selection0001", "x": 15, "y": 15},
            {"selection_id": "selection0002", "x": 22, "y": 15},
            {"selection_id": "selection0003", "x": 60, "y": 50},
            {"selection_id": "selection0004", "x": 99, "y": 0},
        ]

        matches = match_points_to_somas(points, instances, maximum_distance=8)

        self.assertEqual(matches[0]["status"], "matched")
        self.assertEqual(matches[0]["reason"], "inside_soma")
        self.assertEqual(matches[1]["status"], "duplicate")
        self.assertEqual(matches[1]["duplicate_of"], "selection0001")
        self.assertEqual(matches[2]["status"], "matched")
        self.assertEqual(matches[2]["soma_id"], 2)
        self.assertEqual(matches[3]["status"], "unmatched")
        self.assertEqual(len(matches), len(points))

    def test_conflict_queries_all_competing_somas_but_exports_only_selected_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = np.zeros((128, 128), dtype=np.uint16)
            semantic = np.zeros((128, 128), dtype=np.uint8)
            semantic[58:66, 28:42] = 2
            semantic[58:66, 86:100] = 2
            semantic[61, 40:88] = 1
            instances = np.zeros_like(semantic, dtype=np.uint16)
            instances[58:66, 28:42] = 1
            instances[58:66, 86:100] = 2
            for name, array in (("raw.tif", raw), ("prediction.tif", semantic), ("semantic.tif", semantic)):
                tifffile.imwrite(root / name, array)
            settings = BlindSelectionSettings(
                semantic=default_settings(PROJECT_ROOT, finalize_fiji=False),
                separator_checkpoint=root / "unused.pth",
            )
            queried_ids: list[int] = []

            def fake_predict(_model, _raw, local_semantic, local_instances, _device):
                ids = np.asarray(sorted(int(v) for v in np.unique(local_instances) if v > 0))
                queried_ids.extend(ids.tolist())
                foreground = local_semantic > 0
                columns = np.indices(local_semantic.shape)[1]
                probabilities = np.stack(
                    (
                        np.where(foreground & (columns < local_semantic.shape[1] // 2), 0.95, 0.05),
                        np.where(foreground & (columns >= local_semantic.shape[1] // 2), 0.95, 0.05),
                    )
                ).astype(np.float32)
                return ids, probabilities

            exported_ids: list[int] = []

            def fake_export(*args, **kwargs):
                exported_ids.append(int(kwargs["qc"]["matched_soma_id"]))
                return {
                    "folder": "cell0001", "source": kwargs["source"],
                    "source_component": kwargs["source_component"], "conflict_group": 1,
                    "skeleton_pixels": 20, "soma_pixels": 100,
                    "x_min": 0, "y_min": 0, "x_max_exclusive": 64,
                    "y_max_exclusive": 64, "preview": "cell0001/preview.png",
                }

            with (
                mock.patch("blind_selection_web.pipeline._conservative_soma_instances", return_value=(instances, [])),
                mock.patch("blind_selection_web.pipeline._load_export_helpers", return_value=object()),
                mock.patch("blind_selection_web.pipeline._load_separator", return_value=(object(), object(), 80)),
                mock.patch("blind_selection_web.pipeline.predict_memberships", side_effect=fake_predict),
                mock.patch("blind_selection_web.pipeline._export_cell", side_effect=fake_export),
            ):
                result = extract_selected_cells(
                    original_path=root / "raw.tif",
                    prediction_path=root / "prediction.tif",
                    semantic_path=root / "semantic.tif",
                    output_root=root / "output",
                    selection={"points": [{"selection_id": "selection0001", "x": 34, "y": 61}]},
                    callback=lambda *_args: None,
                    settings=settings,
                )

            self.assertEqual(queried_ids, [1, 2])
            self.assertEqual(exported_ids, [1])
            self.assertEqual(result["exported_cell_count"], 1)


class SelectionStoreTests(unittest.TestCase):
    @staticmethod
    def _tiff_bytes() -> bytes:
        stream = io.BytesIO()
        tifffile.imwrite(stream, np.arange(96 * 112, dtype=np.uint16).reshape(96, 112))
        return stream.getvalue()

    @staticmethod
    def _fake_pipeline(input_path, run_root, selection, callback, settings):
        callback("prediction", 20, "fake prediction", None)
        cell = run_root / "04_evo_single_cells" / "cell0001"
        cell.mkdir(parents=True)
        raw = np.zeros((64, 64), dtype=np.uint16)
        semantic = np.zeros((64, 64), dtype=np.uint8)
        semantic[28, 14:51] = 1
        semantic[23:34, 9:20] = 2
        for name, array in (
            ("raw.tif", raw),
            ("prediction.tif", semantic),
            ("postprocessed.tif", semantic),
            ("skeleton.tif", (semantic == 1).astype(np.uint8)),
            ("soma.tif", (semantic == 2).astype(np.uint8)),
            ("seg.tif", semantic),
            ("cell_mask.tif", (semantic > 0).astype(np.uint8)),
        ):
            tifffile.imwrite(cell / name, array)
        for name in ("raw_preview.png", "prediction_preview.png", "postprocessing_preview.png", "preview.png"):
            (cell / name).write_bytes(b"PNG")
        (cell / "seg-000.swc").write_text("1 1 12 28 0 1 -1\n", encoding="utf-8")
        (cell / "seg.traces").write_text("<tracings/>", encoding="utf-8")
        for name in ("soma.zip", "bounds.zip"):
            with zipfile.ZipFile(cell / name, "w") as archive:
                archive.writestr("test.roi", b"ROI")
        for name in ("bounds.json", "location.json", "metadata.json"):
            (cell / name).write_text("{}", encoding="utf-8")
        (run_root / "04_evo_single_cells" / "manifest.csv").write_text(
            "folder,source,source_component,conflict_group,skeleton_pixels,soma_pixels,x_min,y_min,x_max_exclusive,y_max_exclusive,preview\n"
            "cell0001,manual_blind_selection_direct,1,,36,121,0,0,64,64,cell0001/preview.png\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(run_root / "evo_candidate_cells.zip", "w"):
            pass
        matching = {"points": [{"selection_id": "selection0001", "status": "exported_candidate"}]}
        (run_root / "model_matching.json").write_text(json.dumps(matching), encoding="utf-8")
        return {
            "exported_cell_count": 1,
            "failed_selections": 0,
            "manifest": [{"folder": "cell0001", "selection_id": "selection0001", "selection_index": 1}],
        }

    def test_selection_is_locked_before_processing_and_good_cell_is_evo_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = BlindSelectionSettings(
                semantic=default_settings(PROJECT_ROOT, finalize_fiji=True),
                separator_checkpoint=root / "unused.pth",
            )
            store = SelectionJobStore(
                root / "runs", root / "evo", root / "learning", settings,
                pipeline_runner=self._fake_pipeline,
            )
            data = self._tiff_bytes()
            job = store.create("overview.tif", len(data), io.BytesIO(data))
            job_id = str(job["id"])
            self.assertEqual(job["status"], "selecting")
            self.assertTrue((root / "runs" / job_id / "raw_preview.png").is_file())

            store.submit_selection(job_id, [{"x": 22.4, "y": 31.6}])
            selection_path = root / "runs" / job_id / "manual_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            self.assertTrue(selection["blind_to_model_output"])
            self.assertEqual(selection["points"][0], {"selection_id": "selection0001", "x": 22, "y": 32})
            with self.assertRaisesRegex(ValueError, "already locked"):
                store.submit_selection(job_id, [{"x": 2, "y": 3}])

            deadline = time.time() + 5
            current = store.public(job_id)
            while time.time() < deadline and current["status"] not in {"completed", "failed"}:
                time.sleep(0.02)
                current = store.public(job_id)
            self.assertEqual(current["status"], "completed")
            self.assertEqual(current["crops"][0]["selection_id"], "selection0001")

            result = store.rate_cell(job_id, "cell0001", "good")
            output_folder = str(result["review"]["evo_cell_folder"])
            output = root / "evo" / "cells" / output_folder
            self.assertTrue((output / "seg.traces").is_file())
            self.assertTrue((output / "soma.zip").is_file())
            learning_result = json.loads(
                (root / "learning" / "jobs" / job_id / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(learning_result["reviews"]["cell0001"]["result"], "good")

            store.rate_cell(job_id, "cell0001", "bad")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
