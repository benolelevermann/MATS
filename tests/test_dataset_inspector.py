from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cell_pipeline_web.dataset_inspector import (
    TrainingDatasetInspector,
    padding_fraction,
    render_overlay_png,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DatasetInspectorHelperTests(unittest.TestCase):
    def test_padding_fraction_uses_only_real_pixels_inside_patch(self) -> None:
        self.assertEqual(padding_fraction(160, 160, 160), 0.0)
        self.assertEqual(padding_fraction(320, 320, 160), 0.0)
        self.assertAlmostEqual(padding_fraction(80, 160, 160), 0.5)
        self.assertAlmostEqual(padding_fraction(40, 40, 160), 0.9375)

    def test_overlay_is_a_valid_rgb_png_with_requested_size(self) -> None:
        image = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
        label = np.zeros((20, 30), dtype=np.uint8)
        label[4:8, 4:8] = 2
        label[10, 3:25] = 1
        valid = np.ones((20, 30), dtype=bool)
        valid[:, :2] = False

        rendered = Image.open(io.BytesIO(render_overlay_png(image, label, valid)))

        self.assertEqual(rendered.format, "PNG")
        self.assertEqual(rendered.mode, "RGB")
        self.assertEqual(rendered.size, (30, 20))


class Dataset139InspectorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inspector = TrainingDatasetInspector.from_project(PROJECT_ROOT)
        if not cls.inspector.available():
            raise unittest.SkipTest("Dataset139 preprocessing is not available in this checkout.")

    def test_index_is_json_serializable_and_matches_manifest(self) -> None:
        payload = self.inspector.index()
        json.dumps(payload, ensure_ascii=False)

        self.assertEqual(payload["dataset_id"], 139)
        self.assertEqual(payload["summary"]["total_cases"], len(payload["cases"]))
        self.assertEqual(
            payload["summary"]["training_cases"] + payload["summary"]["validation_cases"],
            payload["summary"]["total_cases"],
        )
        self.assertEqual(payload["summary"]["patch_size"], [160, 160])

    def test_detail_uses_exact_training_patch_and_all_views_render(self) -> None:
        case_id = self.inspector.index()["cases"][0]["case_id"]
        detail = self.inspector.detail(case_id, seed=7)
        json.dumps(detail, ensure_ascii=False)

        self.assertEqual(detail["training_sample"]["shape"], [160, 160])
        self.assertEqual(
            detail["raw"]["label_counts"], detail["preprocessed"]["label_counts"]
        )
        for view in ("raw", "raw-label", "preprocessed", "training"):
            image = Image.open(io.BytesIO(self.inspector.render_view(case_id, view, seed=7)))
            self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
