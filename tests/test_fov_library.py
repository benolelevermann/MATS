from __future__ import annotations

import csv
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from cell_pipeline_web.fov_library import FovLibrary
from cell_pipeline_web.pipeline import default_settings
from cell_pipeline_web.server import JobStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FovLibraryTests(unittest.TestCase):
    def test_expands_good_examples_with_additional_non_overlapping_fovs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops = root / "02_random_fovs_64px"
            crops.mkdir(parents=True)
            source = crops / "250317_MCS24GFP_div04_plate1_C03_FOV01_x00100_y00100_64x64.tif"
            tifffile.imwrite(source, np.ones((64, 64), dtype=np.uint16))
            source_lof = root / "source.lof"
            source_lof.write_bytes(b"test")
            (root / "fov_manifest.csv").write_text(
                "acquisition,div,plate,well,provenance,fov,x,y,width,height,source_width,source_height,source_lof,crop_tiff\n"
                f"250317,4,1,C03,250317_MCS24GFP_div04_plate1_C03,1,100,100,64,64,512,512,{source_lof},{source}\n",
                encoding="utf-8",
            )
            library = FovLibrary(root)

            def fake_cropper(_source_lof, outputs, coordinates, crop_size):
                self.assertEqual(crop_size, 64)
                self.assertEqual(len(outputs), len(coordinates))
                for output in outputs:
                    tifffile.imwrite(output, np.full((64, 64), 2, dtype=np.uint16))

            created = library.expand_from_examples(
                [source.stem, source.stem], count_per_overview=2, crop_writer=fake_cropper
            )
            self.assertEqual(len(created), 2)
            self.assertTrue(all(record.generation == "expanded" for record in created))
            self.assertEqual([record.fov for record in created], [2, 3])
            occupied = [(100, 100)]
            for record in created:
                self.assertTrue(record.source.is_file())
                self.assertTrue(all(
                    ((record.x - x) ** 2 + (record.y - y) ** 2) ** 0.5 >= 64
                    for x, y in occupied
                ))
                occupied.append((record.x, record.y))
            reloaded = FovLibrary(root)
            self.assertEqual(reloaded.public()["count"], 3)
            self.assertEqual(
                sum(item["generation"] == "expanded" for item in reloaded.public()["fovs"]), 2
            )

    def test_loads_manifest_and_prepares_browser_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops = root / "02_random_fovs_2048px"
            crops.mkdir()
            filename = "250317_MCS24GFP_div04_plate1_C03_FOV01_x00010_y00020_64x64.tif"
            source = crops / filename
            image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
            tifffile.imwrite(source, image)
            with (root / "fov_manifest.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "acquisition",
                        "div",
                        "plate",
                        "well",
                        "fov",
                        "x",
                        "y",
                        "width",
                        "height",
                        "crop_tiff",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "acquisition": "250317",
                        "div": 4,
                        "plate": 1,
                        "well": "C03",
                        "fov": 1,
                        "x": 10,
                        "y": 20,
                        "width": 64,
                        "height": 64,
                        "crop_tiff": str(source),
                    }
                )

            library = FovLibrary(root)
            public = library.public()
            self.assertEqual(public["count"], 1)
            self.assertEqual(public["fovs"][0]["well"], "C03")
            summary = library.prepare_previews(maximum_side=96)
            self.assertEqual(summary["status"], "PASS")
            preview = library.get(source.stem).preview
            self.assertTrue(preview.is_file())
            with Image.open(preview) as rendered:
                self.assertEqual(rendered.size, (64, 64))

    def test_rejects_source_outside_library_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            source = Path(outside) / "outside.tif"
            tifffile.imwrite(source, np.zeros((64, 64), dtype=np.uint16))
            (root / "fov_manifest.csv").write_text(
                "acquisition,div,plate,well,fov,x,y,width,height,crop_tiff\n"
                f"250317,4,1,C03,1,0,0,64,64,{source}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                FovLibrary(root)

    def test_selected_library_fov_is_copied_and_queued_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_root = root / "library"
            crops = library_root / "02_random_fovs_2048px"
            crops.mkdir(parents=True)
            source = crops / "250317_MCS24GFP_div04_plate1_C03_FOV01_x00010_y00020_64x64.tif"
            tifffile.imwrite(source, np.ones((64, 64), dtype=np.uint16))
            (library_root / "fov_manifest.csv").write_text(
                "acquisition,div,plate,well,fov,x,y,width,height,crop_tiff\n"
                f"250317,4,1,C03,1,10,20,64,64,{source}\n",
                encoding="utf-8",
            )
            library = FovLibrary(library_root)

            def fake_pipeline(input_path, run_root, callback, settings):
                self.assertTrue(input_path.is_file())
                callback("prediction", 30, "fake", None)
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
                root / "runs",
                default_settings(PROJECT_ROOT, finalize_fiji=False),
                pipeline_runner=fake_pipeline,
                fov_library=library,
            )
            jobs = store.create_fov_jobs([source.stem])
            self.assertEqual(len(jobs), 1)
            job_id = str(jobs[0]["id"])
            deadline = time.time() + 5
            while time.time() < deadline:
                current = store.public(job_id)
                if current["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual(current["status"], "completed")
            self.assertEqual(current["source"]["fov_id"], source.stem)
            copied = root / "runs" / job_id / "01_input" / f"{source.stem}_0000.tif"
            self.assertTrue(copied.is_file())


if __name__ == "__main__":
    unittest.main()
