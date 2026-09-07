from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


DEFAULT_FOV_ROOT = Path(
    r"X:\Niro\04_Raw_Data\InVitro\mica\250317_MCS24GFP\C3-C10_training_expansion"
)
DEFAULT_FIJI_ROOT = Path(r"C:\Users\01.144_1\Fiji")
DEFAULT_EXPANSION_SEED = 20260831


@dataclass(frozen=True)
class FovRecord:
    identifier: str
    filename: str
    source: Path
    preview: Path
    acquisition: str
    div: int
    plate: int
    well: str
    fov: int
    x: int
    y: int
    width: int
    height: int
    provenance: str = ""
    source_width: int | None = None
    source_height: int | None = None
    source_lof: Path | None = None
    generation: str = "initial"
    mean: float | None = None
    std: float | None = None

    def public(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "filename": self.filename,
            "acquisition": self.acquisition,
            "div": self.div,
            "plate": self.plate,
            "well": self.well,
            "fov": self.fov,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "provenance": self.provenance,
            "generation": self.generation,
            "mean": self.mean,
            "std": self.std,
            "preview_url": f"/fov-previews/{self.identifier}.png",
        }


class FovLibrary:
    def __init__(self, root: Path, fiji_root: Path = DEFAULT_FIJI_ROOT) -> None:
        self.root = root.resolve()
        self.fiji_root = fiji_root.resolve()
        self.manifest_path = self.root / "fov_manifest.csv"
        self.preview_root = self.root / "03_fov_selection" / "previews"
        self.summary_path = self.root / "03_fov_selection" / "selection_summary.json"
        self._lock = threading.RLock()
        self._records = self._load_records()

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    def _load_records(self) -> dict[str, FovRecord]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"FOV manifest not found: {self.manifest_path}")
        quality_path = self.root / "fov_quality_metrics.csv"
        quality = {
            row["filename"]: row
            for row in self._read_csv(quality_path)
        } if quality_path.is_file() else {}
        records: dict[str, FovRecord] = {}
        for row in self._read_csv(self.manifest_path):
            source = Path(row["crop_tiff"]).resolve()
            try:
                source.relative_to(self.root)
            except ValueError as error:
                raise ValueError(f"FOV path lies outside the library root: {source}") from error
            identifier = source.stem
            if identifier in records:
                raise ValueError(f"Duplicate FOV identifier: {identifier}")
            metrics = quality.get(source.name, {})
            records[identifier] = FovRecord(
                identifier=identifier,
                filename=source.name,
                source=source,
                preview=self.preview_root / f"{identifier}.png",
                acquisition=row["acquisition"],
                div=int(row["div"]),
                plate=int(row["plate"]),
                well=row["well"],
                fov=int(row["fov"]),
                x=int(row["x"]),
                y=int(row["y"]),
                width=int(row["width"]),
                height=int(row["height"]),
                provenance=row.get("provenance") or (
                    f"{row['acquisition']}_div{int(row['div']):02d}_plate{int(row['plate'])}_{row['well']}"
                ),
                source_width=int(row["source_width"]) if row.get("source_width") else None,
                source_height=int(row["source_height"]) if row.get("source_height") else None,
                source_lof=Path(row["source_lof"]).resolve() if row.get("source_lof") else None,
                generation=row.get("generation") or "initial",
                mean=float(metrics["mean"]) if metrics.get("mean") else None,
                std=float(metrics["std"]) if metrics.get("std") else None,
            )
        if not records:
            raise ValueError(f"FOV manifest is empty: {self.manifest_path}")
        return records

    def records(self) -> list[FovRecord]:
        return sorted(
            self._records.values(),
            key=lambda item: (item.div, item.plate, item.well, item.fov),
        )

    def get(self, identifier: str) -> FovRecord:
        if identifier not in self._records:
            raise KeyError(identifier)
        return self._records[identifier]

    def public(self) -> dict[str, object]:
        records = self.records()
        return {
            "available": True,
            "count": len(records),
            "fovs": [record.public() for record in records],
        }

    @staticmethod
    def _stable_random(token: str) -> random.Random:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @staticmethod
    def _additional_coordinates(
        width: int,
        height: int,
        crop_size: int,
        count: int,
        occupied: list[tuple[int, int]],
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        if crop_size > width or crop_size > height:
            raise ValueError("FOV crop is larger than the source overview.")
        half_crop = crop_size / 2.0
        radius = min(width, height) / 2.0 - math.sqrt(2.0) * half_crop
        if radius <= 0:
            raise ValueError("The overview is too small for circularly constrained FOV crops.")
        result: list[tuple[int, int]] = []
        for _ in range(50_000):
            x = rng.randint(0, width - crop_size)
            y = rng.randint(0, height - crop_size)
            inside_well = math.hypot(
                x + half_crop - width / 2.0,
                y + half_crop - height / 2.0,
            ) <= radius
            separated = all(
                math.hypot(x - other_x, y - other_y) >= crop_size
                for other_x, other_y in [*occupied, *result]
            )
            if inside_well and separated:
                result.append((x, y))
                if len(result) == count:
                    return result
        raise RuntimeError("Could not place enough additional, non-overlapping FOVs.")

    def _crop_lof(
        self,
        source_lof: Path,
        outputs: list[Path],
        coordinates: list[tuple[int, int]],
        crop_size: int,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        java_candidates = sorted((self.fiji_root / "java").glob("**/bin/java.exe"))
        javac_candidates = sorted((self.fiji_root / "java").glob("**/bin/javac.exe"))
        if len(java_candidates) != 1 or len(javac_candidates) != 1:
            raise FileNotFoundError(f"Fiji Java runtime not found below {self.fiji_root}")
        classpath = ";".join(
            [str(project_root / "tools"), str(self.fiji_root / "jars" / "*"), str(self.fiji_root / "plugins" / "*")]
        )
        source_code = project_root / "tools" / "LeicaLofCropper.java"
        compiled = project_root / "tools" / "LeicaLofCropper.class"
        if not compiled.is_file() or compiled.stat().st_mtime < source_code.stat().st_mtime:
            subprocess.run(
                [str(javac_candidates[0]), "-proc:none", "-cp", classpath, str(source_code)],
                check=True,
                cwd=project_root,
            )
        command = [str(java_candidates[0]), "-cp", classpath, "LeicaLofCropper", str(source_lof)]
        for output, (x, y) in zip(outputs, coordinates):
            command.extend([str(output), str(x), str(y), str(crop_size), str(crop_size)])
        subprocess.run(command, check=True, cwd=project_root, capture_output=True, text=True)

    def expand_from_examples(
        self,
        identifiers: list[str],
        count_per_overview: int = 6,
        crop_writer=None,
    ) -> list[FovRecord]:
        """Create more random FOVs from the overview represented by good examples."""
        if count_per_overview < 1 or count_per_overview > 12:
            raise ValueError("Choose between 1 and 12 additional FOVs per overview.")
        unique_ids = list(dict.fromkeys(str(value) for value in identifiers))
        if not unique_ids:
            raise ValueError("Select at least one good example FOV.")
        with self._lock:
            examples = [self.get(identifier) for identifier in unique_ids]
            grouped: dict[str, FovRecord] = {}
            for record in examples:
                grouped.setdefault(record.provenance, record)
            manifest_rows = self._read_csv(self.manifest_path)
            original_manifest = self.manifest_path.read_bytes()
            manifest_committed = False
            fieldnames = list(manifest_rows[0])
            for field in ("generation", "expanded_from"):
                if field not in fieldnames:
                    fieldnames.append(field)
            new_rows: list[dict[str, object]] = []
            created_paths: list[Path] = []
            try:
                for provenance, example in grouped.items():
                    if example.source_width is None or example.source_height is None or example.source_lof is None:
                        raise ValueError(
                            f"The manifest lacks source dimensions or source_lof for {provenance}."
                        )
                    if not example.source_lof.is_file():
                        raise FileNotFoundError(f"Source overview not found: {example.source_lof}")
                    related = [record for record in self._records.values() if record.provenance == provenance]
                    crop_size = example.width
                    if example.width != example.height or any(
                        record.width != crop_size or record.height != crop_size for record in related
                    ):
                        raise ValueError(f"All FOVs for {provenance} must use one square crop size.")
                    occupied = [(record.x, record.y) for record in related]
                    round_token = f"{DEFAULT_EXPANSION_SEED}|{provenance}|{len(related)}"
                    coordinates = self._additional_coordinates(
                        example.source_width,
                        example.source_height,
                        crop_size,
                        count_per_overview,
                        occupied,
                        self._stable_random(round_token),
                    )
                    first_index = max(record.fov for record in related) + 1
                    output_root = example.source.parent
                    outputs = [
                        output_root / (
                            f"{provenance}_FOV{first_index + offset:02d}_"
                            f"x{x:05d}_y{y:05d}_{crop_size}x{crop_size}.tif"
                        )
                        for offset, (x, y) in enumerate(coordinates)
                    ]
                    for output in outputs:
                        if output.exists():
                            raise FileExistsError(f"Additional FOV already exists: {output}")
                    created_paths.extend(outputs)
                    writer = crop_writer or self._crop_lof
                    writer(example.source_lof, outputs, coordinates, crop_size)
                    missing = [path for path in outputs if not path.is_file()]
                    if missing:
                        raise IOError(f"FOV cropper did not create {missing[0]}")
                    template = next(
                        row for row in manifest_rows if Path(str(row["crop_tiff"])).resolve() == example.source
                    )
                    for offset, (output, (x, y)) in enumerate(zip(outputs, coordinates)):
                        row = dict(template)
                        row.update(
                            {
                                "fov": first_index + offset,
                                "x": x,
                                "y": y,
                                "crop_tiff": str(output),
                                "generation": "expanded",
                                "expanded_from": example.identifier,
                            }
                        )
                        new_rows.append(row)
                temporary = self.manifest_path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows([*manifest_rows, *new_rows])
                temporary.replace(self.manifest_path)
                manifest_committed = True
                self._records = self._load_records()
                self.prepare_previews(maximum_side=384, overwrite=False)
                return [self._records[Path(str(row["crop_tiff"])).stem] for row in new_rows]
            except Exception:
                for path in created_paths:
                    path.unlink(missing_ok=True)
                    (self.preview_root / f"{path.stem}.png").unlink(missing_ok=True)
                if manifest_committed:
                    rollback = self.manifest_path.with_suffix(".csv.rollback")
                    rollback.write_bytes(original_manifest)
                    rollback.replace(self.manifest_path)
                    self._records = self._load_records()
                raise

    @staticmethod
    def _preview(image: np.ndarray, maximum_side: int) -> Image.Image:
        values = image.astype(np.float32, copy=False)
        low, high = np.percentile(values, [0.5, 99.8])
        if high <= low:
            scaled = np.zeros(image.shape, dtype=np.uint8)
        else:
            scaled = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        result = Image.fromarray(scaled, mode="L")
        result.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
        return result

    def prepare_previews(self, maximum_side: int = 384, overwrite: bool = False) -> dict[str, object]:
        if maximum_side < 96:
            raise ValueError("Preview size must be at least 96 px.")
        self.preview_root.mkdir(parents=True, exist_ok=True)
        created = 0
        existing = 0
        for index, record in enumerate(self.records(), start=1):
            if record.preview.is_file() and not overwrite:
                existing += 1
                continue
            image = np.squeeze(np.asarray(tifffile.imread(record.source)))
            if image.ndim != 2:
                raise ValueError(f"FOV is not two-dimensional: {record.source} {image.shape}")
            self._preview(image, maximum_side).save(record.preview, optimize=True)
            created += 1
            if index % 24 == 0:
                print(f"Prepared {index}/{len(self._records)} FOV previews", flush=True)
        missing = [record.identifier for record in self.records() if not record.preview.is_file()]
        summary = {
            "status": "PASS" if not missing else "FAIL",
            "fov_count": len(self._records),
            "created": created,
            "existing": existing,
            "preview_size": maximum_side,
            "missing": missing,
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare browser previews for the MICA FOV library.")
    parser.add_argument("--root", type=Path, default=DEFAULT_FOV_ROOT)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = FovLibrary(args.root)
    print(json.dumps(library.prepare_previews(args.size, args.overwrite), indent=2), flush=True)


if __name__ == "__main__":
    main()
