from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PLATE_PATTERN = re.compile(
    r"(?P<token>\d{6}_MCS24GFP_div(?P<div>\d+)_plate(?P<plate>\d+))$",
    re.IGNORECASE,
)
DEFAULT_SOURCE = Path(r"X:\Niro\04_Raw_Data\InVitro\mica\250317_MCS24GFP")
DEFAULT_OUTPUT = DEFAULT_SOURCE / "C3-C10_training_expansion"
DEFAULT_FIJI = Path(r"C:\Users\01.144_1\Fiji")
DEFAULT_SEED = 20260831


@dataclass(frozen=True)
class Overview:
    plate_folder: Path
    token: str
    div: int
    plate: int
    well: str
    lof: Path
    xlif: Path
    xlcf: Path
    size_x: int
    size_y: int

    @property
    def provenance(self) -> str:
        return f"{self.token}_{self.well}"


def parse_plate_folder(name: str) -> tuple[str, int, int]:
    match = PLATE_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Cannot read acquisition/DIV/plate from folder name: {name}")
    return match.group("token"), int(match.group("div")), int(match.group("plate"))


def read_xy_dimensions(xlif: Path) -> tuple[int, int]:
    root = ET.parse(xlif).getroot()
    dimensions: dict[int, int] = {}
    for element in root.findall(".//DimensionDescription"):
        dim_id = int(element.attrib["DimID"])
        if dim_id in (1, 2) and dim_id not in dimensions:
            dimensions[dim_id] = int(element.attrib["NumberOfElements"])
    if set(dimensions) != {1, 2}:
        raise ValueError(f"Could not read X/Y dimensions from {xlif}")
    return dimensions[1], dimensions[2]


def discover_overviews(source: Path) -> list[Overview]:
    overviews: list[Overview] = []
    for plate_folder in sorted(path for path in source.iterdir() if path.is_dir()):
        try:
            token, div, plate = parse_plate_folder(plate_folder.name)
        except ValueError:
            continue
        for well_number in range(3, 11):
            well_name = f"C{well_number:02d}"
            well_folder = plate_folder / "Sequence 001" / "C" / str(well_number)
            lof_files = sorted(well_folder.glob("*.lof"))
            xlif_files = sorted((well_folder / "Metadata").glob("*.xlif"))
            xlcf_files = sorted((well_folder / "Metadata").glob("*.xlcf"))
            if not (len(lof_files) == len(xlif_files) == len(xlcf_files) == 1):
                raise ValueError(
                    f"Expected exactly one LOF/XLIF/XLCF set in {well_folder}, found "
                    f"{len(lof_files)}/{len(xlif_files)}/{len(xlcf_files)}"
                )
            size_x, size_y = read_xy_dimensions(xlif_files[0])
            overviews.append(
                Overview(
                    plate_folder=plate_folder,
                    token=token,
                    div=div,
                    plate=plate,
                    well=well_name,
                    lof=lof_files[0],
                    xlif=xlif_files[0],
                    xlcf=xlcf_files[0],
                    size_x=size_x,
                    size_y=size_y,
                )
            )
    if len(overviews) != 64:
        raise ValueError(f"Expected 8 plates × 8 C-wells = 64 overviews, found {len(overviews)}")
    provenances = [item.provenance.casefold() for item in overviews]
    if len(provenances) != len(set(provenances)):
        raise ValueError("Duplicate provenance names detected")
    return overviews


def stable_random(provenance: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{provenance}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def random_fov_coordinates(
    size_x: int,
    size_y: int,
    crop_size: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if crop_size > size_x or crop_size > size_y:
        raise ValueError("Crop is larger than the overview")
    center_x = size_x / 2.0
    center_y = size_y / 2.0
    half_crop = crop_size / 2.0
    # The montage covers a circular well. Requiring the complete crop to fit
    # inside an inscribed circle avoids empty corner FOVs while remaining random.
    radius = min(size_x, size_y) / 2.0 - math.sqrt(2.0) * half_crop
    if radius <= 0:
        raise ValueError("Overview is too small for circularly constrained crops")

    result: list[tuple[int, int]] = []
    for _ in range(10_000):
        x = rng.randint(0, size_x - crop_size)
        y = rng.randint(0, size_y - crop_size)
        crop_center_x = x + half_crop
        crop_center_y = y + half_crop
        inside_well = math.hypot(
            crop_center_x - center_x, crop_center_y - center_y
        ) <= radius
        separated = all(
            math.hypot(x - other_x, y - other_y) >= crop_size
            for other_x, other_y in result
        )
        if inside_well and separated:
            result.append((x, y))
            if len(result) == count:
                return result
    raise RuntimeError("Could not draw enough separated random FOVs")


def copy_if_needed(source: Path, target: Path) -> str:
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing copy has the wrong size: {target}")
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.stat().st_size != source.stat().st_size:
        raise IOError(f"Copy size verification failed: {target}")
    return "copied"


def java_paths(fiji: Path, project: Path) -> tuple[Path, Path, str]:
    java_candidates = sorted((fiji / "java").glob("**/bin/java.exe"))
    javac_candidates = sorted((fiji / "java").glob("**/bin/javac.exe"))
    if len(java_candidates) != 1 or len(javac_candidates) != 1:
        raise FileNotFoundError(f"Expected one Fiji Java runtime under {fiji / 'java'}")
    classpath = ";".join(
        [str(project / "tools"), str(fiji / "jars" / "*"), str(fiji / "plugins" / "*")]
    )
    return java_candidates[0], javac_candidates[0], classpath


def compile_cropper(project: Path, javac: Path, classpath: str) -> None:
    source = project / "tools" / "LeicaLofCropper.java"
    compiled = project / "tools" / "LeicaLofCropper.class"
    if compiled.exists() and compiled.stat().st_mtime >= source.stat().st_mtime:
        return
    subprocess.run(
        [str(javac), "-proc:none", "-cp", classpath, str(source)],
        check=True,
        cwd=project,
    )


def crop_overview(
    java: Path,
    classpath: str,
    overview: Overview,
    output_files: list[Path],
    coordinates: list[tuple[int, int]],
    crop_size: int,
) -> None:
    existing = [path.is_file() for path in output_files]
    if all(existing):
        return
    if any(existing):
        raise FileExistsError(
            f"Partial crop set exists for {overview.provenance}; remove or complete it explicitly"
        )
    command = [str(java), "-cp", classpath, "LeicaLofCropper", str(overview.lof)]
    for output, (x, y) in zip(output_files, coordinates):
        command.extend([str(output), str(x), str(y), str(crop_size), str(crop_size)])
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    for line in result.stdout.splitlines():
        if line.startswith(("INPUT=", "WROTE=")):
            print(line, flush=True)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MICA C3-C10 overviews and extract reproducible random FOVs."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fiji", type=Path, default=DEFAULT_FIJI)
    parser.add_argument("--crop-size", type=int, default=2048)
    parser.add_argument("--crops-per-overview", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-overview-copy",
        action="store_true",
        help="Create FOVs and manifests without duplicating the 43.5-GB Leica containers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    overviews_dir = args.output / "01_overviews_leica"
    crops_dir = args.output / f"02_random_fovs_{args.crop_size}px"
    args.output.mkdir(parents=True, exist_ok=True)
    overviews_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    overviews = discover_overviews(args.source)
    java, javac, classpath = java_paths(args.fiji, project)
    compile_cropper(project, javac, classpath)
    rows: list[dict[str, object]] = []

    for overview_index, overview in enumerate(overviews, start=1):
        print(
            f"[{overview_index:02d}/{len(overviews)}] {overview.provenance} "
            f"({overview.size_x}x{overview.size_y})",
            flush=True,
        )
        copied_lof = overviews_dir / f"{overview.provenance}_overview.lof"
        copied_xlif = overviews_dir / f"{overview.provenance}_overview.xlif"
        copied_xlcf = overviews_dir / f"{overview.provenance}_well_metadata.xlcf"
        copy_status = "skipped"
        if not args.skip_overview_copy:
            copy_status = copy_if_needed(overview.lof, copied_lof)
            copy_if_needed(overview.xlif, copied_xlif)
            copy_if_needed(overview.xlcf, copied_xlcf)

        coordinates = random_fov_coordinates(
            overview.size_x,
            overview.size_y,
            args.crop_size,
            args.crops_per_overview,
            stable_random(overview.provenance, args.seed),
        )
        output_files = [
            crops_dir
            / (
                f"{overview.provenance}_FOV{fov_index:02d}_"
                f"x{x:05d}_y{y:05d}_{args.crop_size}x{args.crop_size}.tif"
            )
            for fov_index, (x, y) in enumerate(coordinates, start=1)
        ]
        crop_overview(
            java,
            classpath,
            overview,
            output_files,
            coordinates,
            args.crop_size,
        )
        for fov_index, (output, (x, y)) in enumerate(
            zip(output_files, coordinates), start=1
        ):
            rows.append(
                {
                    "acquisition": overview.token.split("_", 1)[0],
                    "div": overview.div,
                    "plate": overview.plate,
                    "well": overview.well,
                    "provenance": overview.provenance,
                    "fov": fov_index,
                    "x": x,
                    "y": y,
                    "width": args.crop_size,
                    "height": args.crop_size,
                    "source_width": overview.size_x,
                    "source_height": overview.size_y,
                    "random_seed": args.seed,
                    "source_lof": str(overview.lof),
                    "copied_lof": str(copied_lof) if not args.skip_overview_copy else "",
                    "copy_status": copy_status,
                    "crop_tiff": str(output),
                }
            )
        write_manifest(args.output / "fov_manifest.csv", rows)

    overview_rows = [
        {
            "acquisition": item.token.split("_", 1)[0],
            "div": item.div,
            "plate": item.plate,
            "well": item.well,
            "provenance": item.provenance,
            "width": item.size_x,
            "height": item.size_y,
            "source_lof": str(item.lof),
            "copied_lof": str(overviews_dir / f"{item.provenance}_overview.lof"),
            "copied_xlif": str(overviews_dir / f"{item.provenance}_overview.xlif"),
            "copied_xlcf": str(overviews_dir / f"{item.provenance}_well_metadata.xlcf"),
        }
        for item in overviews
    ]
    write_manifest(args.output / "overview_manifest.csv", overview_rows)
    print(
        f"DONE overviews={len(overviews)} crops={len(rows)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
