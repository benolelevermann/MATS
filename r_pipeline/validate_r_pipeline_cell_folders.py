from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile


REQUIRED_FILES = (
    "raw.tif",
    "skeleton.tif",
    "soma.tif",
    "seg.tif",
    "seg-000.swc",
    "seg.traces",
    "soma.zip",
    "bounds.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exported cell folders for the copied R pipeline."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def traces_contains_xml(path: Path) -> bool:
    with path.open("rb") as handle:
        signature = handle.read(2)
    if signature == b"\x1f\x8b":
        with gzip.open(path, "rb") as handle:
            sample = handle.read(4096)
    else:
        sample = path.read_bytes()[:4096]
    return b"<tracings" in sample or b"<?xml" in sample


def validate_cell(cell_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        path = cell_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {name}")

    for name in ("soma.zip", "bounds.zip"):
        path = cell_dir / name
        if path.is_file() and not zipfile.is_zipfile(path):
            errors.append(f"not a valid ZIP: {name}")
        elif path.is_file():
            with zipfile.ZipFile(path) as archive:
                if not any(item.lower().endswith(".roi") for item in archive.namelist()):
                    errors.append(f"contains no .roi: {name}")

    traces = cell_dir / "seg.traces"
    if traces.is_file() and not traces_contains_xml(traces):
        errors.append("seg.traces is not SNT XML/gzip XML")

    swc = cell_dir / "seg-000.swc"
    if swc.is_file():
        data_lines = [
            line
            for line in swc.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not data_lines:
            errors.append("seg-000.swc has no nodes")
        elif any(len(line.split()) != 7 for line in data_lines):
            errors.append("seg-000.swc contains a malformed node row")

    arrays: dict[str, np.ndarray] = {}
    for name in ("raw.tif", "skeleton.tif", "soma.tif", "seg.tif"):
        path = cell_dir / name
        if path.is_file():
            array = np.squeeze(np.asarray(tifffile.imread(path)))
            if array.ndim != 2:
                errors.append(f"{name} is not 2D: {array.shape}")
            else:
                arrays[name] = array
    shapes = {name: array.shape for name, array in arrays.items()}
    if len(set(shapes.values())) > 1:
        errors.append(f"TIFF shape mismatch: {shapes}")
    if "seg.tif" in arrays:
        values = set(np.unique(arrays["seg.tif"]).astype(int).tolist())
        if not values.issubset({0, 1, 2}):
            errors.append(f"seg.tif values are not 0/1/2: {sorted(values)}")

    for json_name in ("bounds.json", "location.json", "metadata.json"):
        path = cell_dir / json_name
        if path.is_file():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:
                errors.append(f"invalid {json_name}: {error}")
    return errors


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    pattern = re.compile(r"^cell\d+$")
    cell_dirs = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_dir() and pattern.match(path.name)
    )
    if not cell_dirs:
        raise RuntimeError(f"No cell<number> folders found: {args.input_dir}")

    failed = 0
    for cell_dir in cell_dirs:
        errors = validate_cell(cell_dir)
        if errors:
            failed += 1
            print(f"ERROR {cell_dir.name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK    {cell_dir.name}")
    print()
    print(f"Checked: {len(cell_dirs)}")
    print(f"Valid:   {len(cell_dirs) - failed}")
    print(f"Failed:  {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
