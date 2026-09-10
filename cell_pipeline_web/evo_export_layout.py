from __future__ import annotations

import hashlib
import re
from pathlib import Path


def source_image_folder(filename: str) -> str:
    """Return a Windows-safe, stable folder name derived from an input image."""

    original = Path(filename).name
    suffix = Path(original).suffix.lower()
    stem = Path(original).stem if suffix in {".tif", ".tiff"} else original
    stem = re.sub(r"_0000$", "", stem)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "uploaded_image"
    if len(safe) <= 140:
        return safe
    digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:129]}__{digest}"


def relative_evo_cell_folder(image_folder: str, cell_folder: str) -> str:
    if not image_folder or Path(image_folder).name != image_folder:
        raise ValueError(f"Invalid Evo image folder: {image_folder!r}")
    if not re.fullmatch(r"cell\d+", cell_folder):
        raise ValueError(f"Invalid Evo cell folder: {cell_folder!r}")
    return (Path(image_folder) / cell_folder).as_posix()


def iter_evo_cell_folders(cells_root: Path):
    for path in cells_root.rglob("cell*"):
        if path.is_dir() and re.fullmatch(r"cell\d+", path.name):
            yield path
