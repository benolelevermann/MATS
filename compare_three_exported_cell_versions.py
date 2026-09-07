#!/usr/bin/env python
"""Create a mask-faithful three-way HTML comparison of exported cell crops.

The three input directories are cell-export roots such as
``.../10_evo_cells_safe/<case>`` or ``.../13b_evo_cells_single_instance/<case>``.
Cells are associated through their global soma centroids, rather than their
``cell####`` names, because export numbering is not stable between runs.

The report reads the actual exported ``raw.tif``, ``skeleton.tif`` and
``soma.tif`` files. It never skeletonizes, dilates, erodes, or otherwise
changes the two masks. Raw TIFFs are only percentile-normalized to make a
browser-viewable PNG; the original crop TIFFs remain untouched.

Version C is treated as the current/new selection in the HTML filters. This
makes it practical to review which cells were retained, added, or no longer
selected after a stricter QC stage.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SCRIPT_VERSION = "three-way-mask-faithful-crop-comparison-v1-2026-08-16"
SKELETON_RGB = (0, 220, 255)
SOMA_RGB = (255, 45, 170)


@dataclass(frozen=True)
class Version:
    """One exported-crop version supplied by the user."""

    key: str
    label: str
    root: Path


@dataclass(frozen=True)
class Cell:
    """A cell crop and its stable global position."""

    version_key: str
    folder: str
    path: Path
    global_x: float
    global_y: float
    skeleton_pixels_from_manifest: int | None
    soma_pixels_from_manifest: int | None


@dataclass(frozen=True)
class Edge:
    """A centroid-based, cross-version candidate match."""

    distance: float
    left_index: int
    right_index: int


@dataclass(frozen=True)
class MatchGroup:
    """At most one spatially corresponding crop from each version."""

    cells: dict[str, Cell]
    selected_edges: tuple[Edge, ...]

    @property
    def membership(self) -> tuple[str, ...]:
        return tuple(sorted(self.cells))

    @property
    def centroid(self) -> tuple[float, float]:
        values = list(self.cells.values())
        return (
            sum(cell.global_x for cell in values) / len(values),
            sum(cell.global_y for cell in values) / len(values),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a three-way, mask-faithful HTML comparison of exported cell crops. "
            "Version C is treated as the current selection."
        )
    )
    for key, default_label in (
        ("a", "Earlier workflow"),
        ("b", "Previous workflow"),
        ("c", "Current workflow"),
    ):
        parser.add_argument(f"--version-{key}-root", type=Path, required=True)
        parser.add_argument(f"--version-{key}-label", default=default_label)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--match-distance",
        type=float,
        default=40.0,
        help=(
            "Maximum global soma-centroid distance in pixels for association across "
            "versions (default: 40)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_directory(path: Path, argument: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{argument} is not a directory: {path}")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}\n"
                "Use a fresh directory or pass --overwrite to replace this comparison output."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON: {path}") from error


def optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def cells_from_manifest(version: Version) -> list[Cell]:
    manifest = version.root / "manifest.csv"
    if not manifest.is_file():
        return []

    cells: list[Cell] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            folder = (row.get("folder") or "").strip()
            if not folder:
                continue
            cell_path = version.root / folder
            if not cell_path.is_dir():
                continue
            try:
                cells.append(
                    Cell(
                        version_key=version.key,
                        folder=folder,
                        path=cell_path,
                        global_x=float(row["global_centroid_x"]),
                        global_y=float(row["global_centroid_y"]),
                        skeleton_pixels_from_manifest=optional_int(row.get("skeleton_pixels")),
                        soma_pixels_from_manifest=optional_int(row.get("soma_pixels")),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid manifest row for {folder} in {manifest}") from error
    return cells


def cells_from_metadata(version: Version) -> list[Cell]:
    """Fallback for legacy exports that do not have ``manifest.csv``."""

    cells: list[Cell] = []
    for cell_path in sorted(version.root.glob("cell*")):
        if not cell_path.is_dir():
            continue
        location_path = cell_path / "location.json"
        if not location_path.is_file():
            continue
        location = read_json(location_path)
        try:
            cells.append(
                Cell(
                    version_key=version.key,
                    folder=cell_path.name,
                    path=cell_path,
                    global_x=float(location["global_x"]),
                    global_y=float(location["global_y"]),
                    skeleton_pixels_from_manifest=None,
                    soma_pixels_from_manifest=None,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid location metadata: {location_path}") from error
    return cells


def load_cells(version: Version) -> list[Cell]:
    cells = cells_from_manifest(version) or cells_from_metadata(version)
    if not cells:
        raise RuntimeError(
            f"No usable cell folders were found in {version.root}. "
            "Expected manifest.csv or cell####/location.json files."
        )

    required_files = ("raw.tif", "skeleton.tif", "soma.tif")
    incomplete = [
        cell.folder
        for cell in cells
        if any(not (cell.path / filename).is_file() for filename in required_files)
    ]
    if incomplete:
        raise RuntimeError(
            f"Missing crop TIFFs in {version.label}: {', '.join(incomplete[:10])}"
        )
    return sorted(cells, key=lambda cell: (cell.global_y, cell.global_x, cell.folder))


def centroid_distance(first: Cell, second: Cell) -> float:
    return math.hypot(first.global_x - second.global_x, first.global_y - second.global_y)


class DisjointSet:
    """Union-find that refuses groups containing the same version twice."""

    def __init__(self, cells: list[Cell]) -> None:
        self.parent = list(range(len(cells)))
        self.size = [1] * len(cells)
        self.versions: list[set[str]] = [{cell.version_key} for cell in cells]
        self.edges: list[list[Edge]] = [[] for _ in cells]

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def merge(self, first: int, second: int, edge: Edge) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.versions[first_root] & self.versions[second_root]:
            return False
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        self.versions[first_root] |= self.versions[second_root]
        self.edges[first_root].extend(self.edges[second_root])
        self.edges[first_root].append(edge)
        return True


def build_match_groups(
    cells_by_version: dict[str, list[Cell]], maximum_distance: float
) -> list[MatchGroup]:
    """Build one-to-one cross-version groups using stable spatial centroids.

    Candidate pairs are processed from smallest to largest distance. A pair is
    accepted only when joining their groups would still leave at most one crop
    from each version. This prevents a close pair from forcing two crops of one
    version into the same group.
    """

    cells: list[Cell] = [
        cell
        for version_cells in cells_by_version.values()
        for cell in version_cells
    ]
    candidates = sorted(
        (
            Edge(
                distance=centroid_distance(first, second),
                left_index=first_index,
                right_index=second_index,
            )
            for first_index, first in enumerate(cells)
            for second_index, second in enumerate(cells[first_index + 1 :], start=first_index + 1)
            if first.version_key != second.version_key
            and centroid_distance(first, second) <= maximum_distance
        ),
        key=lambda edge: (edge.distance, edge.left_index, edge.right_index),
    )

    groups = DisjointSet(cells)
    for edge in candidates:
        groups.merge(edge.left_index, edge.right_index, edge)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(cells)):
        members[groups.find(index)].append(index)

    result: list[MatchGroup] = []
    for root, indexes in members.items():
        by_version = {cells[index].version_key: cells[index] for index in indexes}
        result.append(MatchGroup(by_version, tuple(sorted(groups.edges[root], key=lambda edge: edge.distance))))
    return sorted(
        result,
        key=lambda group: (
            group.centroid[1],
            group.centroid[0],
            "+".join(group.membership),
        ),
    )


def read_2d(path: Path) -> np.ndarray:
    """Read a single-plane exported TIFF without changing its values."""

    with Image.open(path) as image:
        array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2-D TIFF, got {array.shape}: {path}")
    return array


def normalize_raw_for_display(raw: np.ndarray) -> np.ndarray:
    """Use only a display contrast conversion; never change saved TIFF data."""

    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.zeros(raw.shape, dtype=np.uint8)
    lower, upper = np.percentile(finite, (1.0, 99.5))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros(raw.shape, dtype=np.uint8)
    scaled = (raw.astype(np.float32) - float(lower)) / float(upper - lower)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def build_assets(cell: Cell, assets_dir: Path) -> dict[str, str | int]:
    """Create browser-viewable raw and exact binary mask PNG layers."""

    raw = read_2d(cell.path / "raw.tif")
    skeleton = read_2d(cell.path / "skeleton.tif") > 0
    soma = read_2d(cell.path / "soma.tif") > 0
    if raw.shape != skeleton.shape or raw.shape != soma.shape:
        raise RuntimeError(
            f"Shape mismatch in {cell.path}: raw={raw.shape}, skeleton={skeleton.shape}, soma={soma.shape}"
        )

    safe_stem = f"{cell.version_key}_{cell.folder}"
    raw_name = f"{safe_stem}_raw.png"
    mask_name = f"{safe_stem}_mask.png"
    Image.fromarray(normalize_raw_for_display(raw)).save(assets_dir / raw_name, optimize=True)

    # This is direct pixel coloring only. There is deliberately no morphology
    # operation such as dilation, skeletonization, closing, or smoothing here.
    mask = np.zeros((*raw.shape, 4), dtype=np.uint8)
    mask[skeleton] = (*SKELETON_RGB, 255)
    mask[soma] = (*SOMA_RGB, 255)
    Image.fromarray(mask, mode="RGBA").save(assets_dir / mask_name, optimize=True)

    return {
        "raw": f"assets/{raw_name}",
        "mask": f"assets/{mask_name}",
        "width": int(raw.shape[1]),
        "height": int(raw.shape[0]),
        "skeleton_pixels": int(skeleton.sum()),
        "soma_pixels": int(soma.sum()),
    }


def membership_label(group: MatchGroup, versions: list[Version]) -> str:
    return " + ".join(version.label for version in versions if version.key in group.cells)


def group_category(group: MatchGroup) -> str:
    present = set(group.cells)
    if present == {"a", "b", "c"}:
        return "all-three"
    if "c" not in present:
        return "not-current"
    if present == {"c"}:
        return "current-only"
    return "changed-current"


def group_title(index: int, group: MatchGroup, versions: list[Version]) -> str:
    category = group_category(group)
    if category == "all-three":
        prefix = "Selected in all three versions"
    elif category == "not-current":
        prefix = "Selected before, not selected in current version"
    elif category == "current-only":
        prefix = "Selected only in current version"
    else:
        prefix = "Selection changed between versions"
    x, y = group.centroid
    return f"Group {index:03d} · {prefix} · global centroid ({x:.1f}, {y:.1f}) px"


def cell_caption(cell: Cell, assets: dict[str, str | int]) -> str:
    return (
        f"{cell.folder} · global centroid ({cell.global_x:.1f}, {cell.global_y:.1f}) px"
        f" · skeleton {assets['skeleton_pixels']} px · soma {assets['soma_pixels']} px"
        f" · {assets['width']} × {assets['height']} px"
    )


def render_pane(version: Version, cell: Cell | None, assets: dict[str, str | int] | None) -> str:
    label = html.escape(version.label)
    if cell is None or assets is None:
        return (
            "<section class='pane missing'>"
            f"<h3>{label}</h3><p>No exported crop selected for this spatial cell.</p>"
            "</section>"
        )

    caption = html.escape(cell_caption(cell, assets))
    raw = html.escape(str(assets["raw"]), quote=True)
    mask = html.escape(str(assets["mask"]), quote=True)
    return f"""<section class='pane'>
  <h3>{label}</h3>
  <button class='crop-button' type='button' data-raw='{raw}' data-mask='{mask}' data-label='{caption}'>
    <span class='image-stack'>
      <img class='raw-layer' src='{raw}' alt='Raw image for {html.escape(cell.folder)}'>
      <img class='mask-layer' src='{mask}' alt='Exact skeleton and soma masks for {html.escape(cell.folder)}'>
    </span>
  </button>
  <p>{caption}</p>
</section>"""


def render_group_card(
    index: int,
    group: MatchGroup,
    versions: list[Version],
    assets_by_version: dict[str, dict[str, dict[str, str | int]]],
) -> str:
    category = group_category(group)
    title = html.escape(group_title(index, group, versions))
    membership = html.escape(membership_label(group, versions))
    edge_distances = ", ".join(f"{edge.distance:.1f}" for edge in group.selected_edges)
    matching_note = (
        f"centroid links: {html.escape(edge_distances)} px" if edge_distances else "no cross-version centroid link"
    )
    panes = "\n".join(
        render_pane(
            version,
            group.cells.get(version.key),
            assets_by_version[version.key].get(group.cells[version.key].folder)
            if version.key in group.cells
            else None,
        )
        for version in versions
    )
    return f"""<article class='group {category}' data-category='{category}'>
  <div class='group-heading'>
    <h2>{title}</h2>
    <p><strong>Present:</strong> {membership} · {matching_note}</p>
  </div>
  <div class='group-grid'>{panes}</div>
</article>"""


def write_group_manifest(groups: list[MatchGroup], versions: list[Version], output_path: Path) -> None:
    fields = [
        "group",
        "category",
        "membership",
        "mean_global_x",
        "mean_global_y",
        "selected_centroid_link_distances_px",
    ]
    for version in versions:
        prefix = f"version_{version.key}_"
        fields.extend(
            [
                f"{prefix}label",
                f"{prefix}folder",
                f"{prefix}global_x",
                f"{prefix}global_y",
                f"{prefix}skeleton_pixels",
                f"{prefix}soma_pixels",
                f"{prefix}source_path",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, group in enumerate(groups, start=1):
            x, y = group.centroid
            row: dict[str, str | int] = {
                "group": index,
                "category": group_category(group),
                "membership": membership_label(group, versions),
                "mean_global_x": f"{x:.4f}",
                "mean_global_y": f"{y:.4f}",
                "selected_centroid_link_distances_px": ";".join(
                    f"{edge.distance:.4f}" for edge in group.selected_edges
                ),
            }
            for version in versions:
                prefix = f"version_{version.key}_"
                cell = group.cells.get(version.key)
                row[f"{prefix}label"] = version.label
                row[f"{prefix}folder"] = cell.folder if cell else ""
                row[f"{prefix}global_x"] = f"{cell.global_x:.4f}" if cell else ""
                row[f"{prefix}global_y"] = f"{cell.global_y:.4f}" if cell else ""
                row[f"{prefix}skeleton_pixels"] = (
                    cell.skeleton_pixels_from_manifest if cell else ""
                )
                row[f"{prefix}soma_pixels"] = cell.soma_pixels_from_manifest if cell else ""
                row[f"{prefix}source_path"] = str(cell.path) if cell else ""
            writer.writerow(row)


def write_html(
    output_path: Path,
    groups: list[MatchGroup],
    versions: list[Version],
    version_counts: dict[str, int],
    match_distance: float,
    assets_by_version: dict[str, dict[str, dict[str, str | int]]],
) -> None:
    category_counts = Counter(group_category(group) for group in groups)
    cards = "\n".join(
        render_group_card(index, group, versions, assets_by_version)
        for index, group in enumerate(groups, start=1)
    )
    safe_labels = {version.key: html.escape(version.label) for version in versions}
    output_path.write_text(
        f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Three-way mask-faithful crop comparison</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: Canvas; color: CanvasText; }}
main {{ max-width: 1840px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 6px; font-size: 1.5rem; }}
.subtitle {{ margin: 0 0 14px; max-width: 1100px; color: color-mix(in srgb, CanvasText 67%, Canvas 33%); }}
.facts {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 12px 0 17px; font-size: .94rem; }}
.facts strong {{ font-weight: 650; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 10px; }}
button {{ font: inherit; cursor: pointer; }}
.mode-button, .filter-button {{ padding: 7px 11px; border: 1px solid color-mix(in srgb, CanvasText 30%, Canvas 70%); background: Canvas; color: CanvasText; border-radius: 5px; }}
.mode-button[aria-pressed='true'], .filter-button[aria-pressed='true'] {{ background: CanvasText; color: Canvas; }}
.shown-count {{ margin: 0 0 18px; color: color-mix(in srgb, CanvasText 65%, Canvas 35%); font-size: .9rem; }}
.group {{ border-top: 1px solid color-mix(in srgb, CanvasText 22%, Canvas 78%); padding: 17px 0 25px; }}
.group[hidden] {{ display: none; }}
.group-heading {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 4px 18px; margin-bottom: 12px; }}
.group h2 {{ margin: 0; font-size: 1rem; font-weight: 650; }}
.group-heading p {{ margin: 0; color: color-mix(in srgb, CanvasText 64%, Canvas 36%); font-size: .8rem; }}
.group-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 15px; }}
.pane {{ min-width: 0; }}
.pane h3 {{ margin: 0 0 7px; font-size: .95rem; font-weight: 550; }}
.pane p {{ margin: 7px 0 0; font-size: .77rem; color: color-mix(in srgb, CanvasText 65%, Canvas 35%); overflow-wrap: anywhere; }}
.missing {{ display: grid; align-content: center; min-height: 230px; color: color-mix(in srgb, CanvasText 55%, Canvas 45%); }}
.crop-button {{ display: block; padding: 0; border: 1px solid color-mix(in srgb, CanvasText 30%, Canvas 70%); background: #000; width: 100%; }}
.image-stack {{ display: grid; width: 100%; aspect-ratio: 1; place-items: center; overflow: hidden; }}
.image-stack img {{ grid-area: 1 / 1; max-width: 100%; max-height: 100%; width: 100%; height: 100%; object-fit: contain; }}
.raw-layer {{ opacity: 1; image-rendering: auto; }}
.mask-layer {{ opacity: .88; image-rendering: pixelated; }}
main[data-mode='raw'] .mask-layer {{ opacity: 0; }}
main[data-mode='mask'] .raw-layer {{ opacity: 0; }}
main[data-mode='mask'] .mask-layer {{ opacity: 1; }}
dialog {{ width: min(96vw, 1500px); height: min(92vh, 1100px); padding: 14px; border: 1px solid color-mix(in srgb, CanvasText 30%, Canvas 70%); background: Canvas; color: CanvasText; }}
dialog::backdrop {{ background: rgb(0 0 0 / .72); }}
.dialog-head {{ display: flex; align-items: start; justify-content: space-between; gap: 10px; margin-bottom: 10px; }}
.dialog-head p {{ margin: 0; font-size: .85rem; overflow-wrap: anywhere; }}
.dialog-view {{ width: 100%; height: calc(100% - 50px); overflow: auto; background: #000; display: grid; place-items: start center; }}
.dialog-stack {{ position: relative; display: inline-grid; transform-origin: top center; }}
.dialog-stack img {{ grid-area: 1 / 1; display: block; }}
.dialog-stack .mask-layer {{ opacity: .88; image-rendering: pixelated; }}
@media (max-width: 960px) {{ .group-grid {{ grid-template-columns: 1fr; }} main {{ padding: 16px; }} }}
</style>
</head>
<body>
<main id='comparison' data-mode='overlay'>
  <h1>Three-way, mask-faithful exported-crop comparison</h1>
  <p class='subtitle'>Every pane uses the actual exported <code>raw.tif</code>, <code>skeleton.tif</code>, and <code>soma.tif</code>. Skeleton and soma masks are not dilated, skeletonized, or otherwise changed. Raw signal is only percentile-normalized for browser display. Version C is the current selection.</p>
  <div class='facts'>
    <span><strong>{safe_labels['a']}:</strong> {version_counts['a']} crops</span>
    <span><strong>{safe_labels['b']}:</strong> {version_counts['b']} crops</span>
    <span><strong>{safe_labels['c']}:</strong> {version_counts['c']} crops</span>
    <span><strong>All three:</strong> {category_counts['all-three']}</span>
    <span><strong>Not current:</strong> {category_counts['not-current']}</span>
    <span><strong>Current only:</strong> {category_counts['current-only']}</span>
    <span><strong>Changed:</strong> {category_counts['changed-current']}</span>
    <span><strong>Centroid threshold:</strong> {match_distance:.1f} px</span>
  </div>
  <div class='controls' aria-label='Image display mode'>
    <button class='mode-button' type='button' data-mode='overlay' aria-pressed='true'>Raw + exact masks</button>
    <button class='mode-button' type='button' data-mode='raw' aria-pressed='false'>Raw only</button>
    <button class='mode-button' type='button' data-mode='mask' aria-pressed='false'>Exact masks only</button>
  </div>
  <div class='controls' aria-label='Group filter'>
    <button class='filter-button' type='button' data-filter='all' aria-pressed='true'>All groups</button>
    <button class='filter-button' type='button' data-filter='all-three' aria-pressed='false'>Selected in all three</button>
    <button class='filter-button' type='button' data-filter='changed-current' aria-pressed='false'>Changed selection</button>
    <button class='filter-button' type='button' data-filter='not-current' aria-pressed='false'>Not selected now</button>
    <button class='filter-button' type='button' data-filter='current-only' aria-pressed='false'>Only selected now</button>
  </div>
  <p id='shown-count' class='shown-count'></p>
  {cards}
</main>
<dialog id='crop-dialog'>
  <div class='dialog-head'><p id='dialog-label'></p><button id='dialog-close' type='button'>Close</button></div>
  <div class='dialog-view'><span class='dialog-stack'><img id='dialog-raw' class='raw-layer' alt='Raw crop'><img id='dialog-mask' class='mask-layer' alt='Exact skeleton and soma masks'></span></div>
</dialog>
<script>
(() => {{
  const root = document.getElementById('comparison');
  const groups = Array.from(document.querySelectorAll('.group'));
  const count = document.getElementById('shown-count');
  const dialog = document.getElementById('crop-dialog');
  const dialogRaw = document.getElementById('dialog-raw');
  const dialogMask = document.getElementById('dialog-mask');
  const dialogLabel = document.getElementById('dialog-label');
  function updateCount() {{
    const shown = groups.filter((group) => !group.hidden).length;
    count.textContent = `${{shown}} of ${{groups.length}} spatial cell groups shown.`;
  }}
  document.querySelectorAll('.mode-button').forEach((button) => {{
    button.addEventListener('click', () => {{
      root.dataset.mode = button.dataset.mode;
      document.querySelectorAll('.mode-button').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    }});
  }});
  document.querySelectorAll('.filter-button').forEach((button) => {{
    button.addEventListener('click', () => {{
      const filter = button.dataset.filter;
      groups.forEach((group) => {{ group.hidden = filter !== 'all' && group.dataset.category !== filter; }});
      document.querySelectorAll('.filter-button').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      updateCount();
    }});
  }});
  document.querySelectorAll('.crop-button').forEach((button) => {{
    button.addEventListener('click', () => {{
      dialogRaw.src = button.dataset.raw;
      dialogMask.src = button.dataset.mask;
      dialogLabel.textContent = button.dataset.label;
      dialog.showModal();
    }});
  }});
  document.getElementById('dialog-close').addEventListener('click', () => dialog.close());
  updateCount();
}})();
</script>
</body>
</html>""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.match_distance <= 0:
        raise ValueError("--match-distance must be positive.")

    versions = [
        Version("a", args.version_a_label, args.version_a_root),
        Version("b", args.version_b_label, args.version_b_root),
        Version("c", args.version_c_label, args.version_c_root),
    ]
    for version in versions:
        require_directory(version.root, f"--version-{version.key}-root")

    cells_by_version = {version.key: load_cells(version) for version in versions}
    groups = build_match_groups(cells_by_version, args.match_distance)

    prepare_output_dir(args.output_dir, args.overwrite)
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir()
    assets_by_version = {
        version.key: {
            cell.folder: build_assets(cell, assets_dir)
            for cell in cells_by_version[version.key]
        }
        for version in versions
    }
    version_counts = {key: len(cells) for key, cells in cells_by_version.items()}

    write_group_manifest(groups, versions, args.output_dir / "three_way_matching_manifest.csv")
    write_html(
        args.output_dir / "index.html",
        groups,
        versions,
        version_counts,
        args.match_distance,
        assets_by_version,
    )

    category_counts = Counter(group_category(group) for group in groups)
    summary = {
        "script_version": SCRIPT_VERSION,
        "versions": [
            {"key": version.key, "label": version.label, "root": str(version.root.resolve()), "crop_count": version_counts[version.key]}
            for version in versions
        ],
        "match_distance_px": args.match_distance,
        "spatial_group_count": len(groups),
        "group_category_counts": dict(category_counts),
        "display_note": (
            "Masks are rendered from exact exported skeleton.tif and soma.tif pixels. "
            "Raw TIFFs use only percentile contrast normalization for browser display."
        ),
        "matching_note": (
            "Groups are built from global soma-centroid distances and contain at most one crop per version. "
            "Crop folder numbers are not used for matching."
        ),
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("THREE-WAY MASK-FAITHFUL CROP COMPARISON COMPLETE")
    print("=" * 72)
    for version in versions:
        print(f"{version.key.upper()} {version.label}: {version_counts[version.key]} crops")
    print(f"Spatial groups: {len(groups)}")
    for category in ("all-three", "changed-current", "not-current", "current-only"):
        print(f"{category}: {category_counts[category]}")
    print(f"HTML: {args.output_dir / 'index.html'}")
    print(f"Manifest: {args.output_dir / 'three_way_matching_manifest.csv'}")


if __name__ == "__main__":
    main()
