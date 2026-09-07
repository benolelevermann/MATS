from pathlib import Path
import csv
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import watershed

# ============================================================
# HIER ANPASSEN
# ============================================================

project_root = Path(r"C:\Ole\20260721_CellClassification_v2")

raw_path = project_root / "overview_inference" / "input_2d" / "overview_max_0000.tif"
pred_path = project_root / "overview_inference" / "output_2d" / "overview_max.tif"

out_dir = project_root / "overview_inference" / "postprocessed_mark_ambiguous"

# ============================================================
# PARAMETER
# ============================================================

# Nutzt eine erweiterte Maske NUR für die Zuordnung.
# Die originale Skeleton-Maske wird dadurch NICHT verändert.
# Größer = mehr fragmentierte Skeletons können einem Soma zugeordnet werden.
assignment_bridge_px = 5

# Wenn ein Skeleton-Komponent innerhalb dieses Radius mehrere Somata sieht,
# wird es als ambiguous markiert.
ambiguity_distance_px = 25

# Wenn ein Skeleton-Komponent sehr nah mehrere Somata berührt,
# ist es sehr wahrscheinlich unklar.
touch_distance_px = 3

# Wenn Skeleton-Pixel weiter als dieser Wert vom nächsten Soma entfernt sind,
# werden sie als "far_skeleton" markiert.
# Sie werden NICHT gelöscht.
far_distance_px = 120

# Wenn True, wird jedes Skeleton-Pixel wenigstens dem nächsten Soma zugeordnet.
# Unklare Fälle werden trotzdem markiert.
assign_all_skeleton_to_nearest_soma = True

# Kleine Skeleton-Komponenten werden NICHT gelöscht,
# aber im Report markiert.
tiny_skeleton_component_px = 10

# Kleine Soma-Komponenten werden NICHT gelöscht,
# aber im Report markiert.
tiny_soma_px = 20

# Preview-Größe
max_preview_size = 2500

random_seed = 42

# ============================================================

out_dir.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(img):
    img = np.asarray(img)
    img = np.squeeze(img).astype(np.float32)

    p1, p99 = np.percentile(img, [1, 99])

    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max())

    if p99 <= p1:
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - p1) / (p99 - p1)
    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)


def save_preview(arr, out_path, max_size=2500):
    img = Image.fromarray(arr)

    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))

    img.save(out_path)


def instance_dtype(max_id):
    if max_id <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def make_instance_colors(max_id):
    rng = np.random.default_rng(random_seed)

    colors = rng.integers(
        low=40,
        high=255,
        size=(max_id + 1, 3),
        dtype=np.uint8,
    )

    colors[0] = [0, 0, 0]
    return colors


def make_instance_preview(instances):
    max_id = int(instances.max())
    colors = make_instance_colors(max_id)
    return colors[instances]


def make_ambiguity_overlay(
    raw_u8,
    pred,
    cell_instances,
    ambiguous_skeleton,
    far_skeleton,
):
    raw_rgb = np.stack([raw_u8, raw_u8, raw_u8], axis=-1)
    overlay = raw_rgb.copy()

    max_id = int(cell_instances.max())
    colors = make_instance_colors(max_id)

    inst_mask = cell_instances > 0
    inst_color = colors[cell_instances]

    # dezente Instanzfarbe
    alpha = 0.25
    overlay[inst_mask] = (
        (1 - alpha) * overlay[inst_mask].astype(np.float32)
        + alpha * inst_color[inst_mask].astype(np.float32)
    ).astype(np.uint8)

    skeleton = pred == 1
    soma = pred == 2

    normal_skeleton = skeleton & (~ambiguous_skeleton) & (~far_skeleton)

    # normales Skeleton rot
    overlay[normal_skeleton, 0] = 255
    overlay[normal_skeleton, 1] = (overlay[normal_skeleton, 1] * 0.25).astype(np.uint8)
    overlay[normal_skeleton, 2] = (overlay[normal_skeleton, 2] * 0.25).astype(np.uint8)

    # far skeleton cyan
    overlay[far_skeleton, 0] = 0
    overlay[far_skeleton, 1] = 255
    overlay[far_skeleton, 2] = 255

    # ambiguous skeleton gelb
    overlay[ambiguous_skeleton, 0] = 255
    overlay[ambiguous_skeleton, 1] = 220
    overlay[ambiguous_skeleton, 2] = 0

    # Soma lila
    overlay[soma, 0] = 180
    overlay[soma, 1] = 40
    overlay[soma, 2] = 255

    return overlay


def expand_slice(sl, margin, shape):
    y0 = max(0, sl[0].start - margin)
    y1 = min(shape[0], sl[0].stop + margin)
    x0 = max(0, sl[1].start - margin)
    x1 = min(shape[1], sl[1].stop + margin)

    return slice(y0, y1), slice(x0, x1)


def ids_to_string(ids):
    ids = [int(x) for x in ids if int(x) > 0]
    ids = sorted(set(ids))
    return ";".join(map(str, ids))


# ============================================================
# MAIN
# ============================================================

print("Reading raw:")
print(raw_path)

raw = np.squeeze(tifffile.imread(raw_path))

print("Reading prediction:")
print(pred_path)

pred = np.squeeze(tifffile.imread(pred_path)).astype(np.uint8)

if raw.shape != pred.shape:
    raise RuntimeError(f"Raw und Prediction haben unterschiedliche Größe: {raw.shape} vs {pred.shape}")

print("Shape:", pred.shape)
print("Prediction values:", np.unique(pred).tolist())

raw_u8 = normalize_to_uint8(raw)

skeleton_mask = pred == 1
soma_mask = pred == 2

print()
print("Original pixels:")
print("Skeleton:", int(skeleton_mask.sum()))
print("Soma:", int(soma_mask.sum()))

# ------------------------------------------------------------
# 1. Soma-Instanzen und Skeleton-Komponenten
# ------------------------------------------------------------

soma_instances = cc_label(soma_mask, connectivity=2)
skeleton_components = cc_label(skeleton_mask, connectivity=2)

num_somata = int(soma_instances.max())
num_skel_components = int(skeleton_components.max())

print()
print("Instances:")
print("Somata:", num_somata)
print("Skeleton components:", num_skel_components)

if num_somata == 0:
    raise RuntimeError("Keine Somata gefunden. Ohne Soma kann keine Zuordnung gemacht werden.")

# ------------------------------------------------------------
# 2. Soma-seeded Watershed-Zuordnung
#    Nur für Zuordnung, nicht zum Löschen oder Ändern der Prediction.
# ------------------------------------------------------------

assignment_domain = soma_mask | skeleton_mask

if assignment_bridge_px > 0:
    assignment_domain = binary_dilation(
        assignment_domain,
        footprint=disk(assignment_bridge_px),
    )

print()
print("Running soma-seeded watershed assignment...")

assigned_full = watershed(
    np.zeros(pred.shape, dtype=np.uint8),
    markers=soma_instances.astype(np.int32),
    mask=assignment_domain,
    connectivity=2,
)

cell_instances = np.zeros(
    pred.shape,
    dtype=instance_dtype(num_somata),
)

# Soma direkt übernehmen
cell_instances[soma_mask] = soma_instances[soma_mask]

# Skeleton via Watershed übernehmen
cell_instances[skeleton_mask] = assigned_full[skeleton_mask]

# ------------------------------------------------------------
# 3. Falls Skeleton-Pixel weiterhin unzugeordnet sind:
#    optional nächstes Soma zuweisen, aber NICHT löschen.
# ------------------------------------------------------------

unassigned_skeleton = skeleton_mask & (cell_instances == 0)

print("Unassigned skeleton pixels after watershed:", int(unassigned_skeleton.sum()))

dist_to_soma, nearest_indices = ndi.distance_transform_edt(
    soma_instances == 0,
    return_indices=True,
)

nearest_soma = soma_instances[nearest_indices[0], nearest_indices[1]]

if assign_all_skeleton_to_nearest_soma and unassigned_skeleton.any():
    fallback = unassigned_skeleton & (nearest_soma > 0)
    cell_instances[fallback] = nearest_soma[fallback]
    print("Fallback assigned skeleton pixels:", int(fallback.sum()))

unassigned_skeleton = skeleton_mask & (cell_instances == 0)

print("Unassigned skeleton pixels final:", int(unassigned_skeleton.sum()))

# ------------------------------------------------------------
# 4. Ambiguity-Analyse
# ------------------------------------------------------------

ambiguous_skeleton = np.zeros(pred.shape, dtype=bool)
far_skeleton = np.zeros(pred.shape, dtype=bool)

component_report_rows = []

objects = ndi.find_objects(skeleton_components)

for comp_id, sl in enumerate(objects, start=1):
    if sl is None:
        continue

    comp_mask_full = skeleton_components == comp_id
    pixels = int(comp_mask_full.sum())

    # assigned IDs im aktuellen Component
    assigned_ids = np.unique(cell_instances[comp_mask_full])
    assigned_ids = assigned_ids[assigned_ids > 0]

    # Distanz zum nächsten Soma
    comp_dist = dist_to_soma[comp_mask_full]
    min_dist = float(comp_dist.min()) if comp_dist.size else np.nan
    mean_dist = float(comp_dist.mean()) if comp_dist.size else np.nan
    max_dist = float(comp_dist.max()) if comp_dist.size else np.nan

    # Somata in unmittelbarer Nähe / Berührung
    touch_sl = expand_slice(sl, touch_distance_px, pred.shape)
    comp_crop_touch = skeleton_components[touch_sl] == comp_id
    soma_crop_touch = soma_instances[touch_sl]

    if touch_distance_px > 0:
        comp_touch_dilated = binary_dilation(
            comp_crop_touch,
            footprint=disk(touch_distance_px),
        )
    else:
        comp_touch_dilated = comp_crop_touch

    touched_soma_ids = np.unique(soma_crop_touch[comp_touch_dilated])
    touched_soma_ids = touched_soma_ids[touched_soma_ids > 0]

    # Somata in Ambiguity-Radius
    amb_sl = expand_slice(sl, ambiguity_distance_px, pred.shape)
    comp_crop_amb = skeleton_components[amb_sl] == comp_id
    soma_crop_amb = soma_instances[amb_sl]

    if ambiguity_distance_px > 0:
        comp_amb_dilated = binary_dilation(
            comp_crop_amb,
            footprint=disk(ambiguity_distance_px),
        )
    else:
        comp_amb_dilated = comp_crop_amb

    near_soma_ids = np.unique(soma_crop_amb[comp_amb_dilated])
    near_soma_ids = near_soma_ids[near_soma_ids > 0]

    flags = []

    if pixels < tiny_skeleton_component_px:
        flags.append("tiny_skeleton_component")

    if len(assigned_ids) == 0:
        flags.append("unassigned_component")

    if len(assigned_ids) > 1:
        flags.append("assigned_to_multiple_somata")
        ambiguous_skeleton[comp_mask_full] = True

    if len(touched_soma_ids) > 1:
        flags.append("touches_multiple_somata")
        ambiguous_skeleton[comp_mask_full] = True

    if len(near_soma_ids) > 1:
        flags.append("near_multiple_somata")
        ambiguous_skeleton[comp_mask_full] = True

    if max_dist > far_distance_px:
        flags.append("far_from_soma")
        far_skeleton[comp_mask_full] = True

    # Wenn unassigned, auch ambiguous markieren
    if len(assigned_ids) == 0:
        ambiguous_skeleton[comp_mask_full] = True

    component_report_rows.append({
        "skeleton_component_id": comp_id,
        "pixels": pixels,
        "assigned_cell_ids": ids_to_string(assigned_ids),
        "num_assigned_cell_ids": int(len(assigned_ids)),
        "touched_soma_ids": ids_to_string(touched_soma_ids),
        "num_touched_somata": int(len(touched_soma_ids)),
        "near_soma_ids": ids_to_string(near_soma_ids),
        "num_near_somata": int(len(near_soma_ids)),
        "min_distance_to_soma": min_dist,
        "mean_distance_to_soma": mean_dist,
        "max_distance_to_soma": max_dist,
        "flags": ";".join(flags),
    })

print()
print("Marked ambiguous skeleton pixels:", int(ambiguous_skeleton.sum()))
print("Marked far skeleton pixels:", int(far_skeleton.sum()))

# ------------------------------------------------------------
# 5. Soma-Report
# ------------------------------------------------------------

soma_report_rows = []

for region in regionprops(soma_instances):
    soma_id = int(region.label)
    area = int(region.area)

    flags = []

    if area < tiny_soma_px:
        flags.append("tiny_soma")

    assigned_skeleton_pixels = int(((cell_instances == soma_id) & skeleton_mask).sum())

    if assigned_skeleton_pixels == 0:
        flags.append("no_assigned_skeleton")

    soma_report_rows.append({
        "soma_id": soma_id,
        "soma_pixels": area,
        "assigned_skeleton_pixels": assigned_skeleton_pixels,
        "flags": ";".join(flags),
    })

# ------------------------------------------------------------
# 6. Outputs schreiben
# ------------------------------------------------------------

print()
print("Writing outputs...")

tifffile.imwrite(out_dir / "original_prediction.tif", pred.astype(np.uint8))
tifffile.imwrite(out_dir / "soma_instances.tif", soma_instances.astype(instance_dtype(num_somata)))
tifffile.imwrite(out_dir / "skeleton_components.tif", skeleton_components.astype(instance_dtype(num_skel_components)))
tifffile.imwrite(out_dir / "cell_instances_assigned.tif", cell_instances.astype(instance_dtype(num_somata)))
tifffile.imwrite(out_dir / "ambiguous_skeleton.tif", ambiguous_skeleton.astype(np.uint8))
tifffile.imwrite(out_dir / "far_skeleton.tif", far_skeleton.astype(np.uint8))
tifffile.imwrite(out_dir / "unassigned_skeleton.tif", unassigned_skeleton.astype(np.uint8))

# CSV Reports
component_csv = out_dir / "skeleton_component_report.csv"

with open(component_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "skeleton_component_id",
        "pixels",
        "assigned_cell_ids",
        "num_assigned_cell_ids",
        "touched_soma_ids",
        "num_touched_somata",
        "near_soma_ids",
        "num_near_somata",
        "min_distance_to_soma",
        "mean_distance_to_soma",
        "max_distance_to_soma",
        "flags",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(component_report_rows)

soma_csv = out_dir / "soma_report.csv"

with open(soma_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "soma_id",
        "soma_pixels",
        "assigned_skeleton_pixels",
        "flags",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(soma_report_rows)

# Preview-Bilder
instance_preview = make_instance_preview(cell_instances)

ambiguity_overlay = make_ambiguity_overlay(
    raw_u8=raw_u8,
    pred=pred,
    cell_instances=cell_instances,
    ambiguous_skeleton=ambiguous_skeleton,
    far_skeleton=far_skeleton,
)

save_preview(instance_preview, out_dir / "overview_instances.png", max_preview_size)
save_preview(ambiguity_overlay, out_dir / "overview_ambiguity_overlay.png", max_preview_size)

print()
print("Fertig.")
print("Output-Ordner:")
print(out_dir)
print()
print("Wichtigste Dateien:")
print(out_dir / "overview_ambiguity_overlay.png")
print(out_dir / "overview_instances.png")
print(out_dir / "cell_instances_assigned.tif")
print(out_dir / "ambiguous_skeleton.tif")
print(out_dir / "skeleton_component_report.csv")
print(out_dir / "soma_report.csv")
