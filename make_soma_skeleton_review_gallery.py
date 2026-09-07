from pathlib import Path
import os
import sys
import random
import shutil
import subprocess
import numpy as np
import tifffile
from PIL import Image, ImageDraw
import html

# ============================================================
# HIER ANPASSEN
# ============================================================

project_root = Path(r"C:\Ole\20260721_CellClassification_v2")

dataset_id = "129"
dataset_name = "Dataset129_invitrotracings_skeleton_soma"

configuration = "2d"
fold = "0"

# Am besten checkpoint_best.pth verwenden
checkpoint_name = "checkpoint_best.pth"

device = "cuda"

# Trainingsdaten kontrollieren
images_dir = project_root / "nnUNet_raw" / dataset_name / "imagesTr"
labels_dir = project_root / "nnUNet_raw" / dataset_name / "labelsTr"

# Review-Output
review_root = project_root / "soma_skeleton_prediction_review"
review_input = review_root / "input_images"
review_output = review_root / "predictions"
thumb_dir = review_root / "thumbs"
html_path = review_root / "review.html"

# Anzahl zufälliger Bilder
num_cases = 80
random_sample = True
random_seed = 42

# Prediction wirklich neu laufen lassen?
run_prediction = True

thumbnail_size = 230

# ============================================================
# nnU-Net Environment
# ============================================================

os.environ["nnUNet_raw"] = str(project_root / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(project_root / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(project_root / "nnUNet_results")

review_root.mkdir(parents=True, exist_ok=True)
review_input.mkdir(parents=True, exist_ok=True)
review_output.mkdir(parents=True, exist_ok=True)
thumb_dir.mkdir(parents=True, exist_ok=True)


def case_sort_key(path):
    case_id = path.stem.replace("_0000", "")
    return int(case_id) if case_id.isdigit() else case_id


def normalize_image(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr).astype(np.float32)

    lo, hi = np.percentile(arr, [1, 99])

    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0, 1)

    return (arr * 255).astype(np.uint8)


def label_to_rgb(label):
    label = np.asarray(label)
    label = np.squeeze(label)

    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Klasse 1 = Skeleton
    rgb[label == 1] = [255, 80, 60]

    # Klasse 2 = Soma
    rgb[label == 2] = [180, 60, 255]

    # Falls andere Werte vorkommen
    rgb[label > 2] = [255, 220, 0]

    return rgb


def make_overlay(image_u8, label):
    image_u8 = np.asarray(image_u8)
    label = np.asarray(label)
    label = np.squeeze(label)

    rgb = np.stack([image_u8, image_u8, image_u8], axis=-1)
    overlay = rgb.copy()

    # Skeleton rot
    mask1 = label == 1
    overlay[mask1, 0] = 255
    overlay[mask1, 1] = (overlay[mask1, 1] * 0.25).astype(np.uint8)
    overlay[mask1, 2] = (overlay[mask1, 2] * 0.25).astype(np.uint8)

    # Soma lila
    mask2 = label == 2
    overlay[mask2, 0] = 180
    overlay[mask2, 1] = 40
    overlay[mask2, 2] = 255

    return overlay


def resize_on_canvas(arr, size):
    img = Image.fromarray(arr)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((size, size))

    canvas = Image.new("RGB", (size, size), "white")

    x = (size - img.width) // 2
    y = (size - img.height) // 2

    canvas.paste(img, (x, y))

    return canvas


def foreground_fraction(label, value):
    return float((label == value).mean())


def make_case_panel(case_id, image_path, gt_path, pred_path):
    image = tifffile.imread(image_path)
    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)

    image = np.squeeze(image)
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

    if image.shape != gt.shape:
        raise RuntimeError(f"Image und GT haben unterschiedliche Form: {image.shape} vs {gt.shape}")

    if image.shape != pred.shape:
        raise RuntimeError(f"Image und Prediction haben unterschiedliche Form: {image.shape} vs {pred.shape}")

    image_u8 = normalize_image(image)

    gt_rgb = label_to_rgb(gt)
    pred_rgb = label_to_rgb(pred)

    gt_overlay = make_overlay(image_u8, gt)
    pred_overlay = make_overlay(image_u8, pred)

    panels = [
        ("Original", image_u8),
        ("Manual Label", gt_rgb),
        ("Prediction", pred_rgb),
        ("GT Overlay", gt_overlay),
        ("Pred Overlay", pred_overlay),
    ]

    w = thumbnail_size * len(panels)
    h = thumbnail_size + 36

    combined = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(combined)

    for i, (title, arr) in enumerate(panels):
        x = i * thumbnail_size
        draw.text((x + 8, 8), title, fill="black")
        thumb = resize_on_canvas(arr, thumbnail_size)
        combined.paste(thumb, (x, 36))

    out_path = thumb_dir / f"{case_id}.jpg"
    combined.save(out_path, quality=92)

    gt_vals = np.unique(gt).tolist()
    pred_vals = np.unique(pred).tolist()

    return {
        "case_id": case_id,
        "thumb_name": out_path.name,
        "image_shape": tuple(image.shape),
        "gt_shape": tuple(gt.shape),
        "pred_shape": tuple(pred.shape),
        "gt_values": gt_vals,
        "pred_values": pred_vals,
        "gt_skeleton_fraction": foreground_fraction(gt, 1),
        "gt_soma_fraction": foreground_fraction(gt, 2),
        "pred_skeleton_fraction": foreground_fraction(pred, 1),
        "pred_soma_fraction": foreground_fraction(pred, 2),
    }


# ============================================================
# Dateien prüfen
# ============================================================

model_folder = (
    project_root
    / "nnUNet_results"
    / dataset_name
    / "nnUNetTrainer__nnUNetPlans__2d"
)

checkpoint_path = model_folder / f"fold_{fold}" / checkpoint_name

print("Dataset:")
print(project_root / "nnUNet_raw" / dataset_name)
print()
print("Model folder:")
print(model_folder)
print()
print("Checkpoint:")
print(checkpoint_path)
print()

if not images_dir.exists():
    raise RuntimeError(f"images_dir existiert nicht: {images_dir}")

if not labels_dir.exists():
    raise RuntimeError(f"labels_dir existiert nicht: {labels_dir}")

if not checkpoint_path.exists():
    raise RuntimeError(f"Checkpoint existiert nicht: {checkpoint_path}")


# ============================================================
# Fälle auswählen
# ============================================================

image_files = [
    p for p in images_dir.iterdir()
    if p.is_file()
    and p.suffix.lower() in [".tif", ".tiff"]
    and p.stem.endswith("_0000")
]

image_files = sorted(image_files, key=case_sort_key)

print(f"Gefundene Trainingsbilder: {len(image_files)}")

if len(image_files) == 0:
    raise RuntimeError(f"Keine Bilder gefunden in: {images_dir}")

if random_sample:
    random.seed(random_seed)
    selected = random.sample(image_files, min(num_cases, len(image_files)))
    selected = sorted(selected, key=case_sort_key)
else:
    selected = image_files[:num_cases]

print(f"Ausgewählte Fälle: {len(selected)}")

# alten Review-Input leeren
for p in review_input.glob("*"):
    if p.is_file():
        p.unlink()

# Bilder für Prediction kopieren
selected_case_ids = []

for img_path in selected:
    case_id = img_path.stem.replace("_0000", "")
    selected_case_ids.append(case_id)

    target = review_input / img_path.name
    shutil.copy2(img_path, target)

print(f"Review-Input: {review_input}")


# ============================================================
# Prediction laufen lassen
# ============================================================

if run_prediction:
    print()
    print("Starte nnU-Net Prediction...")

    # alten Output leeren
    for p in review_output.glob("*"):
        if p.is_file():
            p.unlink()

    code = f"""
import sys
from nnunetv2.inference.predict_from_raw_data import predict_entry_point

sys.argv = [
    'nnUNetv2_predict',
    '-i', r'{review_input}',
    '-o', r'{review_output}',
    '-d', '{dataset_id}',
    '-c', '{configuration}',
    '-f', '{fold}',
    '-chk', '{checkpoint_name}',
    '-device', '{device}'
]

predict_entry_point()
"""

    subprocess.run([sys.executable, "-c", code], check=True)

    print("Prediction fertig.")


# ============================================================
# HTML-Galerie bauen
# ============================================================

rows = []
problems = []

for case_id in selected_case_ids:
    image_path = review_input / f"{case_id}_0000.tif"
    gt_path = labels_dir / f"{case_id}.tif"
    pred_path = review_output / f"{case_id}.tif"

    if not gt_path.exists():
        problems.append((case_id, "Manual label fehlt"))
        continue

    if not pred_path.exists():
        problems.append((case_id, "Prediction fehlt"))
        continue

    try:
        info = make_case_panel(case_id, image_path, gt_path, pred_path)
        rows.append(info)
    except Exception as e:
        problems.append((case_id, str(e)))


with open(html_path, "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Skeleton + Soma nnU-Net Review</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 20px;
}
.legend {
    padding: 10px;
    background: #f1f1f1;
    border-radius: 8px;
    margin-bottom: 16px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(1180px, 1fr));
    gap: 18px;
}
.card {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 10px;
    background: #fafafa;
}
.card img {
    width: 1150px;
    max-width: 100%;
    border: 1px solid #ddd;
}
.meta {
    font-size: 12px;
    color: #444;
    line-height: 1.4;
}
.problem {
    color: #b00020;
    font-weight: bold;
}
.skeleton {
    color: rgb(220, 30, 20);
    font-weight: bold;
}
.soma {
    color: rgb(150, 40, 220);
    font-weight: bold;
}
</style>
</head>
<body>
<h1>Skeleton + Soma nnU-Net Review</h1>

<div class="legend">
<b>Ansichten:</b> Original | Manual Label | Prediction | GT Overlay | Pred Overlay<br>
<span class="skeleton">Skeleton = Klasse 1, rot</span><br>
<span class="soma">Soma = Klasse 2, lila</span>
</div>
""")

    f.write(f"<p>Geprüfte Fälle: {len(rows)}</p>\n")

    if problems:
        f.write("<h2>Probleme</h2><ul>\n")
        for case_id, msg in problems:
            f.write(f"<li class='problem'>{html.escape(case_id)}: {html.escape(msg)}</li>\n")
        f.write("</ul>\n")

    f.write("<div class='grid'>\n")

    for info in rows:
        f.write("<div class='card'>\n")
        f.write(f"<h2>{html.escape(info['case_id'])}</h2>\n")
        f.write(f"<img src='thumbs/{html.escape(info['thumb_name'])}'>\n")
        f.write("<div class='meta'>\n")
        f.write(f"Image shape: {info['image_shape']}<br>\n")
        f.write(f"GT values: {info['gt_values']}<br>\n")
        f.write(f"Pred values: {info['pred_values']}<br>\n")
        f.write(f"GT skeleton fraction: {info['gt_skeleton_fraction']:.5f}<br>\n")
        f.write(f"GT soma fraction: {info['gt_soma_fraction']:.5f}<br>\n")
        f.write(f"Pred skeleton fraction: {info['pred_skeleton_fraction']:.5f}<br>\n")
        f.write(f"Pred soma fraction: {info['pred_soma_fraction']:.5f}<br>\n")
        f.write("</div>\n")
        f.write("</div>\n")

    f.write("</div>\n</body>\n</html>\n")

print()
print("Fertig.")
print("HTML öffnen:")
print(html_path)