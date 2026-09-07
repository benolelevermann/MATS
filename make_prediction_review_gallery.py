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

dataset_id = "128"
dataset_name = "Dataset128_invitrotracings"
configuration = "2d"
fold = "0"

# Für fertige Modelle am besten checkpoint_best.pth
checkpoint_name = "checkpoint_best.pth"

device = "cuda"

images_dir = project_root / "nnUNet_raw" / dataset_name / "imagesTr"
labels_dir = project_root / "nnUNet_raw" / dataset_name / "labelsTr"

review_root = project_root / "prediction_review"
review_input = review_root / "input_images"
review_output = review_root / "predictions"
thumb_dir = review_root / "thumbs"
html_path = review_root / "review.html"

# Wie viele Fälle anschauen?
num_cases = 50

# Zufällige Auswahl oder erste N Bilder?
random_sample = True
random_seed = 42

# Wenn Prediction schon existiert, kann man das auf False setzen
run_prediction = True

thumbnail_size = 220

# ============================================================
# nnU-Net ENVIRONMENT SETZEN
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

    # Klasse 1 = Skeleton/Zelle
    rgb[label == 1] = [255, 255, 255]

    # Klasse 2 = Soma, falls vorhanden
    rgb[label == 2] = [180, 80, 255]

    # Alles andere > 2 gelb markieren
    rgb[label > 2] = [255, 220, 0]

    return rgb


def make_overlay(image_u8, label):
    image_u8 = np.asarray(image_u8)
    label = np.asarray(label)
    label = np.squeeze(label)

    rgb = np.stack([image_u8, image_u8, image_u8], axis=-1)

    overlay = rgb.copy()

    # Klasse 1 rot
    mask1 = label == 1
    overlay[mask1, 0] = 255
    overlay[mask1, 1] = (overlay[mask1, 1] * 0.25).astype(np.uint8)
    overlay[mask1, 2] = (overlay[mask1, 2] * 0.25).astype(np.uint8)

    # Klasse 2 lila
    mask2 = label == 2
    overlay[mask2, 0] = 180
    overlay[mask2, 1] = 60
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


def make_case_panel(case_id, image_path, gt_path, pred_path):
    image = tifffile.imread(image_path)
    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)

    image = np.squeeze(image)
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

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
    h = thumbnail_size + 34

    combined = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(combined)

    for i, (title, arr) in enumerate(panels):
        x = i * thumbnail_size
        draw.text((x + 8, 8), title, fill="black")
        thumb = resize_on_canvas(arr, thumbnail_size)
        combined.paste(thumb, (x, 34))

    out_path = thumb_dir / f"{case_id}.jpg"
    combined.save(out_path, quality=90)

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
        "gt_fraction": float((gt > 0).mean()),
        "pred_fraction": float((pred > 0).mean()),
    }


# ============================================================
# FÄLLE AUSWÄHLEN
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
# PREDICTION LAUFEN LASSEN
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
# HTML-GALERIE ERSTELLEN
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
<title>nnU-Net Prediction Review</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 20px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(1150px, 1fr));
    gap: 18px;
}
.card {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 10px;
    background: #fafafa;
}
.card img {
    width: 1100px;
    max-width: 100%;
    border: 1px solid #ddd;
}
.meta {
    font-size: 12px;
    color: #444;
}
.problem {
    color: #b00020;
    font-weight: bold;
}
</style>
</head>
<body>
<h1>nnU-Net Prediction Review</h1>
<p>Original | Manual Label | Prediction | GT Overlay | Pred Overlay</p>
""")

    f.write(f"<p>Fälle: {len(rows)}</p>\n")

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
        f.write(f"GT shape: {info['gt_shape']}<br>\n")
        f.write(f"Pred shape: {info['pred_shape']}<br>\n")
        f.write(f"GT values: {info['gt_values']}<br>\n")
        f.write(f"Pred values: {info['pred_values']}<br>\n")
        f.write(f"GT foreground fraction: {info['gt_fraction']:.5f}<br>\n")
        f.write(f"Pred foreground fraction: {info['pred_fraction']:.5f}<br>\n")
        f.write("</div>\n")
        f.write("</div>\n")

    f.write("</div>\n</body>\n</html>\n")

print()
print("Fertig.")
print(f"HTML öffnen:")
print(html_path)