# Netz-129-Zellen für die R/PCA-Pipeline exportieren

Die kopierte Funktion `getMetaData_singlecell(...)` erwartet pro Zelle:

```text
cell0001/
├── raw.tif
├── seg-000.swc
├── seg.traces
├── soma.zip
└── bounds.zip
```

Der neue Export erzeugt zusätzlich `skeleton.tif`, `soma.tif`, `seg.tif`,
`cell_mask.tif`, `location.zip` und Metadaten.

## 1. Skripte in den Projektordner kopieren

Kopiere diese Dateien nach:

```text
C:\Ole\20260721_CellClassification_v2\r_pipeline
```

- `export_cells_for_r_pipeline.py`
- `finalize_cells_with_fiji.py`
- `validate_r_pipeline_cell_folders.py`

## 2. Safe-Zellen klassifizieren und Crops erzeugen

```powershell
cd C:\Ole\20260721_CellClassification_v2
.\.venv\Scripts\Activate.ps1

$original = "C:\Ole\20260721_CellClassification_v2\PFAD_ZUM_ORIGINAL\overview_max.tif"
$semantic = "C:\Ole\20260721_CellClassification_v2\PFAD_ZUM_POSTPROCESSING\15_completed_skeleton_and_soma_0-1-2.tif"
$cells = "C:\Ole\20260721_CellClassification_v2\cells_for_r_pipeline"

python .\r_pipeline\export_cells_for_r_pipeline.py `
  --original $original `
  --semantic $semantic `
  --output-dir $cells `
  --contact-radius 5 `
  --min-soma-area 20 `
  --min-skeleton-pixels 8 `
  --margin 32 `
  --min-crop-size 128
```

Die Klassifizierung verwirft kein Skeletonmaterial. Sie arbeitet mit der
fertigen No-Loss-Datei `15_completed_skeleton_and_soma_0-1-2.tif`.

Eine Zelle ist `safe`, wenn ihre Skeletonkomponenten genau einem Soma
zugeordnet werden können. Komponenten mit Kontakt zu mehreren Somata bleiben
ambig; Komponenten ohne Soma bleiben unzugeordnet.

QC-Dateien liegen anschließend in:

```text
cells_for_r_pipeline\_classification
```

Besonders wichtig:

```text
cell_classification_overlay.png
safe_cell_instances_overview.tif
ambiguous_cell_instances_overview.tif
unassigned_skeleton_overview.tif
cell_status_overview.tif
```

## 3. Echte SNT-TRACES und ImageJ-ROI-ZIPs erzeugen

Die Datei `seg.traces` darf nicht als leere Platzhalterdatei angelegt werden.
Dieses Skript importiert jedes `seg-000.swc` mit der installierten
Simple-Neurite-Tracer-Version und speichert eine echte gzip-komprimierte
TRACES-XML-Datei.

```powershell
$env:CELL_EXPORT_ROOT = $cells

& "C:\Program Files\Fiji.app\ImageJ-win64.exe" `
  --allow-multiple `
  --headless `
  --console `
  --run "C:\Ole\20260721_CellClassification_v2\r_pipeline\finalize_cells_with_fiji.py"
```

Der Lauf erzeugt in jedem `cell####`-Ordner:

```text
seg.traces
soma.zip
bounds.zip
location.zip
```

Die Fiji-Version kann alte Java-Warnungen ausgeben. Maßgeblich ist die Zeile:

```text
Finalized N cell folders
```

und die Datei:

```text
cells_for_r_pipeline\_fiji_finalize_log.txt
```

## 4. Struktur prüfen

```powershell
python .\r_pipeline\validate_r_pipeline_cell_folders.py `
  --input-dir $cells
```

Am Ende muss stehen:

```text
Failed: 0
```

## 5. In `prepare_feature_extraction.Rmd`

Benutze den Abschnitt mit:

```r
getMetaData_singlecell(...)
```

und setze:

```r
cell_folder <- "C:/Ole/20260721_CellClassification_v2/cells_for_r_pipeline"
```

Außerdem muss `stack_directory` auf einen tatsächlich existierenden Ordner
mit den zugehörigen Originalstacks zeigen. Die Funktion prüft diesen Pfad.

## Parameter

- `--contact-radius`: Zuordnung Skeleton zu Soma. Bei sichtbaren kleinen
  Kontaktlücken zuerst von `5` auf `7` erhöhen. Ein zu großer Wert kann bei
  eng benachbarten Somata mehr Ambiguitäten erzeugen.
- `--min-soma-area`: Kleinere Soma-Komponenten gelten als Rauschen und werden
  nicht als Zellkern verwendet. Die Soma-Pixel werden in der semantischen
  Eingabe nicht gelöscht.
- `--min-skeleton-pixels`: Mindestmenge zugeordneten Skeletonmaterials für
  eine PCA-Zelle. Bei sehr kurzen, aber echten Fortsätzen auf `3` bis `5`
  reduzieren.
- `--margin`: Kontext um die komplette Zelle im Crop.
- `--min-crop-size`: Mindestgröße eines Crops.
