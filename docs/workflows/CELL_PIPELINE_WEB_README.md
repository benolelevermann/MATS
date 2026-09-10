# Single Cell Extractor - local web pipeline

[Zur Dokumentationsübersicht](../README.md)

The local website runs the deliberately small processing chain requested for new overview images:

1. Optional visual preselection from the prepared MICA C3-C10 FOV library. Good example FOVs can
   generate six more non-overlapping random FOVs from the same Leica overview before inference.
2. Dataset 138 inference with `nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x`.
3. Adaptive hysteresis of the skeleton probability map. No soma filling and no no-loss gap completion.
4. Direct export of connected components that contain exactly one valid soma.
5. NeuroTreeTracer tree extraction only for connected components containing multiple somas. The tracer
   receives a temporary 3 px dilation of the skeleton as its search body; that dilation is never exported.
6. Atomic rejection of an entire conflict group if all somas cannot be separated safely.
7. One labelled cell per crop, optional Fiji/SNT finalization, validation, ZIP archive and browser gallery.
8. Non-destructive manual crop adjustment plus one final-result decision per cell while Dataset 138,
   hysteresis and the isolated single-cell mask remain visible next to each other.

## Start

In PowerShell, from this project folder:

```powershell
.\run_cell_pipeline_web.ps1
```

The page opens at `http://127.0.0.1:8765`. Runs and all uploaded/generated files are kept under
`web_pipeline_runs/<job-id>/`. Processing is strictly sequential so two GPU/NeuroTreeTracer jobs
cannot exhaust memory at the same time.

Manually uploaded TIFFs have priority over FOV-library batch jobs. A one-off image therefore runs
next even when a large FOV selection is already queued; the background FOV batch remains saved and
continues afterward.

For a quick UI or CPU-only check:

```powershell
.\run_cell_pipeline_web.ps1 -Device cpu -NoFiji
```

`-NoFiji` produces cell folders with TIFF, CSV and SWC files, but without the final SNT/ROI files
required by the strict Evo validator.

For training-data curation only, Fiji finalization is unnecessary:

```powershell
.\run_cell_pipeline_web.ps1 -NoFiji
```

For Dataset 141 there are two deliberately separate launchers:

```powershell
.\run_net141_training_data_pipeline.ps1  # good -> review_dataset/approved
.\run_net141_evo_pipeline.ps1            # good -> evo_pipeline_input_net141/cells
```

The Evo launcher always performs Fiji/SNT finalization. Its right-arrow decision copies the
complete validated cell folder (`raw.tif`, masks, SWC, `seg.traces`, `soma.zip`, `bounds.zip`)
into a sequential `cell######` directory. It never writes to `review_dataset/approved`. Manual
recropping is disabled in this mode because changing the pixel geometry would invalidate the
already finalized trace and ROI files.

If port 8765 is already occupied by an older instance, choose another local port:

```powershell
.\run_cell_pipeline_web.ps1 -NoFiji -Port 8766
```

Queued or interrupted analyses can be marked permanently as cancelled after stopping the server:

```powershell
.\.venv\Scripts\python.exe .\cancel_cell_pipeline_jobs.py
```

This changes only jobs whose state is `queued` or `running`. Completed reviews and the global
`review_dataset/approved` training pairs are not modified, and cancelled jobs are not resumed on
the next server start.

## MICA FOV preselection

When `fov_manifest.csv` exists below the configured `-FovRoot`, the start page shows a filterable
gallery of all prepared FOVs. Filters are available for DIV, plate and well. Selections remain in
the browser on the same machine until they are cleared. Clicking **Auswahl analysieren** copies the
selected TIFFs into separate jobs; the existing single-worker queue processes them one after the
other so GPU memory cannot be exhausted by concurrent predictions.

For a second sampling round, mark one or more good example FOVs and click
**6 weitere je Übersicht**. The examples are grouped by source overview, so selecting two examples
from the same well still creates only six new fields for that overview. Existing coordinates are
excluded, new TIFFs and manifest rows are added without overwriting the first three FOVs, and the
new browser previews are marked **NEU**. They can then be selected normally for analysis. Repeating
the action creates the next non-overlapping sampling round.

The default library is:

```text
X:\Niro\04_Raw_Data\InVitro\mica\250317_MCS24GFP\C3-C10_training_expansion
```

Browser previews are prepared once with:

```powershell
.\.venv\Scripts\python.exe -m cell_pipeline_web.fov_library
```

## Compare fixed T_low values

`sweep_hysteresis_tlow.py` applies the same hysteresis implementation as the web pipeline several
times to one Dataset-138 probability file. `T_high` stays fixed while only `T_low` changes; the main
pipeline configuration is not modified.

Example for the current `NewImage2` run:

```powershell
.\.venv\Scripts\python.exe .\sweep_hysteresis_tlow.py `
  --probabilities .\web_pipeline_runs\20260824_214604_51a75d31\02_prediction_dataset138\NewImage2.npz `
  --raw .\web_pipeline_runs\20260824_214604_51a75d31\01_input\NewImage2_0000.tif `
  --output-dir .\web_pipeline_runs\20260824_214604_51a75d31\06_tlow_sweep `
  --t-high 0.60 `
  --t-low 0.30 0.25 0.20 0.15 `
  --overwrite
```

The output root contains `tlow_sweep_report.html`, `tlow_sweep.csv` and a JSON summary. Every
`tlow_*` directory contains the semantic 0/1/2 TIFF, skeleton TIFF, added/removed-pixel masks and a
PNG overview. The report additionally shows one fixed 320×320 crop per valid soma: raw image and
all requested `T_low` variants appear directly next to each other. The crop size can be changed with
`--cell-crop-size`. Filters show all somas, only cells whose topology status changes, or cells that
are isolated at least once. Matching PNG files are stored under `cell_crops/soma_*/`, and
`tlow_cell_comparison.csv` contains the per-cell status for every threshold.

In the previews, retained skeleton is green/cyan, pixels newly accepted by hysteresis are yellow,
removed argmax pixels are red, the target soma is pink and other somas are purple. The report also
compares connectivity, soma attachment, isolated candidates, conflict groups and orphan
components. On the current image one full-image variant occupies about 98 MB, so start with a
small, targeted threshold list.

## Direct single-cell acceptance

The adaptive semantic map keeps labels 0 (background), 1 (skeleton) and 2 (soma). Soma components
smaller than 20 px are treated as prediction noise. The remaining soma and skeleton pixels are
grouped with 8-connectivity. A group is exported directly when it contains exactly one soma and at
least 8 skeleton pixels. Candidates touching the source border or lying within 3 px of the final crop
edge are rejected. Accepted crops are square where the source dimensions permit it, use a 48 px
margin, and are at least 128 px wide/high where the source image permits it.

## Conservative conflict acceptance

A NeuroTreeTracer conflict group is accepted only if all of these are true:

- the entire connected group fits in a square crop of at most 384 px per side;
- every input soma receives a trace;
- no trace pixel is assigned to two different somas;
- every trace is attached to its own soma and does not touch another soma;
- every soma has at least 8 unique trace pixels;
- at least 98% of trace points remain on the dilated segmented structure;
- the union of traces covers at least 60% of the original group skeleton within a 3 px tolerance.

If one check fails, no cell from that conflict group is exported. This follows the requested
"rather leave both out" behavior and reflects the ambiguity limitation described in the
NeuroTreeTracer paper.

A run in which every candidate is rejected still finishes normally with zero crops. The empty
manifest and `rejected_groups.csv` remain available in the ZIP so the rejection reasons are visible.

The 384 px limit predicts about 9.6 GB for NeuroTreeTracer's 64,800 full-crop search masks. Larger
connected conflict groups are rejected instead of risking an out-of-memory run.

## Outputs

Each accepted crop folder contains at least:

- `raw.tif`, `skeleton.tif`, `soma.tif`, `seg.tif`, `cell_mask.tif`
- `prediction.tif` (plain Dataset-138 argmax) and `postprocessed.tif` (adaptive hysteresis)
- `seg.csv`, `seg-000.swc`
- `bounds.json`, `location.json`, `metadata.json`
- `raw_preview.png`, `prediction_preview.png`, `postprocessing_preview.png`, `preview.png`

With Fiji finalization enabled it additionally contains `seg.traces`, `soma.zip` and `bounds.zip`;
the case root also receives `bounds.zip` and `locations.zip`. The browser download is
`evo_single_cells.zip`.

## Manual review: training dataset or Evo selection

The review view shows one cell at a time as Dataset-138 prediction, hysteresis result and final
isolated one-cell mask. The reviewer records one final `result`: press the right arrow for `good`
or the left arrow for `bad`. The next open cell is displayed immediately. The two old columns
`training` and `postprocessing` remain synchronized in new records so older exports stay readable;
existing two-part reviews are migrated logically without changing the previous files. Reviews
survive server restarts in:

```text
web_pipeline_runs/<job-id>/05_review/reviews.json
web_pipeline_runs/<job-id>/05_review/reviews.csv
```

**Crop anpassen** opens a canvas editor for the automatic cell crop. The yellow frame can be moved
and resized; a dashed green rectangle marks the mandatory label region. The server rejects any crop
that cuts skeleton or soma pixels or leaves less than 3 px clearance. The automatic crop is never
overwritten. The selected version is stored in `04_evo_single_cells/<cell>/curated/`, and previous
manual coordinates remain in `manual_crop_history.jsonl`.

The reviewed cell artifacts are also copied into `training/good|bad` and
`postprocessing/good|bad` below both the job review directory and the global `review_dataset`
directory. Changing a review removes the old generated copy before creating the new one.

Only cells whose final result is `good` are placed in the global ready-to-train structure:

```text
review_dataset/approved/
  imagesTr/<source-FOV>__<job-id>__<cell>_0000.tif  # raw curated crop
  labelsTr/<source-FOV>__<job-id>__<cell>.tif       # curated 0/1/2 label
  skeletons/<source-FOV>__<job-id>__<cell>.tif
  somas/<source-FOV>__<job-id>__<cell>.tif
  metadata/<source-FOV>__<job-id>__<cell>.json
  dataset.json
```

`review_dataset/review_index.csv` is the cross-run decision index. The website button
**Freigegebene Trainingszellen** builds a ZIP of the `approved` directory on demand.

In Evo review mode the same arrow keys mean **discard** and **use for Evo**. Approved cells are
kept separately in:

```text
evo_pipeline_input_net141/
  cells/<source-image-name>/cell000001/...
  selected_cells.csv
  selection_summary.json
  evo_selected_cells.zip
```

Each input TIFF receives its own directory below `cells`. Use one such image
directory directly as the cell-folder input of the Evo/R pipeline. This keeps
cells from different overview images separate during feature extraction and
comparison.

## Manuell getracte ROCKi-Zellen prüfen

`prepare_rocki_training_review.ps1` liest die drei fest eingetragenen
`TracingsFelix`-Sammlungen auf `X:` ausschließlich lesend ein. `raw.tif`,
`seg.swc` und `soma.zip` werden lokal rasterisiert und erscheinen danach als
eigener fertiger Lauf **ROCKi-Tracings · manueller Trainingsdaten-Review**.

- Pfeil links verwirft den Crop.
- Pfeil rechts übernimmt ihn in `review_dataset/approved`.
- **Crop anpassen** schneidet eine zweite, nicht annotierte Zelle heraus; alle
  Skelett- und Somapixel der Zielzelle müssen mit Sicherheitsabstand erhalten bleiben.
- Quellpfad, Sammlung, FOV und ursprüngliche Zellnummer bleiben in den Metadaten
  und im späteren Dateinamen erhalten. Die Dateien auf `X:` werden nie verändert.

Nach abgeschlossenem Review erstellt `run_build_dataset139_review_expansion.ps1`
ein neues `Dataset139_dataset138_plus_reviewed_cells`. Dataset138 bleibt
unverändert. Mit `-RunPreprocess -Train -Device cuda` kann anschließend direkt
mit demselben bekannten Skeleton-Recall-Trainer trainiert werden. Der bestehende
Dataset138-Split bleibt erhalten; neue Review-Zellen werden nur dem Training
hinzugefügt.
