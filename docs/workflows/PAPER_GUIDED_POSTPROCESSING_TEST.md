# Paper-guided, no-loss test workflow

[Zur Dokumentationsübersicht](../README.md)

This is a separate test pipeline for predictions from the Dataset136
`nnUNetTrainerSkeletonRecallCells` model. It does **not** overwrite the prior
workflow.

## Central material rule

At every connection stage the postprocessor enforces:

```text
completed_skeleton = input_skeleton OR accepted_new_connections
```

No pre-existing skeleton pixel is deleted. The full binary material mask is
also carried to the next pass independently of the 0/1/2 semantic TIFF, so a
connection entering a soma remains represented in the material map even though
the semantic TIFF correctly renders that overlap as class `2 = soma`.
Skeletonization is used only inside graph analysis, never as the final material
mask. This guarantee applies to skeleton material; tiny soma candidates are
intentionally excluded as invalid *seeds* by the soma-preparation stage.

## New files

| File | Purpose |
| --- | --- |
| `build_neurite_ridge_evidence.py` | Tiled multi-scale Hessian/Frangi evidence map from the raw image. This is positive route evidence, not a new segmentation. |
| `prepare_somas_paper_guided.py` | Keeps only sufficiently large soma seeds, fills enclosed holes, and performs very local intensity-supported recovery without crossing skeleton or joining soma seeds. |
| `quality_gate_cells_for_evo.py` | Separates selected cells into `safe`, `review`, and `excluded` selection maps. It never edits the semantic segmentation. |
| `run_paper_guided_skeleton_to_evo.ps1` | Runs the stages in a fresh versioned result folder. |

`postprocess_net129_no_loss.py` was extended with:

```text
--ridge-evidence <combined evidence TIFF>
--ridge-weight 0.35
```

The raw image remains the QC background and the ridge map contributes only to
the path cost.

## Recommended first run: stop after the short, strict pass

Run from the project folder in an activated virtual environment:

```powershell
cd C:\Ole\20260721_CellClassification_v2
.\.venv\Scripts\Activate.ps1

.\run_paper_guided_skeleton_to_evo.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\YYYYMMDD_paper_guided_test" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\PATH_TO_INPUT_IMAGES" `
  -Device cuda `
  -StopAfterShortGapPass
```

Input images must have names such as:

```text
Bleb_0000.tif
DMSO_0000.tif
```

If Dataset136 inference already exists and the folder contains both
`<case>.tif` and `<case>.npz`, reuse it rather than predicting again:

```powershell
.\run_paper_guided_skeleton_to_evo.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\YYYYMMDD_paper_guided_test" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\PATH_TO_INPUT_IMAGES" `
  -ExistingPredictionDir "C:\Ole\20260721_CellClassification_v2\PATH_TO_EXISTING_PREDICTIONS" `
  -StopAfterShortGapPass
```

Inspect these files for every case before continuing:

```text
02_ridge_hessian_evidence\<case>\05_ridge_evidence_qc.png
03_paper_guided_soma_preparation\<case>\09_soma_preparation_qc.png
04_gap_pass_1_short_high_precision\<case>\14_qc_overlay.png
04_gap_pass_1_short_high_precision\<case>\02_added_connections_accepted.tif
```

## Continue with the medium pass

Use the same `RunRoot`. Completed stages are detected and are not recomputed.

```powershell
.\run_paper_guided_skeleton_to_evo.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\YYYYMMDD_paper_guided_test" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\PATH_TO_INPUT_IMAGES" `
  -Device cuda `
  -AllowExistingRunRoot `
  -StopAfterMediumGapPass
```

Inspect:

```text
05_gap_pass_2_medium\<case>\14_qc_overlay.png
05_gap_pass_2_medium\<case>\02_added_connections_accepted.tif
05_gap_pass_2_medium\<case>\03_competing_connections_ambiguous.tif
```

## Continue with the long, strongly aligned rescue pass

```powershell
.\run_paper_guided_skeleton_to_evo.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\YYYYMMDD_paper_guided_test" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\PATH_TO_INPUT_IMAGES" `
  -Device cuda `
  -AllowExistingRunRoot `
  -StopAfterLongGapPass
```

The final no-loss semantic file is:

```text
06_gap_pass_3_long_aligned\<case>\15_completed_skeleton_and_soma_0-1-2.tif
```

Inspect its `14_qc_overlay.png`, particularly yellow (accepted) and blue
(competing/ambiguous) connections. If a stage creates unacceptable false links,
use a new `RunRoot` and tune that pass rather than continuing from it.

## Continue through cell assignment and crop generation

The default selection is `matrix_forest_inspired`, which is the closest of the
existing implementations to soma-seeded NeuronCyto-II-style propagation. The
workflow still produces `topology_baseline`, `matrix_forest_inspired`, and
`gcut_inspired` outputs for comparison.

```powershell
.\run_paper_guided_skeleton_to_evo.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\YYYYMMDD_paper_guided_test" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\PATH_TO_INPUT_IMAGES" `
  -Device cuda `
  -AllowExistingRunRoot `
  -AssignmentMethod matrix_forest_inspired
```

The outputs to inspect are:

```text
07_cell_assignment_comparison\<case>\comparison.html
08_overlap_representative_selection\<case>\07_qc_overlap_representatives.png
09_cell_quality_gate\<case>\06_cell_quality_gate_qc.png
09_cell_quality_gate\<case>\manual_review_template.csv
10_evo_cells_safe\<case>\cell####\
11_evo_crop_overviews_safe\<case>\all_cells_overview.png
```

`01_cells_safe_for_evo.tif` is the only map automatically exported to Evo cell
folders. `02_cells_for_manual_review.tif` is deliberately kept separate so
uncertain cells do not silently enter the downstream analysis. This review map
also includes candidates that were not selected as the one representative of
an overlap group; they appear in `manual_review_template.csv` rather than
silently disappearing.

To produce the Fiji/SNT files (`seg.traces`, `soma.zip`, `bounds.zip`, and
`locations.zip`) only after you approve the crops, rerun the last command with
`-RunFiji`.

## Key connection parameters

| Stage | Search range | Purpose |
| --- | --- | --- |
| Short pass | endpoint 14 px; soma 12 px | High-precision closure of visibly tiny gaps; geometry rescue disabled. |
| Medium pass | endpoint 28 px; soma 20 px | Evidence-guided closure of medium curved or straight gaps; geometry rescue still disabled. |
| Long pass | endpoint 42 px; geometry rescue up to 78 px | Only strongly aligned, low-detour connections with ridge/raw support. |

The most important tuning values are:

```text
RidgeEvidenceWeight     higher = evidence map matters more than raw intensity
--min-score             higher = fewer accepted paths
--min-image-support     higher = a path needs clearer raw/ridge signal
--ambiguity-margin      higher = more candidates are visibly marked ambiguous
--geometry-rescue-cosine higher = long links must be straighter/more aligned
```

For a conservative first test, keep the defaults. Do not increase all radii at
once: inspect the newly added material of one stage before changing the next.
