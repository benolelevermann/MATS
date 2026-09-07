# Dataset 136: Skeleton-Recall → cell crops → Evo workflow

This is the executable workflow implemented by
`run_skeleton_recall_to_evo_workflow.ps1`.

```text
Input overview image(s), named <case>_0000.tif
        ↓
Dataset 136 skeleton-recall nnU-Net prediction
        ↓
Soma preparation: fill enclosed holes; reject tiny soma seeds
        ↓
No-loss skeleton repair: only add plausible paths, never delete input skeleton
        ↓
NeuronCyto-II-inspired soma-seeded instance assignment
        ↓
For every multi-cell overlap group: retain one eligible representative cell
        ↓
Per-cell Evo crop folders → optional Fiji/SNT finalization → Evo R pipeline
```

## Required input layout

Create a new run directory and put the overview images in `01_inputimages`.
Every input filename needs the nnU-Net channel suffix `_0000.tif`.

```text
C:\Ole\20260721_CellClassification_v2\20260812_my_run\
└── 01_inputimages\
    ├── Bleb_0000.tif
    └── DMSO_0000.tif
```

## Phase 1 — prediction through cell-assignment QC

Open PowerShell and run:

```powershell
cd C:\Ole\20260721_CellClassification_v2
Set-ExecutionPolicy -Scope Process Bypass

.\run_skeleton_recall_to_evo_workflow.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\20260812_my_run" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\20260812_my_run\01_inputimages" `
  -Device cuda `
  -AssignmentMethod matrix_forest_inspired `
  -StopAfterAssignment
```

This executes Dataset 136, soma preparation, no-loss gap completion and all
three assignment variants. It then stops deliberately before any cells are
exported.

For every case, inspect:

```text
05_neurocytoII_inspired_assignment\<case>\comparison.html
```

`matrix_forest_inspired` is the closest conceptual match to NeuronCyto II:
soma seeds propagate through the skeleton graph. `gcut_inspired` remains an
alternative to compare, and `topology_baseline` is the conservative reference.

## Phase 2 — select one cell per ambiguous overlap group

After choosing the method, rerun the same workflow. The completed stages are
recognized and skipped automatically; these flags make that explicit:

```powershell
.\run_skeleton_recall_to_evo_workflow.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\20260812_my_run" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\20260812_my_run\01_inputimages" `
  -Device cuda `
  -AssignmentMethod matrix_forest_inspired `
  -SkipPrediction `
  -SkipSomaPreparation `
  -SkipGapCompletion `
  -SkipAssignment `
  -StopAfterRepresentativeSelection
```

Inspect the representative-selection QC image:

```text
06_overlap_representative_selection\<case>\07_qc_overlap_representatives.png
```

Interpretation:

- coloured material: retained crop candidates;
- blue: a representative selected from an overlap group;
- orange: a competing cell not exported from that overlap group;
- cyan: unresolved ambiguity; it is not forced into a crop;
- magenta: soma material; grey: source skeleton material.

The selection is intentionally separate from the final segmentation: it does
not delete or reclassify skeleton pixels in the complete 0/1/2 mask.

## Phase 3 — export crops and create Fiji/SNT artifacts

When the selection QC looks acceptable, run:

```powershell
.\run_skeleton_recall_to_evo_workflow.ps1 `
  -RunRoot "C:\Ole\20260721_CellClassification_v2\20260812_my_run" `
  -InputDir "C:\Ole\20260721_CellClassification_v2\20260812_my_run\01_inputimages" `
  -Device cuda `
  -AssignmentMethod matrix_forest_inspired `
  -SkipPrediction `
  -SkipSomaPreparation `
  -SkipGapCompletion `
  -SkipAssignment `
  -SkipRepresentativeSelection `
  -RunFiji
```

This creates crop folders at:

```text
07_evo_cells_matrix_forest_inspired\<case>\
├── cell0001\
│   ├── raw.tif
│   ├── skeleton.tif
│   ├── soma.tif
│   ├── seg.tif
│   ├── seg.traces
│   ├── soma.zip
│   ├── bounds.zip
│   ├── location.zip
│   └── ...
└── bounds.zip / locations.zip
```

The crop overview is written to:

```text
08_evo_crop_overviews_matrix_forest_inspired\<case>\all_cells_overview.png
```

The final Fiji validation runs automatically. If it succeeds, the cell folders
can be used with the normal Evo R metadata and feature-extraction scripts.

## Important parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `-MinSomaArea` | 80 px | Smaller predicted soma components are not used as cell seeds. |
| `-MaxFillHoleArea` | 2500 px | Largest enclosed hole that may be filled inside one soma. |
| `-MaxFillRatio` | 1.5 | Filled hole may not exceed 150% of that soma's original area. |
| `-EndpointGap` | 24 px | Endpoint-to-endpoint repair range. |
| `-SomaGap` | 18 px | Endpoint-to-soma repair range. |
| `-SegmentGap` | 16 px | Endpoint-to-nearby-skeleton repair range. |
| `-GeometryRescueGap` | 60 px | Longer, strongly aligned and image-supported repair search. |

The no-loss postprocessing output is always the authoritative material mask:

```text
04_conservative_no_loss_postprocessing\<case>\04_completed_skeleton_fullwidth_NO_LOSS.tif
```

It obeys exactly:

```text
completed_skeleton = original_skeleton OR accepted_added_connections
```

No downstream selection/cropping output should be mistaken for the full
material-preserving segmentation; those files are intentionally cell-specific
subsets for export.
