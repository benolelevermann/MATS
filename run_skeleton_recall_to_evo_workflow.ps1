<#
End-to-end workflow for Dataset136 skeleton-recall predictions.

It keeps stages separate and resume-safe:
  02 prediction -> 03 soma preparation -> 04 no-loss gap completion
  -> 05 NeuronCyto-II-inspired assignment -> 06 representative overlap selection
  -> 07 Evo-compatible crops -> optional Fiji finalization.

The "matrix_forest_inspired" assignment is an adaptation of the NeuronCyto II
idea (soma-seeded graph propagation), not a claim to be the original software.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,

    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [int]$DatasetId = 136,
    [string]$Configuration = "2d",
    [int]$Fold = 0,
    [string]$Trainer = "nnUNetTrainerSkeletonRecallCells",
    [string]$Plans = "nnUNetPlans",
    [string]$Checkpoint = "checkpoint_best.pth",

    [ValidateSet("topology_baseline", "matrix_forest_inspired", "gcut_inspired")]
    [string]$AssignmentMethod = "matrix_forest_inspired",

    [int]$MinSomaArea = 80,
    [int]$MaxFillHoleArea = 2500,
    [double]$MaxFillRatio = 1.5,
    [int]$EndpointGap = 24,
    [int]$SomaGap = 18,
    [int]$SegmentGap = 16,
    [int]$GeometryRescueGap = 60,

    # Adaptive hysteresis thresholding of the skeleton class instead of nnU-Net's argmax.
    # Method: Du et al., Comput Biol Med 153:106416 (2023), section 3.3. Otsu on the skeleton
    # probability map yields T; T+ and T- bracket the ambiguous band by cutting off the
    # fraction HysteresisAlpha of the between-class variance mass on either side. Skeleton
    # pixels above T+ seed, pixels above T- are kept where 8-connected to a seed.
    # Measured on 17 images: components -35 percent (median), skeleton +9.2 percent,
    # skeleton attached to a soma +3.3 percentage points. Soma class is untouched.
    [switch]$AdaptiveHysteresis,
    [double]$HysteresisAlpha = 0.33333333,

    [switch]$ForcePrediction,
    [switch]$SkipPrediction,
    [switch]$SkipSomaPreparation,
    [switch]$SkipGapCompletion,
    [switch]$SkipAssignment,
    [switch]$SkipRepresentativeSelection,
    [switch]$SkipCropExport,
    [switch]$StopAfterAssignment,
    [switch]$StopAfterRepresentativeSelection,
    [switch]$RunFiji
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($ForcePrediction -and $SkipPrediction) {
    throw "Use either -ForcePrediction or -SkipPrediction, not both."
}

$Project = "C:\Ole\20260721_CellClassification_v2"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Predictor = Join-Path $Project ".venv\Scripts\nnUNetv2_predict.exe"
$SomaPreparation = Join-Path $Project "prepare_somas_for_assignment.py"
$NoLossPostprocess = Join-Path $Project "postprocess_net129_no_loss.py"
$AssignmentScript = Join-Path $Project "compare_cell_assignment_methods.py"
$RepresentativeSelector = Join-Path $Project "select_overlap_representative_cells.py"
$AssignedExporter = Join-Path $Project "export_assigned_cells_for_evo.py"
$ExistingExporter = Join-Path $Project "r_pipeline\export_cells_for_r_pipeline.py"
$CropOverview = Join-Path $Project "make_cells_overview.py"
$HysteresisScript = Join-Path $Project "apply_adaptive_hysteresis.py"
$Finalizer = Join-Path $Project "r_pipeline\finalize_cells_with_fiji.py"
$Validator = Join-Path $Project "r_pipeline\validate_r_pipeline_cell_folders.py"
$Fiji = "C:\Program Files\Fiji.app\ImageJ-win64.exe"

$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$InputDir = [System.IO.Path]::GetFullPath($InputDir)
$PredictionRoot = Join-Path $RunRoot "02_predictions_skeleton_recall"
$HysteresisRoot = Join-Path $RunRoot "02b_adaptive_hysteresis"
$SomaRoot = Join-Path $RunRoot "03_soma_preparation"
$PostprocessRoot = Join-Path $RunRoot "04_conservative_no_loss_postprocessing"
$AssignmentRoot = Join-Path $RunRoot "05_neurocytoII_inspired_assignment"
$RepresentativeRoot = Join-Path $RunRoot "06_overlap_representative_selection"
$CellRoot = Join-Path $RunRoot "07_evo_cells_$AssignmentMethod"
$GalleryRoot = Join-Path $RunRoot "08_evo_crop_overviews_$AssignmentMethod"

$env:nnUNet_raw = Join-Path $Project "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Project "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Project "nnUNet_results"
Remove-Item Env:nnUNet_extTrainer -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

function Assert-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}

function Assert-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required directory not found: $Path"
    }
}

function Invoke-Python([string]$Label, [string[]]$Arguments, [string]$ExpectedFile) {
    if ($ExpectedFile -and (Test-Path -LiteralPath $ExpectedFile -PathType Leaf)) {
        Write-Host "=== ${Label}: already complete; skipped ===" -ForegroundColor DarkGray
        return
    }
    Write-Host ""; Write-Host "=== $Label ===" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit code $LASTEXITCODE)." }
    if ($ExpectedFile -and -not (Test-Path -LiteralPath $ExpectedFile -PathType Leaf)) {
        throw "$Label finished without the expected output: $ExpectedFile"
    }
}

Assert-Directory $InputDir
foreach ($file in @(
    $Python, $Predictor, $SomaPreparation, $NoLossPostprocess, $AssignmentScript,
    $RepresentativeSelector, $AssignedExporter, $ExistingExporter, $CropOverview
)) {
    Assert-File $file
}

$caseIds = @(
    Get-ChildItem -LiteralPath $InputDir -Filter "*_0000.tif" -File |
        Sort-Object Name |
        ForEach-Object { $_.BaseName -replace "_0000$", "" }
)
if ($caseIds.Count -eq 0) {
    throw "No *_0000.tif input files found in: $InputDir"
}

New-Item -ItemType Directory -Force -Path $RunRoot, $PredictionRoot | Out-Null

# Stage 1: one Dataset136 inference call handles all input overview images.
$missingPredictions = @($caseIds | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $PredictionRoot "$_.tif") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $PredictionRoot "$_.npz") -PathType Leaf)
})
if ($SkipPrediction) {
    if ($missingPredictions.Count -gt 0) {
        throw "Predictions are missing although -SkipPrediction was used: $($missingPredictions -join ', ')"
    }
} elseif ($ForcePrediction -or $missingPredictions.Count -gt 0) {
    Write-Host ""; Write-Host "=== Dataset$DatasetId skeleton-recall inference ===" -ForegroundColor Cyan
    & $Predictor `
        -i $InputDir `
        -o $PredictionRoot `
        -d $DatasetId `
        -c $Configuration `
        -f $Fold `
        -tr $Trainer `
        -p $Plans `
        -chk $Checkpoint `
        -device $Device `
        --save_probabilities
    if ($LASTEXITCODE -ne 0) { throw "Dataset$DatasetId inference failed (exit code $LASTEXITCODE)." }
} else {
    Write-Host "=== Dataset$DatasetId predictions already exist; skipped ===" -ForegroundColor DarkGray
}
foreach ($caseId in $caseIds) {
    Assert-File (Join-Path $PredictionRoot "$caseId.tif")
    Assert-File (Join-Path $PredictionRoot "$caseId.npz")
}

# Stage 1b (optional): replace argmax by adaptive hysteresis on the skeleton class.
# argmax decides each pixel independently, so a single weak pixel inside an otherwise confident
# neurite severs it. Hysteresis keeps such a pixel when it is 8-connected to confident
# skeleton, which is why it removes fragments without inflating the skeleton much.
if ($AdaptiveHysteresis) {
    Assert-File $HysteresisScript
    $hysteresisSemantic = @{}
    foreach ($caseId in $caseIds) {
        $hysteresisSemantic[$caseId] = Join-Path $HysteresisRoot "${caseId}_adaptive_hysteresis_0-1-2.tif"
    }
    $missingHysteresis = @($caseIds | Where-Object { -not (Test-Path -LiteralPath $hysteresisSemantic[$_] -PathType Leaf) })
    if ($missingHysteresis.Count -gt 0) {
        Write-Host ""; Write-Host "=== Adaptive hysteresis thresholding (alpha = $HysteresisAlpha) ===" -ForegroundColor Cyan
        $probabilityFiles = $caseIds | ForEach-Object { Join-Path $PredictionRoot "$_.npz" }
        & $Python $HysteresisScript `
            --probabilities @probabilityFiles `
            --output-dir $HysteresisRoot `
            --alpha $HysteresisAlpha `
            --write-semantic `
            --overwrite
        if ($LASTEXITCODE -ne 0) { throw "Adaptive hysteresis thresholding failed (exit code $LASTEXITCODE)." }
    } else {
        Write-Host "=== Adaptive hysteresis already complete; skipped ===" -ForegroundColor DarkGray
    }
    foreach ($caseId in $caseIds) { Assert-File $hysteresisSemantic[$caseId] }
}

foreach ($caseId in $caseIds) {
    $original = Join-Path $InputDir "${caseId}_0000.tif"
    $prediction = if ($AdaptiveHysteresis) { $hysteresisSemantic[$caseId] } else { Join-Path $PredictionRoot "$caseId.tif" }
    $probabilities = Join-Path $PredictionRoot "$caseId.npz"   # always the raw softmax, unaffected by thresholding
    $caseSomaRoot = Join-Path $SomaRoot $caseId
    $somaSemantic = Join-Path $caseSomaRoot "04_semantic_soma_filled_and_filtered_0-1-2.tif"
    $casePostprocessRoot = Join-Path $PostprocessRoot $caseId
    $completedSemantic = Join-Path $casePostprocessRoot "15_completed_skeleton_and_soma_0-1-2.tif"
    $caseAssignmentRoot = Join-Path $AssignmentRoot $caseId
    $assignmentHtml = Join-Path $caseAssignmentRoot "comparison.html"
    $methodRoot = Join-Path $caseAssignmentRoot $AssignmentMethod
    $caseRepresentativeRoot = Join-Path $RepresentativeRoot $caseId
    $representativeInstances = Join-Path $caseRepresentativeRoot "01_representative_cells_safe.tif"
    $caseCellRoot = Join-Path $CellRoot $caseId
    $exportSummary = Join-Path $caseCellRoot "export_summary.json"

    # Stage 2: fill only enclosed soma holes and exclude tiny soma seeds.
    if ($SkipSomaPreparation) {
        Assert-File $somaSemantic
    } else {
        Invoke-Python "${caseId}: soma filling and seed filtering" @(
            $SomaPreparation,
            "--prediction", $prediction,
            "--original", $original,
            "--output-dir", $caseSomaRoot,
            "--min-soma-area", "$MinSomaArea",
            "--max-fill-hole-area", "$MaxFillHoleArea",
            "--max-fill-ratio", "$MaxFillRatio",
            "--overwrite"
        ) $somaSemantic
    }

    # Stage 3: conservative, material-preserving gap completion. The filled
    # soma semantic is used as the hard input; probabilities remain those of
    # Dataset136 and guide the path cost.
    if ($SkipGapCompletion) {
        Assert-File $completedSemantic
    } else {
        Invoke-Python "${caseId}: no-loss conservative skeleton repair" @(
            $NoLossPostprocess,
            "--prediction", $somaSemantic,
            "--probabilities", $probabilities,
            "--original", $original,
            "--output-dir", $casePostprocessRoot,
            "--endpoint-gap", "$EndpointGap",
            "--soma-gap", "$SomaGap",
            "--segment-gap", "$SegmentGap",
            "--geometry-rescue-gap", "$GeometryRescueGap",
            "--geometry-rescue-cosine", "0.82",
            "--geometry-rescue-min-score", "0.30",
            "--geometry-rescue-min-support", "0.38",
            "--passes", "2",
            "--min-score", "0.40",
            "--min-path-probability", "0.12",
            "--min-image-support", "0.55",
            "--ambiguity-margin", "0.06"
        ) $completedSemantic
    }

    # Stage 4: all three assignment methods are written for transparent QC.
    # matrix_forest_inspired is the NeuronCyto-II-inspired default used below.
    if ($SkipAssignment) {
        Assert-File $assignmentHtml
    } else {
        Invoke-Python "${caseId}: NeuronCyto-II-inspired cell assignment comparison" @(
            $AssignmentScript,
            "--segmentation", $completedSemantic,
            "--original", $original,
            "--output-dir", $caseAssignmentRoot,
            "--root-radius", "3",
            "--mft-tau", "1.0",
            "--mft-seed-strength", "25",
            "--mft-min-confidence", "0.58",
            "--mft-margin", "0.12",
            "--gcut-turn-weight", "1.35",
            "--gcut-radial-weight", "0.22",
            "--gcut-margin", "0.10",
            "--safe-max-ambiguous-fraction", "0.05",
            "--safe-min-skeleton-pixels", "12",
            "--border-margin", "2",
            "--qc-max-size", "2200"
        ) $assignmentHtml
    }
}

Write-Host ""; Write-Host "=== Assignment QC is ready ===" -ForegroundColor Green
foreach ($caseId in $caseIds) {
    Write-Host "Review: $(Join-Path $AssignmentRoot "$caseId\comparison.html")"
}
if ($StopAfterAssignment) {
    Write-Host "Stopped after assignment as requested. Resume the same command without -StopAfterAssignment." -ForegroundColor Yellow
    return
}

foreach ($caseId in $caseIds) {
    $original = Join-Path $InputDir "${caseId}_0000.tif"
    $completedSemantic = Join-Path $PostprocessRoot "$caseId\15_completed_skeleton_and_soma_0-1-2.tif"
    $methodRoot = Join-Path $AssignmentRoot "$caseId\$AssignmentMethod"
    $caseRepresentativeRoot = Join-Path $RepresentativeRoot $caseId
    $representativeInstances = Join-Path $caseRepresentativeRoot "01_representative_cells_safe.tif"
    $caseCellRoot = Join-Path $CellRoot $caseId
    $exportSummary = Join-Path $caseCellRoot "export_summary.json"

    # Stage 5: in an ambiguous overlap retain one eligible representative cell
    # rather than dropping every candidate in that overlap group.
    if ($SkipRepresentativeSelection) {
        Assert-File $representativeInstances
    } else {
        Invoke-Python "${caseId}: select one cell per overlap group" @(
            $RepresentativeSelector,
            "--semantic", $completedSemantic,
            "--assigned-instances", (Join-Path $methodRoot "01_cell_instances_assigned.tif"),
            "--safe-instances", (Join-Path $methodRoot "02_cell_instances_safe.tif"),
            "--ambiguous-regions", (Join-Path $methodRoot "05_ambiguous_regions_instances.tif"),
            "--status-map", (Join-Path $methodRoot "08_assignment_status_map.tif"),
            "--cell-report", (Join-Path $methodRoot "cell_report.csv"),
            "--original", $original,
            "--output-dir", $caseRepresentativeRoot,
            "--contact-radius", "8",
            "--min-skeleton-pixels", "12",
            "--border-margin", "2",
            "--overwrite"
        ) $representativeInstances
    }
}

Write-Host ""; Write-Host "=== Representative-cell QC is ready ===" -ForegroundColor Green
foreach ($caseId in $caseIds) {
    Write-Host "Review: $(Join-Path $RepresentativeRoot "$caseId\07_qc_overlap_representatives.png")"
}
if ($StopAfterRepresentativeSelection) {
    Write-Host "Stopped after representative selection as requested. Resume the same command without -StopAfterRepresentativeSelection." -ForegroundColor Yellow
    return
}

foreach ($caseId in $caseIds) {
    $original = Join-Path $InputDir "${caseId}_0000.tif"
    $completedSemantic = Join-Path $PostprocessRoot "$caseId\15_completed_skeleton_and_soma_0-1-2.tif"
    $representativeInstances = Join-Path $RepresentativeRoot "$caseId\01_representative_cells_safe.tif"
    $caseCellRoot = Join-Path $CellRoot $caseId
    $exportSummary = Join-Path $caseCellRoot "export_summary.json"

    # Stage 6: per-cell crop folders with raw.tif, skeleton.tif, soma.tif,
    # seg.tif, SWC/CSV metadata, etc. No crop is allowed to touch the overview
    # border after its required margin has been applied.
    if ($SkipCropExport) {
        Assert-File $exportSummary
    } else {
        Invoke-Python "${caseId}: export representative cells for Evo" @(
            $AssignedExporter,
            "--original", $original,
            "--semantic", $completedSemantic,
            "--safe-instances", $representativeInstances,
            "--output-dir", $caseCellRoot,
            "--helper-script", $ExistingExporter,
            "--margin", "48",
            "--min-crop-size", "128",
            "--edge-clearance", "3",
            "--square"
        ) $exportSummary

        $overviewRoot = Join-Path $GalleryRoot $caseId
        Invoke-Python "${caseId}: create Evo crop overview" @(
            $CropOverview,
            "--cells-root", $caseCellRoot,
            "--output-dir", $overviewRoot,
            "--columns", "0",
            "--thumbnail-size", "180",
            "--skeleton-thickness", "1",
            "--make-pages",
            "--page-columns", "5",
            "--page-rows", "4",
            "--page-thumbnail-size", "320"
        ) (Join-Path $overviewRoot "all_cells_overview.png")
    }
}

if ($RunFiji) {
    Assert-File $Fiji
    Assert-File $Finalizer
    Assert-File $Validator
    foreach ($caseId in $caseIds) {
        $caseCellRoot = Join-Path $CellRoot $caseId
        $logPath = Join-Path $caseCellRoot "_fiji_finalize_log.txt"
        $boundsPath = Join-Path $caseCellRoot "bounds.zip"
        $locationsPath = Join-Path $caseCellRoot "locations.zip"
        $ready = $false
        if ((Test-Path $logPath) -and (Test-Path $boundsPath) -and (Test-Path $locationsPath)) {
            $ready = (Get-Content -LiteralPath $logPath -Raw) -match "GLOBAL OK bounds\.zip and locations\.zip"
        }
        if (-not $ready) {
            Write-Host ""; Write-Host "=== ${caseId}: Fiji/SNT finalization ===" -ForegroundColor Cyan
            $env:CELL_EXPORT_ROOT = $caseCellRoot
            & $Fiji --allow-multiple --headless --console --run $Finalizer
            if (-not ((Test-Path $logPath) -and (Test-Path $boundsPath) -and (Test-Path $locationsPath))) {
                throw "Fiji finalization did not create its expected global ZIP files. Read: $logPath"
            }
        }
        Write-Host ""; Write-Host "=== ${caseId}: validate Evo cell folders ===" -ForegroundColor Cyan
        & $Python $Validator --input-dir $caseCellRoot
        if ($LASTEXITCODE -ne 0) { throw "Cell-folder validation failed for $caseId." }
    }
}

Write-Host ""; Write-Host ("=" * 72); Write-Host "SKELETON-RECALL TO EVO WORKFLOW COMPLETE" -ForegroundColor Green; Write-Host ("=" * 72)
Write-Host "Prediction root:      $PredictionRoot"
Write-Host "Soma preparation:     $SomaRoot"
Write-Host "No-loss postprocess:  $PostprocessRoot"
Write-Host "Assignment QC:        $AssignmentRoot"
Write-Host "Representative QC:    $RepresentativeRoot"
Write-Host "Evo cell folders:     $CellRoot"
Write-Host "Crop galleries:       $GalleryRoot"
if (-not $RunFiji) {
    Write-Host ""
    Write-Host "The crop folders are not yet Fiji/SNT-finalized. To create seg.traces and the ZIP files, rerun this command with -RunFiji." -ForegroundColor Yellow
}
