<#
Paper-guided, material-preserving test workflow for Dataset136 predictions.

Stages (each is written to a separate folder):
  01 prediction -> 02 Hessian/ridge evidence -> 03 soma preparation
  -> 04 short high-precision gaps -> 05 medium gaps
  -> 06 long aligned gaps -> 07 assignment comparison
  -> 08 representative overlap selection -> 09 quality gate
  -> 10 Evo-safe crops -> optional Fiji/SNT finalization.

The semantic material invariant is preserved by the postprocessor at every
connection stage: final skeleton = previous skeleton OR accepted connections.
No stage removes pre-existing skeleton pixels.
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

    [string]$ExistingPredictionDir = "",

    [ValidateSet("topology_baseline", "matrix_forest_inspired", "gcut_inspired")]
    [string]$AssignmentMethod = "matrix_forest_inspired",

    [int]$MinSomaArea = 80,
    [int]$CropMargin = 64,
    [int]$CropMinimumSize = 160,
    [double]$RidgeEvidenceWeight = 0.35,

    [switch]$SkipPrediction,
    [switch]$SkipEvidence,
    [switch]$SkipSomaPreparation,
    [switch]$SkipShortGapPass,
    [switch]$SkipMediumGapPass,
    [switch]$SkipLongGapPass,
    [switch]$SkipAssignment,
    [switch]$SkipRepresentativeSelection,
    [switch]$SkipQualityGate,
    [switch]$SkipCropExport,
    [switch]$StopAfterSoma,
    [switch]$StopAfterShortGapPass,
    [switch]$StopAfterMediumGapPass,
    [switch]$StopAfterLongGapPass,
    [switch]$StopAfterAssignment,
    [switch]$StopAfterQualityGate,
    [switch]$RunFiji,
    [switch]$AllowExistingRunRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file was not found: $Path"
    }
}

function Assert-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required directory was not found: $Path"
    }
}

function Test-StageDone([string]$ExpectedFile) {
    return Test-Path -LiteralPath $ExpectedFile -PathType Leaf
}

function Assert-StageCanStart([string]$StageDirectory, [string]$ExpectedFile) {
    if (Test-StageDone $ExpectedFile) { return $false }
    if ((Test-Path -LiteralPath $StageDirectory -PathType Container) -and
        (Get-ChildItem -LiteralPath $StageDirectory -Force | Select-Object -First 1)) {
        throw "Partial stage exists but its expected result is missing: $StageDirectory`nUse a fresh RunRoot or inspect/remove this partial stage deliberately."
    }
    New-Item -ItemType Directory -Path $StageDirectory -Force | Out-Null
    return $true
}

function Invoke-Python([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed with exit code $LASTEXITCODE." }
}

function Get-InputCases([string]$Directory) {
    $files = @(
        Get-ChildItem -LiteralPath $Directory -File |
        Where-Object { $_.Extension -in ".tif", ".tiff" } |
        Where-Object { $_.BaseName -match "_0000$" } |
        Sort-Object Name
    )
    if ($files.Count -eq 0) {
        throw "No *_0000.tif or *_0000.tiff input images were found in: $Directory"
    }
    return @(
        foreach ($file in $files) {
            [pscustomobject]@{
                Id = $file.BaseName -replace "_0000$", ""
                Raw = $file.FullName
            }
        }
    )
}

$Project = "C:\Ole\20260721_CellClassification_v2"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Predictor = Join-Path $Project ".venv\Scripts\nnUNetv2_predict.exe"
$EvidenceScript = Join-Path $Project "build_neurite_ridge_evidence.py"
$SomaScript = Join-Path $Project "prepare_somas_paper_guided.py"
$Postprocessor = Join-Path $Project "postprocess_net129_no_loss.py"
$AssignmentScript = Join-Path $Project "compare_cell_assignment_methods.py"
$RepresentativeScript = Join-Path $Project "select_overlap_representative_cells.py"
$QualityGateScript = Join-Path $Project "quality_gate_cells_for_evo.py"
$Exporter = Join-Path $Project "export_assigned_cells_for_evo.py"
$CropOverview = Join-Path $Project "make_cells_overview.py"
$Finalizer = Join-Path $Project "r_pipeline\finalize_cells_with_fiji.py"
$Validator = Join-Path $Project "r_pipeline\validate_r_pipeline_cell_folders.py"
$Fiji = "C:\Program Files\Fiji.app\ImageJ-win64.exe"

foreach ($required in @($Python, $EvidenceScript, $SomaScript, $Postprocessor, $AssignmentScript, $RepresentativeScript, $QualityGateScript, $Exporter, $CropOverview)) {
    Assert-File $required
}
if (-not $SkipPrediction) { Assert-File $Predictor }
Assert-Directory $InputDir

$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$InputDir = [System.IO.Path]::GetFullPath($InputDir)
if ((Test-Path -LiteralPath $RunRoot -PathType Container) -and
    (Get-ChildItem -LiteralPath $RunRoot -Force | Select-Object -First 1) -and
    -not $AllowExistingRunRoot) {
    throw "RunRoot already contains files: $RunRoot`nUse a new versioned folder. Only use -AllowExistingRunRoot when resuming verified completed stages."
}
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

$UseExistingPredictions = -not [string]::IsNullOrWhiteSpace($ExistingPredictionDir)
$PredictionRoot = if ($UseExistingPredictions) {
    [System.IO.Path]::GetFullPath($ExistingPredictionDir)
} else {
    Join-Path $RunRoot "01_predictions_skeleton_recall"
}
$EvidenceRoot = Join-Path $RunRoot "02_ridge_hessian_evidence"
$SomaRoot = Join-Path $RunRoot "03_paper_guided_soma_preparation"
$ShortRoot = Join-Path $RunRoot "04_gap_pass_1_short_high_precision"
$MediumRoot = Join-Path $RunRoot "05_gap_pass_2_medium"
$LongRoot = Join-Path $RunRoot "06_gap_pass_3_long_aligned"
$AssignmentRoot = Join-Path $RunRoot "07_cell_assignment_comparison"
$RepresentativeRoot = Join-Path $RunRoot "08_overlap_representative_selection"
$QualityRoot = Join-Path $RunRoot "09_cell_quality_gate"
$CellsRoot = Join-Path $RunRoot "10_evo_cells_safe"
$GalleryRoot = Join-Path $RunRoot "11_evo_crop_overviews_safe"

$env:nnUNet_raw = Join-Path $Project "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Project "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Project "nnUNet_results"
Remove-Item Env:nnUNet_extTrainer -ErrorAction SilentlyContinue

$cases = Get-InputCases $InputDir
Write-Host ""; Write-Host "Cases: $($cases.Id -join ', ')" -ForegroundColor Cyan

if ($UseExistingPredictions) {
    Assert-Directory $PredictionRoot
    Write-Host "Using existing skeleton-recall predictions: $PredictionRoot" -ForegroundColor Yellow
} elseif (-not $SkipPrediction) {
    $missingPrediction = @(
        foreach ($case in $cases) {
            if (-not (Test-Path (Join-Path $PredictionRoot "$($case.Id).tif")) -or
                -not (Test-Path (Join-Path $PredictionRoot "$($case.Id).npz"))) { $case.Id }
        }
    )
    if ($missingPrediction.Count -gt 0) {
        New-Item -ItemType Directory -Path $PredictionRoot -Force | Out-Null
        Write-Host ""; Write-Host "=== Dataset $DatasetId skeleton-recall inference ===" -ForegroundColor Cyan
        & $Predictor -i $InputDir -o $PredictionRoot -d $DatasetId -c $Configuration -f $Fold `
            -tr $Trainer -p $Plans -chk $Checkpoint -device $Device --save_probabilities
        if ($LASTEXITCODE -ne 0) { throw "nnU-Net inference failed with exit code $LASTEXITCODE." }
    }
}
foreach ($case in $cases) {
    Assert-File (Join-Path $PredictionRoot "$($case.Id).tif")
    Assert-File (Join-Path $PredictionRoot "$($case.Id).npz")
}

foreach ($case in $cases) {
    $stage = Join-Path $EvidenceRoot $case.Id
    $expected = Join-Path $stage "04_combined_neurite_evidence.tif"
    if ($SkipEvidence) {
        Write-Host "Skipping ridge evidence for $($case.Id)" -ForegroundColor Yellow
        continue
    }
    if (Assert-StageCanStart $stage $expected) {
        Write-Host ""; Write-Host "=== Ridge/Hessian evidence: $($case.Id) ===" -ForegroundColor Cyan
        Invoke-Python @($EvidenceScript, "--original", $case.Raw, "--output-dir", $stage,
            "--polarity", "bright", "--sigma-min", "0.8", "--sigma-max", "4.0", "--sigma-steps", "5",
            "--background-sigma", "18", "--tile-size", "2048", "--tile-overlap", "96")
    }
}

foreach ($case in $cases) {
    $stage = Join-Path $SomaRoot $case.Id
    $expected = Join-Path $stage "06_semantic_soma_prepared_0-1-2.tif"
    if ($SkipSomaPreparation) { continue }
    if (Assert-StageCanStart $stage $expected) {
        Write-Host ""; Write-Host "=== Paper-guided soma preparation: $($case.Id) ===" -ForegroundColor Cyan
        Invoke-Python @($SomaScript, "--prediction", (Join-Path $PredictionRoot "$($case.Id).tif"),
            "--original", $case.Raw, "--output-dir", $stage, "--polarity", "bright",
            "--min-soma-area", "$MinSomaArea", "--recovery-radius", "5", "--recovery-min-support", "0.62",
            "--max-recovery-ratio", "0.35")
    }
}
if ($StopAfterSoma) { Write-Host "Stopped after soma QC by request." -ForegroundColor Yellow; return }

function Invoke-GapPass([string]$Name, [string]$Root, [scriptblock]$InputPathProvider, [scriptblock]$PriorMaterialProvider, [string[]]$Settings, [switch]$Skip) {
    foreach ($case in $cases) {
        $stage = Join-Path $Root $case.Id
        $expected = Join-Path $stage "15_completed_skeleton_and_soma_0-1-2.tif"
        if ($Skip) { continue }
        if (Assert-StageCanStart $stage $expected) {
            $inputSemantic = & $InputPathProvider $case
            Assert-File $inputSemantic
            $arguments = @($Postprocessor, "--prediction", $inputSemantic,
                "--probabilities", (Join-Path $PredictionRoot "$($case.Id).npz"),
                "--original", $case.Raw, "--output-dir", $stage,
                "--original-polarity", "bright", "--ridge-weight", "$RidgeEvidenceWeight")
            $evidence = Join-Path (Join-Path $EvidenceRoot $case.Id) "04_combined_neurite_evidence.tif"
            if (Test-Path -LiteralPath $evidence -PathType Leaf) {
                $arguments += @("--ridge-evidence", $evidence)
            }
            $priorMaterial = & $PriorMaterialProvider $case
            if (-not [string]::IsNullOrWhiteSpace($priorMaterial)) {
                Assert-File $priorMaterial
                $arguments += @("--prior-skeleton-material", $priorMaterial)
            }
            $arguments += $Settings
            Write-Host ""; Write-Host "=== $Name : $($case.Id) ===" -ForegroundColor Cyan
            Invoke-Python $arguments
        }
    }
}

Invoke-GapPass "Gap pass 1 (short, high precision)" $ShortRoot `
    { param($case) Join-Path (Join-Path $SomaRoot $case.Id) "06_semantic_soma_prepared_0-1-2.tif" } `
    { param($case) $null } `
    @("--endpoint-gap", "14", "--soma-gap", "12", "--segment-gap", "12", "--passes", "1",
      "--disable-geometry-rescue", "--max-candidates-per-type", "1", "--min-score", "0.48",
      "--min-path-probability", "0.16", "--min-image-support", "0.60", "--min-direction-cosine", "0.10",
      "--max-detour", "1.25", "--ambiguity-margin", "0.06", "--route-margin", "6") -Skip:$SkipShortGapPass
if ($StopAfterShortGapPass) { Write-Host "Stopped after short-gap QC by request." -ForegroundColor Yellow; return }

Invoke-GapPass "Gap pass 2 (medium, evidence guided)" $MediumRoot `
    {
        param($case)
        $shortSemantic = Join-Path (Join-Path $ShortRoot $case.Id) "15_completed_skeleton_and_soma_0-1-2.tif"
        if (Test-Path -LiteralPath $shortSemantic -PathType Leaf) { return $shortSemantic }
        return (Join-Path (Join-Path $SomaRoot $case.Id) "06_semantic_soma_prepared_0-1-2.tif")
    } `
    {
        param($case)
        $shortMaterial = Join-Path (Join-Path $ShortRoot $case.Id) "04_completed_skeleton_fullwidth_NO_LOSS.tif"
        if (Test-Path -LiteralPath $shortMaterial -PathType Leaf) { return $shortMaterial }
        return $null
    } `
    @("--endpoint-gap", "28", "--soma-gap", "20", "--segment-gap", "20", "--passes", "1",
      "--disable-geometry-rescue", "--max-candidates-per-type", "2", "--min-score", "0.40",
      "--min-path-probability", "0.11", "--min-image-support", "0.54", "--min-direction-cosine", "-0.10",
      "--max-detour", "1.55", "--ambiguity-margin", "0.07", "--route-margin", "8") -Skip:$SkipMediumGapPass
if ($StopAfterMediumGapPass) { Write-Host "Stopped after medium-gap QC by request." -ForegroundColor Yellow; return }

Invoke-GapPass "Gap pass 3 (long, strongly aligned only)" $LongRoot `
    {
        param($case)
        foreach ($root in @($MediumRoot, $ShortRoot)) {
            $semantic = Join-Path (Join-Path $root $case.Id) "15_completed_skeleton_and_soma_0-1-2.tif"
            if (Test-Path -LiteralPath $semantic -PathType Leaf) { return $semantic }
        }
        return (Join-Path (Join-Path $SomaRoot $case.Id) "06_semantic_soma_prepared_0-1-2.tif")
    } `
    {
        param($case)
        foreach ($root in @($MediumRoot, $ShortRoot)) {
            $material = Join-Path (Join-Path $root $case.Id) "04_completed_skeleton_fullwidth_NO_LOSS.tif"
            if (Test-Path -LiteralPath $material -PathType Leaf) { return $material }
        }
        return $null
    } `
    @("--endpoint-gap", "42", "--soma-gap", "30", "--segment-gap", "26", "--passes", "1",
      "--geometry-rescue-gap", "78", "--geometry-rescue-cosine", "0.84", "--geometry-rescue-min-score", "0.32",
      "--geometry-rescue-min-support", "0.45", "--geometry-rescue-max-detour", "1.40",
      "--max-candidates-per-type", "2", "--min-score", "0.37", "--min-path-probability", "0.10",
      "--min-image-support", "0.51", "--min-direction-cosine", "-0.05", "--max-detour", "1.45",
      "--ambiguity-margin", "0.08", "--route-margin", "10") -Skip:$SkipLongGapPass
if ($StopAfterLongGapPass) { Write-Host "Stopped after connection QC by request." -ForegroundColor Yellow; return }

function Get-FinalSemantic([object]$Case) {
    foreach ($root in @($LongRoot, $MediumRoot, $ShortRoot)) {
        $candidate = Join-Path (Join-Path $root $Case.Id) "15_completed_skeleton_and_soma_0-1-2.tif"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    $somaOnly = Join-Path (Join-Path $SomaRoot $Case.Id) "06_semantic_soma_prepared_0-1-2.tif"
    Assert-File $somaOnly
    return $somaOnly
}

foreach ($case in $cases) {
    $finalSemantic = Get-FinalSemantic $case
    Assert-File $finalSemantic
    $stage = Join-Path $AssignmentRoot $case.Id
    $expected = Join-Path $stage "comparison.html"
    if (-not $SkipAssignment -and (Assert-StageCanStart $stage $expected)) {
        Write-Host ""; Write-Host "=== Assignment comparison: $($case.Id) ===" -ForegroundColor Cyan
        Invoke-Python @($AssignmentScript, "--segmentation", $finalSemantic, "--original", $case.Raw,
            "--output-dir", $stage, "--mft-min-confidence", "0.58", "--mft-margin", "0.12",
            "--gcut-turn-weight", "1.35", "--gcut-radial-weight", "0.22", "--gcut-margin", "0.10",
            "--safe-max-ambiguous-fraction", "0.05", "--safe-min-skeleton-pixels", "20", "--border-margin", "4")
    }
}
if ($StopAfterAssignment) { Write-Host "Stopped after assignment comparison by request." -ForegroundColor Yellow; return }

foreach ($case in $cases) {
    $finalSemantic = Get-FinalSemantic $case
    $method = Join-Path (Join-Path $AssignmentRoot $case.Id) $AssignmentMethod
    $stage = Join-Path $RepresentativeRoot $case.Id
    $expected = Join-Path $stage "01_representative_cells_safe.tif"
    if (-not $SkipRepresentativeSelection -and (Assert-StageCanStart $stage $expected)) {
        Write-Host ""; Write-Host "=== One representative per overlap: $($case.Id) ===" -ForegroundColor Cyan
        Invoke-Python @($RepresentativeScript, "--semantic", $finalSemantic, "--original", $case.Raw,
            "--assigned-instances", (Join-Path $method "01_cell_instances_assigned.tif"),
            "--safe-instances", (Join-Path $method "02_cell_instances_safe.tif"),
            "--ambiguous-regions", (Join-Path $method "05_ambiguous_regions_instances.tif"),
            "--status-map", (Join-Path $method "08_assignment_status_map.tif"),
            "--cell-report", (Join-Path $method "cell_report.csv"), "--output-dir", $stage,
            "--contact-radius", "8", "--min-skeleton-pixels", "20", "--border-margin", "4")
    }
}

foreach ($case in $cases) {
    $finalSemantic = Get-FinalSemantic $case
    $stage = Join-Path $QualityRoot $case.Id
    $expected = Join-Path $stage "01_cells_safe_for_evo.tif"
    if (-not $SkipQualityGate -and (Assert-StageCanStart $stage $expected)) {
        $method = Join-Path (Join-Path $AssignmentRoot $case.Id) $AssignmentMethod
        $arguments = @($QualityGateScript, "--semantic", $finalSemantic,
            "--instances", (Join-Path (Join-Path $RepresentativeRoot $case.Id) "01_representative_cells_safe.tif"),
            "--review-instances", (Join-Path $method "01_cell_instances_assigned.tif"),
            "--original", $case.Raw, "--output-dir", $stage, "--min-soma-area", "$MinSomaArea",
            "--min-skeleton-pixels", "20", "--contact-radius", "4", "--border-margin", "4")
        $evidence = Join-Path (Join-Path $EvidenceRoot $case.Id) "04_combined_neurite_evidence.tif"
        if (Test-Path -LiteralPath $evidence -PathType Leaf) {
            $arguments += @("--ridge-evidence", $evidence, "--ridge-weight", "$RidgeEvidenceWeight")
        }
        Write-Host ""; Write-Host "=== Export quality gate: $($case.Id) ===" -ForegroundColor Cyan
        Invoke-Python $arguments
    }
}
if ($StopAfterQualityGate) { Write-Host "Stopped after crop-quality QC by request." -ForegroundColor Yellow; return }

foreach ($case in $cases) {
    $finalSemantic = Get-FinalSemantic $case
    $stage = Join-Path $CellsRoot $case.Id
    $expected = Join-Path $stage "export_summary.json"
    if (-not $SkipCropExport -and (Assert-StageCanStart $stage $expected)) {
        Write-Host ""; Write-Host "=== Export Evo-safe crops: $($case.Id) ===" -ForegroundColor Cyan
        Invoke-Python @($Exporter, "--original", $case.Raw, "--semantic", $finalSemantic,
            "--safe-instances", (Join-Path (Join-Path $QualityRoot $case.Id) "01_cells_safe_for_evo.tif"),
            "--output-dir", $stage, "--square", "--margin", "$CropMargin",
            "--min-crop-size", "$CropMinimumSize", "--edge-clearance", "4")
    }
    if (Test-StageDone $expected) {
        $gallery = Join-Path $GalleryRoot $case.Id
        New-Item -ItemType Directory -Path $gallery -Force | Out-Null
        if (-not (Test-Path (Join-Path $gallery "all_cells_overview.png"))) {
            Invoke-Python @($CropOverview, "--cells-root", $stage, "--output-dir", $gallery,
                "--make-pages", "--page-columns", "6", "--page-rows", "4")
        }
    }
}

if ($RunFiji) {
    Assert-File $Fiji
    Assert-File $Finalizer
    Assert-File $Validator
    foreach ($case in $cases) {
        $cellRoot = Join-Path $CellsRoot $case.Id
        $env:CELL_EXPORT_ROOT = $cellRoot
        Write-Host ""; Write-Host "=== Fiji/SNT finalization: $($case.Id) ===" -ForegroundColor Cyan
        & $Fiji --allow-multiple --headless --console --run $Finalizer
        if (-not (Test-Path (Join-Path $cellRoot "bounds.zip"))) {
            throw "Fiji did not create bounds.zip. Read the finalization log in: $cellRoot"
        }
        Invoke-Python @($Validator, "--input-dir", $cellRoot)
    }
}

Write-Host ""; Write-Host ("=" * 76) -ForegroundColor Green
Write-Host "PAPER-GUIDED SKELETON-TO-EVO TEST WORKFLOW COMPLETE" -ForegroundColor Green
Write-Host ("=" * 76) -ForegroundColor Green
Write-Host "Final semantic (per case):  $LongRoot"
Write-Host "Assignment comparison:      $AssignmentRoot"
Write-Host "Safe/review quality maps:   $QualityRoot"
Write-Host "Evo-safe cell folders:      $CellsRoot"
Write-Host "Crop galleries:             $GalleryRoot"
