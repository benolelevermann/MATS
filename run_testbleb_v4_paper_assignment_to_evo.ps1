param(
    [ValidateSet("topology_baseline", "matrix_forest_inspired", "gcut_inspired")]
    [string]$Method = "gcut_inspired",
    [switch]$SkipGapCompletion,
    [switch]$SkipAssignment,
    [switch]$StopAfterAssignment,
    [switch]$SkipExport,
    [switch]$SkipFiji,
    [switch]$RunFeatureExtraction
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Project = "C:\Ole\20260721_CellClassification_v2"
$RunRoot = Join-Path $Project "20260730_TestBleb_v4"
$Pipeline = Join-Path $Project "r_pipeline"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Fiji = "C:\Program Files\Fiji.app\ImageJ-win64.exe"
$Rscript = "C:\Program Files\R\R-4.5.1\bin\Rscript.exe"

$NoLossPostprocess = Join-Path $Project "postprocess_net129_no_loss.py"
$AssignmentScript = Join-Path $Project "compare_cell_assignment_methods.py"
$AssignedExporter = Join-Path $Project "export_assigned_cells_for_evo.py"
$ExistingExporter = Join-Path $Pipeline "export_cells_for_r_pipeline.py"
$CropOverview = Join-Path $Project "make_cells_overview.py"
$Finalizer = Join-Path $Pipeline "finalize_cells_with_fiji.py"
$Validator = Join-Path $Pipeline "validate_r_pipeline_cell_folders.py"
$PrepareMetadata = Join-Path $Pipeline "prepare_testbleb_v4_paper_metadata.R"
$RunExtraction = Join-Path $Pipeline "run_testbleb_v4_paper_feature_extraction.R"
$SaveExtraction = Join-Path $Pipeline "save_testbleb_v4_paper_extraction.R"

$PaperRoot = Join-Path $RunRoot "05_paper_postprocessing_skeleton_recall"
$SharedAnalysisRoot = Join-Path $RunRoot "07_evo_pipeline_$Method"

$Cases = @(
    [pscustomobject]@{
        Id = "Bleb"
        Condition = "Blebbistatin"
        Original = Join-Path $RunRoot "01_inputimages\Bleb_0000.tif"
        Prediction = Join-Path $RunRoot "02_predictions_skeleton_recall\Bleb.tif"
        Probabilities = Join-Path $RunRoot "02_predictions_skeleton_recall\Bleb.npz"
    },
    [pscustomobject]@{
        Id = "DMSO"
        Condition = "DMSO"
        Original = Join-Path $RunRoot "01_inputimages\DMSO_0000.tif"
        Prediction = Join-Path $RunRoot "02_predictions_skeleton_recall\DMSO.tif"
        Probabilities = Join-Path $RunRoot "02_predictions_skeleton_recall\DMSO.npz"
    }
)

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

foreach ($file in @($Python, $NoLossPostprocess, $AssignmentScript, $AssignedExporter, $ExistingExporter, $CropOverview)) {
    Assert-File $file
}
foreach ($case in $Cases) {
    foreach ($file in @($case.Original, $case.Prediction, $case.Probabilities)) { Assert-File $file }
}

foreach ($case in $Cases) {
    $CaseRoot = Join-Path $PaperRoot $case.Id
    $GapRoot = Join-Path $CaseRoot "00_gap_completion_no_loss"
    $Semantic = Join-Path $GapRoot "15_completed_skeleton_and_soma_0-1-2.tif"
    $AssignmentRoot = Join-Path $CaseRoot "01_cell_assignment_comparison"
    $AssignmentHtml = Join-Path $AssignmentRoot "comparison.html"

    if (-not $SkipGapCompletion) {
        Invoke-Python `
            "$($case.Id): No-Loss gap completion" `
            @(
                $NoLossPostprocess,
                "--prediction", $case.Prediction,
                "--probabilities", $case.Probabilities,
                "--original", $case.Original,
                "--output-dir", $GapRoot,
                "--endpoint-gap", "24",
                "--soma-gap", "18",
                "--segment-gap", "16",
                "--geometry-rescue-gap", "60",
                "--geometry-rescue-cosine", "0.82",
                "--geometry-rescue-min-score", "0.30",
                "--geometry-rescue-min-support", "0.38",
                "--passes", "2",
                "--min-score", "0.40",
                "--min-path-probability", "0.12",
                "--min-image-support", "0.55",
                "--ambiguity-margin", "0.06"
            ) `
            $Semantic
    } else {
        Assert-File $Semantic
    }

    if (-not $SkipAssignment) {
        Invoke-Python `
            "$($case.Id): graph-based cell assignment comparison" `
            @(
                $AssignmentScript,
                "--segmentation", $Semantic,
                "--original", $case.Original,
                "--output-dir", $AssignmentRoot,
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
                "--border-margin", "2"
            ) `
            $AssignmentHtml
    } else {
        Assert-File $AssignmentHtml
    }
}

Write-Host ""; Write-Host "=== Assignment comparison complete ===" -ForegroundColor Green
foreach ($case in $Cases) {
    $summary = Join-Path $PaperRoot "$($case.Id)\01_cell_assignment_comparison\comparison_summary.csv"
    Write-Host "$($case.Id) comparison: $summary"
}

if ($StopAfterAssignment) {
    Write-Host ""
    Write-Host "The shared skeleton completion and all three cell-assignment methods are ready." -ForegroundColor Green
    Write-Host "Review the Bleb and DMSO comparison.html files, then export the selected method with:"
    Write-Host ".\run_testbleb_v4_paper_assignment_to_evo.ps1 -Method $Method -SkipGapCompletion -SkipAssignment"
    exit 0
}

foreach ($case in $Cases) {
    $CaseRoot = Join-Path $PaperRoot $case.Id
    $Semantic = Join-Path $CaseRoot "00_gap_completion_no_loss\15_completed_skeleton_and_soma_0-1-2.tif"
    $SafeInstances = Join-Path $CaseRoot "01_cell_assignment_comparison\$Method\02_cell_instances_safe.tif"
    $CellRoot = Join-Path $RunRoot "06_evo_cells_${Method}_$($case.Id)"
    $ExportSummary = Join-Path $CellRoot "export_summary.json"

    if (-not $SkipExport) {
        if (
            (Test-Path -LiteralPath $CellRoot -PathType Container) -and
            @(Get-ChildItem -LiteralPath $CellRoot -Force).Count -gt 0 -and
            -not (Test-Path -LiteralPath $ExportSummary -PathType Leaf)
        ) {
            throw "Existing non-empty cell export folder without export_summary.json: $CellRoot"
        }
        Invoke-Python `
            "$($case.Id): export $Method safe cells for Evo" `
            @(
                $AssignedExporter,
                "--original", $case.Original,
                "--semantic", $Semantic,
                "--safe-instances", $SafeInstances,
                "--output-dir", $CellRoot,
                "--helper-script", $ExistingExporter,
                "--margin", "48",
                "--min-crop-size", "128",
                "--edge-clearance", "3",
                "--square"
            ) `
            $ExportSummary
    } else {
        Assert-Directory $CellRoot
    }

    $OverviewRoot = Join-Path $RunRoot "06_evo_cells_${Method}_$($case.Id)_overview"
    $OverviewPng = Join-Path $OverviewRoot "all_cells_overview.png"
    Invoke-Python `
        "$($case.Id): create crop overview" `
        @(
            $CropOverview,
            "--cells-root", $CellRoot,
            "--output-dir", $OverviewRoot,
            "--columns", "0",
            "--thumbnail-size", "180",
            "--skeleton-thickness", "1",
            "--make-pages",
            "--page-columns", "5",
            "--page-rows", "4",
            "--page-thumbnail-size", "320"
        ) `
        $OverviewPng
}

if (-not $SkipFiji) {
    Assert-File $Fiji
    Assert-File $Finalizer
    foreach ($case in $Cases) {
        $CellRoot = Join-Path $RunRoot "06_evo_cells_${Method}_$($case.Id)"
        $logPath = Join-Path $CellRoot "_fiji_finalize_log.txt"
        $boundsPath = Join-Path $CellRoot "bounds.zip"
        $locationsPath = Join-Path $CellRoot "locations.zip"
        $ready = $false
        if ((Test-Path $logPath) -and (Test-Path $boundsPath) -and (Test-Path $locationsPath)) {
            $existingLog = Get-Content -LiteralPath $logPath -Raw
            $ready = $existingLog -match "GLOBAL OK bounds\.zip and locations\.zip"
        }
        if ($ready) {
            Write-Host "=== $($case.Id): Fiji already complete; skipped ===" -ForegroundColor DarkGray
            continue
        }
        Write-Host ""; Write-Host "=== $($case.Id): create seg.traces and ROI ZIPs with Fiji ===" -ForegroundColor Cyan
        $env:CELL_EXPORT_ROOT = $CellRoot
        & $Fiji --allow-multiple --headless --console --run $Finalizer
        $fijiExitCode = $LASTEXITCODE
        $success = $false
        if ((Test-Path $logPath) -and (Test-Path $boundsPath) -and (Test-Path $locationsPath)) {
            $newLog = Get-Content -LiteralPath $logPath -Raw
            $success = (
                $newLog -match "GLOBAL OK bounds\.zip and locations\.zip" -and
                $newLog -notmatch "(?m)^\[\d+/\d+\] ERROR "
            )
        }
        if (-not $success) { throw "Fiji finalization failed for $($case.Id). Read: $logPath" }
        if ($fijiExitCode -ne 0) { Write-Warning "Fiji returned $fijiExitCode but the completion log is valid." }
    }
}

Assert-File $Validator
foreach ($case in $Cases) {
    $CellRoot = Join-Path $RunRoot "06_evo_cells_${Method}_$($case.Id)"
    Write-Host ""; Write-Host "=== Validate $($case.Id) Evo cell folders ===" -ForegroundColor Cyan
    & $Python $Validator --input-dir $CellRoot
    if ($LASTEXITCODE -ne 0) { throw "Cell-folder validation failed for $($case.Id)." }
}

Assert-File $Rscript
Assert-File $PrepareMetadata
Write-Host ""; Write-Host "=== Prepare joint Bleb/DMSO Evo metadata ===" -ForegroundColor Cyan
& $Rscript $PrepareMetadata $Method
if ($LASTEXITCODE -ne 0) { throw "Bleb/DMSO metadata preparation failed." }

if (-not $RunFeatureExtraction) {
    Write-Host ""; Write-Host "Bleb and DMSO are exported, finalized, validated and have common metadata." -ForegroundColor Green
    Write-Host "Run feature extraction later with:"
    Write-Host ".\run_testbleb_v4_paper_assignment_to_evo.ps1 -Method $Method -SkipGapCompletion -SkipAssignment -SkipExport -SkipFiji -RunFeatureExtraction"
    exit 0
}

foreach ($file in @($RunExtraction, $SaveExtraction)) { Assert-File $file }
Write-Host ""; Write-Host "=== Run joint Bleb/DMSO Evo feature extraction ===" -ForegroundColor Cyan
& $Rscript $RunExtraction $Method
if ($LASTEXITCODE -ne 0) { throw "Bleb/DMSO Evo feature extraction failed." }
& $Rscript $SaveExtraction $Method
if ($LASTEXITCODE -ne 0) { throw "Saving Bleb/DMSO extraction failed." }

$latestPathFile = Join-Path $SharedAnalysisRoot "05_r_pipeline\saved_extraction\_LATEST_EXTRACTION_PATH.txt"
Assert-File $latestPathFile
$latestExtraction = (Get-Content -LiteralPath $latestPathFile -Raw).Trim()
Write-Host ""; Write-Host ("=" * 72); Write-Host "BLEB VS DMSO EVO EXTRACTION READY" -ForegroundColor Green; Write-Host ("=" * 72)
Write-Host "Use this path as input_dir in setupNewObject_MAPPING.Rmd:"
Write-Host $latestExtraction
