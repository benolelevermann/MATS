<#
Recheck an already completed paper-guided run with configurable soma / multi-cell QC.

This script starts at the FINAL semantic image. It never reruns nnU-Net or
postprocessing, and it never edits the semantic segmentation. It writes a new
selection stage and, unless -StopAfterQualityGate is used, new Evo cell folders.

Copy this file AND quality_gate_cells_for_evo_soma_strict.py into the project
root before running it. Defaults preserve the original strict behaviour; a
separate stage can opt into a hole-tolerant, coherent-single-instance test.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,

    [Parameter(Mandatory = $true)]
    [string]$CaseId,

    [Parameter(Mandatory = $true)]
    [string]$RawImage,

    [string]$ProjectRoot = $PSScriptRoot,

    [ValidateSet("topology_baseline", "matrix_forest_inspired", "gcut_inspired")]
    [string]$AssignmentMethod = "matrix_forest_inspired",

    [int]$MinSomaArea = 300,
    [double]$MinSomaCoreRadius = 6.0,
    [double]$MinSomaSolidity = 0.80,
    [double]$MaxSomaCoreDiskRatio = 5.0,
    [int]$CoreMinSeparation = 12,
    [double]$CorePeakRelativeHeight = 0.55,
    [double]$CorePeakProminenceFraction = 0.18,
    [ValidateSet("exclude", "review", "allow-if-single-instance")]
    [string]$MultiCorePolicy = "exclude",
    [ValidateSet("ignore", "review", "exclude")]
    [string]$SomaHolePolicy = "review",
    [int]$MinSomaHolePixelsForReview = 1,
    [int]$ForeignSomaMinArea = 300,
    [int]$CropMargin = 64,
    [int]$CropMinimumSize = 160,
    [int]$EdgeClearance = 4,

    [string]$QualityStageName = "12_strict_soma_multicell_quality_gate",
    [string]$CellsStageName = "13_evo_cells_soma_strict",
    [string]$GalleryStageName = "14_evo_crop_overviews_soma_strict",

    [switch]$StopAfterQualityGate,

    [switch]$UseExistingQualityGate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Assert-EmptyOrMissingDirectory([string]$Path, [string]$Description) {
    if ((Test-Path -LiteralPath $Path -PathType Container) -and
        (Get-ChildItem -LiteralPath $Path -Force | Measure-Object).Count -gt 0) {
        throw "$Description already contains files. Choose a new run root or move this stage first: $Path"
    }
}

function Invoke-Python([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code $LASTEXITCODE."
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$RawImage = [System.IO.Path]::GetFullPath($RawImage)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$QualityGate = Join-Path $ProjectRoot "quality_gate_cells_for_evo_soma_strict.py"
$Exporter = Join-Path $ProjectRoot "export_assigned_cells_for_evo.py"
$Overview = Join-Path $ProjectRoot "make_cells_overview.py"

foreach ($item in @(
    @($Python, "Project Python"),
    @($QualityGate, "Strict soma quality gate"),
    @($Exporter, "Evo crop exporter"),
    @($Overview, "Crop overview script"),
    @($RawImage, "Raw image")
)) {
    Assert-File $item[0] $item[1]
}

$Semantic = Join-Path $RunRoot "06_gap_pass_3_long_aligned\$CaseId\15_completed_skeleton_and_soma_0-1-2.tif"
$SelectedInstances = Join-Path $RunRoot "08_overlap_representative_selection\$CaseId\01_representative_cells_safe.tif"
$ReviewInstances = Join-Path $RunRoot "07_cell_assignment_comparison\$CaseId\$AssignmentMethod\01_cell_instances_assigned.tif"
$RidgeEvidence = Join-Path $RunRoot "02_ridge_hessian_evidence\$CaseId\04_combined_neurite_evidence.tif"

foreach ($item in @(
    @($Semantic, "Final semantic segmentation"),
    @($SelectedInstances, "Overlap representative instances"),
    @($ReviewInstances, "Assignment instance map")
)) {
    Assert-File $item[0] $item[1]
}

foreach ($stageName in @($QualityStageName, $CellsStageName, $GalleryStageName)) {
    if ([string]::IsNullOrWhiteSpace($stageName) -or $stageName.IndexOfAny([System.IO.Path]::GetInvalidFileNameChars()) -ge 0) {
        throw "Stage names must be non-empty folder names: $stageName"
    }
}
$QualityOutput = Join-Path $RunRoot "$QualityStageName\$CaseId"
$CellsOutput = Join-Path $RunRoot "$CellsStageName\$CaseId"
$GalleryOutput = Join-Path $RunRoot "$GalleryStageName\$CaseId"
if ($UseExistingQualityGate) {
    Assert-File (Join-Path $QualityOutput "01_cells_safe_for_evo.tif") "Existing quality-gate safe map"
} else {
    Assert-EmptyOrMissingDirectory $QualityOutput "Strict quality-gate output"
}
if (-not $StopAfterQualityGate) {
    Assert-EmptyOrMissingDirectory $CellsOutput "Strict Evo crop output"
}

$arguments = @(
    $QualityGate,
    "--semantic", $Semantic,
    "--instances", $SelectedInstances,
    "--review-instances", $ReviewInstances,
    "--original", $RawImage,
    "--output-dir", $QualityOutput,
    "--min-soma-area", "$MinSomaArea",
    "--min-soma-core-radius", "$MinSomaCoreRadius",
    "--min-soma-solidity", "$MinSomaSolidity",
    "--max-soma-core-disk-ratio", "$MaxSomaCoreDiskRatio",
    "--core-min-separation", "$CoreMinSeparation",
    "--core-peak-relative-height", "$CorePeakRelativeHeight",
    "--core-peak-prominence-fraction", "$CorePeakProminenceFraction",
    "--multi-core-policy", $MultiCorePolicy,
    "--soma-hole-policy", $SomaHolePolicy,
    "--min-soma-hole-pixels-for-review", "$MinSomaHolePixelsForReview",
    "--foreign-soma-min-area", "$ForeignSomaMinArea",
    "--foreign-soma-policy", "allow",
    "--foreign-skeleton-policy", "allow",
    "--crop-margin", "$CropMargin",
    "--crop-min-size", "$CropMinimumSize"
)
if (Test-Path -LiteralPath $RidgeEvidence -PathType Leaf) {
    $arguments += @("--ridge-evidence", $RidgeEvidence, "--ridge-weight", "0.35")
}

if (-not $UseExistingQualityGate) {
    Write-Host ""; Write-Host "=== Soma / single-instance quality gate: $CaseId ===" -ForegroundColor Cyan
    Invoke-Python $arguments
} else {
    Write-Host ""; Write-Host "=== Reusing existing soma / single-instance quality gate: $CaseId ===" -ForegroundColor Cyan
}

if ($StopAfterQualityGate) {
    Write-Host ""; Write-Host "Stopped after QC by request." -ForegroundColor Yellow
    Write-Host "Inspect these first:" -ForegroundColor Yellow
    Write-Host "  $QualityOutput\08_cell_quality_gate_qc.png"
    Write-Host "  $QualityOutput\07_flagged_soma_qc"
    Write-Host "  $QualityOutput\cell_quality_report.csv"
    return
}

Write-Host ""; Write-Host "=== Export strict Evo-safe crops: $CaseId ===" -ForegroundColor Cyan
Invoke-Python @(
    $Exporter,
    "--original", $RawImage,
    "--semantic", $Semantic,
    "--safe-instances", (Join-Path $QualityOutput "01_cells_safe_for_evo.tif"),
    "--output-dir", $CellsOutput,
    "--square",
    "--margin", "$CropMargin",
    "--min-crop-size", "$CropMinimumSize",
    "--edge-clearance", "$EdgeClearance"
)

Write-Host ""; Write-Host "=== Mask-faithful overview: $CaseId ===" -ForegroundColor Cyan
Invoke-Python @(
    $Overview,
    "--cells-root", $CellsOutput,
    "--output-dir", $GalleryOutput,
    "--skeleton-thickness", "0",
    "--make-pages"
)

Write-Host ""; Write-Host "=" * 72
Write-Host "SOMA QC + EVO CROP EXPORT COMPLETE" -ForegroundColor Green
Write-Host "Quality report:        $QualityOutput\cell_quality_report.csv"
Write-Host "Safe cell folders:     $CellsOutput"
Write-Host "Mask-faithful overview: $GalleryOutput\all_cells_overview.png"
