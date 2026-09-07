param(
    [string]$Project = "C:\Ole\20260721_CellClassification_v2",
    [string]$TestRoot = "C:\Ole\20260721_CellClassification_v2\20280812_newTest2"
)

$ErrorActionPreference = "Stop"

$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Script = Join-Path $Project "compare_cell_assignment_methods.py"
$Segmentation = Join-Path $TestRoot "05_paper_postprocessing_test\00_gap_completion_no_loss\15_completed_skeleton_and_soma_0-1-2.tif"
$Original = Join-Path $TestRoot "01_inputimages\NewImage2_0000.tif"
$Output = Join-Path $TestRoot "05_paper_postprocessing_test\01_cell_assignment_comparison"

foreach ($RequiredFile in @($Python, $Script, $Segmentation, $Original)) {
    if (-not (Test-Path -LiteralPath $RequiredFile)) {
        throw "Required file not found: $RequiredFile"
    }
}

Write-Host "Input segmentation: $Segmentation"
Write-Host "Original image:     $Original"
Write-Host "Output directory:   $Output"

& $Python $Script `
    --segmentation $Segmentation `
    --original $Original `
    --output-dir $Output `
    --root-radius 3 `
    --mft-tau 1.0 `
    --mft-seed-strength 25 `
    --mft-min-confidence 0.58 `
    --mft-margin 0.12 `
    --gcut-turn-weight 1.35 `
    --gcut-radial-weight 0.22 `
    --gcut-margin 0.10 `
    --safe-max-ambiguous-fraction 0.05 `
    --safe-min-skeleton-pixels 12 `
    --border-margin 2 `
    --qc-max-size 2200

if ($LASTEXITCODE -ne 0) {
    throw "Cell-assignment comparison failed with exit code $LASTEXITCODE."
}

$Html = Join-Path $Output "comparison.html"
Write-Host ""
Write-Host "Comparison completed: $Html"
Start-Process $Html
