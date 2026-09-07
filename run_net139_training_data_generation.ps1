[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$Port = 8772,
    [switch]$PrepareOnly,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$preparer = Join-Path $projectRoot "prepare_net139_reanalysis_from_batch.py"
$webRunner = Join-Path $projectRoot "run_cell_pipeline_web.ps1"
$sourceRuns = Join-Path $projectRoot "web_pipeline_runs"
$targetRuns = Join-Path $projectRoot "web_pipeline_runs_net139"
$sourceBatch = "batch_20260831_153636_b2d626"
$targetBatch = "net139_reanalysis_20260903"
$trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug"
$checkpoint = Join-Path $projectRoot "nnUNet_results\Dataset139_dataset138_plus_reviewed_cells\${trainer}__nnUNetPlans__2d\fold_0\checkpoint_final.pth"

foreach ($required in @($python, $preparer, $webRunner, $sourceRuns, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}

& $python $preparer `
    --source-runs $sourceRuns `
    --source-batch $sourceBatch `
    --target-runs $targetRuns `
    --target-batch $targetBatch
if ($LASTEXITCODE -ne 0) {
    throw "Der Netz-139-Reviewbatch konnte nicht vorbereitet werden."
}
if ($PrepareOnly) {
    Write-Host "Vorbereitung abgeschlossen: $targetRuns" -ForegroundColor Green
    return
}

Write-Host "Netz 139 verarbeitet erneut die 189 ausgewaehlten MICA-FOVs." -ForegroundColor Cyan
Write-Host "Review-Seite: http://127.0.0.1:$Port/"
Write-Host "Freigegebene Zellen werden weiterhin nach review_dataset\approved geschrieben."

& $webRunner `
    -Device $Device `
    -Port $Port `
    -RunsRoot $targetRuns `
    -DatasetId 139 `
    -Trainer $trainer `
    -Plans "nnUNetPlans" `
    -Checkpoint "checkpoint_final.pth" `
    -NoFiji `
    -NoBrowser:$NoBrowser
