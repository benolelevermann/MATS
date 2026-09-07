[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$Port = 8777,
    [ValidateSet("adaptive", "tlow-0300", "tlow-0250", "tlow-0200")]
    [string]$HysteresisProfile = "tlow-0200",
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug"
$checkpoint = Join-Path $projectRoot "nnUNet_results\Dataset141_dataset139_plus_net139_reviewed_cells\${trainer}__nnUNetPlans139Exact__2d\fold_0\checkpoint_final.pth"
$runner = Join-Path $projectRoot "run_cell_pipeline_web.ps1"
$runsRoot = Join-Path $projectRoot "web_pipeline_runs_net141_training"
$reviewRoot = Join-Path $projectRoot "review_dataset"

foreach ($required in @($checkpoint, $runner)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Erforderliche Datei fehlt: $required"
    }
}

Write-Host "Netz 141 final: neue Trainingsdaten erzeugen" -ForegroundColor Cyan
Write-Host "Hysterese-Standard: $HysteresisProfile"
Write-Host "Webseite: http://127.0.0.1:$Port/"
Write-Host "Freigegebene Trainingsdaten: $(Join-Path $reviewRoot 'approved')"

& $runner `
    -Device $Device `
    -Port $Port `
    -RunsRoot $runsRoot `
    -ReviewMode "training" `
    -ReviewRoot $reviewRoot `
    -DatasetId 141 `
    -Trainer $trainer `
    -Plans "nnUNetPlans139Exact" `
    -Checkpoint "checkpoint_final.pth" `
    -HysteresisProfile $HysteresisProfile `
    -NoFiji `
    -NoBrowser:$NoBrowser
