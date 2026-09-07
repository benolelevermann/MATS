[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$Port = 8776,
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
$fiji = "C:\Program Files\Fiji.app\ImageJ-win64.exe"
$runsRoot = Join-Path $projectRoot "web_pipeline_runs_net141_final"
$reviewRoot = Join-Path $projectRoot "evo_pipeline_input_net141"

foreach ($required in @($checkpoint, $runner, $fiji)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Erforderliche Datei fehlt: $required"
    }
}

Write-Host "Netz 141 final: Evo-Zellauswahl" -ForegroundColor Cyan
Write-Host "Hysterese-Standard: $HysteresisProfile"
Write-Host "Webseite: http://127.0.0.1:$Port/"
Write-Host "Freigegebene Evo-Zellen: $(Join-Path $reviewRoot 'cells')"

& $runner `
    -Device $Device `
    -Port $Port `
    -RunsRoot $runsRoot `
    -ReviewMode "evo" `
    -ReviewRoot $reviewRoot `
    -DatasetId 141 `
    -Trainer $trainer `
    -Plans "nnUNetPlans139Exact" `
    -Checkpoint "checkpoint_final.pth" `
    -HysteresisProfile $HysteresisProfile `
    -NoBrowser:$NoBrowser
