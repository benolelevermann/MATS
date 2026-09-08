[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$Port = 8783,
    [switch]$NoBrowser,
    [switch]$NoFiji
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$semanticCheckpoint = Join-Path $projectRoot "nnUNet_results\Dataset141_dataset139_plus_net139_reviewed_cells\nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug__nnUNetPlans139Exact__2d\fold_0\checkpoint_final.pth"
$separatorCheckpoint = Join-Path $projectRoot "instance_separator_results\seeded_full80\checkpoint_final.pth"
$fiji = "C:\Program Files\Fiji.app\ImageJ-win64.exe"

foreach ($required in @($python, $semanticCheckpoint, $separatorCheckpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Erforderliche Datei fehlt: $required"
    }
}
if (-not $NoFiji -and -not (Test-Path -LiteralPath $fiji -PathType Leaf)) {
    throw "Fiji fehlt: $fiji"
}

$arguments = @(
    "-m", "blind_selection_web.server",
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--device", $Device,
    "--separator-checkpoint", $separatorCheckpoint
)
if (-not $NoBrowser) { $arguments += "--open-browser" }
if ($NoFiji) { $arguments += "--no-fiji" }

Write-Host "Blinde Zellauswahl + Evo-Pipeline" -ForegroundColor Cyan
Write-Host "Webseite: http://127.0.0.1:$Port/"
Write-Host "Netz 141 final | Hysterese T_low=0.20 | Zelltrenner Checkpoint 80"
Write-Host "Freigegebene Zellen: $(Join-Path $projectRoot 'evo_manual_selection_output\cells')"
Write-Host "Auswahldaten für ein späteres Netz: $(Join-Path $projectRoot 'blind_selection_learning_dataset')"

& $python @arguments
