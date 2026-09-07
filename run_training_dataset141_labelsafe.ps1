[CmdletBinding()]
param(
    [switch]$FullRun,
    [switch]$Continue,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataset = "Dataset141_dataset139_plus_net139_reviewed_cells"
$plans = "nnUNetPlans139Exact"
$trainer = if ($FullRun) {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug"
} else {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugDebug50"
}
$preprocessedDataset = Join-Path $projectRoot (Join-Path "nnUNet_preprocessed" $dataset)
$splitFile = Join-Path $preprocessedDataset "splits_final.json"
$plansFile = Join-Path $preprocessedDataset "$plans.json"

foreach ($required in @($python, $preprocessedDataset, $splitFile, $plansFile)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_results"
$externalTrainerPath = Join-Path $projectRoot "custom_trainers\skeleton_recall"
$env:nnUNet_extTrainer = $externalTrainerPath
$pythonPathEntries = @($externalTrainerPath)
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $pythonPathEntries += $env:PYTHONPATH
}
$env:PYTHONPATH = $pythonPathEntries -join [IO.Path]::PathSeparator

$foldFolder = Join-Path $env:nnUNet_results "$dataset\${trainer}__${plans}__2d\fold_0"
$finalCheckpoint = Join-Path $foldFolder "checkpoint_final.pth"
$latestCheckpoint = Join-Path $foldFolder "checkpoint_latest.pth"
$bestCheckpoint = Join-Path $foldFolder "checkpoint_best.pth"

if ((Test-Path -LiteralPath $finalCheckpoint -PathType Leaf) -and -not $Continue) {
    throw "Dieser Dataset141-Lauf ist bereits abgeschlossen: $finalCheckpoint"
}
if ((Test-Path -LiteralPath $foldFolder -PathType Container) -and -not $Continue) {
    $existingFiles = @(Get-ChildItem -LiteralPath $foldFolder -File -ErrorAction SilentlyContinue)
    $nonLogFiles = @($existingFiles | Where-Object { $_.Name -notlike "training_log_*.txt" })
    if ($nonLogFiles.Count -gt 0) {
        throw "Ein angefangener Dataset141-Lauf existiert bereits. Zum Fortsetzen -Continue verwenden: $foldFolder"
    }
}
if ($Continue -and -not (
    (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf)
)) {
    throw "Es gibt keinen Dataset141-Checkpoint zum Fortsetzen: $foldFolder"
}

$mode = if ($FullRun) { "1000 Epochen" } else { "50-Epochen-Techniktest" }
Write-Host "Dataset141: dieselbe Netz-139-Konfiguration mit 597 neuen Review-Zellen" -ForegroundColor Cyan
Write-Host "Modus: $mode"
Write-Host "Trainer: $trainer"
Write-Host "Plans: $plans (exakte Kopie von Dataset139)"
Write-Host "Ergebnisse: $foldFolder"

$arguments = @(
    "141", "2d", "0",
    "-tr", $trainer,
    "-p", $plans,
    "-device", $Device,
    "--npz"
)
if ($Continue) { $arguments += "--c" }

$trainingBootstrap = "import nnUNetTrainerSkeletonRecallCells, runpy; runpy.run_module('nnunetv2.run.run_training', run_name='__main__')"
& $python "-c" $trainingBootstrap @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Dataset141-Training ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}

Write-Host "Dataset141-Training ist abgeschlossen." -ForegroundColor Green
Write-Host "Ergebnisse: $foldFolder"
