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
$dataset = "Dataset143_dataset141_plus_synthetic_multicell"
$datasetId = "143"
$plans = "nnUNetPlans139Exact"
$trainer = if ($FullRun) {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugSyntheticMultiCellFineTune200"
} else {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugSyntheticMultiCellFineTune20"
}
$preprocessedDataset = Join-Path $projectRoot (Join-Path "nnUNet_preprocessed" $dataset)
$checkpoint141 = Join-Path $projectRoot "nnUNet_results\Dataset141_dataset139_plus_net139_reviewed_cells\nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug__nnUNetPlans139Exact__2d\fold_0\checkpoint_final.pth"
$externalTrainerPath = Join-Path $projectRoot "custom_trainers\skeleton_recall"

foreach ($required in @(
    $python,
    $preprocessedDataset,
    (Join-Path $preprocessedDataset "$plans.json"),
    (Join-Path $preprocessedDataset "splits_final.json"),
    $checkpoint141,
    $externalTrainerPath
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required. Dataset143 zuerst mit -Preprocess vorbereiten."
    }
}

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_results"
$holdoutGuard = Join-Path $projectRoot "evo_test_holdout.py"
& $python $holdoutGuard check --dataset (Join-Path $env:nnUNet_raw $dataset)
if ($LASTEXITCODE -ne 0) {
    throw "Training abgebrochen: Der permanente EvoTest/div10_CC-Hold-out ist im Datensatz enthalten."
}
$env:nnUNet_extTrainer = $externalTrainerPath
$pythonPathEntries = @($externalTrainerPath)
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $pythonPathEntries += $env:PYTHONPATH
}
$env:PYTHONPATH = $pythonPathEntries -join [IO.Path]::PathSeparator

$resultFolder = Join-Path $env:nnUNet_results "$dataset\${trainer}__${plans}__2d\fold_0"
$finalCheckpoint = Join-Path $resultFolder "checkpoint_final.pth"
$latestCheckpoint = Join-Path $resultFolder "checkpoint_latest.pth"
$bestCheckpoint = Join-Path $resultFolder "checkpoint_best.pth"
if ((Test-Path -LiteralPath $finalCheckpoint) -and -not $Continue) {
    throw "Dieser Dataset143-Lauf ist bereits abgeschlossen: $finalCheckpoint"
}
if ((Test-Path -LiteralPath $resultFolder) -and -not $Continue) {
    $existing = @(Get-ChildItem -LiteralPath $resultFolder -File -ErrorAction SilentlyContinue)
    $meaningful = @($existing | Where-Object { $_.Name -notlike "training_log_*.txt" })
    if ($meaningful.Count -gt 0) {
        throw "Ein Dataset143-Lauf wurde bereits begonnen. Zum Fortsetzen -Continue verwenden: $resultFolder"
    }
}
if ($Continue -and -not (
    (Test-Path -LiteralPath $latestCheckpoint) -or
    (Test-Path -LiteralPath $bestCheckpoint) -or
    (Test-Path -LiteralPath $finalCheckpoint)
)) {
    throw "Kein Dataset143-Checkpoint zum Fortsetzen vorhanden: $resultFolder"
}

$mode = if ($FullRun) { "200 Epochen" } else { "20-Epochen-Techniktest" }
Write-Host "Fine-Tuning von Netz 141 mit echten Einzelzellen und synthetischen Mehrzellbildern" -ForegroundColor Cyan
Write-Host "Modus: $mode; Lernrate: 0.001"
Write-Host "Startgewichte: $checkpoint141"
Write-Host "Trainer: $trainer"
Write-Host "Ergebnisse: $resultFolder"

$arguments = @(
    $datasetId, "2d", "0",
    "-tr", $trainer,
    "-p", $plans,
    "-device", $Device,
    "--npz"
)
if ($Continue) {
    Remove-Item Env:NNUNET_SYNTHETIC_MULTICELL_FINETUNE_CHECKPOINT -ErrorAction SilentlyContinue
    $arguments += "--c"
} else {
    $env:NNUNET_SYNTHETIC_MULTICELL_FINETUNE_CHECKPOINT = $checkpoint141
}

$trainingBootstrap = "import nnUNetTrainerSkeletonRecallCells, runpy; runpy.run_module('nnunetv2.run.run_training', run_name='__main__')"
& $python "-c" $trainingBootstrap @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Dataset143-Fine-Tuning ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}
Write-Host "Dataset143-Fine-Tuning ist abgeschlossen." -ForegroundColor Green
Write-Host "Ergebnisse: $resultFolder"
