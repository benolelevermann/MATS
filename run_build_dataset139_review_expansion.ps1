[CmdletBinding()]
param(
    [switch]$RunPreprocess,
    [switch]$Train,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$builder = Join-Path $projectRoot "build_dataset139_from_dataset138_and_reviewed.py"
$baseDataset = Join-Path $projectRoot "nnUNet_raw\Dataset138_cleanSingleCell_soma_skeleton_recrop"
$approvedDataset = Join-Path $projectRoot "review_dataset\approved"
$targetName = "Dataset139_dataset138_plus_reviewed_cells"
$targetDataset = Join-Path $projectRoot (Join-Path "nnUNet_raw" $targetName)
$baseSplits = Join-Path $projectRoot "nnUNet_preprocessed\Dataset138_cleanSingleCell_soma_skeleton_recrop\splits_final.json"
$targetSplits = Join-Path $targetDataset "splits_final_preserve_dataset138.json"

foreach ($required in @($python, $builder, $baseDataset, $approvedDataset, $baseSplits)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Erforderlicher Pfad fehlt: $required" }
}
if (Test-Path -LiteralPath $targetDataset) {
    throw "Dataset139 existiert bereits: $targetDataset`nEs wird aus Sicherheitsgruenden nicht ueberschrieben."
}
if ($Train -and -not $RunPreprocess) {
    throw "Fuer einen frischen Aufbau bitte -Train zusammen mit -RunPreprocess verwenden."
}

& $python $builder `
    --base-dataset $baseDataset `
    --approved-dataset $approvedDataset `
    --output-dataset $targetDataset `
    --base-splits $baseSplits
if ($LASTEXITCODE -ne 0) { throw "Dataset139 konnte nicht erstellt werden." }

Write-Host "Dataset139 wurde erstellt: $targetDataset" -ForegroundColor Green
if (-not $RunPreprocess) {
    Write-Host "Naechster Schritt: denselben Befehl nicht erneut starten, sondern Dataset139 preprocessen und trainieren."
    exit 0
}

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_results"
$preprocess = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_plan_and_preprocess.exe"
& $preprocess -d 139 -c 2d --verify_dataset_integrity
if ($LASTEXITCODE -ne 0) { throw "Preprocessing von Dataset139 ist fehlgeschlagen." }

$preprocessedTarget = Join-Path $env:nnUNet_preprocessed $targetName
Copy-Item -LiteralPath $targetSplits -Destination (Join-Path $preprocessedTarget "splits_final.json")
Write-Host "Dataset138-Split beibehalten; neue Review-Zellen liegen nur im Training." -ForegroundColor Green

if (-not $Train) { exit 0 }
& $python -m nnunetv2.run.run_training `
    139 2d 0 `
    -tr nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x `
    -p nnUNetPlans `
    -device $Device
if ($LASTEXITCODE -ne 0) { throw "Training von Dataset139 ist fehlgeschlagen." }

