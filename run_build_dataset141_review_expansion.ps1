[CmdletBinding()]
param(
    [switch]$RunPreprocess,
    [switch]$ResumePreprocess,
    [switch]$Train,
    [switch]$FullRun,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$builder = Join-Path $projectRoot "build_dataset141_from_dataset139_and_reviewed.py"
$baseName = "Dataset139_dataset138_plus_reviewed_cells"
$targetName = "Dataset141_dataset139_plus_net139_reviewed_cells"
$baseDataset = Join-Path $projectRoot (Join-Path "nnUNet_raw" $baseName)
$approvedDataset = Join-Path $projectRoot "review_dataset\approved"
$targetDataset = Join-Path $projectRoot (Join-Path "nnUNet_raw" $targetName)
$basePreprocessed = Join-Path $projectRoot (Join-Path "nnUNet_preprocessed" $baseName)
$baseSplits = Join-Path $basePreprocessed "splits_final.json"
$targetSplits = Join-Path $targetDataset "splits_final_preserve_dataset139.json"
$plansName = "nnUNetPlans139Exact"
$trainingRunner = Join-Path $projectRoot "run_training_dataset141_labelsafe.ps1"

foreach ($required in @($python, $builder, $baseDataset, $approvedDataset, $baseSplits)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}
if ($ResumePreprocess) {
    $RunPreprocess = $true
}
if ((Test-Path -LiteralPath $targetDataset) -and -not $ResumePreprocess) {
    throw "Dataset141 existiert bereits: $targetDataset`nZum Fortsetzen nach einem abgebrochenen Preprocessing -ResumePreprocess verwenden."
}
if ($Train -and -not $RunPreprocess) {
    throw "Zum Trainieren muss auch -RunPreprocess gesetzt sein."
}

if (-not $ResumePreprocess) {
    & $python $builder `
        --base-dataset $baseDataset `
        --approved-dataset $approvedDataset `
        --output-dataset $targetDataset `
        --base-splits $baseSplits
    if ($LASTEXITCODE -ne 0) { throw "Dataset141 konnte nicht erstellt werden." }
    Write-Host "Dataset141 wurde erstellt: $targetDataset" -ForegroundColor Green
} else {
    $summaryPath = Join-Path $targetDataset "build_summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "Dataset141 ist kein vollstaendig aufgebauter Rohdatensatz: $summaryPath fehlt."
    }
    Write-Host "Vorhandenes Dataset141 wird nur ab dem Preprocessing fortgesetzt." -ForegroundColor Yellow
}
$holdoutGuard = Join-Path $projectRoot "evo_test_holdout.py"
& $python $holdoutGuard check --dataset $targetDataset
if ($LASTEXITCODE -ne 0) {
    throw "Dataset141 enthaelt den permanenten EvoTest/div10_CC-Hold-out und wird nicht weiterverarbeitet."
}
if (-not $RunPreprocess) { return }

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_results"
$fingerprint = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_extract_fingerprint.exe"
$movePlans = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_move_plans_between_datasets.exe"
$preprocess = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_preprocess.exe"

& $fingerprint -d 141 --verify_dataset_integrity --clean
if ($LASTEXITCODE -ne 0) { throw "Integritaetspruefung von Dataset141 ist fehlgeschlagen." }

& $movePlans -s 139 -t 141 -sp nnUNetPlans -tp $plansName
if ($LASTEXITCODE -ne 0) { throw "Die exakten Dataset139-Plans konnten nicht uebernommen werden." }

$targetPreprocessed = Join-Path $env:nnUNet_preprocessed $targetName
Copy-Item -LiteralPath (Join-Path $targetDataset "dataset.json") -Destination (Join-Path $targetPreprocessed "dataset.json") -Force

& $preprocess -d 141 -plans_name $plansName -c 2d
if ($LASTEXITCODE -ne 0) { throw "Preprocessing von Dataset141 ist fehlgeschlagen." }

Copy-Item -LiteralPath $targetSplits -Destination (Join-Path $targetPreprocessed "splits_final.json")

$sourcePlans = Get-Content -LiteralPath (Join-Path $basePreprocessed "nnUNetPlans.json") -Raw | ConvertFrom-Json
$targetPlans = Get-Content -LiteralPath (Join-Path $targetPreprocessed "$plansName.json") -Raw | ConvertFrom-Json
$sourceConfiguration = $sourcePlans.configurations.'2d'
$targetConfiguration = $targetPlans.configurations.'2d'
foreach ($property in @("patch_size", "batch_size", "spacing", "normalization_schemes", "architecture", "batch_dice")) {
    $sourceValue = $sourceConfiguration.$property | ConvertTo-Json -Depth 100 -Compress
    $targetValue = $targetConfiguration.$property | ConvertTo-Json -Depth 100 -Compress
    if ($sourceValue -ne $targetValue) {
        throw "Dataset141 weicht bei '$property' von den Dataset139-Plans ab."
    }
}
Write-Host "Dataset139-Split und exakte 160x160-Plans wurden uebernommen." -ForegroundColor Green

if ($Train) {
    & $trainingRunner -Device $Device -FullRun:$FullRun
    if ($LASTEXITCODE -ne 0) { throw "Training von Dataset141 ist fehlgeschlagen." }
}
