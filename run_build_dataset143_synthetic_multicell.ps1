[CmdletBinding()]
param(
    [switch]$Preprocess,
    [int]$Seed = 143,
    [int]$PreviewCount = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$builder = Join-Path $projectRoot "build_dataset143_synthetic_multicell.py"
$baseDataset = Join-Path $projectRoot "nnUNet_raw\Dataset141_dataset139_plus_net139_reviewed_cells"
$baseSplits = Join-Path $projectRoot "nnUNet_preprocessed\Dataset141_dataset139_plus_net139_reviewed_cells\splits_final.json"
$dataset = "Dataset143_dataset141_plus_synthetic_multicell"
$datasetId = "143"
$rawDataset = Join-Path $projectRoot (Join-Path "nnUNet_raw" $dataset)
$preprocessedDataset = Join-Path $projectRoot (Join-Path "nnUNet_preprocessed" $dataset)
$plans = "nnUNetPlans139Exact"

foreach ($required in @($python, $builder, $baseDataset, $baseSplits)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_results"

if (Test-Path -LiteralPath $rawDataset) {
    foreach ($required in @(
        (Join-Path $rawDataset "dataset.json"),
        (Join-Path $rawDataset "build_summary.json"),
        (Join-Path $rawDataset "splits_final.json"),
        (Join-Path $rawDataset "synthetic_multicell_review.html")
    )) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Dataset143 existiert, ist aber unvollstaendig. Es wird nicht automatisch ueberschrieben: $required"
        }
    }
    Write-Host "Der vorhandene, vollstaendige Dataset143-Rohordner wird wiederverwendet."
} else {
    Write-Host "Erzeuge reproduzierbare Mehrzellbilder aus allen geeigneten Dataset141-Zellen ..." -ForegroundColor Cyan
    & $python $builder `
        --base-dataset $baseDataset `
        --base-splits $baseSplits `
        --output-dataset $rawDataset `
        --seed $Seed `
        --preview-count $PreviewCount
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset143 konnte nicht aufgebaut werden."
    }
}

Write-Host "Rohdaten: $rawDataset" -ForegroundColor Green
Write-Host "HTML-Uebersicht: $(Join-Path $rawDataset 'synthetic_multicell_review.html')"
if (-not $Preprocess) {
    Write-Host "Zum anschliessenden nnU-Net-Preprocessing dasselbe Skript mit -Preprocess starten."
    exit 0
}

$fingerprint = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_extract_fingerprint.exe"
$movePlans = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_move_plans_between_datasets.exe"
$preprocessExecutable = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_preprocess.exe"
foreach ($required in @($fingerprint, $movePlans, $preprocessExecutable)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "nnU-Net-Werkzeug fehlt: $required"
    }
}

Write-Host "Pruefe Dataset143 und bereite es mit den exakten Netz-141-Plans vor ..." -ForegroundColor Cyan
& $fingerprint -d $datasetId --verify_dataset_integrity -np 2
if ($LASTEXITCODE -ne 0) { throw "Integritaetspruefung/Fingerprint von Dataset143 ist fehlgeschlagen." }
Copy-Item -LiteralPath (Join-Path $rawDataset "dataset.json") `
    -Destination (Join-Path $preprocessedDataset "dataset.json") -Force
& $movePlans -s 141 -t $datasetId -sp $plans -tp $plans
if ($LASTEXITCODE -ne 0) { throw "Die Dataset141-Plans konnten nicht uebernommen werden." }
& $preprocessExecutable -d $datasetId -plans_name $plans -c 2d -np 2
if ($LASTEXITCODE -ne 0) { throw "Preprocessing von Dataset143 ist fehlgeschlagen." }
Copy-Item -LiteralPath (Join-Path $rawDataset "splits_final.json") `
    -Destination (Join-Path $preprocessedDataset "splits_final.json") -Force

Write-Host "Dataset143 ist bereit fuer das Fine-Tuning." -ForegroundColor Green
Write-Host "Trainingsstart (Techniktest): .\run_training_dataset143_synthetic_multicell.ps1"
Write-Host "Vollstaendiger Lauf: .\run_training_dataset143_synthetic_multicell.ps1 -FullRun"
