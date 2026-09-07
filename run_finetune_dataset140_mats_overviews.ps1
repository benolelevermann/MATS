[CmdletBinding()]
param(
    [switch]$Prepare,
    [switch]$PrepareOnly,
    [switch]$FullRun,
    [switch]$Continue,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$builder = Join-Path $projectRoot "build_dataset140_mats_overview_finetune.py"
$resumeRunner = Join-Path $projectRoot "continue_nnunet_from_checkpoint.py"
$exports = "X:\Ole\Projects\evoView\20260831_OverViewImage\MATS_annotations\exports"
$dataset = "Dataset140_matsOverview_finetune139"
$datasetId = "140"
$rawDataset = Join-Path $projectRoot (Join-Path "nnUNet_raw" $dataset)
$preprocessedDataset = Join-Path $projectRoot (Join-Path "nnUNet_preprocessed" $dataset)
$sourcePlansName = "nnUNetPlans"
$targetPlansName = "nnUNetPlans139Transfer"
$splitSource = Join-Path $rawDataset "splits_final_overview.json"
$splitTarget = Join-Path $preprocessedDataset "splits_final.json"
$externalTrainerPath = Join-Path $projectRoot "custom_trainers\skeleton_recall"
$checkpoint139 = Join-Path $projectRoot "nnUNet_results\Dataset139_dataset138_plus_reviewed_cells\nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug__nnUNetPlans__2d\fold_0\checkpoint_final.pth"
$trainer = if ($FullRun) {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugMATSFineTune100"
} else {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugMATSFineTune5"
}

foreach ($required in @($python, $builder, $resumeRunner, $exports, $externalTrainerPath, $checkpoint139)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}
if ($Prepare -and $Continue) {
    throw "-Prepare und -Continue duerfen nicht zusammen verwendet werden."
}
if ($PrepareOnly -and -not $Prepare) {
    throw "-PrepareOnly muss zusammen mit -Prepare verwendet werden."
}

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_results"
$env:nnUNet_extTrainer = $externalTrainerPath
$pythonPathEntries = @($externalTrainerPath)
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $pythonPathEntries += $env:PYTHONPATH
}
$env:PYTHONPATH = $pythonPathEntries -join [IO.Path]::PathSeparator

if ($Prepare) {
    if (Test-Path -LiteralPath $rawDataset) {
        foreach ($required in @(
            (Join-Path $rawDataset "dataset.json"),
            (Join-Path $rawDataset "build_summary.json"),
            $splitSource
        )) {
            if (-not (Test-Path -LiteralPath $required)) {
                throw "Dataset140 ist unvollstaendig; bitte nicht automatisch ueberschreiben: $required"
            }
        }
        Write-Host "Der gepruefte Dataset140-Rohordner ist bereits vorhanden; Aufbau wird uebersprungen."
    } else {
        Write-Host "Setze die neuesten MATS-Patches zu vollstaendigen Uebersichtsbildern zusammen ..." -ForegroundColor Cyan
        & $python $builder `
            --exports-dir $exports `
            --output-dataset $rawDataset `
            --validation-image-id "Mosaic001_Merged-3"
        if ($LASTEXITCODE -ne 0) { throw "Dataset140 konnte nicht aufgebaut werden." }
    }

    $fingerprint = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_extract_fingerprint.exe"
    $movePlans = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_move_plans_between_datasets.exe"
    $preprocess = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_preprocess.exe"
    & $fingerprint -d $datasetId --verify_dataset_integrity -np 2
    if ($LASTEXITCODE -ne 0) { throw "Integritaetspruefung/Fingerprint von Dataset140 ist fehlgeschlagen." }
    # The standalone fingerprint command creates the target directory but does
    # not copy dataset.json (the combined planner normally does this).
    Copy-Item -LiteralPath (Join-Path $rawDataset "dataset.json") `
        -Destination (Join-Path $preprocessedDataset "dataset.json") -Force
    & $movePlans -s 139 -t $datasetId -sp $sourcePlansName -tp $targetPlansName
    if ($LASTEXITCODE -ne 0) { throw "Die kompatiblen Dataset139-Plans konnten nicht uebernommen werden." }
    & $preprocess -d $datasetId -plans_name $targetPlansName -c 2d -np 2
    if ($LASTEXITCODE -ne 0) { throw "Preprocessing von Dataset140 ist fehlgeschlagen." }
    Copy-Item -LiteralPath $splitSource -Destination $splitTarget -Force
    Write-Host "Dataset140 ist vorbereitet: Mosaic 1+2 Training, Mosaic 3 Validierung." -ForegroundColor Green
    if ($PrepareOnly) {
        Write-Host "Vorbereitung abgeschlossen; Training wurde noch nicht gestartet."
        exit 0
    }
}

foreach ($required in @($rawDataset, $preprocessedDataset, $splitTarget)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Dataset140 ist noch nicht vorbereitet. Zuerst mit -Prepare starten. Fehlend: $required"
    }
}

$resultFolder = Join-Path $env:nnUNet_results "$dataset\${trainer}__${targetPlansName}__2d\fold_0"
$finalCheckpoint = Join-Path $resultFolder "checkpoint_final.pth"
$latestCheckpoint = Join-Path $resultFolder "checkpoint_latest.pth"
$bestCheckpoint = Join-Path $resultFolder "checkpoint_best.pth"
if ((Test-Path -LiteralPath $finalCheckpoint) -and -not $Continue) {
    throw "Dieser Fine-Tuning-Lauf ist bereits abgeschlossen: $finalCheckpoint"
}
if ($Continue -and -not (
    (Test-Path -LiteralPath $latestCheckpoint) -or
    (Test-Path -LiteralPath $bestCheckpoint) -or
    (Test-Path -LiteralPath $finalCheckpoint)
)) {
    throw "Kein Checkpoint zum Fortsetzen vorhanden: $resultFolder"
}
if ((Test-Path -LiteralPath $resultFolder) -and -not $Continue) {
    $existing = @(Get-ChildItem -LiteralPath $resultFolder -File -ErrorAction SilentlyContinue)
    $meaningful = @($existing | Where-Object { $_.Name -notlike "training_log_*.txt" })
    if ($meaningful.Count -gt 0) {
        throw "Ein Fine-Tuning-Lauf wurde bereits begonnen. Zum Fortsetzen -Continue verwenden: $resultFolder"
    }
}

$mode = if ($FullRun) { "100 Epochen, Lernrate 0.001" } else { "5-Epochen-Techniktest, Lernrate 0.001" }
Write-Host "Fine-Tuning von Netz 139 auf vollstaendigen MATS-Uebersichtsbildern" -ForegroundColor Cyan
Write-Host "Modus: $mode"
Write-Host "Training: Mosaic001_Merged-1 und -2"
Write-Host "Validierung: Mosaic001_Merged-3"
Write-Host "Netz-Patches: 160 x 160 px (identisch zu Netz 139)"
Write-Host "Startgewichte: $checkpoint139"
Write-Host "Ergebnisse: $resultFolder"

$arguments = @(
    $datasetId, "2d", "0",
    "-tr", $trainer,
    "-p", $targetPlansName,
    "-device", $Device,
    "--npz"
)
if ($Continue) {
    Remove-Item Env:NNUNET_MATS_FINETUNE_CHECKPOINT -ErrorAction SilentlyContinue
    $resumeCheckpoint = @($latestCheckpoint, $bestCheckpoint, $finalCheckpoint) |
        Where-Object { Test-Path -LiteralPath $_ } |
        ForEach-Object { Get-Item -LiteralPath $_ } |
        Sort-Object LastWriteTime |
        Select-Object -Last 1
    Write-Host "Fortsetzung ab neuestem Checkpoint: $($resumeCheckpoint.Name)" -ForegroundColor Green
    & $python $resumeRunner `
        $datasetId "2d" "0" `
        --trainer $trainer `
        --plans $targetPlansName `
        --checkpoint $resumeCheckpoint.FullName `
        --bootstrap-module "nnUNetTrainerSkeletonRecallCells" `
        --device $Device `
        --npz
} else {
    # The generic nnU-Net pretrained-weights option deliberately skips the
    # segmentation heads. This dedicated trainer loads the complete state.
    $env:NNUNET_MATS_FINETUNE_CHECKPOINT = $checkpoint139
    $trainingBootstrap = "import nnUNetTrainerSkeletonRecallCells, runpy; runpy.run_module('nnunetv2.run.run_training', run_name='__main__')"
    & $python "-c" $trainingBootstrap @arguments
}
if ($LASTEXITCODE -ne 0) {
    throw "Dataset140-Fine-Tuning ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}
Write-Host "Dataset140-Fine-Tuning ist abgeschlossen." -ForegroundColor Green
Write-Host "Ergebnisse: $resultFolder"
