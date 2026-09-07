param(
    [switch]$FullRun,
    [switch]$Continue,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$DatasetName = "Dataset138_cleanSingleCell_soma_skeleton_recrop"

if ($FullRun) {
    $Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly"
}
else {
    $Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnlyDebug50"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}

$env:nnUNet_raw = Join-Path $Project "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Project "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Project "nnUNet_results"
Remove-Item Env:nnUNet_extTrainer -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

$PreprocessedDataset = Join-Path $env:nnUNet_preprocessed $DatasetName
if (-not (Test-Path -LiteralPath $PreprocessedDataset -PathType Container)) {
    throw "Preprocessed Dataset138 not found: $PreprocessedDataset"
}

$ResultRoot = Join-Path $env:nnUNet_results $DatasetName
$ModelFolder = Join-Path $ResultRoot "${Trainer}__nnUNetPlans__2d"
$FoldFolder = Join-Path $ModelFolder "fold_0"
$FinalCheckpoint = Join-Path $FoldFolder "checkpoint_final.pth"
$LatestCheckpoint = Join-Path $FoldFolder "checkpoint_latest.pth"

if ((Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -and -not $Continue) {
    throw "This run is already complete: $FinalCheckpoint"
}

if ((Test-Path -LiteralPath $FoldFolder -PathType Container) -and -not $Continue) {
    $ExistingFiles = @(Get-ChildItem -LiteralPath $FoldFolder -File -ErrorAction SilentlyContinue)
    $NonLogFiles = @(
        $ExistingFiles | Where-Object { $_.Name -notlike "training_log_*.txt" }
    )
    if ($NonLogFiles.Count -gt 0) {
        throw "An incomplete result folder already exists. Use -Continue to resume it: $FoldFolder"
    }
    if ($ExistingFiles.Count -gt 0) {
        Write-Host "A previous start stopped before epoch 0. Keeping its log and retrying safely."
    }
}

if ($Continue -and -not (
    (Test-Path -LiteralPath $LatestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath (Join-Path $FoldFolder "checkpoint_best.pth") -PathType Leaf)
)) {
    throw "No checkpoint is available to continue: $FoldFolder"
}

$Mode = if ($FullRun) { "1000-epoch full run" } else { "50-epoch technical test" }
Write-Host "Experiment 01: full-resolution loss only"
Write-Host "Mode: $Mode"
Write-Host "Trainer: $Trainer"
Write-Host "Device: $Device"
Write-Host "Result folder: $FoldFolder"

$TrainingArguments = @(
    "-m", "nnunetv2.run.run_training",
    "138", "2d", "0",
    "-tr", $Trainer,
    "-p", "nnUNetPlans",
    "-device", $Device
)
if ($Continue) {
    $TrainingArguments += "--c"
}

& $Python @TrainingArguments
if ($LASTEXITCODE -ne 0) {
    throw "nnU-Net training failed with exit code $LASTEXITCODE"
}

$ValidationFolder = Join-Path $FoldFolder "validation"
$ValidationCount = 0
if (Test-Path -LiteralPath $ValidationFolder -PathType Container) {
    $ValidationCount = @(
        Get-ChildItem -LiteralPath $ValidationFolder -Filter "*.tif" -File
    ).Count
}

Write-Host ""
Write-Host "Training finished."
Write-Host "Final checkpoint: $FinalCheckpoint"
Write-Host "Validation TIFF files: $ValidationCount"
Write-Host "Progress image: $(Join-Path $FoldFolder 'progress.png')"
