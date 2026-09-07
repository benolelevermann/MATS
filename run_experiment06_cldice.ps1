param(
    [switch]$FullRun,
    [switch]$Continue,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Dataset = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
$Trainer = if ($FullRun) {
    "nnUNetTrainerClDiceCellsAlpha01"
} else {
    "nnUNetTrainerClDiceCellsAlpha01Debug50"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment not found: $Python" }
$env:nnUNet_raw = Join-Path $Project "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Project "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Project "nnUNet_results"
$env:nnUNet_extTrainer = Join-Path $Project "custom_trainers\cldice"

$FoldFolder = Join-Path $env:nnUNet_results "$Dataset\${Trainer}__nnUNetPlans__2d\fold_0"
$FinalCheckpoint = Join-Path $FoldFolder "checkpoint_final.pth"
$LatestCheckpoint = Join-Path $FoldFolder "checkpoint_latest.pth"
$BestCheckpoint = Join-Path $FoldFolder "checkpoint_best.pth"
if ((Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -and -not $Continue) {
    throw "This clDice run is already complete: $FinalCheckpoint"
}
if ((Test-Path -LiteralPath $FoldFolder -PathType Container) -and -not $Continue) {
    $ExistingFiles = @(Get-ChildItem -LiteralPath $FoldFolder -File -ErrorAction SilentlyContinue)
    $NonLogFiles = @($ExistingFiles | Where-Object { $_.Name -notlike "training_log_*.txt" })
    if ($NonLogFiles.Count -gt 0) {
        throw "An incomplete clDice result folder already exists. Use -Continue: $FoldFolder"
    }
    if ($ExistingFiles.Count -gt 0) {
        Write-Host "A previous start stopped before epoch 0. Keeping its log and retrying safely."
    }
}
if ($Continue -and -not (
    (Test-Path -LiteralPath $LatestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $BestCheckpoint -PathType Leaf)
)) { throw "No clDice checkpoint is available to continue: $FoldFolder" }

$Mode = if ($FullRun) { "1000-epoch full run" } else { "50-epoch technical test" }
Write-Host "Experiment R6: class-1 soft-clDice"
Write-Host "Mode: $Mode"
Write-Host "Trainer: $Trainer"
Write-Host "clDice alpha/iterations: 0.1 / 5"
Write-Host "Deep supervision: standard Dice/CE on all configured scales; clDice only full resolution"
Write-Host "Inference output: unchanged semantic labels 0/1/2"

$Arguments = @(
    "-m", "nnunetv2.run.run_training",
    "138", "2d", "0",
    "-tr", $Trainer,
    "-p", "nnUNetPlans",
    "-device", $Device,
    "--npz"
)
if ($Continue) { $Arguments += "--c" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "clDice training failed with exit code $LASTEXITCODE" }

$Validation = Join-Path $FoldFolder "validation"
$TiffCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.tif" -File -ErrorAction SilentlyContinue).Count
$NpzCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count
if ($TiffCount -eq 0 -or $NpzCount -eq 0) {
    throw "Training ended, but validation TIFF/NPZ output is incomplete: $TiffCount/$NpzCount"
}
Write-Host "clDice run finished. Validation TIFF/NPZ: $TiffCount/$NpzCount"
Write-Host "Result folder: $FoldFolder"
