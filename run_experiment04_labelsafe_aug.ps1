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
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug"
} else {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugDebug50"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment not found: $Python" }
$env:nnUNet_raw = Join-Path $Project "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Project "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Project "nnUNet_results"
$env:nnUNet_extTrainer = Join-Path $Project "custom_trainers\skeleton_recall"

$FoldFolder = Join-Path $env:nnUNet_results "$Dataset\${Trainer}__nnUNetPlans__2d\fold_0"
$FinalCheckpoint = Join-Path $FoldFolder "checkpoint_final.pth"
$LatestCheckpoint = Join-Path $FoldFolder "checkpoint_latest.pth"
if ((Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -and -not $Continue) {
    throw "This R4 run is already complete: $FinalCheckpoint"
}
if ((Test-Path -LiteralPath $FoldFolder -PathType Container) -and -not $Continue) {
    $ExistingFiles = @(Get-ChildItem -LiteralPath $FoldFolder -File -ErrorAction SilentlyContinue)
    $NonLogFiles = @($ExistingFiles | Where-Object { $_.Name -notlike "training_log_*.txt" })
    if ($NonLogFiles.Count -gt 0) {
        throw "An incomplete R4 result folder already exists. Use -Continue to resume it: $FoldFolder"
    }
    if ($ExistingFiles.Count -gt 0) {
        Write-Host "A previous R4 start stopped before epoch 0. Keeping its log and retrying safely."
    }
}
if ($Continue -and -not (
    (Test-Path -LiteralPath $LatestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath (Join-Path $FoldFolder "checkpoint_best.pth") -PathType Leaf)
)) { throw "No R4 checkpoint is available to continue: $FoldFolder" }

$Mode = if ($FullRun) { "1000-epoch full run" } else { "50-epoch technical test" }
Write-Host "Experiment R4: label-safe spatial augmentation"
Write-Host "Mode: $Mode"
Write-Host "Trainer: $Trainer"
Write-Host "Continuous rotation/scaling: disabled"
Write-Host "Exact Rot90 and mirroring: enabled"

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
if ($LASTEXITCODE -ne 0) { throw "R4 training failed with exit code $LASTEXITCODE" }

$Validation = Join-Path $FoldFolder "validation"
$TiffCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.tif" -File -ErrorAction SilentlyContinue).Count
$NpzCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count
Write-Host "R4 finished. Validation TIFF/NPZ: $TiffCount/$NpzCount"
Write-Host "Result folder: $FoldFolder"
