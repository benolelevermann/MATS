param(
    [switch]$Force,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Dataset = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
$Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
$Validation = Join-Path $Project "nnUNet_results\$Dataset\${Trainer}__nnUNetPlans__2d\fold_0\validation"
$Checkpoint = Join-Path (Split-Path -Parent $Validation) "checkpoint_final.pth"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "R0 final checkpoint not found: $Checkpoint"
}

$TiffCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.tif" -File -ErrorAction SilentlyContinue).Count
$NpzCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count
if (-not $Force -and $TiffCount -gt 0 -and $NpzCount -eq $TiffCount) {
    Write-Host "R0 validation probabilities are already complete: $NpzCount files"
    Write-Host "Folder: $Validation"
    exit 0
}

$env:nnUNet_raw = Join-Path $Project "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Project "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Project "nnUNet_results"
$env:nnUNet_extTrainer = Join-Path $Project "custom_trainers\skeleton_recall"

Write-Host "Exporting R0 fold-0 validation probabilities without training"
Write-Host "Existing TIFF files: $TiffCount"
Write-Host "Existing NPZ files:  $NpzCount"
& $Python -m nnunetv2.run.run_training 138 2d 0 -tr $Trainer -p nnUNetPlans -device $Device --val --npz
if ($LASTEXITCODE -ne 0) {
    throw "R0 validation probability export failed with exit code $LASTEXITCODE"
}

$FinalNpzCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.npz" -File).Count
if ($FinalNpzCount -ne $TiffCount) {
    throw "Probability export is incomplete: $FinalNpzCount NPZ for $TiffCount TIFF files"
}
Write-Host "Probability export complete: $FinalNpzCount files"
Write-Host "Folder: $Validation"
