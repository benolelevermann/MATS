[CmdletBinding()]
param(
    [switch]$FullRun,
    [switch]$Continue,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataset = "Dataset139_dataset138_plus_reviewed_cells"
$trainer = if ($FullRun) {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug"
} else {
    "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugDebug50"
}
$preprocessedDataset = Join-Path $projectRoot (Join-Path "nnUNet_preprocessed" $dataset)
$splitFile = Join-Path $preprocessedDataset "splits_final.json"

foreach ($required in @($python, $preprocessedDataset, $splitFile)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}

$env:nnUNet_raw = Join-Path $projectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "nnUNet_results"
$externalTrainerPath = Join-Path $projectRoot "custom_trainers\skeleton_recall"
$env:nnUNet_extTrainer = $externalTrainerPath

# nnU-Net temporarily adds external trainer folders only while looking up the
# class. Windows worker processes must be able to import the same module again
# when the custom data loader is unpickled, so keep this folder on PYTHONPATH.
$pythonPathEntries = @($externalTrainerPath)
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $pythonPathEntries += $env:PYTHONPATH
}
$env:PYTHONPATH = $pythonPathEntries -join [IO.Path]::PathSeparator

$foldFolder = Join-Path $env:nnUNet_results "$dataset\${trainer}__nnUNetPlans__2d\fold_0"
$finalCheckpoint = Join-Path $foldFolder "checkpoint_final.pth"
$latestCheckpoint = Join-Path $foldFolder "checkpoint_latest.pth"
$bestCheckpoint = Join-Path $foldFolder "checkpoint_best.pth"

if ((Test-Path -LiteralPath $finalCheckpoint -PathType Leaf) -and -not $Continue) {
    throw "Dieser LabelSafe-Lauf ist bereits abgeschlossen: $finalCheckpoint"
}
if ((Test-Path -LiteralPath $foldFolder -PathType Container) -and -not $Continue) {
    $existingFiles = @(Get-ChildItem -LiteralPath $foldFolder -File -ErrorAction SilentlyContinue)
    $nonLogFiles = @($existingFiles | Where-Object { $_.Name -notlike "training_log_*.txt" })
    if ($nonLogFiles.Count -gt 0) {
        throw "Ein angefangener LabelSafe-Lauf existiert bereits. Zum Fortsetzen -Continue verwenden: $foldFolder"
    }
    if ($existingFiles.Count -gt 0) {
        Write-Host "Ein früherer Start endete vor Epoche 0. Das Log bleibt erhalten; der Start wird sicher wiederholt."
    }
}
if ($Continue -and -not (
    (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf)
)) {
    throw "Es gibt keinen LabelSafe-Checkpoint zum Fortsetzen: $foldFolder"
}

$mode = if ($FullRun) { "1000 Epochen" } else { "50-Epochen-Techniktest" }
Write-Host "Dataset139: Skeleton-Recall mit label-sicherer Augmentation" -ForegroundColor Cyan
Write-Host "Modus: $mode"
Write-Host "Trainer: $trainer"
Write-Host "Freie Rotation und Skalierung: deaktiviert"
Write-Host "Exakte 90-Grad-Rotation und Spiegelung: aktiviert"
Write-Host "Ergebnisse: $foldFolder"

$arguments = @(
    "139", "2d", "0",
    "-tr", $trainer,
    "-p", "nnUNetPlans",
    "-device", $Device,
    "--npz"
)
if ($Continue) { $arguments += "--c" }

# Preload the flat external module before nnU-Net performs its temporary class
# search. This keeps the exact same class object registered for Windows spawn.
$trainingBootstrap = "import nnUNetTrainerSkeletonRecallCells, runpy; runpy.run_module('nnunetv2.run.run_training', run_name='__main__')"
& $python "-c" $trainingBootstrap @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Dataset139-LabelSafe-Training ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}

Write-Host "Dataset139-LabelSafe-Training ist abgeschlossen." -ForegroundColor Green
Write-Host "Ergebnisse: $foldFolder"
