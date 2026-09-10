[CmdletBinding()]
param(
    [switch]$Continue,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataset = "Dataset139_dataset138_plus_reviewed_cells"
$trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
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
$holdoutGuard = Join-Path $projectRoot "evo_test_holdout.py"
& $python $holdoutGuard check --dataset (Join-Path $env:nnUNet_raw $dataset)
if ($LASTEXITCODE -ne 0) {
    throw "Training abgebrochen: Der permanente EvoTest/div10_CC-Hold-out ist im Datensatz enthalten."
}

$foldFolder = Join-Path $env:nnUNet_results "$dataset\${trainer}__nnUNetPlans__2d\fold_0"
$finalCheckpoint = Join-Path $foldFolder "checkpoint_final.pth"
$latestCheckpoint = Join-Path $foldFolder "checkpoint_latest.pth"
$bestCheckpoint = Join-Path $foldFolder "checkpoint_best.pth"

if ((Test-Path -LiteralPath $finalCheckpoint -PathType Leaf) -and -not $Continue) {
    throw "Das Training ist bereits abgeschlossen: $finalCheckpoint"
}
if ((Test-Path -LiteralPath $foldFolder -PathType Container) -and -not $Continue) {
    $existingFiles = @(Get-ChildItem -LiteralPath $foldFolder -File -ErrorAction SilentlyContinue)
    if ($existingFiles.Count -gt 0) {
        throw "Ein angefangener Lauf existiert bereits. Zum Fortsetzen bitte -Continue verwenden: $foldFolder"
    }
}
if ($Continue -and -not (
    (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf) -or
    (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf)
)) {
    throw "Es gibt noch keinen Checkpoint zum Fortsetzen: $foldFolder"
}

Write-Host "Dataset139: bekanntes Skeleton-Recall-Modell mit den freigegebenen Review-Zellen"
Write-Host "Trainer: $trainer"
Write-Host "Geraet: $Device"
Write-Host "Ergebnisse: $foldFolder"

$arguments = @(
    "-m", "nnunetv2.run.run_training",
    "139", "2d", "0",
    "-tr", $trainer,
    "-p", "nnUNetPlans",
    "-device", $Device,
    "--npz"
)
if ($Continue) { $arguments += "--c" }

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Training von Dataset139 ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}

Write-Host "Training von Dataset139 ist abgeschlossen." -ForegroundColor Green
Write-Host "Ergebnisse: $foldFolder"
