# Predicts the three 10X objective-test images with the Dataset138 network.
$ErrorActionPreference = "Stop"
$root = "C:\Ole\20260721_CellClassification_v2"
$env:nnUNet_raw          = "$root\nnUNet_raw"
$env:nnUNet_preprocessed = "$root\nnUNet_preprocessed"
$env:nnUNet_results      = "$root\nnUNet_results"
$predict = "$root\.venv\Scripts\nnUNetv2_predict.exe"
$inBase  = "$root\20260817_ObjectiveTest\01_inputimages"
$outBase = "$root\20260817_ObjectiveTest\03_dataset138_predictions"

$folders = @("10X_CF", "10X_WF", "10X_CF_rollingball50pixel")

foreach ($f in $folders) {
    $in  = Join-Path $inBase $f
    $out = Join-Path $outBase $f
    if (-not (Test-Path $in)) { throw "Eingabeordner fehlt: $in" }
    Write-Host ""
    Write-Host "=== $f ===" -ForegroundColor Cyan
    $t0 = Get-Date
    & $predict -i $in -o $out -d 138 -c 2d -f 0 `
        -tr nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x -p nnUNetPlans `
        --save_probabilities
    if ($LASTEXITCODE -ne 0) { throw "Prediction fehlgeschlagen fuer $f" }
    Write-Host ("--> $f fertig in {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
}
Write-Host ""
Write-Host "ALLE DREI FERTIG" -ForegroundColor Green
