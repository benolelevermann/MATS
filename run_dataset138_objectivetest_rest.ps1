# Predicts the remaining objective-test folders with Dataset138 (40X excluded).
$ErrorActionPreference = "Stop"
$root = "C:\Ole\20260721_CellClassification_v2"
$env:nnUNet_raw          = "$root\nnUNet_raw"
$env:nnUNet_preprocessed = "$root\nnUNet_preprocessed"
$env:nnUNet_results      = "$root\nnUNet_results"
$predict = "$root\.venv\Scripts\nnUNetv2_predict.exe"
$inBase  = "$root\20260817_ObjectiveTest\01_inputimages"
$outBase = "$root\20260817_ObjectiveTest\03_dataset138_predictions"

$skip = @("10X_CF","10X_WF","10X_CF_rollingball50pixel","40X_CF","40X_WF")
$folders = Get-ChildItem $inBase -Directory | Where-Object { $skip -notcontains $_.Name } | Sort-Object Name

Write-Host ("Zu predicten: {0} Ordner" -f $folders.Count)
$i = 0
foreach ($f in $folders) {
    $i++
    $out = Join-Path $outBase $f.Name
    Write-Host ""
    Write-Host ("=== [{0}/{1}] {2} ===" -f $i, $folders.Count, $f.Name) -ForegroundColor Cyan
    $t0 = Get-Date
    & $predict -i $f.FullName -o $out -d 138 -c 2d -f 0 `
        -tr nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x -p nnUNetPlans `
        --save_probabilities
    if ($LASTEXITCODE -ne 0) { throw "Prediction fehlgeschlagen fuer $($f.Name)" }
    Write-Host ("--> {0} fertig in {1:N1} min" -f $f.Name, ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
}
Write-Host ""
Write-Host "REST FERTIG (ohne 40X)" -ForegroundColor Green
