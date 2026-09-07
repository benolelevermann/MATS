param(
    [switch]$ForcePrediction,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Project = "C:\Ole\20260721_CellClassification_v2"
$RunRoot = Join-Path $Project "20260730_TestBleb_v4"
$InputDir = Join-Path $RunRoot "01_inputimages"
$BaselineDir = Join-Path $RunRoot "02_predictions"
$RecallDir = Join-Path $RunRoot "02_predictions_skeleton_recall"
$ReportDir = Join-Path $RunRoot "06_model_comparison_skeleton_recall"
$Predictor = Join-Path $Project ".venv\Scripts\nnUNetv2_predict.exe"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$ReportScript = Join-Path $Project "make_skeleton_recall_comparison_html.py"

$env:nnUNet_raw = Join-Path $Project "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Project "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $Project "nnUNet_results"
Remove-Item Env:nnUNet_extTrainer -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

foreach ($required in @($Predictor, $Python, $ReportScript, $InputDir, $BaselineDir)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

$caseIds = Get-ChildItem -LiteralPath $InputDir -Filter "*_0000.tif" -File |
    ForEach-Object { $_.BaseName -replace "_0000$", "" }

if (@($caseIds).Count -eq 0) {
    throw "No *_0000.tif inputs found in $InputDir"
}

$missingBaseline = @($caseIds | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $BaselineDir "$_.tif") -PathType Leaf)
})
if ($missingBaseline.Count -gt 0) {
    throw "Baseline predictions are missing for: $($missingBaseline -join ', ')"
}

$missingRecall = @($caseIds | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $RecallDir "$_.tif") -PathType Leaf)
})

if ($ForcePrediction -or $missingRecall.Count -gt 0) {
    New-Item -ItemType Directory -Force -Path $RecallDir | Out-Null
    Write-Host ""
    Write-Host "=== Dataset136 Skeleton-Recall inference ==="
    & $Predictor `
        -i $InputDir `
        -o $RecallDir `
        -d 136 `
        -c 2d `
        -f 0 `
        -tr nnUNetTrainerSkeletonRecallCells `
        -p nnUNetPlans `
        -chk checkpoint_best.pth `
        -device $Device `
        --save_probabilities

    if ($LASTEXITCODE -ne 0) {
        throw "Skeleton-Recall inference failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "Skeleton-Recall predictions already exist; inference skipped."
    Write-Host "Use -ForcePrediction to regenerate them."
}

$missingAfterPrediction = @($caseIds | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $RecallDir "$_.tif") -PathType Leaf)
})
if ($missingAfterPrediction.Count -gt 0) {
    throw "Skeleton-Recall predictions are still missing for: $($missingAfterPrediction -join ', ')"
}

Write-Host ""
Write-Host "=== Creating side-by-side HTML report ==="
& $Python $ReportScript `
    --input-dir $InputDir `
    --baseline-dir $BaselineDir `
    --skeleton-recall-dir $RecallDir `
    --output-dir $ReportDir `
    --preview-max-size 1800 `
    --crop-size 768 `
    --num-crops 8

if ($LASTEXITCODE -ne 0) {
    throw "HTML report creation failed with exit code $LASTEXITCODE"
}

$HtmlPath = Join-Path $ReportDir "comparison.html"
Write-Host ""
Write-Host "Done. Open this file in a browser:"
Write-Host $HtmlPath
