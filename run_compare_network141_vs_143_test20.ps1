[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$OutputRoot = "",
    [switch]$ForcePrediction
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$predictor = Join-Path $projectRoot ".venv\Scripts\nnUNetv2_predict.exe"
$hysteresisScript = Join-Path $projectRoot "apply_adaptive_hysteresis.py"
$extractScript = Join-Path $projectRoot "extract_network_comparison_cells.py"
$cellGalleryScript = Join-Path $projectRoot "make_multi_network_cell_comparison_gallery.py"
$overviewScript = Join-Path $projectRoot "make_multi_network_overview_comparison.py"
$externalTrainerPath = Join-Path $projectRoot "custom_trainers\skeleton_recall"
$inputs = Join-Path $projectRoot "20280812_newTest2\01_inputimages"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot "20260908_network141_vs_143_test20_newtest2"
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

$networks = @(
    [pscustomobject]@{
        Id = "141"
        DatasetId = "141"
        Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug"
        Plans = "nnUNetPlans139Exact"
        Checkpoint = "checkpoint_final.pth"
    },
    [pscustomobject]@{
        Id = "143-test20"
        DatasetId = "143"
        Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugSyntheticMultiCellFineTune20"
        Plans = "nnUNetPlans139Exact"
        Checkpoint = "checkpoint_best.pth"
    }
)

foreach ($required in @(
    $python, $predictor, $hysteresisScript, $extractScript,
    $cellGalleryScript, $overviewScript, $externalTrainerPath, $inputs
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
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
Remove-Item Env:NNUNET_MATS_FINETUNE_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:NNUNET_SYNTHETIC_MULTICELL_FINETUNE_CHECKPOINT -ErrorAction SilentlyContinue

function Assert-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Erwartete Datei fehlt: $Path"
    }
}

$caseIds = @(
    Get-ChildItem -LiteralPath $inputs -Filter "*_0000.tif" -File |
        Sort-Object Name |
        ForEach-Object { $_.BaseName -replace "_0000$", "" }
)
if ($caseIds.Count -eq 0) {
    throw "Keine *_0000.tif-Dateien in $inputs"
}

foreach ($network in $networks) {
    $datasetFolder = Get-ChildItem -LiteralPath $env:nnUNet_results -Directory |
        Where-Object { $_.Name -like "Dataset$($network.DatasetId)_*" } |
        Select-Object -First 1
    if ($null -eq $datasetFolder) {
        throw "Kein Ergebnisordner fuer Dataset$($network.DatasetId) gefunden."
    }
    $checkpoint = Join-Path $datasetFolder.FullName "$($network.Trainer)__$($network.Plans)__2d\fold_0\$($network.Checkpoint)"
    Assert-File $checkpoint
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$predictionDirs = @{}
foreach ($network in $networks) {
    $predictionDir = Join-Path $OutputRoot "predictions_net$($network.Id)"
    $predictionDirs[$network.Id] = $predictionDir
    New-Item -ItemType Directory -Force -Path $predictionDir | Out-Null
    $missing = @($caseIds | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $predictionDir "$_.tif")) -or
        -not (Test-Path -LiteralPath (Join-Path $predictionDir "$_.npz"))
    })
    if ($ForcePrediction -or $missing.Count -gt 0) {
        Write-Host "Netz $($network.Id): Prediction mit $($network.Checkpoint)" -ForegroundColor Cyan
        $arguments = @(
            "-i", $inputs,
            "-o", $predictionDir,
            "-d", $network.DatasetId,
            "-c", "2d",
            "-f", "0",
            "-tr", $network.Trainer,
            "-p", $network.Plans,
            "-chk", $network.Checkpoint,
            "-device", $Device,
            "-npp", "2",
            "-nps", "2",
            "--save_probabilities",
            "--disable_progress_bar"
        )
        if (-not $ForcePrediction) { $arguments += "--continue_prediction" }
        & $predictor @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Prediction mit Netz $($network.Id) ist fehlgeschlagen."
        }
    }
    foreach ($caseId in $caseIds) {
        Assert-File (Join-Path $predictionDir "$caseId.tif")
        Assert-File (Join-Path $predictionDir "$caseId.npz")
    }

    $hysteresisDir = Join-Path $OutputRoot "hysteresis_net$($network.Id)"
    $summary = Join-Path $hysteresisDir "run_summary.json"
    if ($ForcePrediction -or -not (Test-Path -LiteralPath $summary)) {
        Write-Host "Netz $($network.Id): Hysterese T_high=0.647, T_low=0.20" -ForegroundColor Cyan
        $probabilities = @($caseIds | ForEach-Object { Join-Path $predictionDir "$_.npz" })
        & $python $hysteresisScript `
            --probabilities @probabilities `
            --output-dir $hysteresisDir `
            --t-high 0.647 `
            --t-low 0.20 `
            --write-semantic `
            --overwrite
        if ($LASTEXITCODE -ne 0) {
            throw "Hysterese fuer Netz $($network.Id) ist fehlgeschlagen."
        }
    }
}

$extractionSummary = Join-Path $OutputRoot "cell_extraction_summary.json"
if (-not (Test-Path -LiteralPath $extractionSummary)) {
    $arguments = @(
        $extractScript,
        "--project-root", $projectRoot,
        "--comparison-root", $OutputRoot,
        "--inputs", $inputs
    )
    foreach ($network in $networks) {
        $arguments += @("--network", $network.Id, $predictionDirs[$network.Id])
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Einzelzell-Extraktion ist fehlgeschlagen." }
}

$cellGallery = Join-Path $OutputRoot "comparison_cells\index.html"
if (-not (Test-Path -LiteralPath $cellGallery)) {
    $arguments = @(
        $cellGalleryScript,
        "--comparison-root", $OutputRoot,
        "--inputs", $inputs,
        "--output-dir", (Split-Path -Parent $cellGallery)
    )
    foreach ($network in $networks) {
        $arguments += @("--network", $network.Id, $predictionDirs[$network.Id])
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Einzelzell-HTML ist fehlgeschlagen." }
}

$overviewGallery = Join-Path $OutputRoot "comparison_overview\index.html"
if (-not (Test-Path -LiteralPath $overviewGallery)) {
    $arguments = @(
        $overviewScript,
        "--inputs", $inputs,
        "--cell-summary", $extractionSummary,
        "--cell-gallery", $cellGallery,
        "--output-dir", (Split-Path -Parent $overviewGallery)
    )
    foreach ($network in $networks) {
        $arguments += @(
            "--network", $network.Id, $predictionDirs[$network.Id],
            (Join-Path $OutputRoot "hysteresis_net$($network.Id)")
        )
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Uebersichts-HTML ist fehlgeschlagen." }
}

$indexPath = Join-Path $OutputRoot "index.html"
$index = @"
<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Netz 141 vs. Dataset143-Test20</title>
<style>body{font:15px/1.5 Segoe UI,Arial,sans-serif;margin:32px;background:#101318;color:#edf1f5}a{color:#7fc8ff}li{margin:10px}</style></head>
<body><h1>Netz 141 final vs. Netz 143 nach 20 Epochen</h1>
<p>Identisches NewTest2-Eingabebild, T_high=0,647, T_low=0,20 und identische Einzelzell-Extraktion.</p>
<ul><li><a href="comparison_overview/index.html">Uebersicht, Prediction und Hysterese</a></li>
<li><a href="comparison_cells/index.html">Zugeordnete Einzelzellen</a></li></ul></body></html>
"@
[IO.File]::WriteAllText($indexPath, $index, [Text.UTF8Encoding]::new($false))

Write-Host "Vergleich abgeschlossen: $indexPath" -ForegroundColor Green
