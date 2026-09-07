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

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot "20260903_network138_139_140_comparison"
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

$collections = @(
    [pscustomobject]@{
        Name = "TrogoCellStateTest"
        Inputs = Join-Path $projectRoot "20260817_TrogoCellStateTest\01_inputimages"
    },
    [pscustomobject]@{
        Name = "NewTest2"
        Inputs = Join-Path $projectRoot "20280812_newTest2\01_inputimages"
    }
)
$networks = @(
    [pscustomobject]@{
        Id = "138"
        Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
        Plans = "nnUNetPlans"
    },
    [pscustomobject]@{
        Id = "139"
        Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug"
        Plans = "nnUNetPlans"
    },
    [pscustomobject]@{
        Id = "140"
        Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugMATSFineTune100"
        Plans = "nnUNetPlans139Transfer"
    }
)

foreach ($required in @(
    $python, $predictor, $hysteresisScript, $extractScript, $cellGalleryScript,
    $overviewScript, $externalTrainerPath
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

function Assert-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Erwartete Datei fehlt: $Path"
    }
}

foreach ($network in $networks) {
    $datasetFolder = Get-ChildItem -LiteralPath $env:nnUNet_results -Directory |
        Where-Object { $_.Name -like "Dataset$($network.Id)_*" } |
        Select-Object -First 1
    if ($null -eq $datasetFolder) {
        throw "Kein Ergebnisordner fuer Dataset$($network.Id) gefunden."
    }
    $checkpoint = Join-Path $datasetFolder.FullName "$($network.Trainer)__$($network.Plans)__2d\fold_0\checkpoint_final.pth"
    Assert-File $checkpoint
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$rootLinks = @()

foreach ($collection in $collections) {
    if (-not (Test-Path -LiteralPath $collection.Inputs -PathType Container)) {
        throw "Eingabeordner fehlt: $($collection.Inputs)"
    }
    $caseIds = @(
        Get-ChildItem -LiteralPath $collection.Inputs -Filter "*_0000.tif" -File |
            Sort-Object Name |
            ForEach-Object { $_.BaseName -replace "_0000$", "" }
    )
    if ($caseIds.Count -eq 0) {
        throw "Keine *_0000.tif-Dateien in $($collection.Inputs)"
    }
    $collectionRoot = Join-Path $OutputRoot $collection.Name
    New-Item -ItemType Directory -Force -Path $collectionRoot | Out-Null
    Write-Host ""
    Write-Host ("=" * 76) -ForegroundColor DarkCyan
    Write-Host "$($collection.Name): $($caseIds -join ', ')" -ForegroundColor Cyan
    Write-Host ("=" * 76) -ForegroundColor DarkCyan

    $predictionDirs = @{}
    foreach ($network in $networks) {
        $predictionDir = Join-Path $collectionRoot "predictions_net$($network.Id)"
        $predictionDirs[$network.Id] = $predictionDir
        New-Item -ItemType Directory -Force -Path $predictionDir | Out-Null
        $missing = @($caseIds | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $predictionDir "$_.tif") -PathType Leaf) -or
            -not (Test-Path -LiteralPath (Join-Path $predictionDir "$_.npz") -PathType Leaf)
        })
        if ($ForcePrediction -or $missing.Count -gt 0) {
            Write-Host ""
            Write-Host "Netz $($network.Id): Prediction mit checkpoint_final.pth" -ForegroundColor Cyan
            $predictionArguments = @(
                "-i", $collection.Inputs,
                "-o", $predictionDir,
                "-d", $network.Id,
                "-c", "2d",
                "-f", "0",
                "-tr", $network.Trainer,
                "-p", $network.Plans,
                "-chk", "checkpoint_final.pth",
                "-device", $Device,
                "-npp", "2",
                "-nps", "2",
                "--save_probabilities",
                "--disable_progress_bar"
            )
            if (-not $ForcePrediction) {
                $predictionArguments += "--continue_prediction"
            }
            & $predictor @predictionArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Prediction mit Netz $($network.Id) ist fehlgeschlagen (Exit-Code $LASTEXITCODE)."
            }
        } else {
            Write-Host "Netz $($network.Id): vorhandene vollstaendige Prediction wird verwendet." -ForegroundColor DarkGray
        }
        foreach ($caseId in $caseIds) {
            Assert-File (Join-Path $predictionDir "$caseId.tif")
            Assert-File (Join-Path $predictionDir "$caseId.npz")
        }

        $hysteresisDir = Join-Path $collectionRoot "hysteresis_net$($network.Id)"
        $hysteresisSummary = Join-Path $hysteresisDir "run_summary.json"
        if (-not (Test-Path -LiteralPath $hysteresisSummary -PathType Leaf) -or $ForcePrediction) {
            Write-Host "Netz $($network.Id): adaptive Hysterese (alpha=1/3)" -ForegroundColor Cyan
            # Force an array even when this collection contains only one image.
            # Otherwise PowerShell splats the single path character by character.
            $probabilityFiles = @($caseIds | ForEach-Object { Join-Path $predictionDir "$_.npz" })
            & $python $hysteresisScript `
                --probabilities @probabilityFiles `
                --output-dir $hysteresisDir `
                --alpha 0.3333333333333333 `
                --write-semantic `
                --overwrite
            if ($LASTEXITCODE -ne 0) {
                throw "Hysterese fuer Netz $($network.Id) ist fehlgeschlagen (Exit-Code $LASTEXITCODE)."
            }
        } else {
            Write-Host "Netz $($network.Id): vorhandene Hysterese wird verwendet." -ForegroundColor DarkGray
        }
        foreach ($caseId in $caseIds) {
            Assert-File (Join-Path $hysteresisDir "${caseId}_adaptive_hysteresis_0-1-2.tif")
        }
    }

    $extractionSummary = Join-Path $collectionRoot "cell_extraction_summary.json"
    if (-not (Test-Path -LiteralPath $extractionSummary -PathType Leaf)) {
        Write-Host ""
        Write-Host "Topologie-basierte Einzelzell-Extraktion fuer alle drei Netze" -ForegroundColor Cyan
        $extractArguments = @(
            $extractScript,
            "--project-root", $projectRoot,
            "--comparison-root", $collectionRoot,
            "--inputs", $collection.Inputs
        )
        foreach ($network in $networks) {
            $extractArguments += @("--network", $network.Id, $predictionDirs[$network.Id])
        }
        & $python @extractArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Zellextraktion fuer $($collection.Name) ist fehlgeschlagen (Exit-Code $LASTEXITCODE)."
        }
    } else {
        Write-Host "Vorhandene vollstaendige Zellextraktion wird verwendet." -ForegroundColor DarkGray
    }

    $cellGallery = Join-Path $collectionRoot "comparison_cells\index.html"
    if (-not (Test-Path -LiteralPath $cellGallery -PathType Leaf)) {
        Write-Host "Erzeuge zugeordneten Einzelzellvergleich ..." -ForegroundColor Cyan
        $galleryArguments = @(
            $cellGalleryScript,
            "--comparison-root", $collectionRoot,
            "--inputs", $collection.Inputs,
            "--output-dir", (Split-Path -Parent $cellGallery)
        )
        foreach ($network in $networks) {
            $galleryArguments += @("--network", $network.Id, $predictionDirs[$network.Id])
        }
        & $python @galleryArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Einzelzellvergleich fuer $($collection.Name) ist fehlgeschlagen (Exit-Code $LASTEXITCODE)."
        }
    }

    $overviewGallery = Join-Path $collectionRoot "comparison_overview\index.html"
    if (-not (Test-Path -LiteralPath $overviewGallery -PathType Leaf)) {
        Write-Host "Erzeuge Uebersichtsvergleich ..." -ForegroundColor Cyan
        $overviewArguments = @(
            $overviewScript,
            "--inputs", $collection.Inputs,
            "--cell-summary", $extractionSummary,
            "--cell-gallery", $cellGallery,
            "--output-dir", (Split-Path -Parent $overviewGallery)
        )
        foreach ($network in $networks) {
            $overviewArguments += @(
                "--network", $network.Id, $predictionDirs[$network.Id],
                (Join-Path $collectionRoot "hysteresis_net$($network.Id)")
            )
        }
        & $python @overviewArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Uebersichtsvergleich fuer $($collection.Name) ist fehlgeschlagen (Exit-Code $LASTEXITCODE)."
        }
    }
    $rootLinks += [pscustomobject]@{
        Name = $collection.Name
        Overview = "$($collection.Name)/comparison_overview/index.html"
        Cells = "$($collection.Name)/comparison_cells/index.html"
    }
}

$rows = $rootLinks | ForEach-Object {
    "<tr><td>$($_.Name)</td><td><a href='$($_.Overview)'>Übersicht, Prediction und Hysterese</a></td><td><a href='$($_.Cells)'>Einzelzellen</a></td></tr>"
}
$index = @"
<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Netze 138 · 139 · 140</title>
<style>body{font:15px/1.5 Segoe UI,Arial,sans-serif;margin:32px;background:#101318;color:#edf1f5}a{color:#7fc8ff}table{border-collapse:collapse;background:#191e24}th,td{padding:12px 16px;border-bottom:1px solid #303842;text-align:left}</style></head>
<body><h1>Vergleich der Netze 138, 139 und 140</h1><p>Identische Eingaben, finale Checkpoints, adaptive Hysterese mit alpha=1/3 und identische Einzelzell-Extraktion.</p><table><thead><tr><th>Bildgruppe</th><th>Gesamtbilder</th><th>Einzelzellen</th></tr></thead><tbody>$($rows -join '')</tbody></table></body></html>
"@
$indexPath = Join-Path $OutputRoot "index.html"
[IO.File]::WriteAllText($indexPath, $index, [Text.UTF8Encoding]::new($false))

Write-Host ""
Write-Host ("=" * 76) -ForegroundColor Green
Write-Host "VERGLEICH ABGESCHLOSSEN" -ForegroundColor Green
Write-Host "Startseite: $indexPath"
Write-Host ("=" * 76) -ForegroundColor Green
