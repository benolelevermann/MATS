param(
    [switch]$SkipFiji,
    [switch]$RunFeatureExtraction
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Project = "C:\Ole\20260721_CellClassification_v2"
$RunRoot = Join-Path $Project "20260730_TestBleb_v4"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Fiji = "C:\Program Files\Fiji.app\ImageJ-win64.exe"
$Rscript = "C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
$Pipeline = Join-Path $Project "r_pipeline"

$Finalizer = Join-Path $Pipeline "finalize_cells_with_fiji.py"
$Validator = Join-Path $Pipeline "validate_r_pipeline_cell_folders.py"
$PrepareMetadata = Join-Path $Pipeline "prepare_testbleb_v4_metadata.R"
$RunExtraction = Join-Path $Pipeline "run_testbleb_v4_feature_extraction.R"
$SaveExtraction = Join-Path $Pipeline "save_testbleb_v4_extraction.R"

$CellRoots = @(
    (Join-Path $RunRoot "04_exported_cells_DMSO_stage6"),
    (Join-Path $RunRoot "04_exported_cells_Bleb_stage6")
)

function Assert-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Datei fehlt: $Path"
    }
}

function Assert-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Ordner fehlt: $Path"
    }
}

foreach ($file in @($Python, $Validator)) {
    Assert-File $file
}
foreach ($root in $CellRoots) {
    Assert-Directory $root
}

if (-not $SkipFiji) {
    Assert-File $Fiji
    Assert-File $Finalizer

    foreach ($root in $CellRoots) {
        Write-Host ""
        Write-Host "=== Fiji finalisiert: $root ==="

        $env:CELL_EXPORT_ROOT = $root

        # Alte Ergebnisdateien entfernen, damit nicht versehentlich
        # Ergebnisse eines früheren Laufs als Erfolg erkannt werden.
        $logPath = Join-Path $root "_fiji_finalize_log.txt"
        $boundsPath = Join-Path $root "bounds.zip"
        $locationsPath = Join-Path $root "locations.zip"

        foreach ($oldFile in @($logPath, $boundsPath, $locationsPath)) {
            if (Test-Path -LiteralPath $oldFile -PathType Leaf) {
                Remove-Item -LiteralPath $oldFile -Force
            }
        }

        & $Fiji `
            --allow-multiple `
            --headless `
            --console `
            --run $Finalizer

        $fijiExitCode = $LASTEXITCODE

        # Fiji kann wegen des Legacy-Patchers Exitcode 1 liefern,
        # obwohl der Finalizer vollständig erfolgreich war.
        $logExists = Test-Path -LiteralPath $logPath -PathType Leaf
        $boundsExists = Test-Path -LiteralPath $boundsPath -PathType Leaf
        $locationsExist = Test-Path -LiteralPath $locationsPath -PathType Leaf

        $finalizerSucceeded = $false

        if ($logExists -and $boundsExists -and $locationsExist) {
            $logContent = Get-Content -LiteralPath $logPath -Raw

            $finalizerSucceeded = (
                $logContent -match "GLOBAL OK bounds\.zip and locations\.zip" -and
                $logContent -notmatch "(?m)^\[\d+/\d+\] ERROR "
            )
        }

        if (-not $finalizerSucceeded) {
            throw @"
Fiji-Finalisierung ist fehlgeschlagen: $root
Fiji-Exitcode: $fijiExitCode
Log vorhanden: $logExists
bounds.zip vorhanden: $boundsExists
locations.zip vorhanden: $locationsExist
"@
        }

        if ($fijiExitCode -ne 0) {
            Write-Warning (
                "Fiji lieferte Exitcode {0}, aber der Finalizer war erfolgreich. " +
                "Der Lauf wird fortgesetzt."
            ) -f $fijiExitCode
        }

        Write-Host "Fiji-Finalisierung erfolgreich: $root"
    }
}

foreach ($root in $CellRoots) {
    Write-Host ""
    Write-Host "=== Zellordner validieren: $root ==="
    & $Python $Validator --input-dir $root
    if ($LASTEXITCODE -ne 0) {
        throw "Validierung ist fehlgeschlagen: $root"
    }
}

if (-not $RunFeatureExtraction) {
    Write-Host ""
    Write-Host "Bleb und DMSO sind fuer die Evo-Pipeline finalisiert und validiert."
    Write-Host "Feature-Extraktion starten mit:"
    Write-Host ".\finalize_testbleb_v4_for_evo.ps1 -SkipFiji -RunFeatureExtraction"
    exit 0
}

foreach ($file in @(
    $Rscript,
    $PrepareMetadata,
    $RunExtraction,
    $SaveExtraction
)) {
    Assert-File $file
}

Write-Host ""
Write-Host "=== Gemeinsame Bleb/DMSO-Metadaten erzeugen ==="
& $Rscript $PrepareMetadata
if ($LASTEXITCODE -ne 0) {
    throw "Metadaten-Erzeugung ist fehlgeschlagen."
}

Write-Host ""
Write-Host "=== Evo Feature Extraction fuer Bleb und DMSO ==="
& $Rscript $RunExtraction
if ($LASTEXITCODE -ne 0) {
    throw "Feature Extraction ist fehlgeschlagen."
}

Write-Host ""
Write-Host "=== Extraktion fuer Mapping/PCA speichern ==="
& $Rscript $SaveExtraction
if ($LASTEXITCODE -ne 0) {
    throw "Speichern der Extraktion ist fehlgeschlagen."
}

$LatestPathFile = Join-Path $RunRoot "05_r_pipeline\saved_extraction\_LATEST_EXTRACTION_PATH.txt"
Assert-File $LatestPathFile
$LatestExtraction = (Get-Content -LiteralPath $LatestPathFile -Raw).Trim()

Write-Host ""
Write-Host ("=" * 72)
Write-Host "BLEB UND DMSO SIND FUER EVO FERTIG"
Write-Host ("=" * 72)
Write-Host "Diesen Ordner in setupNewObject_MAPPING.Rmd als input_dir verwenden:"
Write-Host $LatestExtraction
Write-Host ""
Write-Host "Condition-Spalte:"
Write-Host "  Blebbistatin"
Write-Host "  DMSO"
