param(
    [ValidateSet("topology_baseline", "matrix_forest_inspired", "gcut_inspired")]
    [string]$Method = "gcut_inspired",
    [switch]$SkipExport,
    [switch]$SkipFiji,
    [switch]$RunFeatureExtraction
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Project = "C:\Ole\20260721_CellClassification_v2"
$TestRoot = Join-Path $Project "20280812_newTest2"
$Pipeline = Join-Path $Project "r_pipeline"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Fiji = "C:\Program Files\Fiji.app\ImageJ-win64.exe"
$Rscript = "C:\Program Files\R\R-4.5.1\bin\Rscript.exe"

$Exporter = Join-Path $Project "export_assigned_cells_for_evo.py"
$HelperExporter = Join-Path $Pipeline "export_cells_for_r_pipeline.py"
$Finalizer = Join-Path $Pipeline "finalize_cells_with_fiji.py"
$Validator = Join-Path $Pipeline "validate_r_pipeline_cell_folders.py"
$PrepareMetadata = Join-Path $Pipeline "prepare_newimage2_metadata.R"
$RunExtraction = Join-Path $Pipeline "run_newimage2_feature_extraction.R"
$SaveExtraction = Join-Path $Pipeline "save_newimage2_extraction.R"

$Comparison = Join-Path $TestRoot "05_paper_postprocessing_test\01_cell_assignment_comparison"
$Original = Join-Path $TestRoot "01_inputimages\NewImage2_0000.tif"
$Semantic = Join-Path $TestRoot "05_paper_postprocessing_test\00_gap_completion_no_loss\15_completed_skeleton_and_soma_0-1-2.tif"
$SafeInstances = Join-Path $Comparison "$Method\02_cell_instances_safe.tif"
$CellRoot = Join-Path $TestRoot "06_evo_cells_$Method"
$AnalysisRoot = Join-Path $TestRoot "07_evo_pipeline_$Method"

function Assert-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}
function Assert-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required directory not found: $Path"
    }
}

foreach ($file in @($Python, $Exporter, $HelperExporter, $Original, $Semantic, $SafeInstances)) {
    Assert-File $file
}

if (-not $SkipExport) {
    Write-Host ""
    Write-Host "=== Export safe $Method cells ===" -ForegroundColor Cyan
    & $Python $Exporter `
        --original $Original `
        --semantic $Semantic `
        --safe-instances $SafeInstances `
        --output-dir $CellRoot `
        --helper-script $HelperExporter `
        --margin 48 `
        --min-crop-size 128 `
        --edge-clearance 3 `
        --square
    if ($LASTEXITCODE -ne 0) {
        throw "Cell export failed."
    }
} else {
    Assert-Directory $CellRoot
}

if (-not $SkipFiji) {
    Assert-File $Fiji
    Assert-File $Finalizer
    Write-Host ""
    Write-Host "=== Create seg.traces and ROI ZIP files with Fiji ===" -ForegroundColor Cyan
    $env:CELL_EXPORT_ROOT = $CellRoot
    $logPath = Join-Path $CellRoot "_fiji_finalize_log.txt"
    $globalBounds = Join-Path $CellRoot "bounds.zip"
    $globalLocations = Join-Path $CellRoot "locations.zip"

    & $Fiji --allow-multiple --headless --console --run $Finalizer
    $fijiExitCode = $LASTEXITCODE

    $success = $false
    if (
        (Test-Path -LiteralPath $logPath -PathType Leaf) -and
        (Test-Path -LiteralPath $globalBounds -PathType Leaf) -and
        (Test-Path -LiteralPath $globalLocations -PathType Leaf)
    ) {
        $logContent = Get-Content -LiteralPath $logPath -Raw
        $success = (
            $logContent -match "GLOBAL OK bounds\.zip and locations\.zip" -and
            $logContent -notmatch "(?m)^\[\d+/\d+\] ERROR "
        )
    }
    if (-not $success) {
        throw "Fiji finalization failed. Check: $logPath (exit code $fijiExitCode)"
    }
    if ($fijiExitCode -ne 0) {
        Write-Warning "Fiji returned $fijiExitCode, but the completion log and global ROI archives are valid."
    }
}

Assert-File $Validator
Write-Host ""
Write-Host "=== Validate Evo cell folders ===" -ForegroundColor Cyan
& $Python $Validator --input-dir $CellRoot
if ($LASTEXITCODE -ne 0) {
    throw "Cell-folder validation failed."
}

Assert-File $Rscript
Assert-File $PrepareMetadata
Write-Host ""
Write-Host "=== Prepare Evo metadata ===" -ForegroundColor Cyan
& $Rscript $PrepareMetadata $Method
if ($LASTEXITCODE -ne 0) {
    throw "Metadata preparation failed."
}

if ($RunFeatureExtraction) {
    Assert-File $RunExtraction
    Assert-File $SaveExtraction
    Write-Host ""
    Write-Host "=== Run Evo feature extraction ===" -ForegroundColor Cyan
    & $Rscript $RunExtraction $Method
    if ($LASTEXITCODE -ne 0) {
        throw "Evo feature extraction failed."
    }
    & $Rscript $SaveExtraction $Method
    if ($LASTEXITCODE -ne 0) {
        throw "Saving Evo extraction failed."
    }

    $latestPathFile = Join-Path $AnalysisRoot "05_r_pipeline\saved_extraction\_LATEST_EXTRACTION_PATH.txt"
    Assert-File $latestPathFile
    $latestExtraction = (Get-Content -LiteralPath $latestPathFile -Raw).Trim()
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "EVO EXTRACTION READY"
    Write-Host ("=" * 72)
    Write-Host "Use this path as input_dir in setupNewObject_MAPPING.Rmd:"
    Write-Host $latestExtraction
} else {
    Write-Host ""
    Write-Host "Cells are finalized, validated and metadata is ready."
    Write-Host "Run feature extraction with:"
    Write-Host ".\export_newimage2_to_evo.ps1 -Method $Method -SkipExport -SkipFiji -RunFeatureExtraction"
}
