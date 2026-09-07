param(
    [switch]$Apply,
    [ValidateRange(1, 4)]
    [int]$Radius = 2
)

$ErrorActionPreference = "Stop"

$project = "C:\Ole\20260721_CellClassification_v2"
$testRoot = Join-Path $project "20260730_TestBleb_v4"
$pythonExe = Join-Path $project ".venv\Scripts\python.exe"
$stage6Script = Join-Path $project "iterative_postprocessing_stage6_endpoint_caps.py"
$prediction = Join-Path $testRoot "03_iterative_postprocessing\Bleb\stage5b_image_guided_RELAXED_APPLIED\09_completed_skeleton_and_soma_0-1-2.tif"
$original = Join-Path $testRoot "01_inputimages\Bleb_0000.tif"

$mode = if ($Apply) { "APPLIED" } else { "PREVIEW" }
$outputDir = Join-Path $testRoot "03_iterative_postprocessing\Bleb\stage6_endpoint_caps_r${Radius}_${mode}"
$maxGap = 2.0 * $Radius + 1.5

foreach ($requiredPath in @(
    $pythonExe,
    $stage6Script,
    $prediction,
    $original
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Fehlender Pfad: $requiredPath"
    }
}

$stage6Arguments = @(
    $stage6Script,
    "--prediction", $prediction,
    "--original", $original,
    "--output-dir", $outputDir,
    "--cap-radius", "$Radius",
    "--max-gap", "$maxGap",
    "--orientation-steps", "5",
    "--min-direction-cos", "-0.35",
    "--ambiguity-margin", "0.08",
    "--soma-contact-radius", "3"
)
if ($Apply) {
    $stage6Arguments += "--apply"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Stage 6: selektive Endpoint-Kappen"
Write-Host "Radius: $Radius px"
Write-Host "Modus:  $mode"
Write-Host "Output: $outputDir"
Write-Host "============================================================"
Write-Host ""

& $pythonExe @stage6Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Stage 6 ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}

Write-Host ""
Write-Host "Pruefen:"
Write-Host "  $(Join-Path $outputDir 'stage6_qc_overlay.png')"
Write-Host "  $(Join-Path $outputDir '03_proposed_selective_endpoint_caps.tif')"
if ($Apply) {
    Write-Host "Final:"
    Write-Host "  $(Join-Path $outputDir '07_completed_skeleton_and_soma_0-1-2.tif')"
}
