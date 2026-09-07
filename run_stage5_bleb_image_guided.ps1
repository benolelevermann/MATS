param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

$project = "C:\Ole\20260721_CellClassification_v2"
$testRoot = Join-Path $project "20260730_TestBleb_v4"
$pythonExe = Join-Path $project ".venv\Scripts\python.exe"
$stage5Script = Join-Path $project "iterative_postprocessing_stage5_image_guided.py"

$prediction = Join-Path $testRoot "03_iterative_postprocessing\Bleb\stage4_long_mixed_conservative\08_skeleton_and_soma_stage1_0-1-2.tif"
$probabilities = Join-Path $testRoot "02_predictions\Bleb.npz"
$original = Join-Path $testRoot "01_inputimages\Bleb_0000.tif"

$modeName = if ($Apply) {
    "stage5_image_guided_APPLIED"
} else {
    "stage5_image_guided_PREVIEW"
}
$outputDir = Join-Path $testRoot "03_iterative_postprocessing\Bleb\$modeName"

$requiredPaths = @(
    $pythonExe,
    $stage5Script,
    $prediction,
    $probabilities,
    $original
)
foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Fehlender Pfad: $requiredPath"
    }
}

$stage5Arguments = @(
    $stage5Script,
    "--prediction", $prediction,
    "--probabilities", $probabilities,
    "--original", $original,
    "--output-dir", $outputDir,
    "--min-gap", "20",
    "--max-gap", "75",
    "--roi-margin", "24",
    "--max-neighbors-per-endpoint", "6",
    "--orientation-steps", "8",
    "--min-chord-direction-cos", "-0.65",
    "--min-path-direction-cos", "0.05",
    "--max-path-ratio", "1.80",
    "--image-weight", "0.55",
    "--probability-weight", "0.45",
    "--min-path-score", "0.44",
    "--min-path-mean", "0.30",
    "--min-path-q20", "0.10",
    "--weak-support-threshold", "0.12",
    "--max-weak-run", "8",
    "--ambiguity-margin", "0.12",
    "--soma-contact-radius", "3"
)

if ($Apply) {
    $stage5Arguments += "--apply"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Stage 5: bildgefuehrte Endpunkt-zu-Endpunkt-Verbindungen"
Write-Host "Modus: $modeName"
Write-Host "Output: $outputDir"
Write-Host "============================================================"
Write-Host ""

& $pythonExe @stage5Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Stage 5 ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}

Write-Host ""
Write-Host "Fertig. Zuerst pruefen:"
Write-Host "  $(Join-Path $outputDir 'stage5_qc_overlay.png')"
Write-Host "  $(Join-Path $outputDir '06_proposed_image_guided_connections.tif')"
if ($Apply) {
    Write-Host "Finale semantische Maske:"
    Write-Host "  $(Join-Path $outputDir '09_completed_skeleton_and_soma_0-1-2.tif')"
}
