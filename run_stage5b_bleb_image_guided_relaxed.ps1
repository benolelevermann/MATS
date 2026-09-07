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
    "stage5b_image_guided_RELAXED_APPLIED"
} else {
    "stage5b_image_guided_RELAXED_PREVIEW"
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
    "--roi-margin", "26",
    "--max-neighbors-per-endpoint", "6",
    "--orientation-steps", "8",
    "--min-chord-direction-cos", "-0.85",
    "--min-path-direction-cos", "-0.20",
    "--max-path-ratio", "1.90",
    "--image-weight", "0.55",
    "--probability-weight", "0.45",
    "--min-path-score", "0.40",
    "--min-path-mean", "0.26",
    "--min-path-q20", "0.06",
    "--weak-support-threshold", "0.10",
    "--max-weak-run", "10",
    "--ambiguity-margin", "0.12",
    "--soma-contact-radius", "3"
)

if ($Apply) {
    $stage5Arguments += "--apply"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Stage 5b: moderat gelockerte bildgefuehrte Endpoint-Pfade"
Write-Host "Soma-Schutz und mutual-best bleiben aktiv"
Write-Host "Modus: $modeName"
Write-Host "Output: $outputDir"
Write-Host "============================================================"
Write-Host ""

& $pythonExe @stage5Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Stage 5b ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}

Write-Host ""
Write-Host "Zuerst pruefen:"
Write-Host "  $(Join-Path $outputDir 'stage5_qc_overlay.png')"
Write-Host "  $(Join-Path $outputDir '06_proposed_image_guided_connections.tif')"
if ($Apply) {
    Write-Host "Finale semantische Maske:"
    Write-Host "  $(Join-Path $outputDir '09_completed_skeleton_and_soma_0-1-2.tif')"
}
