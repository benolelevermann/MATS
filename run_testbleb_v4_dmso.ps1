param(
    [ValidateSet("Preview5", "Preview6", "Finish")]
    [string]$Mode = "Preview5"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Project = "C:\Ole\20260721_CellClassification_v2"
$RunRoot = Join-Path $Project "20260730_TestBleb_v4"
$Python = Join-Path $Project ".venv\Scripts\python.exe"

$StagePolynomial = Join-Path $Project "iterative_postprocessing_stage1_polynomial.py"
$StageImageGuided = Join-Path $Project "iterative_postprocessing_stage5_image_guided.py"
$StageEndpointCaps = Join-Path $Project "iterative_postprocessing_stage6_endpoint_caps.py"
$CellExporter = Join-Path $Project "export_safe_cells_for_pipeline.py"
$OverviewScript = Join-Path $Project "make_cells_overview.py"

$Original = Join-Path $RunRoot "01_inputimages\DMSO_0000.tif"
$Probabilities = Join-Path $RunRoot "02_predictions\DMSO.npz"
$Prediction = Join-Path $RunRoot "02_predictions\DMSO.tif"

$DmsoRoot = Join-Path $RunRoot "03_iterative_postprocessing\DMSO"
$Stage1Dir = Join-Path $DmsoRoot "stage1_short_gaps_soma_guard"
$Stage2Dir = Join-Path $DmsoRoot "stage2_medium_gaps_straight_rescue"
$Stage3Dir = Join-Path $DmsoRoot "stage3_long_straight_only"
$Stage4Dir = Join-Path $DmsoRoot "stage4_long_mixed_conservative"
$Stage5PreviewDir = Join-Path $DmsoRoot "stage5b_image_guided_RELAXED_PREVIEW"
$Stage5AppliedDir = Join-Path $DmsoRoot "stage5b_image_guided_RELAXED_APPLIED"
$Stage6PreviewDir = Join-Path $DmsoRoot "stage6_endpoint_caps_r2_PREVIEW"
$Stage6AppliedDir = Join-Path $DmsoRoot "stage6_endpoint_caps_r2_APPLIED"

$Stage1Semantic = Join-Path $Stage1Dir "08_skeleton_and_soma_stage1_0-1-2.tif"
$Stage2Semantic = Join-Path $Stage2Dir "08_skeleton_and_soma_stage1_0-1-2.tif"
$Stage3Semantic = Join-Path $Stage3Dir "08_skeleton_and_soma_stage1_0-1-2.tif"
$Stage4Semantic = Join-Path $Stage4Dir "08_skeleton_and_soma_stage1_0-1-2.tif"
$Stage5AppliedSemantic = Join-Path $Stage5AppliedDir "09_completed_skeleton_and_soma_0-1-2.tif"
$Stage6AppliedSemantic = Join-Path $Stage6AppliedDir "07_completed_skeleton_and_soma_0-1-2.tif"

$CellsDir = Join-Path $RunRoot "04_exported_cells_DMSO_stage6"
$OverviewDir = Join-Path $RunRoot "05_crop_overview_DMSO_stage6"

function Assert-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Datei fehlt: $Path"
    }
}

function Invoke-PythonStep {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [string]$ExpectedFile
    )

    if (
        $ExpectedFile -and
        (Test-Path -LiteralPath $ExpectedFile -PathType Leaf)
    ) {
        Write-Host ""
        Write-Host "=== ${Label}: bereits vorhanden, wird uebersprungen ==="
        Write-Host $ExpectedFile
        return
    }

    Write-Host ""
    Write-Host "=== $Label ==="
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label ist fehlgeschlagen (Exitcode $LASTEXITCODE)."
    }

    if (
        $ExpectedFile -and
        -not (Test-Path -LiteralPath $ExpectedFile -PathType Leaf)
    ) {
        throw "$Label lief ohne Fehler, aber die erwartete Datei fehlt: $ExpectedFile"
    }
}

foreach ($required in @(
    $Python,
    $StagePolynomial,
    $StageImageGuided,
    $StageEndpointCaps,
    $CellExporter,
    $OverviewScript,
    $Original,
    $Probabilities,
    $Prediction
)) {
    Assert-File $required
}

New-Item -ItemType Directory -Force -Path $DmsoRoot | Out-Null

# Stage 1: kurze Luecken. Straight rescue ist hier bewusst deaktiviert,
# damit exakt die bereits fuer Bleb bestaetigte erste Stufe reproduziert wird.
Invoke-PythonStep `
    -Label "DMSO Stage 1: kurze Luecken" `
    -ExpectedFile $Stage1Semantic `
    -Arguments @(
        $StagePolynomial,
        "--prediction", $Prediction,
        "--probabilities", $Probabilities,
        "--original", $Original,
        "--output-dir", $Stage1Dir,
        "--min-gap", "2",
        "--max-gap", "12",
        "--orientation-steps", "6",
        "--tangent-scale", ".4",
        "--min-direction-cos", "0",
        "--max-curve-ratio", "1.45",
        "--min-score", ".42",
        "--ambiguity-margin", ".045",
        "--max-candidates-per-endpoint", "3",
        "--connection-radius", "0",
        "--soma-contact-radius", "3",
        "--disable-straight-rescue"
    )

Invoke-PythonStep `
    -Label "DMSO Stage 2: mittlere Luecken plus gerade Rettung" `
    -ExpectedFile $Stage2Semantic `
    -Arguments @(
        $StagePolynomial,
        "--prediction", $Stage1Semantic,
        "--probabilities", $Probabilities,
        "--original", $Original,
        "--output-dir", $Stage2Dir,
        "--min-gap", "10",
        "--max-gap", "22",
        "--orientation-steps", "8",
        "--tangent-scale", ".45",
        "--min-direction-cos", ".15",
        "--max-curve-ratio", "1.35",
        "--min-score", ".50",
        "--ambiguity-margin", ".06",
        "--max-candidates-per-endpoint", "2",
        "--connection-radius", "0",
        "--soma-contact-radius", "3",
        "--straight-rescue-min-cos", ".85",
        "--straight-rescue-max-curve-ratio", "1.20",
        "--straight-rescue-min-score", ".43",
        "--straight-rescue-min-image-mean", ".35",
        "--straight-rescue-min-image-q25", ".30"
    )

Invoke-PythonStep `
    -Label "DMSO Stage 3: lange, fast gerade Luecken" `
    -ExpectedFile $Stage3Semantic `
    -Arguments @(
        $StagePolynomial,
        "--prediction", $Stage2Semantic,
        "--probabilities", $Probabilities,
        "--original", $Original,
        "--output-dir", $Stage3Dir,
        "--min-gap", "18",
        "--max-gap", "36",
        "--orientation-steps", "12",
        "--tangent-scale", ".35",
        "--min-direction-cos", ".55",
        "--max-curve-ratio", "1.15",
        "--min-score", ".99",
        "--ambiguity-margin", ".08",
        "--max-candidates-per-endpoint", "2",
        "--connection-radius", "0",
        "--soma-contact-radius", "3",
        "--straight-rescue-min-cos", ".92",
        "--straight-rescue-max-curve-ratio", "1.10",
        "--straight-rescue-min-score", ".40",
        "--straight-rescue-min-image-mean", ".38",
        "--straight-rescue-min-image-q25", ".30"
    )

Invoke-PythonStep `
    -Label "DMSO Stage 4: lange gemischte Luecken, konservativ" `
    -ExpectedFile $Stage4Semantic `
    -Arguments @(
        $StagePolynomial,
        "--prediction", $Stage3Semantic,
        "--probabilities", $Probabilities,
        "--original", $Original,
        "--output-dir", $Stage4Dir,
        "--min-gap", "24",
        "--max-gap", "48",
        "--orientation-steps", "10",
        "--tangent-scale", ".50",
        "--min-direction-cos", ".10",
        "--max-curve-ratio", "1.45",
        "--min-score", ".56",
        "--ambiguity-margin", ".10",
        "--max-candidates-per-endpoint", "2",
        "--connection-radius", "0",
        "--soma-contact-radius", "3",
        "--straight-rescue-min-cos", ".90",
        "--straight-rescue-max-curve-ratio", "1.12",
        "--straight-rescue-min-score", ".42",
        "--straight-rescue-min-image-mean", ".40",
        "--straight-rescue-min-image-q25", ".30"
    )

$Stage5Arguments = @(
    $StageImageGuided,
    "--prediction", $Stage4Semantic,
    "--probabilities", $Probabilities,
    "--original", $Original,
    "--min-gap", "20",
    "--max-gap", "75",
    "--roi-margin", "26",
    "--max-neighbors-per-endpoint", "6",
    "--orientation-steps", "8",
    "--min-chord-direction-cos", "-.85",
    "--min-path-direction-cos", "-.20",
    "--max-path-ratio", "1.90",
    "--image-weight", ".55",
    "--probability-weight", ".45",
    "--min-path-score", ".40",
    "--min-path-mean", ".26",
    "--min-path-q20", ".06",
    "--weak-support-threshold", ".10",
    "--max-weak-run", "10",
    "--ambiguity-margin", ".12",
    "--soma-contact-radius", "3"
)

if ($Mode -eq "Preview5") {
    Invoke-PythonStep `
        -Label "DMSO Stage 5: bildgefuehrte Vorschau" `
        -ExpectedFile (Join-Path $Stage5PreviewDir "stage5_qc_overlay.png") `
        -Arguments (
            $Stage5Arguments +
            @("--output-dir", $Stage5PreviewDir)
        )

    Write-Host ""
    Write-Host "Vorschau 5 ist fertig. Pruefe:"
    Write-Host (Join-Path $Stage5PreviewDir "stage5_qc_overlay.png")
    Write-Host (Join-Path $Stage5PreviewDir "06_proposed_image_guided_connections.tif")
    Write-Host ""
    Write-Host "Wenn die gelben Vorschlaege passen:"
    Write-Host ".\run_testbleb_v4_dmso.ps1 -Mode Preview6"
    exit 0
}

Invoke-PythonStep `
    -Label "DMSO Stage 5: bildgefuehrte Verbindungen anwenden" `
    -ExpectedFile $Stage5AppliedSemantic `
    -Arguments (
        $Stage5Arguments +
        @(
            "--output-dir", $Stage5AppliedDir,
            "--apply"
        )
    )

$Stage6Arguments = @(
    $StageEndpointCaps,
    "--prediction", $Stage5AppliedSemantic,
    "--original", $Original,
    "--cap-radius", "2",
    "--max-gap", "5.5",
    "--orientation-steps", "5",
    "--min-direction-cos", "-.35",
    "--ambiguity-margin", ".08",
    "--soma-contact-radius", "3"
)

if ($Mode -eq "Preview6") {
    Invoke-PythonStep `
        -Label "DMSO Stage 6: Endpunkt-Kappen Vorschau" `
        -ExpectedFile (Join-Path $Stage6PreviewDir "stage6_qc_overlay.png") `
        -Arguments (
            $Stage6Arguments +
            @("--output-dir", $Stage6PreviewDir)
        )

    Write-Host ""
    Write-Host "Vorschau 6 ist fertig. Pruefe:"
    Write-Host (Join-Path $Stage6PreviewDir "stage6_qc_overlay.png")
    Write-Host (Join-Path $Stage6PreviewDir "03_proposed_selective_endpoint_caps.tif")
    Write-Host ""
    Write-Host "Wenn die gelben Kappen passen:"
    Write-Host ".\run_testbleb_v4_dmso.ps1 -Mode Finish"
    exit 0
}

Invoke-PythonStep `
    -Label "DMSO Stage 6: Endpunkt-Kappen anwenden" `
    -ExpectedFile $Stage6AppliedSemantic `
    -Arguments (
        $Stage6Arguments +
        @(
            "--output-dir", $Stage6AppliedDir,
            "--apply"
        )
    )

$ExportSummary = Join-Path $CellsDir "export_summary.json"
if (-not (Test-Path -LiteralPath $ExportSummary -PathType Leaf)) {
    if (
        (Test-Path -LiteralPath $CellsDir -PathType Container) -and
        @(Get-ChildItem -LiteralPath $CellsDir -Force).Count -gt 0
    ) {
        throw (
            "Der DMSO-Zielordner ist nicht leer, aber export_summary.json fehlt. " +
            "Zum Schutz vorhandener Daten bitte den Ordner pruefen oder einen neuen Namen verwenden: " +
            $CellsDir
        )
    }

    Invoke-PythonStep `
        -Label "DMSO: sichere Zellen als vollstaendige Crops exportieren" `
        -ExpectedFile $ExportSummary `
        -Arguments @(
            $CellExporter,
            "--original", $Original,
            "--semantic", $Stage6AppliedSemantic,
            "--output-dir", $CellsDir,
            "--contact-radius", "4",
            "--min-soma-area", "20",
            "--min-skeleton-pixels", "8",
            "--margin", "64",
            "--min-crop-size", "160",
            "--edge-clearance", "4",
            "--square",
            "--save-diagnostic-masks"
        )
}

Invoke-PythonStep `
    -Label "DMSO: Crop-Uebersicht erzeugen" `
    -ExpectedFile (Join-Path $OverviewDir "all_cells_overview.png") `
    -Arguments @(
        $OverviewScript,
        "--cells-root", $CellsDir,
        "--output-dir", $OverviewDir,
        "--columns", "0",
        "--thumbnail-size", "180",
        "--skeleton-thickness", "1",
        "--make-pages",
        "--page-columns", "5",
        "--page-rows", "4",
        "--page-thumbnail-size", "320"
    )

Write-Host ""
Write-Host ("=" * 72)
Write-Host "DMSO IST BIS ZUM ZELLEXPORT FERTIG"
Write-Host ("=" * 72)
Write-Host "Finale 0/1/2-Segmentierung:"
Write-Host $Stage6AppliedSemantic
Write-Host "DMSO-Zellordner:"
Write-Host $CellsDir
Write-Host "Crop-Uebersicht:"
Write-Host (Join-Path $OverviewDir "all_cells_overview.png")
Write-Host ""
Write-Host "Naechster Schritt: Fiji-Finalisierung und Evo-Feature-Extraktion."
