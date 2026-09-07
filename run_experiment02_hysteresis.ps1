param(
    [ValidateSet("Calibration", "Formal")]
    [string]$Stage = "Calibration",
    [double]$TLow = 0.20,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Dataset = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
$Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
$Baseline = Join-Path $Project "nnUNet_results\$Dataset\${Trainer}__nnUNetPlans__2d\fold_0\validation"
$Labels = Join-Path $Project "nnUNet_raw\$Dataset\labelsTr"
$Split = Join-Path $Project "nnUNet_preprocessed\$Dataset\splits_final.json"
$FormalManifest = Join-Path $Project "one_px_skeleton_review\panel_manifest.csv"
$Runs = Join-Path $Project "skeleton_connectivity_runs"
$OutputRoot = Join-Path $Runs "R2_hysteresis"
$CalibrationManifest = Join-Path $Runs "calibration_manifest.csv"
$CalibrationOutput = Join-Path $OutputRoot "calibration_review"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment not found: $Python" }
$NpzCount = @(Get-ChildItem -LiteralPath $Baseline -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count
if ($NpzCount -eq 0) {
    throw "R0 probability archives are missing. Run .\run_r0_probability_export.ps1 first."
}

if (-not (Test-Path -LiteralPath $CalibrationManifest -PathType Leaf)) {
    & $Python (Join-Path $Project "skeleton_connectivity_experiments.py") select-calibration `
        --baseline-dir $Baseline --split-file $Split --formal-manifest $FormalManifest `
        --output-manifest $CalibrationManifest
    if ($LASTEXITCODE -ne 0) { throw "Calibration selection failed" }
}

if ($Stage -eq "Calibration") {
    $Arguments = @(
        (Join-Path $Project "skeleton_connectivity_experiments.py"), "hysteresis",
        "--baseline-dir", $Baseline,
        "--probabilities-dir", $Baseline,
        "--labels-dir", $Labels,
        "--output-root", $OutputRoot,
        "--t-low", "0.30", "0.20", "0.10"
    )
    if ($Overwrite) { $Arguments += "--overwrite" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "R2 candidate generation failed" }

    $ReviewArguments = @(
        (Join-Path $Project "make_skeleton_calibration_review.py"),
        "--base-dir", $Baseline,
        "--base-name", "R0 aktuelle Baseline",
        "--candidate", "R2 Tlow 0.30=$(Join-Path $OutputRoot 'tlow_0p300')",
        "--candidate", "R2 Tlow 0.20=$(Join-Path $OutputRoot 'tlow_0p200')",
        "--candidate", "R2 Tlow 0.10=$(Join-Path $OutputRoot 'tlow_0p100')",
        "--manifest", $CalibrationManifest,
        "--output-dir", $CalibrationOutput
    )
    if ($Overwrite) { $ReviewArguments += "--overwrite" }
    & $Python @ReviewArguments
    if ($LASTEXITCODE -ne 0) { throw "R2 calibration page generation failed" }
    Write-Host "Open: $(Join-Path $CalibrationOutput 'calibration.html')"
    exit 0
}

if ($TLow -notin @(0.30, 0.20, 0.10)) {
    throw "Formal R2 review only accepts the calibrated values 0.30, 0.20, or 0.10."
}
$Token = $TLow.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
$Candidate = Join-Path $OutputRoot "tlow_$Token"
if (-not (Test-Path -LiteralPath $Candidate -PathType Container)) {
    throw "Selected R2 candidate does not exist: $Candidate"
}
$FormalOutput = Join-Path $OutputRoot "formal_review_tlow_$Token"
$FormalArguments = @(
    (Join-Path $Project "make_blinded_skeleton_ab_review.py"),
    "--predictions-a", $Baseline,
    "--predictions-b", $Candidate,
    "--name-a", "R0 aktuelle Baseline",
    "--name-b", "R2 Hysteresis Tlow $TLow",
    "--manifest", $FormalManifest,
    "--output-dir", $FormalOutput
)
if ($Overwrite) { $FormalArguments += "--overwrite" }
& $Python @FormalArguments
if ($LASTEXITCODE -ne 0) { throw "R2 formal review generation failed" }
Write-Host "Open: $(Join-Path $FormalOutput 'review.html')"
Write-Host "Automatic metrics: $(Join-Path $Candidate 'summary.json')"
