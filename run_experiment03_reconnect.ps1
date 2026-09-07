param(
    [ValidateSet("Calibration", "Formal")]
    [string]$Stage = "Calibration",
    [string]$BasePredictions = "",
    [string]$BaseName = "R0 aktuelle Baseline",
    [string]$RunName = "",
    [double]$MaxGap = 6.0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"

function Resolve-UserPath([string]$Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

$Dataset = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
$Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
$DefaultBaseline = Join-Path $Project "nnUNet_results\$Dataset\${Trainer}__nnUNetPlans__2d\fold_0\validation"
$Baseline = if ($BasePredictions) { Resolve-UserPath $BasePredictions } else { $DefaultBaseline }
$Labels = Join-Path $Project "nnUNet_raw\$Dataset\labelsTr"
$FormalManifest = Join-Path $Project "one_px_skeleton_review\panel_manifest.csv"
$CalibrationManifest = Join-Path $Project "skeleton_connectivity_runs\calibration_manifest.csv"
$ResolvedRunName = if ($RunName) { $RunName } else { $BaseName }
$RunToken = ($ResolvedRunName -replace '[^A-Za-z0-9_.-]+', '-').Trim('-')
if (-not $RunToken) { throw "RunName or BaseName must contain at least one letter or digit." }
$OutputRoot = Join-Path $Project "skeleton_connectivity_runs\R3_reconnect_from_$RunToken"

if (-not (Test-Path -LiteralPath $Baseline -PathType Container)) { throw "Base predictions not found: $Baseline" }
if (-not (Test-Path -LiteralPath $CalibrationManifest -PathType Leaf)) {
    throw "Calibration manifest is missing. Run .\run_experiment02_hysteresis.ps1 first."
}

if ($Stage -eq "Calibration") {
    $Arguments = @(
        (Join-Path $Project "skeleton_connectivity_experiments.py"), "reconnect",
        "--baseline-dir", $Baseline,
        "--labels-dir", $Labels,
        "--output-root", $OutputRoot,
        "--max-gap", "6", "10",
        "--minimum-direction-cosine", "0.707"
    )
    if ($Overwrite) { $Arguments += "--overwrite" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "R3 candidate generation failed" }

    $CalibrationOutput = Join-Path $OutputRoot "calibration_review"
    $ReviewArguments = @(
        (Join-Path $Project "make_skeleton_calibration_review.py"),
        "--base-dir", $Baseline,
        "--base-name", $BaseName,
        "--candidate", "R3 gap 6px=$(Join-Path $OutputRoot 'gap_6p000')",
        "--candidate", "R3 gap 10px=$(Join-Path $OutputRoot 'gap_10p000')",
        "--manifest", $CalibrationManifest,
        "--output-dir", $CalibrationOutput
    )
    if ($Overwrite) { $ReviewArguments += "--overwrite" }
    & $Python @ReviewArguments
    if ($LASTEXITCODE -ne 0) { throw "R3 calibration page generation failed" }
    Write-Host "Open: $(Join-Path $CalibrationOutput 'calibration.html')"
    exit 0
}

if ($MaxGap -notin @(6.0, 10.0)) { throw "Formal R3 review accepts max gap 6 or 10 px." }
$Token = $MaxGap.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
$Candidate = Join-Path $OutputRoot "gap_$Token"
if (-not (Test-Path -LiteralPath $Candidate -PathType Container)) { throw "Selected R3 candidate not found: $Candidate" }
$FormalOutput = Join-Path $OutputRoot "formal_review_gap_$Token"
$CandidateName = "R3 Reconnect gap $MaxGap px"
$FormalArguments = @(
    (Join-Path $Project "make_blinded_skeleton_ab_review.py"),
    "--predictions-a", $Baseline,
    "--predictions-b", $Candidate,
    "--name-a", $BaseName,
    "--name-b", $CandidateName,
    "--manifest", $FormalManifest,
    "--output-dir", $FormalOutput
)
if ($Overwrite) { $FormalArguments += "--overwrite" }
& $Python @FormalArguments
if ($LASTEXITCODE -ne 0) { throw "R3 formal review generation failed" }
Write-Host "Open: $(Join-Path $FormalOutput 'review.html')"
Write-Host "Automatic metrics: $(Join-Path $Candidate 'summary.json')"
