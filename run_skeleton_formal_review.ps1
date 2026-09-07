param(
    [Parameter(Mandatory = $true)]
    [string]$CandidatePredictions,
    [Parameter(Mandatory = $true)]
    [string]$CandidateName,
    [string]$BasePredictions = "",
    [string]$BaseName = "R0 aktuelle Baseline",
    [string]$OutputName = "formal_review",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"

function Resolve-UserPath([string]$Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

$Dataset = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
$R0Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x"
$DefaultBase = Join-Path $Project "nnUNet_results\$Dataset\${R0Trainer}__nnUNetPlans__2d\fold_0\validation"
$Base = if ($BasePredictions) { Resolve-UserPath $BasePredictions } else { $DefaultBase }
$Candidate = Resolve-UserPath $CandidatePredictions
$Manifest = Join-Path $Project "one_px_skeleton_review\panel_manifest.csv"
$Output = Join-Path $Project "skeleton_connectivity_runs\$OutputName"

foreach ($Path in @($Base, $Candidate)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw "Prediction directory not found: $Path" }
}
$Arguments = @(
    (Join-Path $Project "make_blinded_skeleton_ab_review.py"),
    "--predictions-a", $Base,
    "--predictions-b", $Candidate,
    "--name-a", $BaseName,
    "--name-b", $CandidateName,
    "--manifest", $Manifest,
    "--output-dir", $Output
)
if ($Overwrite) { $Arguments += "--overwrite" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Formal review generation failed" }
Write-Host "Open: $(Join-Path $Output 'review.html')"
