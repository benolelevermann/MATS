param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$Port = 8765,
    [string]$FovRoot = "X:\Niro\04_Raw_Data\InVitro\mica\250317_MCS24GFP\C3-C10_training_expansion",
    [string]$RunsRoot = "",
    [int]$DatasetId = 138,
    [string]$Trainer = "nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x",
    [string]$Plans = "nnUNetPlans",
    [string]$Checkpoint = "checkpoint_best.pth",
    [ValidateSet("training", "evo")]
    [string]$ReviewMode = "training",
    [string]$ReviewRoot = "",
    [ValidateSet("adaptive", "tlow-0300", "tlow-0250", "tlow-0200")]
    [string]$HysteresisProfile = "adaptive",
    [switch]$NoFiji,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}

$Arguments = @(
    "-m", "cell_pipeline_web.server",
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--device", $Device,
    "--fov-root", $FovRoot,
    "--dataset-id", "$DatasetId",
    "--trainer", $Trainer,
    "--plans", $Plans,
    "--checkpoint", $Checkpoint,
    "--review-mode", $ReviewMode,
    "--hysteresis-profile", $HysteresisProfile
)
if (-not [string]::IsNullOrWhiteSpace($RunsRoot)) { $Arguments += @("--runs-root", $RunsRoot) }
if (-not [string]::IsNullOrWhiteSpace($ReviewRoot)) { $Arguments += @("--review-root", $ReviewRoot) }
if ($NoFiji) { $Arguments += "--no-fiji" }
if (-not $NoBrowser) { $Arguments += "--open-browser" }

Write-Host "Starting Single Cell Extractor at http://127.0.0.1:$Port" -ForegroundColor Cyan
& $Python @Arguments
