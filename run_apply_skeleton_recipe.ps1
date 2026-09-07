param(
    [Parameter(Mandatory = $true)]
    [string]$ModelValidation,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [double]$TLow = -1,
    [double]$MaxGap = -1,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"

function Resolve-UserPath([string]$Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

$Dataset = "Dataset138_cleanSingleCell_soma_skeleton_recrop"
$Labels = Join-Path $Project "nnUNet_raw\$Dataset\labelsTr"
$Validation = Resolve-UserPath $ModelValidation
$Root = Resolve-UserPath $OutputRoot
if (-not (Test-Path -LiteralPath $Validation -PathType Container)) {
    throw "Model validation folder not found: $Validation"
}
if ($TLow -lt 0 -and $MaxGap -lt 0) {
    throw "Specify at least one accepted postprocessing step with -TLow and/or -MaxGap."
}

$Current = $Validation
if ($TLow -ge 0) {
    $NpzCount = @(Get-ChildItem -LiteralPath $Validation -Filter "*.npz" -File -ErrorAction SilentlyContinue).Count
    if ($NpzCount -eq 0) { throw "Hysteresis requires NPZ probabilities in $Validation" }
    $R2Root = Join-Path $Root "R2_hysteresis"
    $Arguments = @(
        (Join-Path $Project "skeleton_connectivity_experiments.py"), "hysteresis",
        "--baseline-dir", $Validation,
        "--probabilities-dir", $Validation,
        "--labels-dir", $Labels,
        "--output-root", $R2Root,
        "--t-low", $TLow.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if ($Overwrite) { $Arguments += "--overwrite" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Selected R2 recipe failed" }
    $Token = $TLow.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
    $Current = Join-Path $R2Root "tlow_$Token"
}

if ($MaxGap -ge 0) {
    $R3Root = Join-Path $Root "R3_reconnect"
    $Arguments = @(
        (Join-Path $Project "skeleton_connectivity_experiments.py"), "reconnect",
        "--baseline-dir", $Current,
        "--labels-dir", $Labels,
        "--output-root", $R3Root,
        "--max-gap", $MaxGap.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--minimum-direction-cosine", "0.707"
    )
    if ($Overwrite) { $Arguments += "--overwrite" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Selected R3 recipe failed" }
    $Token = $MaxGap.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
    $Current = Join-Path $R3Root "gap_$Token"
}

Write-Host "Final prediction directory: $Current"
Write-Host "Apply the same TLow/MaxGap values to R0 and every trained candidate before A/B review."
