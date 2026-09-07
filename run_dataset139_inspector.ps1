param(
    [int]$Port = 8767,
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
    "--device", "cpu",
    "--runs-root", (Join-Path $Project ".dataset_inspector_runtime"),
    "--fov-root", (Join-Path $Project ".dataset_inspector_no_fovs"),
    "--no-fiji",
    "--open-path", "/dataset-inspector"
)
if (-not $NoBrowser) { $Arguments += "--open-browser" }

$Url = "http://127.0.0.1:$Port/dataset-inspector"
Write-Host "Dataset139 Trainingsdaten-Inspector: $Url" -ForegroundColor Cyan
Write-Host "Dieses Fenster offen lassen, solange du die Seite verwendest." -ForegroundColor DarkGray
& $Python @Arguments
