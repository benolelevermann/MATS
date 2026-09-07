[CmdletBinding()]
param(
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $projectRoot "import_rocki_training_review.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Projekt-Python fehlt: $python"
}

$arguments = @($script)
if ($Overwrite) { $arguments += "--overwrite" }

& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Der Trainingsdaten-Import ist fehlgeschlagen." }

Write-Host ""
Write-Host "Der Review ist vorbereitet." -ForegroundColor Green
Write-Host "Falls die Webseite schon offen war, den lokalen Server einmal neu starten und dann im Verlauf 'ROCKi-Tracings' öffnen."

