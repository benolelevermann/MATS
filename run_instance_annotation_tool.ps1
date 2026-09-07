param(
    [int]$Port = 8778,
    [string]$DatasetRoot = "",
    [string]$OutputRoot = "",
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}

$Arguments = @("-m", "instance_annotation_web.server", "--port", "$Port")
if (-not [string]::IsNullOrWhiteSpace($DatasetRoot)) { $Arguments += @("--dataset-root", $DatasetRoot) }
if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) { $Arguments += @("--output-root", $OutputRoot) }
if (-not $NoBrowser) { $Arguments += "--open-browser" }

Write-Host "Starting instance annotation tool at http://127.0.0.1:$Port" -ForegroundColor Cyan
& $Python @Arguments
