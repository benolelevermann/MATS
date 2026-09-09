[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$Message,

    [switch]$NoPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & git -C $projectRoot @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git-Befehl fehlgeschlagen: git $($Arguments -join ' ')"
    }
}

Invoke-Git rev-parse --is-inside-work-tree | Out-Null

$candidates = @(& git -C $projectRoot ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) {
    throw "Die zu versionierenden Dateien konnten nicht ermittelt werden."
}

$largeFiles = foreach ($relativePath in $candidates) {
    $fullPath = Join-Path $projectRoot $relativePath
    if ((Test-Path -LiteralPath $fullPath -PathType Leaf) -and
        (Get-Item -LiteralPath $fullPath).Length -gt 50MB) {
        $relativePath
    }
}
if ($largeFiles) {
    throw "Neue oder versionierte Dateien über 50 MiB gefunden:`n$($largeFiles -join "`n")`nDiese Dateien zuerst aus Git ausschließen oder bewusst über Git LFS verwalten."
}

Invoke-Git add -A
$staged = @(& git -C $projectRoot diff --cached --name-status)
if ($LASTEXITCODE -ne 0) {
    throw "Die vorgemerkten Änderungen konnten nicht gelesen werden."
}
if (-not $staged) {
    Write-Host "Keine Änderungen für GitHub vorhanden." -ForegroundColor Yellow
    exit 0
}

Write-Host "Folgende Änderungen sind für den Commit vorgesehen:" -ForegroundColor Cyan
$staged | ForEach-Object { Write-Host "  $_" }
$confirmation = Read-Host "Commit erstellen? [j/N]"
if ($confirmation -notmatch '^(j|ja|y|yes)$') {
    Invoke-Git restore --staged .
    Write-Host "Abgebrochen; es wurde nichts committed oder gepusht." -ForegroundColor Yellow
    exit 0
}

Invoke-Git commit -m $Message
if (-not $NoPush) {
    Invoke-Git push
    Write-Host "GitHub ist aktuell." -ForegroundColor Green
} else {
    Write-Host "Lokaler Commit erstellt; Push wurde übersprungen." -ForegroundColor Green
}
