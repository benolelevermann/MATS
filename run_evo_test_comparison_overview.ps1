[CmdletBinding()]
param(
    [string[]]$AutomaticRoot = @(
        "20260910_evo_DMSO_canonical1px_snt_shared_root"
    ),
    [double]$MaximumDistancePx = 20.0,
    [int]$Port = 8789
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$builder = Join-Path $projectRoot "r_pipeline\build_evo_test_comparison_overview.py"
$testRoot = Join-Path $projectRoot "EvoTest\div10_CC"
$output = Join-Path $testRoot "comparison_overview"

foreach ($required in @($python, $builder, $testRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}

$arguments = @(
    $builder,
    "--test-root", $testRoot,
    "--output-dir", $output,
    "--maximum-distance-px", $MaximumDistancePx,
    "--replace"
)
foreach ($root in $AutomaticRoot) {
    $resolvedRoot = if ([IO.Path]::IsPathRooted($root)) {
        $root
    } else {
        Join-Path $projectRoot $root
    }
    if (-not (Test-Path -LiteralPath $resolvedRoot -PathType Container)) {
        throw "Ordner mit automatischen Zellen fehlt: $resolvedRoot"
    }
    $arguments += @("--automatic-root", $resolvedRoot)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Die permanente Vergleichsübersicht konnte nicht aktualisiert werden."
}

$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    $server = Start-Process -FilePath $python `
        -ArgumentList @("-m", "http.server", $Port, "--bind", "127.0.0.1", "--directory", $output) `
        -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $output "server.pid") -Value $server.Id
}

$url = "http://127.0.0.1:$Port/index.html"
Write-Host "Vergleichsübersicht: $url" -ForegroundColor Green
