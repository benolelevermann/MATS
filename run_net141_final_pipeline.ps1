[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$Port = 8776,
    [ValidateSet("adaptive", "tlow-0300", "tlow-0250", "tlow-0200")]
    [string]$HysteresisProfile = "tlow-0200",
    [switch]$NoFiji,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$evoRunner = Join-Path $projectRoot "run_net141_evo_pipeline.ps1"
if ($NoFiji) {
    throw "-NoFiji ist fuer eine Evo-fertige Auswahl nicht erlaubt. Nutze fuer Trainingsdaten run_net141_training_data_pipeline.ps1."
}
Write-Warning "Dieser alte Startname startet jetzt die getrennte Evo-Auswahl."
& $evoRunner `
    -Device $Device `
    -Port $Port `
    -HysteresisProfile $HysteresisProfile `
    -NoBrowser:$NoBrowser
