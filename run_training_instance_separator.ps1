[CmdletBinding()]
param(
    [switch]$FullRun,
    [switch]$Continue,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataset = Join-Path $projectRoot "nnUNet_raw\Dataset143_dataset141_plus_synthetic_multicell"
$mode = if ($FullRun) { "seeded_full80" } else { "seeded_debug5" }
$epochs = if ($FullRun) { 80 } else { 5 }
$output = Join-Path $projectRoot "instance_separator_results\$mode"

foreach ($required in @(
    $python,
    $dataset,
    (Join-Path $dataset "synthetic_manifest.json"),
    (Join-Path $projectRoot "train_instance_separator.py")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Erforderlicher Pfad fehlt: $required"
    }
}

$arguments = @(
    (Join-Path $projectRoot "train_instance_separator.py"),
    "--dataset-root", $dataset,
    "--output-dir", $output,
    "--epochs", $epochs,
    "--device", $Device
)
if ($Continue) {
    $arguments += "--continue"
}

Write-Host "Trainiere nur die Zelltrennung; Netz 141 bleibt unveraendert." -ForegroundColor Cyan
Write-Host "Eingabe: Rohbild + Skeleton/Soma von Netz 141"
Write-Host "Ziel: Zugehoerigkeitsmaske je Soma; Ueberlappungen duerfen zwei IDs tragen"
Write-Host "Modus: $epochs Epochen; Ergebnisse: $output"
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Training des Zelltrenners ist mit Exit-Code $LASTEXITCODE fehlgeschlagen."
}
