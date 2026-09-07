# Runs the NeuroTreeTracer tree extraction under GNU Octave.
# Usage: .\run_neurotreetracer_octave.ps1 -InputDir <export dir> -OutputDir <result dir>
param(
    [Parameter(Mandatory=$true)][string]$InputDir,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Octave = "C:\Users\01.144_1\AppData\Local\Programs\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe",
    [string]$ProjectRoot = "C:\Ole\20260721_CellClassification_v2"
)
if (-not (Test-Path $Octave)) { throw "Octave not found: $Octave" }
$code = @"
pkg load image;
addpath('$ProjectRoot\octave_shims');
addpath('$ProjectRoot\external\NeuroTreeTracer');
addpath('$ProjectRoot');
run_neurotreetracer('$InputDir', '$OutputDir', '$ProjectRoot\external\NeuroTreeTracer');
"@
& $Octave --no-gui --quiet --eval $code
