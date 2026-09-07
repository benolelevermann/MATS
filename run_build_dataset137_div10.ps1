<#
Build a fresh nnU-Net Dataset137 from Dataset136 plus manually traced div10
cells, then optionally preprocess and train it.

This script never changes Dataset136 or the source `Tracings` folders.
It intentionally stops on non-empty stage directories, missing mappings, or
failed rasterizations so that mixed/stale datasets cannot be trained by accident.

Usage example:
  .\run_build_dataset137_div10.ps1 `
    -ProjectRoot "C:\Ole\20260721_CellClassification_v2" `
    -TracingsRoot "X:\...\div10\Tracings" `
    -EncryptionKey "X:\...\div10\toTrace_Encrypted\toTrace_EncryptionKey.csv" `
    -RunPreprocess

Run a second time with `-Train` only after the preprocessing QC has passed.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$TracingsRoot,

    [Parameter(Mandatory = $true)]
    [string]$EncryptionKey,

    [string]$DatasetName = "Dataset137_allcells_soma_skeleton_plus_div10_nonBleb_nonDMSO",

    [string]$WorkDirectory,

    [string]$FijiPath = "C:\Program Files\Fiji.app\ImageJ-win64.exe",

    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda",

    [switch]$RunPreprocess,

    [switch]$Train
)

$ErrorActionPreference = "Stop"

function Require-Path {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description does not exist: $Path"
    }
}

function Require-NewOrEmptyDirectory {
    param([string]$Path, [string]$Description)
    if (Test-Path -LiteralPath $Path) {
        $items = @(Get-ChildItem -LiteralPath $Path -Force)
        if ($items.Count -gt 0) {
            throw "$Description is not empty: $Path`nUse a fresh path or inspect the previous run first."
        }
    }
    else {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$TracingsRoot = [System.IO.Path]::GetFullPath($TracingsRoot)
$EncryptionKey = [System.IO.Path]::GetFullPath($EncryptionKey)
if (-not $WorkDirectory) {
    $WorkDirectory = Join-Path $ProjectRoot "dataset137_div10_build"
}
$WorkDirectory = [System.IO.Path]::GetFullPath($WorkDirectory)

Require-Path $ProjectRoot "Project root"
Require-Path $TracingsRoot "Tracings root"
Require-Path $EncryptionKey "Encryption key"
Require-Path $FijiPath "Fiji executable"

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$selectionScript = Join-Path $ProjectRoot "select_div10_tracings_for_dataset137.py"
$rasterScript = Join-Path $ProjectRoot "rasterize_div10_training_masks_fiji.py"
$assemblerScript = Join-Path $ProjectRoot "build_dataset137_from_dataset136_and_div10.py"
$preprocess = Join-Path $ProjectRoot ".venv\Scripts\nnUNetv2_plan_and_preprocess.exe"
$baseDataset = Join-Path $ProjectRoot "nnUNet_raw\Dataset136_allcells_soma_skeleton_fold0_full_run"
$targetDataset = Join-Path $ProjectRoot (Join-Path "nnUNet_raw" $DatasetName)

foreach ($item in @(
    @{ Path = $python; Description = "Project Python" },
    @{ Path = $selectionScript; Description = "Selection script" },
    @{ Path = $rasterScript; Description = "Fiji rasterizer" },
    @{ Path = $assemblerScript; Description = "Dataset assembler" },
    @{ Path = $baseDataset; Description = "Dataset136" }
)) {
    Require-Path $item.Path $item.Description
}

if ($Train -and -not $RunPreprocess) {
    throw "For safety, use -RunPreprocess together with -Train only for a brand-new build, or run training separately after you inspected preprocessing."
}

$selectionDirectory = Join-Path $WorkDirectory "01_selection"
$rasterDirectory = Join-Path $WorkDirectory "02_rasterized"

Require-NewOrEmptyDirectory $WorkDirectory "Work directory"

Write-Host "`n=== 1/3 Select div10 cells ===" -ForegroundColor Cyan
& $python $selectionScript `
    --tracings-root $TracingsRoot `
    --encryption-key $EncryptionKey `
    --output-dir $selectionDirectory
if ($LASTEXITCODE -ne 0) { throw "Selection failed." }

$manifest = Join-Path $selectionDirectory "included_tracings_manifest.csv"
Require-Path $manifest "Included-tracings manifest"

Write-Host "`n=== 2/3 Rasterize manual SWC + Soma ROIs in Fiji ===" -ForegroundColor Cyan
$env:DIV10_INCLUDED_MANIFEST = $manifest
$env:DIV10_RASTER_OUTPUT = $rasterDirectory
& $FijiPath --allow-multiple --headless --console --run $rasterScript
if ($LASTEXITCODE -ne 0) {
    throw "Fiji rasterization failed. Inspect: $rasterDirectory\rasterization_failures.csv"
}

$failures = @(Import-Csv -LiteralPath (Join-Path $rasterDirectory "rasterization_failures.csv"))
if ($failures.Count -gt 0) {
    throw "Rasterization reports $($failures.Count) failed cells. Inspect rasterization_failures.csv; do not build Dataset137 yet."
}

Write-Host "`n=== 3/3 Assemble Dataset137 ===" -ForegroundColor Cyan
if (Test-Path -LiteralPath $targetDataset) {
    throw "Target dataset already exists: $targetDataset`nDo not overwrite it. Choose a new DatasetName/ID after inspecting the existing folder."
}
& $python $assemblerScript `
    --base-dataset $baseDataset `
    --rasterized-new-dir $rasterDirectory `
    --selection-manifest $manifest `
    --output-dataset $targetDataset
if ($LASTEXITCODE -ne 0) { throw "Dataset137 assembly failed." }

Write-Host "`nDataset construction completed: $targetDataset" -ForegroundColor Green
Write-Host "Inspect the selection manifest and a few raw/label TIFF pairs before preprocessing."

if (-not $RunPreprocess) {
    Write-Host "`nNext, after QC:" -ForegroundColor Yellow
    Write-Host "  & `"$preprocess`" -d 137 -c 2d --verify_dataset_integrity"
    exit 0
}

Require-Path $preprocess "nnUNet preprocessing executable"
$env:nnUNet_raw = Join-Path $ProjectRoot "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $ProjectRoot "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $ProjectRoot "nnUNet_results"

Write-Host "`n=== Preprocess Dataset137 ===" -ForegroundColor Cyan
& $preprocess -d 137 -c 2d --verify_dataset_integrity
if ($LASTEXITCODE -ne 0) { throw "Planning/preprocessing failed." }

if (-not $Train) {
    Write-Host "`nPreprocessing completed. Do not copy Dataset136's splits_final.json." -ForegroundColor Green
    Write-Host "Train after reviewing the newly generated split and preprocessing:" -ForegroundColor Yellow
    Write-Host "  & `"$python`" -m nnunetv2.run.run_training 137 2d 0 -tr nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x -p nnUNetPlans -device $Device"
    exit 0
}

Write-Host "`n=== Train Dataset137, Fold 0 ===" -ForegroundColor Cyan
& $python -m nnunetv2.run.run_training `
    137 2d 0 `
    -tr nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x `
    -p nnUNetPlans `
    -device $Device
if ($LASTEXITCODE -ne 0) { throw "Training failed." }
