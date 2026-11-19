# Train candidates, register the best one as EXPERIMENTAL/VALIDATED. Use -Demo for the SYNTHETIC demo dataset.
param([switch]$Demo, [int]$Trials = 30, [switch]$NoTune)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
if ($Demo) {
  & $py -m flightops_ml demo-data
  $env:FLIGHTOPS_DATASET = "data/raw/ai4i2020_synthetic_demo.csv"
  Write-Host "Using SYNTHETIC demo data (NOT the real AI4I 2020 dataset)." -ForegroundColor Yellow
}
& $py -m flightops_ml validate-data
$cliArgs = @("-m", "flightops_ml", "train", "--trials", $Trials)
if ($NoTune) { $cliArgs += "--no-tune" }
& $py @cliArgs
Write-Host "`nNext: review the candidate, then approve + deploy:"
Write-Host "  $py -m flightops_ml registry status"
Write-Host "  $py -m flightops_ml registry approve --version 1 --user <you> --reason '<why>'"
Write-Host "  $py -m flightops_ml registry deploy  --version 1 --user <you>"
