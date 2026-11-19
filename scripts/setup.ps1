# FlightOps ML - one-time setup (Windows PowerShell 5.1+ / PowerShell 7)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python 3.12 not found on PATH" }
python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 required, found ' + sys.version" 
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python.exe -m pip install -e . --no-deps
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - edit FLIGHTOPS_ADMIN_TOKEN" }
if (-not (Test-Path data\raw\ai4i2020.csv)) {
  Write-Host "ai4i2020.csv not found in data\raw." -ForegroundColor Yellow
  Write-Host "Download it from Kaggle, or create the SYNTHETIC demo file with:  .\scripts\train.ps1 -Demo"
}
Write-Host "Setup complete. Activate with:  .\.venv\Scripts\Activate.ps1"
