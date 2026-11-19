# Start MLflow server, API and dashboard locally (three PowerShell windows). No Docker required.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$root = (Get-Location).Path
$py = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force mlflow_local\artifacts | Out-Null
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:PYTHONPATH = "$root\src;$root"
Start-Process powershell -ArgumentList "-NoExit","-Command","& '$py' -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_local/mlflow.db --artifacts-destination mlflow_local/artifacts --serve-artifacts"
Start-Sleep -Seconds 6
Start-Process powershell -ArgumentList "-NoExit","-Command","`$env:MLFLOW_TRACKING_URI='http://127.0.0.1:5000'; `$env:PYTHONPATH='$root\src;$root'; & '$py' -m uvicorn api.main:app --port 8000"
Start-Process powershell -ArgumentList "-NoExit","-Command","`$env:MLFLOW_TRACKING_URI='http://127.0.0.1:5000'; `$env:PYTHONPATH='$root\src;$root'; & '$py' -m streamlit run dashboard\app.py --server.port 8501"
Write-Host "MLflow  http://127.0.0.1:5000`nAPI     http://127.0.0.1:8000/docs`nDashboard http://127.0.0.1:8501"
