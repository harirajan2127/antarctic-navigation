$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw "Virtual environment not found at .\.venv\Scripts\Activate.ps1"
}

& ".\.venv\Scripts\Activate.ps1"

python .\scripts\calculate_total_accuracy.py

Write-Host ""
Write-Host "Accuracy check complete."
