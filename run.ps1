param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$backendPath = Join-Path $projectRoot "backend"
$frontendPath = Join-Path $projectRoot "frontend"
$venvPath = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11+ and run this script again."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js 18+ and run this script again."
}

$venvHealthy = Test-Path $pythonPath
if ($venvHealthy) {
    try {
        & $pythonPath --version 2>$null
        $venvHealthy = $LASTEXITCODE -eq 0
    }
    catch {
        $venvHealthy = $false
    }
}

if (-not $venvHealthy) {
    if (Test-Path $venvPath) {
        Write-Host "Recreating the stale Python virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    }
    Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
    & python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed." }
}

if (-not $SkipInstall) {
    if (-not (Test-Path (Join-Path $frontendPath "node_modules"))) {
        Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
        Push-Location $frontendPath
        try {
            & npm install
            if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
        }
        finally {
            Pop-Location
        }
    }

    $requirementsStamp = Join-Path $venvPath ".requirements-installed"
    if (-not (Test-Path $requirementsStamp)) {
        Write-Host "Installing backend dependencies..." -ForegroundColor Cyan
        & $pythonPath -m pip install -r (Join-Path $backendPath "requirements.txt")
        if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }
        New-Item -ItemType File -Path $requirementsStamp -Force | Out-Null
    }
}

Write-Host "Starting backend at http://localhost:8000 ..." -ForegroundColor Green
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
Start-Process powershell.exe -WorkingDirectory $backendPath -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", "& '$pythonPath' -m uvicorn main:app --reload --port 8000"
)

Write-Host "Starting frontend at http://localhost:5173 ..." -ForegroundColor Green
Start-Process powershell.exe -WorkingDirectory $frontendPath -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", "& npm run dev"
)

Write-Host "Application started. Open http://localhost:5173" -ForegroundColor Green
Write-Host "Use Ctrl+C in each service window to stop it." -ForegroundColor DarkGray