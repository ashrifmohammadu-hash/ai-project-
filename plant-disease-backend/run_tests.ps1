# Run backend test suite (creates .test_venv on first run)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root ".test_venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating test virtual environment..."
    python -m venv (Join-Path $root ".test_venv")
    & (Join-Path $root ".test_venv\Scripts\pip.exe") install -r (Join-Path $root "requirements.txt")
}

Set-Location $root
& $venvPython -m unittest test_app -v
