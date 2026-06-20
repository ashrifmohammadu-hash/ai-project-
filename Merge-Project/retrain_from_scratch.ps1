# Retrain plant disease model from scratch (fresh output + deploy to backend)
# Run from PowerShell:
#   cd "d:\ai data\Final\Merge-Project"
#   .\retrain_from_scratch.ps1

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Output = Join-Path $Root "output"
$Dataset = Join-Path $Root "dataset_sri_lanka"
$BackendModels = Join-Path $Root "..\plant-disease-backend\models"

Write-Host "=== Sri Lanka plant disease — RETRAIN FROM SCRATCH ===" -ForegroundColor Cyan

# Activate plant_env if present
$VenvActivate = Join-Path $Root "plant_env\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    . $VenvActivate
    Write-Host "Using venv: plant_env"
} else {
    Write-Host "Using current Python (no plant_env found)"
}

if (-not (Test-Path $Dataset)) {
    Write-Host "ERROR: Dataset folder missing: $Dataset" -ForegroundColor Red
    Write-Host "Run: python download_banana_and_build.py"
    Write-Host "     python download_sl_crops.py"
    Write-Host "     python build_dataset_sri_lanka.py"
    exit 1
}

$classCount = (Get-ChildItem $Dataset -Directory).Count
Write-Host "Dataset: $Dataset ($classCount class folders)"

Write-Host "Installing training packages..." -ForegroundColor Yellow
python -m pip install -q numpy opencv-python scikit-learn joblib matplotlib seaborn tqdm

Write-Host "Removing old model files (fresh start)..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $Output | Out-Null
Remove-Item "$Output\model.pkl", "$Output\scaler.pkl", "$Output\label_encoder.pkl" -ErrorAction SilentlyContinue
Remove-Item "$Output\results.txt", "$Output\confusion_matrix.png" -ErrorAction SilentlyContinue
if (Test-Path $BackendModels) {
    Remove-Item "$BackendModels\model.pkl", "$BackendModels\scaler.pkl", "$BackendModels\label_encoder.pkl" -ErrorAction SilentlyContinue
}

Write-Host "Training started (may take 30 min – 3 hours for ~97k images)..." -ForegroundColor Green
python train.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training FAILED. Check errors above." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Deploying models to backend..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $BackendModels | Out-Null
Copy-Item "$Output\model.pkl", "$Output\scaler.pkl", "$Output\label_encoder.pkl" -Destination $BackendModels -Force

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host "  Results : $Output\results.txt"
Write-Host "  Models  : $BackendModels"
Write-Host ""
Write-Host "Restart backend:" -ForegroundColor Cyan
Write-Host '  cd "d:\ai data\Final\plant-disease-backend"'
Write-Host "  python app.py"
