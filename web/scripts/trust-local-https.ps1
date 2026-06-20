# Install a trusted local CA so https://localhost works without certificate warnings.
# Run PowerShell as Administrator:
#   cd "D:\ai data\Final\web"
#   .\scripts\trust-local-https.ps1

$ErrorActionPreference = "Stop"
$webRoot = Split-Path $PSScriptRoot -Parent
Set-Location $webRoot

Write-Host "Plant Village AI — trusted HTTPS setup" -ForegroundColor Cyan
Write-Host ""

# mkcert binary downloaded by vite-plugin-mkcert on first `npm run dev`
$mkcertExe = Join-Path $env:USERPROFILE ".vite-plugin-mkcert\mkcert.exe"

if (-not (Test-Path $mkcertExe)) {
    Write-Host "mkcert not found yet. Run once first:" -ForegroundColor Yellow
    Write-Host "  npm run dev" -ForegroundColor White
    Write-Host "Wait until Vite starts, then press Ctrl+C and run this script again as Administrator." -ForegroundColor Yellow
    exit 1
}

Write-Host "Using: $mkcertExe" -ForegroundColor Gray
& $mkcertExe -install
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "If this failed, right-click PowerShell -> Run as administrator, then run this script again." -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Success. Trusted CA installed for localhost." -ForegroundColor Green
Write-Host "1. Stop any old dev server (Ctrl+C)" -ForegroundColor White
Write-Host "2. npm run dev" -ForegroundColor White
Write-Host "3. Open https://localhost:5173/login" -ForegroundColor White
Write-Host ""
Write-Host "Close old tabs that used the bad example.org certificate." -ForegroundColor Yellow
