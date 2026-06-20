# Start Flask API for phone / emulator (run in PowerShell)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

# Show Wi-Fi IP for Flutter constants.dart
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -match 'Wi-Fi|WLAN' -and $_.IPAddress -notlike '169.*' } |
    Select-Object -First 1).IPAddress
if ($ip) {
    Write-Host ""
    Write-Host "Set in project\lib\utils\constants.dart :" -ForegroundColor Cyan
    Write-Host "  pcLanIpForPhysicalDevice = '$ip'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Phone URL: http://${ip}:5000" -ForegroundColor Green
} else {
    Write-Host "Could not detect Wi-Fi IP — run ipconfig" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Web UI: cd ..\web && npm run build  then open http://localhost:5000" -ForegroundColor Cyan
Write-Host "If phone cannot connect, run allow_firewall.ps1 as Administrator once." -ForegroundColor DarkYellow
Write-Host ""

$py = Join-Path $here "..\Merge-Project\plant_env\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Set-Location $here
& $py app.py
