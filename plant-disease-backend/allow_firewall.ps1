# Run as Administrator: right-click PowerShell -> Run as administrator
# Allows phones on same Wi-Fi to reach Flask on port 5000

netsh advfirewall firewall add rule name="Plant Disease API 5000" dir=in action=allow protocol=TCP localport=5000
Write-Host "Firewall rule added for TCP port 5000" -ForegroundColor Green
