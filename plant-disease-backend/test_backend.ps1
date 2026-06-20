param(
    [Parameter(Mandatory=$false, HelpMessage="Path to leaf image (JPEG/PNG)")]
    [string]$ImagePath,
    [Parameter(Mandatory=$false)]
    [string]$ServerUrl = "http://127.0.0.1:5000"
)

# ─── Helper: resolve path & check file ──────────────────────
function Get-ValidImagePath {
    param([string]$Raw)

    if (-not $Raw -or $Raw -eq '') { return $null }

    # Resolve relative to current directory
    $resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Raw)
    if (Test-Path -LiteralPath $resolved -PathType Leaf) {
        return (Resolve-Path -LiteralPath $resolved).ProviderPath
    }
    return $null
}

# ─── Helper: list dataset test images ───────────────────────
function Get-TestImages {
    $base = "D:\ai data\Final\Merge-Project\resized_merged"
    if (-not (Test-Path -LiteralPath $base)) { return @() }

    $images = @()
    $dirs = Get-ChildItem -Path $base -Directory | Where-Object { $_.Name -like "Apple*" }
    foreach ($d in $dirs) {
        $first = Get-ChildItem -Path $d.FullName -File | Select-Object -First 1
        if ($first) {
            $images += [PSCustomObject]@{
                Label = $d.Name.Replace('___', ' ').Replace('_', ' ')
                Path  = $first.FullName
            }
        }
    }
    return $images
}

# ─── Helper: upload image (PS 5.1 compatible) ──────────────
function Upload-Image {
    param([string]$Path, [string]$Endpoint)

    $uri = "$ServerUrl$Endpoint"

    Add-Type -AssemblyName System.Net.Http
    $client = New-Object System.Net.Http.HttpClient
    $content = New-Object System.Net.Http.MultipartFormDataContent
    $fileStream = $null

    try {
        $fileName = [System.IO.Path]::GetFileName($Path)
        $fileStream = [System.IO.File]::OpenRead($Path)
        $fileContent = New-Object System.Net.Http.StreamContent($fileStream)

        # field name "image" matches Flask's request.files.get("image")
        $content.Add($fileContent, "image", $fileName)

        Write-Host "`n>>> POST $uri (file: $fileName)" -ForegroundColor Cyan
        $response = $client.PostAsync($uri, $content).GetAwaiter().GetResult()
        $responseText = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()

        if ($response.IsSuccessStatusCode) {
            Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
            Write-Host "Response:" -ForegroundColor Green
            $parsed = $responseText | ConvertFrom-Json
            $parsed | ConvertTo-Json -Depth 5
        } else {
            Write-Host "Status: $($response.StatusCode)" -ForegroundColor Red
            Write-Host "Error body:" -ForegroundColor Red
            try { $responseText | ConvertFrom-Json | ConvertTo-Json -Depth 5 } catch { $responseText }
        }
    } catch {
        Write-Host "Request failed: $_" -ForegroundColor Red
    } finally {
        if ($fileStream) { $fileStream.Close() }
        $client.Dispose()
    }
}

# ─── Main ───────────────────────────────────────────────────
Write-Host "===== Plant Disease Backend Test =====" -ForegroundColor Yellow
Write-Host "Server: $ServerUrl" -ForegroundColor Gray

# 1) Resolve image path
$imageFile = Get-ValidImagePath $ImagePath

if (-not $imageFile) {
    Write-Host ""
    Write-Host "Image file not found or not provided." -ForegroundColor Yellow
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  [1] Drag & drop an image file onto this window + press Enter" -ForegroundColor Cyan
    Write-Host "  [2] Type/paste a full path (e.g. C:\Users\MIM.ASHRIF\Downloads\leaf.jpg)" -ForegroundColor Cyan
    Write-Host "  [3] Pick from built-in test images below" -ForegroundColor Cyan

    # Show built-in test options
    $testImages = Get-TestImages
    if ($testImages.Count -gt 0) {
        Write-Host ""
        Write-Host "  Built-in test images:" -ForegroundColor Magenta
        for ($i = 0; $i -lt $testImages.Count; $i++) {
            Write-Host "  [$($i+1)] $($testImages[$i].Label)" -ForegroundColor Magenta
        }
        Write-Host "  [0] Custom path (type your own)" -ForegroundColor Magenta

        $choice = Read-Host "`nPick a number (or 0 for custom path, or press Enter to quit)"
        if ($choice -match '^\d+$') {
            $idx = [int]$choice
            if ($idx -ge 1 -and $idx -le $testImages.Count) {
                $imageFile = $testImages[$idx - 1].Path
                Write-Host "Selected: $($testImages[$idx - 1].Label)" -ForegroundColor Green
            }
        }
    }

    # If still no file, prompt for custom path
    if (-not $imageFile) {
        $dropped = Read-Host "`nEnter image path (or drag & drop) and press Enter"
        $imageFile = Get-ValidImagePath $dropped
        if (-not $imageFile) {
            Write-Host "ERROR: File not found at '$dropped'" -ForegroundColor Red
            Write-Host "       Make sure the path is correct and the file exists." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host "Using image: $imageFile" -ForegroundColor Green

# 2) Health check
Write-Host "`n--- Health check ---" -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest -Uri "$ServerUrl/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "Server OK: $($health.Content)" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Health check failed. Is the server running?" -ForegroundColor Red
    Write-Host "  Start it: .\venv\Scripts\python.exe app.py" -ForegroundColor Gray
    Write-Host "  Error: $_" -ForegroundColor Red
    $continue = Read-Host "`nContinue anyway? (y/N)"
    if ($continue -ne 'y') { exit 1 }
}

# 3) Test /upload_predict
Write-Host "`n===== Test 1: /upload_predict =====" -ForegroundColor Yellow
Upload-Image -Path $imageFile -Endpoint "/upload_predict"

# 4) Test /predict
Write-Host "`n===== Test 2: /predict =====" -ForegroundColor Yellow
Upload-Image -Path $imageFile -Endpoint "/predict"

Write-Host "`nDone." -ForegroundColor Cyan
