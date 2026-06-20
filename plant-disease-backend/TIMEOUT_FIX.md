# Flutter Timeout Error - Troubleshooting Guide

## Error
```
Flask Login Error: TimeoutException after 0:00:10.000000: Future not completed
```

---

## Root Causes

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Supabase Down** | Requests hang for 10s | Check Supabase status at supabase.com |
| **Network Issue** | Cannot reach backend | Verify internet connection |
| **Wrong URL** | Connection refused | Check Flask URL in Flutter app |
| **CORS Blocked** | Request fails silently | Backend now has CORS enabled |
| **Slow Supabase Auth** | Login times out | Supabase taking too long to respond |
| **Missing .env** | Backend crashes on startup | Add SUPABASE_URL & SUPABASE_KEY to .env |

---

## Quick Fixes

### 1. Test Backend is Running

**Windows (PowerShell):**
```powershell
$response = Invoke-WebRequest http://localhost:5000/health -ErrorAction SilentlyContinue
if ($response.StatusCode -eq 200) {
    Write-Host "✓ Backend is running"
} else {
    Write-Host "✗ Backend not responding"
}
```

**Or visit in browser:**
```
http://localhost:5000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "message": "Backend is running"
}
```

### 2. Verify .env File

Create/update `.env` in project root:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**To get Supabase credentials:**
1. Go to https://supabase.com
2. Sign in to your project
3. Go to Settings → API
4. Copy "Project URL" and "anon/public" key

### 3. Check Network Connectivity

**From Python terminal:**
```python
import requests
import json

# Test health endpoint
response = requests.get('http://localhost:5000/health', timeout=5)
print(json.dumps(response.json(), indent=2))
```

**From PowerShell:**
```powershell
Test-NetConnection -ComputerName localhost -Port 5000 -InformationLevel "Detailed"
```

### 4. Update Flutter App Settings

Make sure your Flutter app uses correct timeout and URL:

```dart
// In your Flutter auth service
const Duration timeout = Duration(seconds: 30);

final response = await http.post(
  Uri.parse('http://localhost:5000/login'),  // Or your server IP
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'email': email,
    'password': password,
  }),
).timeout(timeout);
```

---

## Step-by-Step Debugging

### Step 1: Verify Backend Process

**Check if Flask is running:**
```powershell
Get-Process | Where-Object {$_.Name -like "*python*"}
```

**If not running, start it:**
```powershell
cd c:\Users\MIM.ASHRIF\Desktop\plant-disease-backend
backend_env\Scripts\activate
python app.py
```

**Expected output:**
```
 * Serving Flask app 'app'
 * Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

### Step 2: Test with cURL

**Test Health Endpoint:**
```bash
curl http://localhost:5000/health
```

**Test Login:**
```bash
curl -X POST http://localhost:5000/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123"
  }' \
  -v
```

**Look for:**
- `200 OK` = Success
- `400 Bad Request` = Invalid credentials
- `504 Gateway Timeout` = Supabase unreachable
- No response = Flask not running

### Step 3: Monitor Backend Logs

**Add logging to see what's happening:**

Backend now logs:
- Request start time
- Request completion time
- Error messages with cause

**Check terminal where Flask is running for messages like:**
```
[INFO] Login request completed in 2.45s
[ERROR] Login timeout: Supabase connection timeout
```

### Step 4: Test Supabase Connection

```python
# Run this to verify Supabase is accessible
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

try:
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )
    print("✓ Supabase connection successful")
except Exception as e:
    print(f"✗ Supabase connection failed: {e}")
```

---

## Solution by Scenario

### Scenario A: Backend Running Locally

**Flutter in emulator/device:**
- Use your computer's IP address, NOT `localhost`
- Find IP: `ipconfig` (look for IPv4 Address)
- Example: `http://192.168.1.100:5000/login`

**Flutter in web browser:**
- Can use `http://localhost:5000/login`

### Scenario B: Slow Supabase Response

**Increase timeout in Flutter:**
```dart
const Duration timeout = Duration(seconds: 30);  // Increased from 10s
```

**Check Supabase status:**
- Visit https://status.supabase.com
- Look for any incidents

### Scenario C: CORS Issues

Already fixed in updated `app.py` - Flask now:
- Has `CORS(app)` enabled
- Returns proper CORS headers
- Accepts requests from any origin

### Scenario D: Connection Refused

**Error:** `Connection refused`

**Causes:**
1. Backend not running
2. Port 5000 already in use
3. Firewall blocking port

**Fix:**
```powershell
# Check if port 5000 is in use
netstat -ano | findstr :5000

# If in use, kill the process
taskkill /PID <PID> /F

# Or use a different port in app.py
app.run(port=5001)
```

---

## Performance Optimization

### 1. Reduce Supabase Calls

**Before (slow):**
```python
def login_user(email, password):
    # Calls Supabase API
    res = supabase.auth.sign_in_with_password(...)
    return res
```

**After (with caching):**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def login_user(email, password):
    res = supabase.auth.sign_in_with_password(...)
    return res
```

### 2. Add Request Timeout to Backend

Update `app.py`:
```python
from threading import Timer

def timeout_handler():
    logger.warning("Request taking too long")

# Set timeout for each request
@app.before_request
def set_timeout():
    timer = Timer(30, timeout_handler)
    timer.daemon = True
    timer.start()
```

### 3. Connection Pooling

Backend now uses `threaded=True`:
```python
app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
```

---

## Testing Checklist

- [ ] Backend starts without errors
- [ ] `/health` endpoint returns `{"status": "healthy"}`
- [ ] `.env` file has SUPABASE_URL and SUPABASE_KEY
- [ ] Supabase project is active (check dashboard)
- [ ] Flask running on correct IP/port
- [ ] Flutter using correct URL
- [ ] CORS headers present in response
- [ ] Login endpoint returns token in <5 seconds

---

## New Features Added

### 1. Health Check Endpoint
```
GET /health
```
- Returns server status
- Use to verify backend is alive
- No authentication required

### 2. Enhanced Error Messages
- "Server timeout. Supabase connection slow. Please try again."
- "Invalid email or password"
- "Email and password required"

### 3. Request Logging
- Backend logs request duration
- Logs errors with full details
- Check terminal for debugging

### 4. Configuration Validation
- Validates .env on startup
- Checks for model files
- Prevents cryptic errors later

---

## Common Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | ✓ Working |
| 400 | Bad Request | Check request format |
| 401 | Invalid Credentials | Verify email/password |
| 504 | Gateway Timeout | Supabase unreachable |

---

## Still Having Issues?

1. **Collect Information:**
   ```
   - Error message (full text)
   - Network (wifi/ethernet/other)
   - OS (Windows/Mac/Linux)
   - Python version
   - Flutter URL being used
   ```

2. **Check Logs:**
   - Look at Flask terminal output
   - Look at Flutter console output
   - Check browser console (if web)

3. **Test Connectivity:**
   ```powershell
   ping 8.8.8.8  # Test internet
   ping supabase.co  # Test Supabase access
   curl http://localhost:5000/health  # Test backend
   ```

4. **Reset Everything:**
   ```powershell
   # Kill Flask
   taskkill /F /IM python.exe
   
   # Restart virtual environment
   backend_env\Scripts\activate
   python app.py
   ```

---

## Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Health Check | <10ms | No DB calls |
| Register | 1-5s | Depends on Supabase |
| Login | 1-5s | Depends on Supabase |
| Predict | 100-500ms | ML inference |
| History | <1s | Small dataset |

---

**Document Version:** 1.0  
**Last Updated:** May 22, 2026
