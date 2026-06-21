# Full Authentication System — File Map & Integration

## Backend auth files (`plant-disease-backend/`)

| File | Role in auth system |
|------|---------------------|
| **`.env`** | Live secrets (not in git): `SUPABASE_URL`, `SUPABASE_KEY`, `JWT_SECRET`, … |
| **`.env.example`** | Template — copy to `.env` and fill values |
| **`config.py`** | Loads env; `JWT_*`, `SUPABASE_AUTH_ENABLED` |
| **`jwt_auth.py`** | Create/verify access & refresh JWT; `refresh_tokens.json` store |
| **`jwt_middleware.py`** | `@jwt_required`, `@optional_jwt` on Flask routes |
| **`auth.py`** | Register, login, logout, email check, password hash, optional Supabase |
| **`app.py`** | HTTP routes: `/register`, `/login`, `/refresh`, `/me`, `/logout`, protected APIs |
| **`local_users.json`** | Local accounts (hashed passwords) |
| **`refresh_tokens.json`** | Active refresh token JTIs |
| **`test_app.py`** | JWT flow tests (register → me → predict → refresh → logout) |

## Flutter auth files (`project/lib/`)

| File | Role in auth system |
|------|---------------------|
| **`services/auth_service.dart`** | Login, register, JWT storage, refresh, `/me`, logout |
| **`services/api_service.dart`** | Sends `Authorization: Bearer` on predict/history/clarify/answer |
| **`providers/auth_provider.dart`** | UI state; shared `AuthService` instance |
| **`providers/prediction_provider.dart`** | Uses same `AuthService` for protected API calls |
| **`screens/auth_gate_screen.dart`** | Sign in / register UI |
| **`screens/splash_screen.dart`** | `validateSession()` → home or auth |
| **`screens/profile_screen.dart`** | Logout → server `/logout` + clear tokens |
| **`utils/constants.dart`** | API paths: `/login`, `/register`, `/refresh`, `/me`, `/logout` |
| **`utils/validators.dart`** | Email/password validation |
| **`models/user_model.dart`** | User + `access_token` parsing |
| **`main.dart`** | Single shared `AuthService()` for all providers |
| **`widgets/auth_scaffold.dart`** | Auth screen layout |
| **`widgets/auth_loading_overlay.dart`** | Loading during auth requests |

Legacy screens (still work via routes): `login_screen.dart`, `register_screen.dart` → redirect to `AuthGateScreen`.

## Auth flow (end-to-end)

```
1. User registers/logs in on Flutter (auth_gate_screen)
2. Flask returns access_token + refresh_token + expires_in
3. auth_service saves tokens in SharedPreferences
4. splash_screen calls validateSession() → GET /me (refresh if 401)
5. predict/history: api_service adds Bearer header
6. On 401: refreshAccessToken() → retry once
7. Logout: POST /logout + clear local tokens
```

## Required `.env` variables

```env
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your_key
JWT_SECRET=long-random-secret-min-32-chars
JWT_ACCESS_MINUTES=60
JWT_REFRESH_DAYS=30
SUPABASE_AUTH_ENABLED=false
```

Copy from `.env.example`:

```powershell
cd plant-disease-backend
copy .env.example .env
# Edit .env with your values
```

## API auth matrix

| Endpoint | JWT required |
|----------|--------------|
| `/register`, `/login`, `/refresh`, `/check-email`, `/health` | No |
| `/me`, `/logout`, `/predict`, `/history` | Yes |
| `/clarify`, `/answer` | Optional |
| `/treatment`, `/diseases` | No |

## Verify everything works

```powershell
cd plant-disease-backend
pip install -r requirements.txt
python app.py
# In another terminal:
.\run_tests.ps1
```

Flutter: restart app after backend is running; sign in again if old `local_xxx` tokens were stored.
