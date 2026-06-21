# Plant Disease Detection — Backend Technical Report

**Project path:** `d:\ai data\Final\plant-disease-backend`  
**Report scope:** Backend only (Flask API, ML, auth, storage, tests)  
**Generated:** May 2026  
**Model version:** Sri Lanka dataset — **69 disease classes**

| Document | Use |
|----------|-----|
| **This file** (`BACKEND_REPORT.md`) | Full technical report |
| `BACKEND_REPORT_SHORT.md` | ~2-page executive summary for submission |
| `BACKEND_REPORT_COVER.md` | Title / cover page |

---

## Table of contents

1. [Executive summary](#1-executive-summary)  
2. [Technology stack](#2-technology-stack)  
3. [Project structure](#3-project-structure)  
4. [Configuration & environment](#4-configuration--environment)  
5. [Authentication (JWT)](#5-authentication-jwt)  
6. [REST API reference](#6-rest-api-reference)  
7. [Machine learning pipeline](#7-machine-learning-pipeline)  
8. [Sri Lanka crop classes](#8-sri-lanka-crop-classes)  
9. [Full model class list (69)](#9-full-model-class-list-69)  
10. [Clarification Q&A engine](#10-clarification-qa-engine)  
11. [Data storage](#11-data-storage)  
12. [Training & model deployment](#12-training--model-deployment)  
13. [Testing & quality assurance](#13-testing--quality-assurance)  
14. [Deployment & operations](#14-deployment--operations)  
15. [Known limitations](#15-known-limitations)  
16. [Architecture diagram](#16-architecture-diagram)  
17. [Related documentation](#17-related-documentation)

---

## 1. Executive summary

The backend is a **Python Flask REST API** that powers the Plant Village mobile app. It provides:

| Capability | Description |
|------------|-------------|
| **User accounts** | Register, login, JWT access/refresh tokens, logout |
| **Leaf disease prediction** | Upload image → sklearn classifier + computer-vision re-ranking |
| **Treatment information** | Symptoms, cause, fertilizer, prevention per disease class |
| **Low-confidence Q&A** | YES/NO questions to refine between similar diseases |
| **Scan history** | Per-user history in `prediction_history.json` (+ optional Supabase backup) |

**Current production model:** 69 classes trained on `dataset_sri_lanka` (PlantVillage + banana, rice, coconut, tea, chili, mango, papaya).  
**Default port:** `5000` on `0.0.0.0` (reachable from phone on same Wi‑Fi).

---

## 2. Technology stack

| Layer | Technology | Version (requirements.txt) |
|-------|------------|----------------------------|
| Language | Python 3 | 3.10+ recommended |
| Web framework | Flask | 3.1.3 |
| CORS | Flask-CORS | 6.0.2 |
| Production server | Waitress (WSGI) | 2.1.2 |
| ML | scikit-learn | 1.8.0 |
| Model I/O | joblib | 1.5.3 |
| Computer vision | OpenCV (headless) | 4.13.0.92 |
| Arrays | NumPy | 2.4.6 |
| Auth tokens | PyJWT | 2.10.1 |
| Password hashing | Werkzeug (via Flask) | — |
| Config | python-dotenv | 0.21.0 |
| Optional cloud | Supabase Python client | 2.30.0 |
| Testing | unittest | stdlib |

---

## 3. Project structure

```
plant-disease-backend/
├── app.py                  # Flask routes (all HTTP endpoints)
├── auth.py                 # Users, login, register, history save/fetch
├── config.py               # Env vars, model paths, startup validation
├── jwt_auth.py             # Create/verify/refresh JWT tokens
├── jwt_middleware.py       # @jwt_required, @optional_jwt decorators
├── predict.py              # ML inference, DISEASE_INFO knowledge base
├── leaf_analysis.py        # Crop shape heuristics, probability re-ranking
├── qa_engine.py            # Clarification questions & answers
├── history_store.py        # Local JSON prediction history
├── wsgi.py                 # Waitress production entry
├── smoke_check.py          # Quick health + model smoke test
├── test_app.py             # API unit tests (22 tests)
├── test_leaf_analysis.py   # Coconut vs grape re-ranking tests (4 tests)
├── run_tests.ps1           # Windows test runner
├── requirements.txt        # Pip dependencies
├── .env.example            # Environment template
├── .env                    # Secrets (not committed)
├── local_users.json        # Local user accounts (runtime)
├── refresh_tokens.json     # JWT refresh token registry (runtime)
├── prediction_history.json # Per-user scan history (runtime)
├── supabase_predictions.sql# Optional Supabase table schema
├── models/
│   ├── model.pkl           # Trained classifier (~1.1 GB)
│   ├── scaler.pkl          # Feature StandardScaler
│   └── label_encoder.pkl   # 69 class label strings
├── BACKEND_REPORT.md       # This document
├── AUTH_SYSTEM.md          # Auth details
├── DEPLOYMENT.md           # Deploy guide
├── TRAIN_SRI_LANKA.md      # How to retrain SL crops
└── MODEL_TRAINING_REPORT.md
```

**Virtual environments (not source):** `venv/`, `backend_env/`, `.test_venv/`

---

## 4. Configuration & environment

### 4.1 `config.py` settings

| Setting | Default / path | Purpose |
|---------|----------------|---------|
| `MODEL_PATH` | `models/model.pkl` | sklearn classifier |
| `SCALER_PATH` | `models/scaler.pkl` | Feature scaling |
| `LABEL_ENCODER_PATH` | `models/label_encoder.pkl` | Class names |
| `IMG_SIZE` | `128` | Resize for feature extraction |
| `JWT_SECRET` | from `.env` | HS256 signing key |
| `JWT_ACCESS_MINUTES` | `60` | Access token lifetime |
| `JWT_REFRESH_DAYS` | `30` | Refresh token lifetime |
| `SUPABASE_URL` | `.env` | Optional cloud DB |
| `SUPABASE_KEY` | `.env` | Supabase API key |
| `SUPABASE_AUTH_ENABLED` | `false` | Use Supabase login (OTP risk if true) |

On import, `validate_config()` ensures all three `.pkl` files exist.

### 4.2 `.env` variables

Copy `.env.example` → `.env`:

```env
SUPABASE_URL=          # Optional — history backup only
SUPABASE_KEY=          # Optional
JWT_SECRET=<long-random-secret>
JWT_ACCESS_MINUTES=60
JWT_REFRESH_DAYS=30
SUPABASE_AUTH_ENABLED=false
```

| Variable | Required | Notes |
|----------|----------|-------|
| `JWT_SECRET` | **Strongly recommended** | Never use dev default in production |
| `SUPABASE_URL` / `KEY` | Optional | History works locally without Supabase |
| `SUPABASE_AUTH_ENABLED` | Optional | Keep `false` to avoid Supabase OTP emails |

---

## 5. Authentication (JWT)

### 5.1 Flow

```
POST /register or /login
    → auth.py validates email/password
    → jwt_auth.create_token_pair()
    → Response: access_token, refresh_token, user_id, email, expires_in

Protected request (e.g. POST /predict)
    → Header: Authorization: Bearer <access_token>
    → jwt_middleware.jwt_required
    → g.current_user = { user_id, email, full_name }

Token expired
    → POST /refresh { "refresh_token": "..." }
    → New access + refresh pair; old refresh JTI rotated
```

### 5.2 Security details

- Passwords stored as **Werkzeug hashes** in `local_users.json` (never plain text).
- **Blocked email domains:** `example.com`, `test.com`, etc. (see `auth.py`).
- **Refresh tokens** stored in `refresh_tokens.json` with JTI rotation on refresh.
- **Local user_id:** MD5-based stable ID from normalized email (works without Supabase).

### 5.3 Auth modules

| File | Role |
|------|------|
| `auth.py` | Register, login, logout, refresh, save/get history |
| `jwt_auth.py` | `create_token_pair`, `decode_token`, `refresh_access_token` |
| `jwt_middleware.py` | `jwt_required`, `optional_jwt` |

---

## 6. REST API reference

**Base URL:** `http://<PC-LAN-IP>:5000`  
**JSON** unless noted. **Multipart** for `/predict`.

### 6.1 Endpoints summary

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | Public | Server status |
| GET | `/check-email?email=` | Public | Format + duplicate check |
| POST | `/register` | Public | Create account → JWT |
| POST | `/login` | Public | Sign in → JWT |
| POST | `/refresh` | Public | New token pair |
| GET | `/me` | JWT | Current user profile |
| POST | `/logout` | JWT | Revoke refresh tokens |
| POST | `/predict` | JWT | Upload leaf image → prediction |
| GET | `/history` | JWT | List user scans |
| POST | `/history` | JWT | Manually append history row |
| POST | `/clarify` | Optional JWT | Q&A for two diseases |
| POST | `/answer` | Optional JWT | Submit YES/NO answer |
| GET | `/treatment/<disease>` | Public | Treatment for one class |
| GET | `/diseases` | Public | All disease labels |

### 6.2 `POST /register` / `POST /login`

**Request body:**
```json
{
  "email": "farmer@gmail.com",
  "password": "SecurePass123",
  "full_name": "Kamal Perera"
}
```

**Success response (201 register / 200 login):**
```json
{
  "message": "Login successful",
  "user_id": "abc123...",
  "email": "farmer@gmail.com",
  "full_name": "Kamal Perera",
  "token": "<access_jwt>",
  "access_token": "<access_jwt>",
  "refresh_token": "<refresh_jwt>",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### 6.3 `POST /predict`

**Headers:** `Authorization: Bearer <access_token>`  
**Body:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | JPEG/PNG leaf photo |
| `force_clarify` | string | No | `true` forces low confidence + Q&A |

**Success response (200):**
```json
{
  "disease": "Coconut_Leaf_rot",
  "display_name": "Coconut - Leaf rot",
  "plant_type": "Coconut",
  "confidence": 72.5,
  "is_confident": true,
  "needs_clarification": false,
  "unsupported_plant": false,
  "detected_plant": null,
  "clarification_questions": [],
  "all_predictions": [
    { "disease": "Coconut_Leaf_rot", "display_name": "Coconut - Leaf rot", "confidence": 72.5 },
    { "disease": "Coconut_healthy", "display_name": "Coconut - Healthy", "confidence": 15.2 }
  ],
  "treatment": {
    "disease": "Coconut_Leaf_rot",
    "display_name": "Coconut - Leaf rot",
    "symptoms": "Brown rotting areas on fronds.",
    "treatment": "Remove severely affected fronds...",
    "prevention": "Good nutrition; avoid waterlogging",
    "fertilizer": "Potassium and micronutrients per soil test",
    "severity": "Medium",
    "cause": "Leaf rot / spotting on coconut palm foliage",
    "weather": "Very wet conditions",
    "insects": "Secondary pests may follow damage"
  }
}
```

**Clarification:** If `confidence < 60` or `force_clarify=true`, `needs_clarification` is true and `clarification_questions` is populated from `qa_engine.py`.

**Unsupported plant:** When leaf shape indicates a plant not in training (rare after SL training), `unsupported_plant: true` and treatment explains limited support.

### 6.4 `GET /history`

**Headers:** JWT required  

**Response:** Array of scan records (newest first), max 50 per request, 100 stored per user locally.

```json
[
  {
    "id": "uuid",
    "user_id": "...",
    "disease": "Coconut_healthy",
    "display_name": "Coconut - Healthy",
    "plant_type": "Coconut",
    "confidence": 81.0,
    "image_name": "leaf.jpg",
    "created_at": "2026-05-24T10:00:00+00:00",
    "is_confident": true,
    "needs_clarification": false,
    "treatment": { ... }
  }
]
```

### 6.5 `POST /clarify` and `POST /answer`

**Clarify request:**
```json
{ "disease1": "Tomato_Early_blight", "disease2": "Tomato_Late_blight" }
```

**Answer request:**
```json
{
  "disease1": "Tomato_Early_blight",
  "disease2": "Tomato_Late_blight",
  "question_index": 0,
  "answer": "yes"
}
```

**Answer response:** `{ "selected_disease": "...", "display_name": "...", "confidence": 85, "treatment": {...} }`

### 6.6 Error format

```json
{ "error": "Human-readable message" }
```

Common HTTP codes: `400` validation, `401` auth, `404` not found, `409` email taken, `504` Supabase timeout.

---

## 7. Machine learning pipeline

### 7.1 Inference flow

```
Image bytes (JPEG/PNG)
    → cv2.imdecode
    → detect_unsupported_plant() [optional block for unknown plants]
    → extract_features() — resize 128×128, HSV histograms (3×32 bins),
      Canny edge mean, per-channel mean/std  → 103 features
    → scaler.transform()
    → model.predict_proba()
    → refine_probabilities() [leaf_analysis.py — crop shape + lesion rules]
    → ML crop preference fix (coconut vs grape, Sri Lanka crops)
    → Top-5 predictions + DISEASE_INFO lookup
    → Return disease, confidence, plant_type, treatment
```

### 7.2 Feature extraction (`predict.py`)

| Step | Detail |
|------|--------|
| Resize | 128 × 128 pixels |
| HSV histogram | 32 bins per channel → 96 values |
| Canny edges | Mean edge strength → 1 value |
| BGR stats | Mean + std per channel → 6 values |
| **Total** | **103 features** |

### 7.3 Vision re-ranking (`leaf_analysis.py`)

Prevents common crop confusion (e.g. **coconut shown as grape**):

| Heuristic | Behavior |
|-----------|----------|
| **Palm leaflet shape** | Long/narrow leaf, ≤2 lobes → boost Coconut, Papaya |
| **Grape lobes** | Round spots boost Grape only if **≥3 lobes** (grapevine shape) |
| **Grape penalty** | Non-lobed leaves → Grape classes × 0.25 |
| **SL crop preference** | If ML top is Coconut/Rice/etc. but vision picked Grape → trust ML |
| **Grape black rot boost** | Only on confirmed lobed grape + round spots |

**Sri Lanka crops in preference set:** `Banana`, `Rice`, `Coconut`, `Tea`, `Chili`, `Mango`, `Papaya`

### 7.4 Model files

| File | Approx. size | Content |
|------|--------------|---------|
| `model.pkl` | ~1.1 GB | Trained sklearn classifier |
| `scaler.pkl` | small | StandardScaler fit on training features |
| `label_encoder.pkl` | small | 69 class name strings |

Models are **lazy-loaded** on first `/predict` (background warmup thread on startup).

### 7.5 Confidence rules (`app.py`)

| Rule | Value |
|------|-------|
| Confident scan | `confidence >= 60` and not unsupported |
| Needs clarification | `confidence < 60` or `force_clarify` |
| `force_clarify` | Caps displayed confidence at 45% |

---

## 8. Sri Lanka crop classes

**31 of 69** classes are Sri Lanka–focused crops (trained in `Merge-Project/dataset_sri_lanka`):

| Crop | Classes | Count |
|------|---------|-------|
| **Banana** | Sigatoka, Xanthomonas wilt, healthy | 3 |
| **Rice** | Bacterial blight, Blast, Brown spot, Tungro | 4 |
| **Coconut** | Gray leaf spot, Leaf rot, healthy | 3 |
| **Tea** | Algal leaf spot, Anthracnose, Bird eye spot, Brown/Gray blight, Red rust, White spot, healthy | 8 |
| **Chili** | Bacterial spot, healthy | 2 |
| **Mango** | Anthracnose, Bacterial canker, Die back, Powdery mildew, Sooty mould, healthy | 6 |
| **Papaya** | Anthracnose, Bacterial spot, Leaf curl, Ringspot, healthy | 5 |

**Note:** Coconut/Chili training used **proxy images** when Mendeley downloads were blocked (papaya/pepper folders mapped to coconut/chili names). Real Sri Lanka field photos improve accuracy.

---

## 9. Full model class list (69)

### PlantVillage + international (38 classes)

```
Apple_Cedar_apple_rust
Apple___Apple_scab
Apple___Black_rot
Apple_healthy
Blueberry_healthy
Cherry_(including_sour)_Powdery_mildew
Cherry_(including_sour)_healthy
Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot
Corn_(maize)_Common_rust_
Corn_(maize)_Northern_Leaf_Blight
Corn_(maize)_healthy
Grape_Black_rot
Grape_Esca_(Black_Measles)
Grape_Leaf_blight_(Isariopsis_Leaf_Spot)
Grape_healthy
Orange_Haunglongbing_(Citrus_greening)
Peach_Bacterial_spot
Peach_healthy
Pepper,_bell_Bacterial_spot
Pepper,_bell_healthy
Potato_Early_blight
Potato_Late_blight
Potato_healthy
Raspberry_healthy
Soybean_healthy
Squash_Powdery_mildew
Strawberry_Leaf_scorch
Strawberry_healthy
Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_Late_blight
Tomato_Leaf_Mold
Tomato_Septoria_leaf_spot
Tomato_Spider_mites_Two-spotted_spider_mite
Tomato_Target_Spot
Tomato_Tomato_Yellow_Leaf_Curl_Virus
Tomato_Tomato_mosaic_virus
Tomato_healthy
```

### Sri Lanka crops (31 classes)

```
Banana_Sigatoka
Banana_Xanthomonas_wilt
Banana_healthy
Chili_Bacterial_spot
Chili_healthy
Coconut_Gray_leaf_spot
Coconut_Leaf_rot
Coconut_healthy
Mango_Anthracnose
Mango_Bacterial_canker
Mango_Die_back
Mango_Powdery_mildew
Mango_Sooty_mould
Mango_healthy
Papaya_Anthracnose
Papaya_Bacterial_spot
Papaya_Leaf_curl
Papaya_Ringspot
Papaya_healthy
Rice_Bacterial_blight
Rice_Blast
Rice_Brown_spot
Rice_Tungro
Tea_Algal_leaf_spot
Tea_Anthracnose
Tea_Bird_eye_spot
Tea_Brown_blight
Tea_Gray_blight
Tea_Red_rust
Tea_White_spot
Tea_healthy
```

**Label format:** `Crop_Disease_name` (underscores). Display names formatted in `format_display_name()` → e.g. `Coconut_Leaf_rot` → **"Coconut - Leaf rot"**.

---

## 10. Clarification Q&A engine

**File:** `qa_engine.py`  
**Purpose:** When two diseases score similarly, ask farmer YES/NO symptom questions.

### 10.1 Configured disease pairs

| Pair | Crops |
|------|-------|
| Tomato Early vs Late blight | Tomato |
| Tomato Bacterial spot vs Septoria | Tomato |
| Potato Early vs Late blight | Potato |
| Apple scab vs Black rot | Apple |
| Rice Blast vs Brown spot | Rice |
| Mango Anthracnose vs Powdery mildew | Mango |
| Grape Black rot vs Leaf blight | Grape |
| Grape Black rot vs Esca | Grape |

If no pair matches, **default generic questions** are returned (may not change disease).

### 10.2 Functions

| Function | Description |
|----------|-------------|
| `get_clarification_questions(d1, d2)` | List of `{ question, yes_disease, no_disease }` |
| `process_answer(d1, d2, index, "yes"/"no")` | Returns selected disease + treatment |
| `get_treatment(disease)` | Lookup from `DISEASE_INFO` in `predict.py` |
| `list_diseases()` | All keys in `DISEASE_INFO` |

---

## 11. Data storage

| Storage | Location | Contents |
|---------|----------|----------|
| **Users** | `local_users.json` | email, password hash, user_id, full_name |
| **Refresh tokens** | `refresh_tokens.json` | JTI → user mapping, expiry |
| **Prediction history** | `prediction_history.json` | Per-user scans (max 100/user) |
| **ML models** | `models/*.pkl` | Classifier, scaler, labels |
| **Secrets** | `.env` | JWT, Supabase keys |
| **Supabase (optional)** | Cloud PostgreSQL | `predictions` table backup |

`history_store.py` always writes locally. `auth.py` optionally syncs to Supabase when URL/key are set.

---

## 12. Training & model deployment

Training is done in **`d:\ai data\Final\Merge-Project`** (not in this folder).

| Step | Command / file |
|------|----------------|
| Build dataset | `python build_dataset_sri_lanka.py` |
| Download SL crops | `python download_sl_crops.py` |
| Train | `python train.py` (uses `config.py`: `dataset_sri_lanka`, `AUTO_DISCOVER_CLASSES`) |
| Deploy to backend | Copy `model.pkl`, `scaler.pkl`, `label_encoder.pkl` → `plant-disease-backend/models/` |
| Restart API | `python app.py` |

See **`TRAIN_SRI_LANKA.md`** and **`MODEL_TRAINING_REPORT.md`** for dataset sources and hyperparameters.

---

## 13. Testing & quality assurance

### 13.1 Test files

| File | Tests | Coverage |
|------|-------|----------|
| `test_app.py` | 22 | Health, auth, predict, history, QA endpoints |
| `test_leaf_analysis.py` | 4 | Coconut vs grape re-ranking |
| `smoke_check.py` | — | Models load, features, predict, `/health`, QA |

### 13.2 How to run

```powershell
cd "d:\ai data\Final\plant-disease-backend"

# All unit tests
python -m unittest test_app test_leaf_analysis -v

# Quick smoke (no unittest)
python smoke_check.py

# Windows script
.\run_tests.ps1
```

**Last verified:** 26 tests OK, smoke check passed (69 classes).

---

## 14. Deployment & operations

### 14.1 Development

```powershell
cd "d:\ai data\Final\plant-disease-backend"
pip install -r requirements.txt
copy .env.example .env
# Edit .env — set JWT_SECRET
python app.py
```

Console output:
```
✓ Flask app starting on 0.0.0.0:5000
Auth: JWT enabled ...
✓ ML models preloaded
```

### 14.2 Production (Waitress)

```powershell
python wsgi.py
```

- Host: `0.0.0.0:5000`  
- Threads: 10  
- Connection limit: 256  

### 14.3 Mobile app connection

Phone must use the PC’s **Wi‑Fi IPv4** (from `ipconfig`), not `localhost`.  
Example: `http://10.124.191.183:5000` — must match Flutter `constants.dart` on the device.

### 14.4 Startup checklist

- [ ] `models/*.pkl` present  
- [ ] `JWT_SECRET` set in `.env`  
- [ ] Firewall allows inbound TCP **5000**  
- [ ] PC and phone on same network  
- [ ] After model update, restart `python app.py`

---

## 15. Known limitations

| Topic | Detail |
|-------|--------|
| **Coconut / Chili training data** | Partially proxy images (papaya/pepper) when external datasets were blocked |
| **Rice healthy class** | No `Rice_healthy` in current 69-class model |
| **Model size** | ~1.1 GB RAM needed to load `model.pkl` |
| **64-bit Python** | Required; 32-bit may fail with "Models too large to load" |
| **Class confusion** | Similar leaf shapes (coconut vs grape) mitigated in code but not 100% on all photos |
| **Q&A pairs** | Limited to configured pairs; unknown pairs use generic questions |
| **Supabase auth** | Off by default; local JWT only |

---

## 16. Architecture diagram

```
┌──────────────────┐     HTTP/JSON + multipart      ┌─────────────────────────────┐
│  Flutter mobile  │ ◄──────────────────────────────► │  Flask app.py (:5000)       │
│  (not in scope)  │     Authorization: Bearer JWT    │                             │
└──────────────────┘                                  │  ├─ jwt_middleware.py       │
                                                      │  ├─ auth.py                 │
                                                      │  ├─ predict.py              │
                                                      │  ├─ leaf_analysis.py        │
                                                      │  └─ qa_engine.py            │
                                                      └──────────┬──────────────────┘
                                                                 │
           ┌─────────────────────────────┬───────────────────────┼────────────────────┐
           ▼                             ▼                       ▼                    ▼
   local_users.json              prediction_history.json    models/*.pkl      Supabase (optional)
   refresh_tokens.json           history_store.py           scikit-learn      predictions table
```

---

## 17. Related documentation

| Document | Content |
|----------|---------|
| `AUTH_SYSTEM.md` | JWT and user storage details |
| `DEPLOYMENT.md` | Server deployment steps |
| `TRAIN_SRI_LANKA.md` | Dataset download and retraining |
| `MODEL_TRAINING_REPORT.md` | Training metrics and methodology |
| `PROJECT_REPORT.md` | Older combined project notes |
| `../Merge-Project/config.py` | Training dataset path and settings |

---

## Appendix A — Source file responsibilities

| File | Key functions / classes |
|------|-------------------------|
| `app.py` | All routes, `_auth_payload`, `_warmup_models` |
| `auth.py` | `register_user`, `login_user`, `save_prediction`, `get_history` |
| `jwt_auth.py` | `create_token_pair`, `refresh_access_token`, `get_user_from_access_token` |
| `jwt_middleware.py` | `jwt_required`, `optional_jwt` |
| `predict.py` | `predict_disease`, `extract_features`, `DISEASE_INFO` |
| `leaf_analysis.py` | `detect_crop_family`, `refine_probabilities`, `best_class_index_for_crop` |
| `qa_engine.py` | `get_clarification_questions`, `process_answer`, `list_diseases` |
| `history_store.py` | `append_prediction`, `list_predictions` |
| `config.py` | `validate_config`, `log_auth_config` |
| `wsgi.py` | Waitress `serve(app, ...)` |

---

## Appendix B — `DISEASE_INFO` coverage

Treatment text in API responses comes from `DISEASE_INFO` in `predict.py`.  
Classes without an explicit entry still return generic symptoms/treatment via `_build_info()`.

Sri Lanka entries include: **Banana**, **Rice**, **Coconut** (healthy, leaf rot, gray leaf spot), **Tea**, **Chili**, **Mango**, **Papaya**, plus all major PlantVillage crops in the deployed model.

---

*End of backend technical report.*
