# Plant Village AI — Full Project Documentation

## 1. Overview

**Plant Village AI** is a full-stack web application that lets farmers take or upload a photo of a plant leaf and receive an AI-powered disease diagnosis. It uses three complementary prediction methods (Groq Vision API, computer vision heuristics, and a PyTorch ResNet50 fallback) and a React frontend with client-side leaf detection via TensorFlow.js MobileNet.

**Tech stack:** Python (Flask) + React (Vite) + TensorFlow.js + OpenCV + PyTorch + Groq API

---

## 2. Project Structure

```
New folder (4)/
├── test_leaf_detection_frontend.md   # Manual test guide for leaf detection
├── PROJECT_DOCUMENTATION.md          # This file
│
├── plant-disease-backend/            # Python Flask backend
│   ├── app.py                        # Flask entry point + REST routes
│   ├── predict.py                    # Core prediction pipeline
│   ├── groq_predict.py               # Groq Vision API integration
│   ├── leaf_analysis.py              # OpenCV shape/color heuristics
│   ├── pytorch_model.py              # PyTorch ResNet50 fallback
│   ├── qa_engine.py                  # Clarification Q&A engine
│   ├── chatbot.py                    # Chatbot (knowledge base + Groq)
│   ├── history_store.py              # Local JSON history storage
│   ├── config.py                     # Legacy config (deprecated)
│   ├── wsgi.py                       # Waitress production server
│   ├── requirements.txt              # Python dependencies
│   ├── .env                          # API keys (Supabase + Groq)
│   ├── .env.example                  # Environment template
│   └── test_*.py                     # Various test files
│
├── web/                              # React frontend
│   ├── package.json                  # npm dependencies
│   ├── vite.config.js                # Vite dev/build config
│   ├── index.html                    # HTML entry point
│   └── src/
│       ├── main.jsx                  # React entry
│       ├── App.jsx                   # Router
│       ├── api/                      # API client functions
│       │   ├── client.js
│       │   ├── predict.js
│       │   ├── auth.js
│       │   ├── chat.js
│       │   └── history.js
│       ├── pages/                    # Page components
│       │   ├── SplashPage.jsx
│       │   ├── UploadPage.jsx
│       │   ├── ResultPage.jsx
│       │   ├── DashboardPage.jsx
│       │   ├── HistoryPage.jsx
│       │   ├── ProfilePage.jsx
│       │   ├── LoginPage.jsx
│       │   └── RegisterPage.jsx
│       ├── components/               # Reusable components
│       │   ├── CameraCapture.jsx
│       │   ├── ImageDropzone.jsx
│       │   ├── PredictionCard.jsx
│       │   ├── ConfidenceRing.jsx
│       │   ├── ExplanationBox.jsx
│       │   ├── ChatWidget.jsx
│       │   ├── ErrorBanner.jsx
│       │   ├── LoadingOverlay.jsx
│       │   ├── Navbar.jsx
│       │   ├── Layout.jsx
│       │   ├── LoginModal.jsx
│       │   └── IconInput.jsx
│       ├── utils/
│       │   ├── leafDetector.js       # TF.js MobileNet leaf detection
│       │   └── format.js             # Formatting helpers
│       ├── context/
│       │   └── ThemeContext.jsx       # Dark/light theme
│       └── styles/
│           ├── variables.css
│           └── global.css
```

---

## 3. Backend Architecture

### 3.1 `app.py` — Flask Server (651 lines)

**Entry point.** Runs on `0.0.0.0:5000`. Serves React build from `web/dist/` and provides REST API endpoints.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves React SPA (or API 404 if not built) |
| `/health` | GET | Backend health check |
| `/upload_predict` | POST | Main prediction — accepts image + farmer_reason |
| `/predict` | POST | Simple prediction (accepts image) |
| `/explain` | POST | Generate AI explanation for a prediction |
| `/clarify` | POST | Get clarification questions for low-confidence pairs |
| `/answer` | POST | Submit yes/no clarification answer |
| `/treatment/<disease>` | GET | Look up treatment info |
| `/chat` | POST | Chatbot |
| `/history` | GET/POST | Scan history |
| `/diseases` | GET | List all diseases with info |
| `/feedback` | POST | User feedback submission |
| `/login` | POST | Simple passwordless login |
| `/logout` | POST | Logout |
| `/register` | POST | Register account |
| `/debug_predict` | POST | Debug prediction |

**Key flow for `/upload_predict`:**
1. Read image + `farmer_reason` from multipart form
2. Build `farmer_context` dict with all farmer-provided info
3. Call `predict_disease(image_bytes, farmer_context)` from `predict.py`
4. If Groq vision model is unavailable → return 503
5. If disease is `"unidentified"` → return `{prediction: "unidentified", message: "...", confidence: 0}`
6. If confidence < 60 → generate clarification questions
7. Save prediction to local JSON history
8. Return diagnosis result with treatment info

### 3.2 `predict.py` — Prediction Pipeline (1342 lines)

**Core orchestration.** Contains the full pipeline that chains Groq → CV → PyTorch.

**`DISEASE_INFO` dict** — 70+ disease entries, each with:
- `display_name`
- `cause` (e.g., "Fungal pathogen Colletotrichum gloeosporioides")
- `symptoms`
- `treatment`
- `prevention`
- `fertilizer`
- `severity` (Low/Medium/High)
- `weather` (related weather conditions)
- `insects` (associated insects)

**Key functions:**

| Function | Purpose |
|---|---|
| `predict_disease(bytes, farmer_context)` | Main pipeline entry |
| `_find_disease_key(plant, disease_name)` | Fuzzy-matches Groq output to DISEASE_INFO keys |
| `_validate_with_leaf_analysis(bytes, groq_result, context)` | CV-based post-processing correction |
| `_bytes_to_cv2(bytes)` | Image bytes → OpenCV ndarray |
| `_map_to_apple_disease(metrics)` | CV metrics → apple disease name |
| `_predict_with_pytorch(bytes)` | PyTorch ResNet50 fallback |
| `_load_pytorch_model()` | Lazy-loads PyTorch model |
| `_build_info(disease, plant, confidence)` | Builds treatment info dict |

**Pipeline steps (in order):**
1. **Groq Vision API** — Send image + prompt with farmer context → get JSON
2. **CV validation** — `_validate_with_leaf_analysis()` corrects apple↔mango, banana↔coconut based on shape
3. **Shape verification** — `is_likely_a_leaf()` checks aspect ratio, solidity, green pixels
4. **Confidence threshold** — If < 30% or `plant_type == "unidentified"`, override to unidentified
5. **PyTorch fallback** — If confidence < 30 and not already unknown, try local PyTorch model
6. **DISEASE_INFO lookup** — Map disease name to structured treatment info

### 3.3 `groq_predict.py` — Groq Vision API (291 lines)

**Model:** `meta-llama/llama-4-scout-17b-16e-instruct`

**Functions:**

| Function | Purpose |
|---|---|
| `predict_with_groq(image_bytes, farmer_context)` | Primary prediction — sends image + structured prompt |
| `re_predict_with_groq(image_bytes, hint)` | Second-opinion call with verification hint |
| `generate_explanation(disease, plant, farmer_reason)` | Text-only explanation generation |

**Prompt structure:**
1. **Farmer ground truth** — If farmer provided info, it's injected as ground truth
2. **STEP 0 (Mandatory)** — Is this a plant leaf? If not, output `{plant_type: "unidentified", disease: "not a leaf", confidence: 0}`
3. **STEP 1** — Identify plant species from: mango, apple, banana, coconut, corn, grape, papaya, chili, potato, rice, tea, tomato
4. **STEP 2** — Identify disease from valid list per plant type
5. **Hard rules** — Coconut vs banana vs strawberry visual rules, apple vs mango edge rules
6. **Confidence calibration** — Guidelines for assigning confidence (0-100)
7. **Good vs Bad examples** — 10 correct examples + 10 wrong examples
8. **Output format** — JSON only, no markdown

**VALID_DISEASES** — 12 plant types with their valid disease names:

| Plant | Diseases |
|---|---|
| mango | Anthracnose, Bacterial_canker, Die_back, Powdery_mildew, Sooty_mould, healthy |
| banana | Sigatoka, Xanthomonas_wilt, healthy |
| coconut | Leaf_rot, Gray_leaf_spot, healthy |
| rice | Blast, Bacterial_blight, Brown_spot, Tungro, healthy |
| tea | Algal_leaf_spot, Anthracnose, Bird_eye_spot, Brown_blight, Gray_blight, Red_rust, White_spot, healthy |
| chili | Bacterial_spot, healthy |
| papaya | Anthracnose, Bacterial_spot, Insect_Hole, Leaf_curl, Ringspot, healthy |
| apple | Cedar_apple_rust, Apple_scab, Black_rot, healthy |
| tomato | Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Target_Spot, Tomato_Yellow_Leaf_Curl_Virus, Tomato_mosaic_virus, healthy |
| potato | Early_blight, Late_blight, healthy |
| corn | Cercospora_leaf_spot, Common_rust, Northern_Leaf_Blight, healthy |
| grape | Black_rot, Esca_(Black_Measles), Leaf_blight, healthy |

### 3.4 `leaf_analysis.py` — Computer Vision Heuristics (1605 lines)

**OpenCV-based shape and color analysis.** Runs as a post-processing layer to correct Groq mispredictions.

**Key functions:**

| Function | Purpose |
|---|---|
| `_leaf_mask(img)` | HSV green masking to extract leaf pixels |
| `_largest_contour(mask)` | Find largest contour (assumed leaf) |
| `_lobe_count(contour)` | Count leaf lobes (grape vs others) |
| `_edge_serration_score(mask, contour)` | Measure edge serration (apple = serrated, mango = smooth) |
| `_lesion_stats(mask, bgr)` | Analyze brown/yellow lesion area and roundness |
| `detect_crop_family(img)` | Guess crop type from shape metrics (21 crop prefixes) |
| `is_likely_a_leaf(metrics)` | Reject non-leaf based on aspect ratio, solidity, green pixels |
| `refine_probabilities(classes, proba, metrics, ...)` | Re-rank ML probabilities using CV features |
| `apply_sl_crop_correction(...)` | Unified Sri Lanka crop correction |
| `finalize_sl_prediction(...)` | Last-pass correction before output |

**Shape metrics computed:**
- `aspect` — width/height ratio of bounding rect
- `solidity` — contour area / convex hull area
- `lobes` — number of leaf lobes (via convexity defects)
- `serration` — edge serration score (0.0 = smooth, >0.25 = serrated)
- `compactness` — perimeter² / (4π × area)
- `area` — contour pixel area
- `has_green` — boolean, whether green pixels are present
- `elongation` — for lesion analysis
- `circularity` — for lesion analysis

**Rejection rules in `is_likely_a_leaf()`:**
- No green pixels → reject
- Aspect ratio > 5.0 (too wide) → reject
- Aspect ratio < 0.2 (too tall) → reject
- Solidity < 0.25 (too irregular) → reject
- Area ≤ 0 (no contour) → reject

**Crop family scoring — 21 prefixes:**
Mango, Apple, Banana, Coconut, Corn_(maize), Grape, Papaya, Chili, Potato, Rice, Tea, Tomato, Peach, Pear, Cherry, Strawberry, Sugarcane, Sunflower, Beans, Pepper,_bell, Soybean

### 3.5 `pytorch_model.py` — PyTorch Fallback (79 lines)

**Model:** ResNet50 with custom FC layer, trained on PlantVillage 38 classes + Sri Lanka crops.

**Weights file:** `pytorch_project/plant_disease_resnet50.pth`

**Function:** `predict_pytorch(image_bytes)` — loads image, runs inference, returns top-5 (class_name, confidence).

### 3.6 `qa_engine.py` — Clarification Engine (178 lines)

When confidence is < 60% for two competing diseases, the engine generates yes/no questions.

**8 question pairs:**
- Mango Anthracnose vs Bacterial Canker
- Mango Die Back vs Powdery Mildew
- Tomato Early Blight vs Late Blight
- Tomato Bacterial Spot vs Septoria Leaf Spot
- Corn Northern Leaf Blight vs Common Rust
- Rice Blast vs Bacterial Blight
- Papaya Ringspot vs Leaf Curl
- Grape Black Rot vs Leaf Blight

Each pair has 4 question characteristics (1 yes, 1 no). `get_clarification_questions()` returns the 4 questions; `process_answer()` selects the disease based on majority vote.

### 3.7 `chatbot.py` — Chatbot (160 lines)

Two-step response:
1. **Local lookup** — Match keywords in DISEASE_INFO for treatment/prevention
2. **Groq fallback** — If no local match, send query to Groq with system prompt

### 3.8 `history_store.py` — History (65 lines)

JSON file-based storage (`prediction_history.json`). Max 100 records per user.

---

## 4. Frontend Architecture

### 4.1 Page Flow

```
SplashPage (/) → DashboardPage (/dashboard) → UploadPage (/upload) → ResultPage (/result)
                       ↕                              ↑
                  HistoryPage (/history)               |
                       ↕                              |
                  ProfilePage (/profile)              |
                       ↕                              |
                  Login/Register                      |
```

### 4.2 Key Features

**Client-side leaf detection (`leafDetector.js`):**
- Uses TensorFlow.js MobileNet v2 (alpha 1.0)
- 80+ plant-related keywords
- Classifies image → checks top-3 predictions
- Rejects if: top-1 probability < 0.5, or top-1 not plant-related, or fewer than 2 of top-3 are plant-related (unless top-1 ≥ 0.7)
- Runs **before** any API call — non-leaf images never reach the backend

**Camera capture (`CameraCapture.jsx`):**
- Native `navigator.mediaDevices.getUserMedia` (no external dependency)
- Rear camera (`facingMode: "environment"`)
- Live preview → capture (canvas) → retake or use photo
- Error handling: denied, no camera, generic
- Captured photo treated as a File → same pipeline as upload

**Upload flow (`UploadPage.jsx`):**
1. Choose: drag/drop file or "📷 Take Photo"
2. Image loads → hidden `<img>` triggers `onLoad`
3. `isImageALeaf()` runs via MobileNet
4. If leaf OK → "What could be the reason?" section appears
5. User types/selects reason, clicks "Get Diagnosis"
6. `uploadAndPredict()` sends image + `farmer_reason` to `/upload_predict`
7. On success → navigate to `/result`
8. On error (unidentified, network, model unavailable) → show error banner

**Result display (`ResultPage.jsx`):**
- Confidence ring (color-coded: green ≥ 70, yellow 40-69, red < 40)
- Disease name + plant type
- AI explanation box (fetched from `/explain`)
- Clarification Q&A (yes/no buttons if confidence < 60)
- Disease info: cause, symptoms, treatment, prevention, fertilizer, severity
- Alternative predictions (top-3)
- "Scan Another" button

**Dashboard:**
- Stats: total scans, healthy count, diseased count
- Recent 5 diagnosis history
- "Scan a leaf now" CTA

**History:**
- Chronological list of past predictions
- Click to view details

**Chat widget:**
- Floating action button
- Knowledge base lookup + Groq fallback
- Markdown formatting (bold, italic)

**Theme:**
- Light/dark mode toggle
- Persisted in localStorage
- CSS custom properties for all colors

### 4.3 API Client (`api/client.js`)

- `getApiBase()` — resolves API URL:
  - Production: same origin
  - Vite dev: proxy to `http://127.0.0.1:5000`
  - Custom: `REACT_APP_API_URL` env var
- `apiRequest()` — fetch wrapper with:
  - JSON or FormData auto-detection
  - 30s timeout (AbortController)
  - Error handling

### 4.4 Routing (`App.jsx`)

```jsx
<Route path="/" element={<SplashPage />} />
<Route element={<Layout />}>
  <Route path="/dashboard" element={<DashboardPage />} />
  <Route path="/upload" element={<UploadPage />} />
  <Route path="/result" element={<ResultPage />} />
  <Route path="/history" element={<HistoryPage />} />
  <Route path="/profile" element={<ProfilePage />} />
</Route>
<Route path="/login" element={<LoginPage />} />
<Route path="/register" element={<RegisterPage />} />
```

### 4.5 Vite Config (`vite.config.js`)

- Dev server: port 5173
- Proxy rules:
  - `/api/...` → `http://127.0.0.1:5000` (strips `/api`)
  - `/upload_predict` → `http://127.0.0.1:5000`
  - `/debug_predict` → `http://127.0.0.1:5000`
  - `/chat` → `http://127.0.0.1:5000`

---

## 5. Data Flow

### 5.1 Prediction Flow (detailed)

```
User uploads/takes photo
        │
        ▼
[Frontend] MobileNet classifies image (leafDetector.js)
        │
        ├── Not a leaf → Show error: "Unidentified image..."
        │
        └── Is a leaf → Show reason input, enable "Get Diagnosis"
                │
                ▼
[Frontend] POST /upload_predict (FormData: image + farmer_reason)
        │
        ▼
[app.py] Read form → build farmer_context → call predict_disease()
        │
        ▼
[predict.py] predict_disease():
        │
        ├── 1. predict_with_groq(image, farmer_context)
        │       │
        │       ├── Groq Vision API → JSON response
        │       │
        │       └── Parse JSON → (plant_type, disease, confidence, symptoms, treatment)
        │
        ├── 2. _validate_with_leaf_analysis(image, groq_result, farmer_context)
        │       │
        │       ├── detect_crop_family() → shape metrics
        │       ├── Apply apple/mango corrections based on serration & aspect
        │       ├── Apply banana/coconut corrections based on shape
        │       └── Return corrected (plant, disease, confidence) or None
        │
        ├── 3. Shape verification via is_likely_a_leaf()
        │       │
        │       └── Reject if not leaf-shaped → override to unidentified
        │
        ├── 4. PyTorch fallback (if confidence < 30%)
        │       │
        │       └── Run local ResNet50 → take higher-confidence result
        │
        └── 5. Confidence threshold check
                │
                ├── If confidence < 30 or plant_type == "unidentified" → override
                │
                └── Return (disease, confidence, info, top_predictions, plant_type, metrics)
        │
        ▼
[app.py] → If disease == "unavailable" → 503
        → If disease == "unidentified" → {prediction: "unidentified", confidence: 0, message: "..."}
        → If confidence < 60 → generate clarification questions
        → Save to history
        → Return full diagnosis JSON
        │
        ▼
[Frontend] → If "unidentified" → show error message
        → Otherwise → navigate to /result page
        → Fetch explanation via /explain endpoint
        → Show diagnosis, treatment, explanation, clarifications
```

### 5.2 Farmer Reason Flow

```
UploadPage.jsx: farmerReason state (user types or selects)
        │
        ▼
predict.js: uploadAndPredict(file, { farmer_reason: farmerReason })
        │
        ▼
FormData: "image" + "farmer_reason" appended
        │
        ▼
app.py: farmer_reason = request.form.get("farmer_reason")
        farmer_context = {"farmer_reason": farmer_reason, ...}
        │
        ▼
predict.py: predict_disease(image_bytes, farmer_context)
        │
        ▼
groq_predict.py: predict_with_groq(image_bytes, farmer_context)
        │
        ├── Reads farmer_context keys (plant_type, symptoms, weather, leaf_age, insect_damage, part_affected, farmer_reason)
        ├── Injects into prompt as "Farmer-provided information"
        └── If farmer_reason = "Too much rain":
            "The farmer suspects the cause is: Too much rain"
        │
        ▼
generate_explanation(disease, plant_type, farmer_reason)
        └── "The farmer says the possible reason is: 'Too much rain'..."
```

---

## 6. Configuration & Setup

### 6.1 Prerequisites

- Python 3.10+
- Node.js 18+
- Groq API key (free tier: 1000 requests/day)

### 6.2 Backend Setup

```bash
cd plant-disease-backend

# Create virtual environment
py -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
copy .env.example .env
# Edit .env: add GROQ_API_KEY=gsk_your_key_here

# Run server
python app.py
# → http://localhost:5000
```

### 6.3 Frontend Setup

```bash
cd web

# Install dependencies
npm install

# Start dev server
npm run dev
# → http://localhost:5173

# Build for production
npm run build
# → Output in web/dist/
```

### 6.4 Environment Variables

**Backend (`.env`):**

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq Vision API key |
| `SUPABASE_URL` | No | Supabase project URL (optional) |
| `SUPABASE_KEY` | No | Supabase anon key (optional) |

**Frontend (`.env.example`):**

| Variable | Required | Purpose |
|---|---|---|
| `VITE_API_URL` | No | Custom API URL (defaults to proxy) |

### 6.5 Requirements

**`requirements.txt`:**
```
Flask==3.1.3
Flask-CORS==6.0.2
numpy==2.4.6
scikit-learn==1.8.0
Pillow==12.2.0
python-dotenv==1.2.2
requests==2.34.2
groq==1.4.0
```

**Frontend `package.json` dependencies:**
```json
{
  "@tensorflow-models/mobilenet": "^2.1.1",
  "@tensorflow/tfjs": "^4.22.0",
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "react-router-dom": "^6.28.0"
}
```

---

## 7. Supported Crops & Diseases

| Plant | Supported Diseases |
|---|---|
| Mango | Anthracnose, Bacterial_canker, Die_back, Powdery_mildew, Sooty_mould, healthy |
| Banana | Sigatoka, Xanthomonas_wilt, healthy |
| Coconut | Leaf_rot, Gray_leaf_spot, healthy |
| Apple | Cedar_apple_rust, Apple_scab, Black_rot, healthy |
| Tomato | Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Target_Spot, Tomato_Yellow_Leaf_Curl_Virus, Tomato_mosaic_virus, healthy |
| Potato | Early_blight, Late_blight, healthy |
| Corn | Cercospora_leaf_spot, Common_rust, Northern_Leaf_Blight, healthy |
| Grape | Black_rot, Esca_(Black_Measles), Leaf_blight, healthy |
| Rice | Blast, Bacterial_blight, Brown_spot, Tungro, healthy |
| Tea | Algal_leaf_spot, Anthracnose, Bird_eye_spot, Brown_blight, Gray_blight, Red_rust, White_spot, healthy |
| Chili | Bacterial_spot, healthy |
| Papaya | Anthracnose, Bacterial_spot, Insect_Hole, Leaf_curl, Ringspot, healthy |

Total: **70+ disease entries** in `DISEASE_INFO` (from `predict.py`), each with cause, symptoms, treatment, prevention, fertilizer, severity, weather, and insects.

---

## 8. Known Architecture Decisions

1. **Groq as primary predictor** — Free, fast, multimodal (vision + text). Rate limit: ~1000 requests/day on free tier.
2. **CV heuristics layer** — Prevents hallucination on apple↔mango, banana↔coconut by checking actual leaf shape. This runs after Groq and overrides if shape contradicts the prediction.
3. **PyTorch fallback** — Local ResNet50 handles cases when Groq is unavailable or produces low confidence. Weaker than Groq but always available.
4. **Client-side leaf detection** — MobileNet via TensorFlow.js prevents unnecessary API calls. Rejects non-leaf images instantly before any network request.
5. **Farmer context injection** — Optional questionnaire data is treated as "ground truth" in the Groq prompt, not just context. If farmer says "mango", the model MUST output mango.
6. **Clarification Q&A** — When two diseases have similar confidence, the app asks binary yes/no questions to narrow down. Each of 8 disease pairs has 4 distinguishing characteristics.
7. **Local JSON storage** — `prediction_history.json` for scan history. No database setup needed. Optional Supabase integration available.
8. **Single-server deployment** — Flask serves both API and built React frontend from `web/dist/`. No separate web server needed.

---

## 9. Known Issues & Limitations

1. **Groq rate limits** — Free tier limited. When exhausted, app falls back to PyTorch which has lower accuracy.
2. **Coconut accuracy** — Model struggles with coconut vs banana discrimination on young/atypical leaves. CV heuristics mitigate but don't eliminate this.
3. **Sri Lanka crop gap** — 31 SL classes have low training data. Real-world accuracy estimated at 15-40% for these.
4. **MobileNet client-side** — The 2MB+ bundle impacts initial page load. Model loading is lazy (on first image upload).
5. **No offline mode** — Requires internet for Groq API. PyTorch fallback is local but less accurate.
6. **Basic auth** — Local JSON-based auth, not production-secure. No password hashing, no email verification.
7. **No image validation** — Large images are accepted without resizing, increasing API latency.

---

## 10. Testing

### 10.1 Test Files

| File | Purpose |
|---|---|
| `test_app.py` | 22 API unit tests (health, auth, predict, history, QA) |
| `test_leaf_analysis.py` | CV heuristic unit tests (crop discrimination, lesion detection) |
| `test_groq.py` | Groq prediction with a real image |
| `test_non_leaf_backend.py` | Non-leaf rejection with synthetic images |
| `test_banana_fix.py` | Banana-specific correction tests |
| `test_coconut_fix.py` | Coconut-specific correction tests |
| `test_mango_fix.py` | Mango-specific correction tests |
| `test_apple_fix.py` | Apple-specific correction tests |
| `test_backend.ps1` | PowerShell test runner |
| `run_tests.ps1` | Unittest runner |

### 10.2 Running Tests

```powershell
cd plant-disease-backend
.venv\Scripts\activate

# Run all tests
python -m unittest discover -p "test_*.py"

# Run specific test
python -m unittest test_leaf_analysis

# Run non-leaf rejection test (requires server running)
python test_non_leaf_backend.py
```

### 10.3 Manual Frontend Tests

See `test_leaf_detection_frontend.md` for 8 manual test cases covering:
- Non-leaf image rejection (person, car, building, food, solid color)
- Leaf image acceptance
- Blurry / low-quality leaf
- Confidence threshold
- Camera capture (leaf, non-leaf, retake)
- Camera fallback (no camera / denied)

---

## 11. Deployment

### 11.1 Production Build

```bash
# 1. Build frontend
cd web
npm run build

# 2. Start backend (serves both API and frontend)
cd ../plant-disease-backend
python app.py
# → http://localhost:5000 (full app)

# For production with waitress:
python wsgi.py
```

### 11.2 Firewall

```powershell
# Allow port 5000 through Windows Firewall
.\plant-disease-backend\allow_firewall.ps1
```

### 11.3 Environment

1. Set `GROQ_API_KEY` in `.env`
2. Optionally set `SUPABASE_URL` and `SUPABASE_KEY` for cloud history storage
3. Frontend env vars are optional (Vite proxy handles dev)

---

## 12. Key Files Summary

| File | Lines | Purpose |
|---|---|---|
| `leaf_analysis.py` | 1605 | CV heuristics — largest file, most complex logic |
| `predict.py` | 1342 | Prediction pipeline + 70+ disease entries |
| `app.py` | 651 | Flask server, all REST routes |
| `groq_predict.py` | 291 | Groq Vision API integration + prompts |
| `UploadPage.jsx` | 198 | Image upload, camera, leaf detection, diagnosis trigger |
| `ResultPage.jsx` | 213 | Diagnosis display, explanations, clarifications |
| `CameraCapture.jsx` | 121 | Live camera capture component |
| `leafDetector.js` | 75 | TF.js MobileNet leaf detection |
| `qa_engine.py` | 178 | Clarification Q&A for 8 disease pairs |
| `chatbot.py` | 160 | Chatbot with knowledge base + Groq fallback |
| `pytorch_model.py` | 79 | PyTorch ResNet50 fallback model loader |

---

## 13. How to Contribute / Extend

### Add a new disease

1. Add to `VALID_DISEASES` in `groq_predict.py`
2. Add entry to `DISEASE_INFO` in `predict.py` with cause, symptoms, treatment, prevention, fertilizer, severity
3. If it competes with existing diseases, add clarification questions in `qa_engine.py`
4. If it requires CV-based discrimination, add scoring rules in `leaf_analysis.py` `detect_crop_family()`

### Add a new crop

1. Add to `VALID_DISEASES` in `groq_predict.py`
2. Add to list of valid plant types in the Groq prompt
3. Add CV scoring rules in `leaf_analysis.py` `detect_crop_family()`
4. Add diseases to `DISEASE_INFO` in `predict.py`
5. Add to CROP_PREFIXES if needed for CV detection

### Retrain PyTorch model

See `TRAIN_SRI_LANKA.md` for full step-by-step retraining guide.

---

## 14. License & Credits

This is a research/agricultural tech prototype built for farmers in Sri Lanka and South Asia. The backend uses Groq's free API tier for multimodal inference, OpenCV for computer vision heuristics, and a PyTorch ResNet50 model trained on PlantVillage + custom Sri Lanka crop datasets.
