# Plant Disease Detection System — Backend Report

## 1. Project Overview

**Goal:** Build a REST API that accepts a leaf image upload and returns a predicted plant disease along with treatment recommendations, symptoms, and confidence metrics.

**Prediction pipeline (current):**

```
User uploads image → Flask endpoint → Groq Vision API (primary)
                                         ↓
                              JSON parsed + mapped to DISEASE_INFO
                                         ↓
                              Post-processing via leaf_analysis CV heuristics
                                         ↓
                              Unified response returned
```

**Fallback chain (partially implemented):**
1. **Groq Vision API** — primary predictor (active)
2. **PyTorch ResNet50** — fallback when Groq returns unknown or confidence < 30 (lazy-loaded, active)
3. **Random Forest** — original ML model, now deprecated and unused

**Current status:** Backend is functional. The Groq integration responds to image uploads. Apple leaf images were historically misclassified as "Mango_Anthracnose" at ~85% confidence. The prompt has been improved and a CV-based post-processing override added to catch this. A PyTorch fallback is registered but rarely invoked. Frontend connection issues via Vite proxy persist.

---

## 2. Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| Web framework | Flask 3.1.3 with Flask-CORS 6.0.2 |
| ML / DL | PyTorch 2.12.0, torchvision 0.27.0, scikit-learn 1.8.0, joblib 1.5.3 |
| External API | Groq SDK 1.4.0 (`meta-llama/llama-4-scout-17b-16e-instruct`) |
| Image processing | OpenCV 4.13.0, Pillow 12.2.0 |
| Data / numerics | NumPy 2.4.6, SciPy 1.17.1 |
| Environment | virtualenv, python-dotenv 1.2.2 |
| HTTP / auth | requests 2.34.2, werkzeug 3.1.8 |
| Production serving | Waitress (optional, via `wsgi.py`) |
| Testing | PowerShell 5.1 (`test_backend.ps1`), Python smoke checks |

---

## 3. Key Packages & Versions

From `requirements.txt` and installed packages in `backend_env`:

| Package | Version | Purpose |
|---|---|---|
| `Flask` | 3.1.3 | REST API framework |
| `Flask-CORS` | 6.0.2 | Cross-origin requests from React frontend |
| `numpy` | 2.4.6 | Array operations, image data |
| `scikit-learn` | 1.8.0 | Random Forest classifier (legacy) |
| `Pillow` | 12.2.0 | Image loading and preprocessing |
| `python-dotenv` | 1.2.2 | Load `.env` with API keys |
| `requests` | 2.34.2 | Supabase auth calls |
| `groq` | 1.4.0 | Groq Vision API client |
| `torch` | 2.12.0 | ResNet50 inference (fallback) |
| `torchvision` | 0.27.0 | Model zoo + transforms |
| `opencv-python` | 4.13.0 | Leaf segmentation, contour analysis |
| `opencv-python-headless` | 4.13.0 | Headless OpenCV for server |
| `joblib` | 1.5.3 | Random Forest model serialization |
| `werkzeug` | 3.1.8 | WSGI utilities, password hashing |

---

## 4. Project Folder Structure (Backend)

```
plant-disease-backend/
├── app.py                          # Flask entry point, routes, CORS
├── predict.py                      # Central predict_disease() + DISEASE_INFO (70 entries)
├── groq_predict.py                 # Groq Vision API call + prompt template
├── leaf_analysis.py                # CV heuristics: crop detection, lesion stats
├── qa_engine.py                    # Clarification question engine (unused in Groq flow)
├── history_store.py                # Local JSON prediction history
├── pytorch_model.py                # Standalone PyTorch ResNet50 loader (WIP)
├── config.py                       # Legacy Random Forest model paths
├── wsgi.py                         # Waitress production server entry
├── requirements.txt                # Pinned dependencies
├── test_backend.ps1                # PowerShell integration test script
├── .env                            # GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY
│
├── scripts/                        # Dataset evaluation & mislabel analysis
│   ├── compute_dataset_counts.py
│   ├── collect_mislabel_samples.py
│   ├── find_mango_to_banana.py
│   ├── generate_misprediction_report.py
│   ├── apply_suggested_corrections.py
│   └── ...
│
├── models/                         # Legacy Random Forest .pkl files
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── uploads/                        # User-uploaded images (created at runtime)
├── eval_output/                    # Dataset evaluation logs
├── prediction_history.json         # Local scan history (created at runtime)
└── feedback_log.json               # User feedback entries (created at runtime)
```

---

## 5. Purpose of Each Important File

### `app.py` — Flask application entry point
- Registers all REST routes: `/health`, `/login`, `/register`, `/logout`, `/upload_predict`, `/predict`, `/history`, `/clarify`, `/answer`, `/treatment`, `/diseases`, `/feedback`, `/debug_predict`
- Handles image upload from `multipart/form-data` field `"image"`
- Saves uploaded file to `uploads/` with timestamped filename
- Optionally adds image to dataset and runs dataset evaluation in background thread
- Serves the built React frontend from `web/dist/` for production
- Includes `_warmup_models()` background thread to preload modules
- Runs on `http://0.0.0.0:5000` (or via Waitress with `python wsgi.py`)

### `predict.py` — Core prediction logic
- Imports `predict_with_groq` from `groq_predict.py`
- Contains **`DISEASE_INFO`** — a dictionary of 70+ entries mapping disease keys to treatment, cause, symptoms, prevention, severity, and fertilizer data
- `_find_disease_key(plant_type, disease_name)` — fuzzy-matches Groq's free-text output to canonical `DISEASE_INFO` keys
- `_validate_with_leaf_analysis(image_bytes, groq_result)` — post-processing override using OpenCV heuristics:
  - Checks if leaf shape matches the predicted plant type
  - Overrides "mango" → "apple" when aspect ratio is round/oval (apple-like) and not elongated (mango-like)
- `_predict_with_pytorch(image_bytes)` — lazy-loaded PyTorch ResNet50 fallback
- `_build_info(disease, plant_type, confidence)` — builds `display_info` dict from `DISEASE_INFO`
- `predict_disease(image_bytes)` — orchestrates the full pipeline: Groq → post-processing → PyTorch fallback → disease info lookup

### `groq_predict.py` — Groq Vision API integration
- Loads `GROQ_API_KEY` from `.env`
- Uses `meta-llama/llama-4-scout-17b-16e-instruct` model (hardcoded, not dynamically listed)
- Encodes image as base64 and sends with a structured prompt asking for `plant_type`, `disease`, `confidence`, `symptoms`, `treatment`
- Prompt includes:
  - Valid disease names per plant (12 plant types, ~40 valid diseases)
  - Apple vs mango visual distinction rules (serrated edges = apple, lanceolate smooth = mango)
  - Negative examples of common mistakes
- Strips markdown fences from response before JSON parsing
- On error, returns `{"plant_type": "unknown", "disease": "unknown", "confidence": 0}`

### `leaf_analysis.py` — Computer vision heuristics
- Extracts metrics from leaf images using OpenCV: aspect ratio, solidity, lobe count, lesion ratio, round/elongated spot detection
- `detect_crop_family(img)` — scores each crop prefix (Apple, Mango, Banana, etc.) based on shape and returns the best match with confidence 0–1
- `is_mango_leaf(metrics)`, `is_banana_leaf(metrics)`, `is_coconut_palm_leaf(metrics)`, etc. — shape-based classifiers
- `has_visible_leaf_disease(metrics)` — detects brown/yellow lesions to determine if the leaf appears diseased
- `refine_probabilities()` — originally designed to re-rank Random Forest probabilities (unused in Groq pipeline)
- `PREDICT_RULES_VERSION = "sl-unified-v14-mango-not-banana"` — version string for debugging

### `qa_engine.py` — Clarification question engine
- Provides structured yes/no questions to distinguish similar disease pairs (e.g., Tomato Early_blight vs Late_blight)
- `get_clarification_questions(disease1, disease2)` — returns questions for a disease pair
- `process_answer(disease1, disease2, question_index, answer)` — narrows down based on user response
- Not currently integrated with the Groq prediction flow (questions appear in `/predict` response when confidence < 60)

### `history_store.py` — Local prediction history
- Reads/writes `prediction_history.json` in the backend root
- `append_prediction(user_id, record)` — inserts a scan record (newest first), capped at 100 entries per user
- `list_predictions(user_id, limit=50)` — returns recent history
- Used by `app.py` `/history` route

### `pytorch_model.py` — Standalone PyTorch loader
- Loads the trained ResNet50 from `pytorch_project/plant_disease_resnet50.pth`
- Requires the dataset directory (`resized_merged`) to extract class names in the correct order
- Contains `_load_model()` and `predict_image_bytes(image_bytes)` functions
- Not actively imported by `predict.py` (a separate inline loader exists there); kept as independent research code

### `config.py` — Legacy configuration
- Holds paths to Random Forest `model.pkl`, `scaler.pkl`, and `label_encoder.pkl`
- `validate_config()` raises `RuntimeError` if model files are missing
- `log_startup_config()` prints auth-disabled message
- All Random Forest dependencies are now deprecated

### `wsgi.py` — Production server
- Runs Flask app via Waitress with 10 threads, connection limit 256
- Alternative to `python app.py` for production deployment

### `test_backend.ps1` — PowerShell test harness
- Accepts `-ImagePath` (path to leaf image) and optional `-ServerUrl` (default `http://127.0.0.1:5000`)
- Uses `System.Net.Http.HttpClient` with `MultipartFormDataContent` to upload files (PS 5.1 does not support `-Form`)
- Runs health check, then tests both `/upload_predict` and `/predict` endpoints
- Pretty-prints JSON response via `ConvertFrom-Json | ConvertTo-Json -Depth 5`
- Offers built-in dataset test image selection when no path is provided

---

## 6. Technology Decisions & Rationale

| Decision | Rationale |
|---|---|
| **Groq over Google Gemini** | Gemini free tier had severe per-minute quota limits, returning 429 errors under moderate load. Groq offers 1000 requests/day free with sub-second inference on LLaMA-4-Scout-17B. |
| **LLaMA-4-Scout over a dedicated vision model** | The model is multimodal (text + image), eliminating the need for a separate image encoder. Single API call per prediction. |
| **CV post-processing (leaf_analysis)** | Groq can hallucinate plant types (especially mango for non-mango leaves). The CV heuristics (`detect_crop_family`) provide a shape-based independent check that can override implausible predictions. |
| **PyTorch ResNet50 kept as fallback** | Originally trained on the merged PlantVillage + Sri Lanka dataset. Kept as a fallback because it runs locally with no API costs, even though its domain gap on real-world photos is a concern. |
| **Random Forest deprecated** | The handcrafted feature pipeline (HSV histograms, edge detection) did not generalize to real-world mobile photos. Replaced by Groq and PyTorch. |
| **Flask + React (Vite)** | Well-understood stack for rapid prototyping. CORS enabled to allow separate frontend/backend development. Vite proxy forwards `/api/*` → Flask. |
| **PowerShell test script** | Windows PowerShell 5.1 aliases `curl` to `Invoke-WebRequest`, which does not accept standard curl syntax. A dedicated `.ps1` script using `System.Net.Http.HttpClient` provides reliable multipart uploads. |
| **Local JSON for history** | Avoids database setup during prototyping. `prediction_history.json` stores up to 100 scans per user. Optional Supabase integration is stubbed. |

---

## 7. Current Known Issues

1. **Apple → mango misclassification (historical):** Groq occasionally classifies apple leaves as "Mango_Anthracnose" at high confidence (~85%). The improved prompt now includes explicit apple vs mango serrated-edge rules, and the post-processing override catches some cases, but not all.

2. **Frontend connection instability:** The Vite dev server on `localhost:5173` proxies `/api/*` to Flask on `localhost:5000`. Configuration mismatches (e.g., missing `/api` prefix stripping, or CORS preflight issues) cause intermittent failures.

3. **Groq hallucination:** The LLaMA-4-Scout model sometimes returns plant types or disease names not in the `VALID_DISEASES` list. The `_find_disease_key` fuzzy matcher mitigates this, but some mismatches still reach the user.

4. **No feedback loop:** When the prediction is wrong, there is no mechanism for the user to correct it. The `/feedback` endpoint exists but is not wired into the frontend.

5. **Apple disease differentiation:** Among apple diseases (scab, black rot, cedar-apple rust), Groq tends to default to "Apple_scab" even when the image shows a different apple disease. This is a model-level limitation.

6. **PyTorch model not actively tested:** The ResNet50 fallback is loaded lazily but has not been stress-tested with the real production flow. Dataset class ordering could break if the dataset directory changes.

---

## 8. Next Steps (Recommended)

1. **Refine Groq prompt further:** Add more negative examples specific to the apple→mango case. Include explicit "do NOT say mango unless you see lanceolate shape with smooth edges".

2. **Strengthen post-processing in `predict.py`:** Use leaf shape metrics from `leaf_analysis` more aggressively. If aspect ratio is < 1.6 and Groq says mango, force a re-evaluation. Currently the override caps confidence at 55; it could also retry with a secondary prompt.

3. **Fix frontend proxy:** Ensure Vite's `vite.config.js` strips the `/api` prefix correctly and that the Flask backend's CORS policy allows credentials.

4. **Wire feedback endpoint into the UI:** Add a "Was this correct?" widget on the result page that calls `POST /feedback`. Collecting corrections will enable prompt refinement and dataset expansion.

5. **Evaluate PyTorch as primary:** If Groq costs become prohibitive or latency is too high, benchmark the PyTorch ResNet50 against a held-out set of real-world photos. Fine-tune on phone photos if the domain gap is manageable.

6. **Add confidence calibration:** Currently the Groq confidence score is an opaque model output. A calibration step (e.g., logistic regression on a validation set) could produce more reliable probability estimates.

7. **Expand DISEASE_INFO:** Some Groq responses include diseases not in the 70-entry dictionary. Add entries for common missing cases (e.g., insect damage, nutrient deficiency) to reduce the number of "unknown" fallbacks.
