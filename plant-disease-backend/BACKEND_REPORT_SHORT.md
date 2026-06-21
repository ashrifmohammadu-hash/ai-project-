# Plant Disease Detection — Backend Report (Executive Summary)

**Project:** `plant-disease-backend` | **Date:** May 2026 | **Scope:** Backend only

---

## 1. Overview

The backend is a **Flask REST API** that classifies plant leaf diseases from uploaded photos and returns treatment advice. It serves the Flutter mobile app over HTTP on **port 5000**.

| Item | Value |
|------|-------|
| Language | Python 3 |
| Framework | Flask 3.1.3 + Flask-CORS |
| Production server | Waitress (`wsgi.py`) |
| ML | scikit-learn + OpenCV + joblib |
| Model classes | **69** (38 PlantVillage + **31 Sri Lanka**) |
| Model files | `models/model.pkl` (~1.1 GB), `scaler.pkl`, `label_encoder.pkl` |
| Image features | 103 (128×128 HSV histograms, edges, RGB stats) |

**Sri Lanka crops in model:** Banana, Rice, Coconut, Tea, Chili, Mango, Papaya.

---

## 2. Main features

1. **User auth** — Register, login, JWT access (60 min) + refresh (30 days), logout.  
2. **Predict** — Upload leaf image → disease name, confidence %, top-5 alternatives, treatment JSON.  
3. **Vision assist** — `leaf_analysis.py` re-ranks results (fixes coconut vs grape confusion).  
4. **Clarification Q&A** — If confidence &lt; 60%, YES/NO symptom questions refine the diagnosis.  
5. **History** — Scans saved per user in `prediction_history.json` (optional Supabase backup).

---

## 3. API endpoints (summary)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | No | Server alive |
| POST | `/register`, `/login`, `/refresh` | No | Account + tokens |
| GET | `/me`, POST `/logout` | JWT | Profile / logout |
| POST | `/predict` | JWT | Image → disease + treatment |
| GET/POST | `/history` | JWT | Scan history |
| POST | `/clarify`, `/answer` | Optional | Q&A refinement |
| GET | `/treatment/<disease>`, `/diseases` | No | Disease lookup |

**Predict request:** `multipart/form-data` with field `image`.  
**Auth header:** `Authorization: Bearer <access_token>`.

---

## 4. Machine learning pipeline

```
Image → OpenCV decode → feature extract (103) → scaler → predict_proba
      → leaf shape/lesion re-rank → Sri Lanka crop preference → top disease
      → DISEASE_INFO (symptoms, treatment, fertilizer)
```

**Confidence:** ≥ 60% = confident; &lt; 60% triggers clarification questions.

**Training (separate folder):** `Merge-Project/` → `train.py` → copy `.pkl` files to `models/`.

---

## 5. Project files (core)

| File | Role |
|------|------|
| `app.py` | All HTTP routes |
| `auth.py` | Users, passwords, history |
| `predict.py` | ML + disease knowledge |
| `leaf_analysis.py` | Crop detection & re-ranking |
| `qa_engine.py` | Clarification questions |
| `jwt_auth.py` / `jwt_middleware.py` | Token security |
| `history_store.py` | Local JSON history |
| `config.py` | Environment & model paths |

---

## 6. Configuration

Copy `.env.example` → `.env`:

```env
JWT_SECRET=<long-random-secret>
JWT_ACCESS_MINUTES=60
JWT_REFRESH_DAYS=30
SUPABASE_AUTH_ENABLED=false
SUPABASE_URL=          # optional
SUPABASE_KEY=          # optional
```

---

## 7. How to run & test

```powershell
cd "d:\ai data\Final\plant-disease-backend"
pip install -r requirements.txt
python app.py                    # Development — http://0.0.0.0:5000
python wsgi.py                   # Production (Waitress)
python -m unittest test_app test_leaf_analysis -v
python smoke_check.py
```

**Phone testing:** Use PC Wi‑Fi IPv4 from `ipconfig` (not `localhost`).

---

## 8. Sri Lanka classes (31)

| Crop | Example classes |
|------|-----------------|
| Banana | Sigatoka, Xanthomonas wilt, healthy |
| Rice | Blast, Brown spot, Bacterial blight, Tungro |
| Coconut | Leaf rot, Gray leaf spot, healthy |
| Tea | 7 diseases + healthy |
| Chili | Bacterial spot, healthy |
| Mango | 5 diseases + healthy |
| Papaya | 4 diseases + healthy |

---

## 9. Testing status

| Check | Result |
|-------|--------|
| `test_app.py` | 22 tests — auth, predict, history, QA |
| `test_leaf_analysis.py` | 4 tests — coconut vs grape |
| `smoke_check.py` | Models, predict, `/health` |
| **Total** | **26 tests passing** |

---

## 10. Limitations

- Coconut/Chili partly trained on **proxy images** (papaya/pepper) when external datasets were unavailable.  
- No `Rice_healthy` class in current model.  
- Model requires **~1.1 GB RAM** and **64-bit Python**.  
- Accuracy depends on clear single-leaf photos and correct PC IP for mobile connection.

---

## 11. Architecture (simplified)

```
Flutter App  ──HTTP/JWT──►  Flask (app.py)
                                ├─ predict.py + leaf_analysis.py
                                ├─ qa_engine.py
                                └─ auth.py / history_store.py
                                    ├─ local_users.json
                                    ├─ prediction_history.json
                                    └─ models/*.pkl
```

---

**Full documentation:** `BACKEND_REPORT.md` (complete API examples, all 69 class names, appendices).  
**Cover page:** `BACKEND_REPORT_COVER.md`

---

*End of executive summary (≈2 pages when printed).*
