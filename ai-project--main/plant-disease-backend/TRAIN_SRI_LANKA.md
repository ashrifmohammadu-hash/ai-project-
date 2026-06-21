# Train the AI for Sri Lanka Leaf Diseases — Step-by-Step Guide

**Goal:** Replace the current US/Europe-focused **PlantVillage** model with classes that matter in **Sri Lanka** (banana, rice, coconut, tea, chili, mango, etc.).

**Your project already has a training pipeline:** `d:\ai data\Final\Merge-Project\`

---

## 1. What you have now (do not skip)

| Item | Location |
|------|----------|
| Training code | `Merge-Project/train.py` |
| Feature extraction | `Merge-Project/features.py` (must not change after training) |
| Live API model | `plant-disease-backend/models/*.pkl` |
| Flutter app | Uses that API |

The app **cannot** learn Sri Lanka crops until you **collect images**, **train**, and **copy new `.pkl` files** into `plant-disease-backend/models/`.

---

## 2. Realistic plan (3 phases)

### Phase A — Start with priority crops (4–8 weeks)

Focus on crops farmers ask about most:

| Crop | Example folder names | Min photos per class |
|------|----------------------|----------------------|
| **Banana** | `Banana_Sigatoka`, `Banana_healthy` | 200+ |
| **Rice** | `Rice_Blast`, `Rice_Bacterial_blight`, `Rice_Brown_spot`, `Rice_healthy` | 200+ |
| **Coconut** | `Coconut_Leaf_rot`, `Coconut_healthy` | 150+ |
| **Tea** | `Tea_Blister_blight`, `Tea_Red_rust`, `Tea_healthy` | 150+ |
| **Chili** | `Chili_Anthracnose`, `Chili_Leaf_curl`, `Chili_healthy` | 150+ |
| **Tomato** | (keep from PlantVillage or reshoot locally) | 100+ |
| **Mango** | `Mango_Anthracnose`, `Mango_Powdery_mildew`, `Mango_healthy` | 150+ |
| **Papaya** | `Papaya_Ringspot`, `Papaya_Black_spot`, `Papaya_healthy` | 150+ |

**Rule:** One folder = one disease (or `healthy`). Folder name = exact label the API will return.

You do **not** need every leaf disease in Sri Lanka on day one. Start with **10–20 classes**, then add more folders and retrain.

### Phase B — Collect Sri Lanka images

**Sources:**

1. **Your own photos** — phone camera, same Wi‑Fi field visits (best for local conditions).
2. **Department of Agriculture / universities** — with permission.
3. **Kaggle / research datasets** — search: rice leaf disease Sri Lanka, banana sigatoka, tea blister blight.
4. **PlantVillage** — only for crops that match SL (tomato, pepper, some fruits); **not** for banana/rice/coconut as primary SL data.

**Photo rules:**

- Clear **single leaf** (or few leaves), fill the frame.
- **Daylight**, avoid heavy blur.
- Include **healthy** leaves for each crop.
- **Balance classes** — similar number of images per folder.
- Formats: `.jpg`, `.png`

**Legal:** Use images you own or have license to use for research/project.

### Phase C — Train → deploy to app

Follow **Section 4** below, then copy models to backend and restart `python app.py`.

---

## 3. Dataset folder structure (required)

Put all images under one root folder, e.g.:

```
d:\ai data\Final\Merge-Project\dataset_sri_lanka\
├── Banana_Sigatoka\
│   ├── img001.jpg
│   └── ...
├── Banana_healthy\
├── Rice_Blast\
├── Rice_healthy\
├── Tea_Blister_blight\
└── ...
```

**Naming rules:**

- Use **English**, **underscores**, no spaces: `Rice_Bacterial_blight`
- Be consistent: `Banana_Sigatoka` not `banana-sigatoka` in one folder and `Banana_sigatoka` in another.
- Optional prefix crop name helps the app: `Crop_Disease`

---

## 4. Train with your project (commands)

### 4.1 Install Python packages (once)

```powershell
cd "d:\ai data\Final\Merge-Project"
pip install numpy opencv-python scikit-learn joblib matplotlib seaborn tqdm
```

### 4.2 Prepare images (resize to 128×128)

If your photos are large, run:

```powershell
cd "d:\ai data\Final\Merge-Project"
python prepare_sri_lanka_dataset.py --input "D:\path\to\raw_photos" --output dataset_sri_lanka
```

(Or manually resize with any tool; training resizes anyway.)

### 4.3 Configure training

Edit `config.py`:

```python
DATASET_PATH = "dataset_sri_lanka"
AUTO_DISCOVER_CLASSES = True   # use every subfolder as a class
```

When `AUTO_DISCOVER_CLASSES = True`, you **do not** need to list every class in `CLASSES` by hand.

### 4.4 Run training

```powershell
cd "d:\ai data\Final\Merge-Project"
python train.py
```

**Outputs:**

- `output/model.pkl`
- `output/scaler.pkl`
- `output/label_encoder.pkl`
- `output/results.txt` — accuracy per class
- `output/confusion_matrix.png`

Check validation accuracy. If one class is always wrong, add more photos for that folder.

### 4.5 Deploy to backend + app

```powershell
copy "d:\ai data\Final\Merge-Project\output\model.pkl" "d:\ai data\Final\plant-disease-backend\models\"
copy "d:\ai data\Final\Merge-Project\output\scaler.pkl" "d:\ai data\Final\plant-disease-backend\models\"
copy "d:\ai data\Final\Merge-Project\output\label_encoder.pkl" "d:\ai data\Final\plant-disease-backend\models\"
```

Then add **treatment text** for each new class in `plant-disease-backend/predict.py` → `DISEASE_INFO` (symptoms, treatment in Sinhala/English as you prefer).

Restart backend:

```powershell
cd "d:\ai data\Final\plant-disease-backend"
python app.py
```

Update `leaf_analysis.py` crop list if you add new crops (e.g. `Banana`, `Rice` in `CROP_PREFIXES`).

Rebuild Flutter app.

---

## 5. Optional: better accuracy with deep learning (CNN)

The current **Random Forest + OpenCV features** is fast but weaker than a **CNN** on photos.

Your repo also has:

`project/training/train.py` — **MobileNetV2** (needs more images, GPU helpful, longer training).

**When to use CNN:**

- 500+ images per class
- GPU available
- You want higher accuracy than Random Forest

**Steps (short):**

1. Put dataset under `project/dataset/<class_name>/` (224×224 training).
2. `pip install tensorflow opencv-python`
3. `cd project/training` → `python train.py`
4. Wire `plant_model.keras` into backend (requires code change in `predict.py` — not automatic today).

For a **university project deadline**, finishing **Phase A with Merge-Project Random Forest** is usually enough if the dataset is good.

---

## 6. Suggested starter class list (Sri Lanka)

Copy this checklist; create folders as you collect data:

```
Banana_Sigatoka
Banana_healthy
Rice_Blast
Rice_Bacterial_leaf_blight
Rice_Brown_spot
Rice_healthy
Coconut_Leaf_wilt
Coconut_healthy
Tea_Blister_blight
Tea_Red_rust
Tea_healthy
Chili_Anthracnose
Chili_healthy
Tomato_Early_blight
Tomato_Late_blight
Tomato_healthy
Mango_Anthracnose
Mango_healthy
Papaya_Ringspot
Papaya_healthy
```

Add **Onion**, **Maize**, **Beans**, **Cucumber** later the same way.

---

## 7. After training — update the app knowledge base

For each new class name, add an entry in:

`plant-disease-backend/predict.py` → `DISEASE_INFO`:

```python
"Banana_Sigatoka": {
    "cause": "...",
    "symptoms": "...",
    "treatment": "...",
    "prevention": "...",
    "fertilizer": "...",
    "severity": "High",
    "weather": "...",
    "insects": "...",
},
```

Without this, the API still predicts the class name but farmer advice stays generic.

---

## 8. Quality checklist before you call it “Sri Lanka ready”

- [ ] At least **150 images** per important class
- [ ] **Healthy** class for every crop
- [ ] Validation accuracy **> 70%** on `results.txt` (higher is better)
- [ ] Tested **banana**, **rice**, **tea** photos from your phone on the app
- [ ] New `.pkl` files copied to `plant-disease-backend/models/`
- [ ] `DISEASE_INFO` updated for each class
- [ ] Backend restarted; Flutter `constants.dart` has correct PC IP

---

## 9. Common mistakes

| Mistake | Fix |
|---------|-----|
| Banana still shows “Grape” | Train with `Banana_*` folders; deploy new `.pkl` files |
| Very low accuracy | More images; balance classes; cleaner photos |
| Folder name typo | Folder name must match exactly across dataset and `DISEASE_INFO` |
| Changed `features.py` after training | Never change — retrain from scratch if you do |
| No backend restart | Always restart `python app.py` after new models |

---

## 10. Who can help you collect data in Sri Lanka

- Rice Research Institute, Bathalagoda (rice diseases)
- Coconut Research Institute, Lunuwila
- Tea Research Institute, Talawakelle
- Provincial agriculture offices
- Your university agriculture department

Ask for **labeled leaf photos** or permission to photograph symptom sheets.

---

## Quick command summary

```powershell
# 1. Download banana + build Sri Lanka dataset (automated)
cd "d:\ai data\Final\Merge-Project"
python download_banana_and_build.py
python build_dataset_sri_lanka.py

# 2. Train (config.py already uses dataset_sri_lanka + AUTO_DISCOVER_CLASSES)
python train.py

# 3. Deploy
copy output\*.pkl "d:\ai data\Final\plant-disease-backend\models\"

# 4. Run API + app
cd "d:\ai data\Final\plant-disease-backend"
python app.py
```

**Already done in this project (May 2026):** 41 classes including `Banana_healthy`, `Banana_Sigatoka`, `Banana_Xanthomonas_wilt` — validation accuracy ~97% overall.

---

*This guide matches your existing `Merge-Project` + `plant-disease-backend` setup. For technical training details see `MODEL_TRAINING_REPORT.md`.*
