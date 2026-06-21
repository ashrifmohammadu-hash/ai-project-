# Plant Disease Detection — Model Training Report

**Report scope:** Model training only (dataset → features → train → save).  
**Does not cover:** Flask API, JWT auth, Flutter app, or deployment server.  
**Project root:** `d:\ai data\Final`  
**Generated:** May 2026  

---

## 1. Executive Summary

This project trains a **plant leaf disease classifier** using the **PlantVillage**-style dataset (images grouped by disease class folders). The **model used in production** (`plant-disease-backend/models/`) was trained with the **Merge-Project** pipeline:

| Item | Choice |
|------|--------|
| **Algorithm** | **Random Forest** (scikit-learn) |
| **Input** | **103 hand-crafted features** per image (OpenCV) |
| **Classes** | **38** disease / healthy labels |
| **Saved files** | `model.pkl`, `scaler.pkl`, `label_encoder.pkl` |

A second, optional training path exists under `project/training/` using **MobileNetV2 + TensorFlow** (deep learning). That path is **not** the model currently loaded by the live backend.

---

## 2. Training Pipeline Used in Production

### 2.1 Folder layout (Merge-Project)

```
Merge-Project/
├── config.py          # Paths, hyperparameters, class list
├── features.py        # OpenCV feature extraction (103 values)
├── dataset.py         # Load images from resized_merged/
├── train.py           # Main training script
├── predict.py         # CLI test after training
└── output/            # Created when you run train.py
    ├── model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── results.txt
    └── confusion_matrix.png
```

Trained files are copied (or already present) in:

```
plant-disease-backend/models/
├── model.pkl           (~635 MB — Random Forest with 100 trees)
├── scaler.pkl
└── label_encoder.pkl
```

### 2.2 How to run training

```powershell
cd "d:\ai data\Final\Merge-Project"
pip install numpy opencv-python scikit-learn joblib matplotlib seaborn tqdm
python train.py
```

**Dataset requirement:** folder `resized_merged/` with one subfolder per class name (see `config.py` → `CLASSES`).

---

## 3. Dataset

| Item | Detail |
|------|--------|
| **Source** | PlantVillage–style leaf images (merged / resized set) |
| **Folder** | `resized_merged/<class_name>/*.jpg` |
| **Image size** | **128 × 128** pixels (BGR) |
| **Labels** | Folder name = disease class (e.g. `Grape_Black_rot`) |
| **Classes** | **38** (see list in `Merge-Project/config.py`) |
| **Train/val split** | **80% train / 20% validation** (`TEST_SIZE = 0.2`) |
| **Stratify** | Yes — keeps class balance in train and val |

**Why this dataset format?**  
Each class is a separate directory, which is standard for image classification and works with simple loading loops (no custom database).

---

## 4. Tools, Packages & Why They Are Used (Training Only)

### 4.1 Core training stack (production path)

| Tool / package | Version (typical) | Role in training | Why used |
|----------------|-------------------|------------------|----------|
| **Python** | 3.10+ | Language | Standard for ML and OpenCV |
| **NumPy** | 2.x | Arrays for features `X`, labels `y` | Fast numeric arrays for scikit-learn |
| **OpenCV** (`opencv-python`) | 4.x | Read/resize images; HSV, Canny, histograms | Industry-standard image I/O and feature extraction without GPU |
| **scikit-learn** | 1.8.x | `RandomForestClassifier`, `StandardScaler`, `LabelEncoder`, `train_test_split`, metrics | Classic ML: trains quickly on CPU, works well on tabular features |
| **joblib** | 1.5.x | Save/load `model.pkl`, `scaler.pkl`, `label_encoder.pkl` | Standard persistence for sklearn models |
| **tqdm** | — | Progress bar while loading thousands of images | Visibility during long dataset load |
| **matplotlib** | — | Save training plots (non-interactive backend `Agg`) | Confusion matrix export |
| **seaborn** | — | Heatmap for confusion matrix | Clear per-class error visualization |

### 4.2 Not used during Random Forest training

These appear elsewhere in the repo but **are not required** to run `Merge-Project/train.py`:

| Package | Where it appears | Note |
|---------|------------------|------|
| Flask, PyJWT, waitress | Backend API | Inference server only |
| TensorFlow / Keras | `project/training/train.py` | Alternate CNN training path |
| Supabase | Backend auth/history | Not part of model training |

### 4.3 Suggested `pip install` for training only

```text
numpy
opencv-python
scikit-learn
joblib
matplotlib
seaborn
tqdm
```

---

## 5. Algorithm — Random Forest

### 5.1 What it is

**Random Forest** is an ensemble of many **decision trees**. Each tree votes for a class; the majority vote is the final prediction. For this project, **`predict_proba`** is used at inference time to get confidence scores.

### 5.2 Why Random Forest was chosen (instead of only deep learning)

| Reason | Explanation |
|--------|-------------|
| **CPU-friendly** | Trains and runs without GPU — suitable for university / local PC setups |
| **Small feature vectors** | Only **103 numbers** per image, not full 128×128×3 tensors |
| **Fast iteration** | Retrain in minutes after dataset or feature changes |
| **Interpretable baseline** | Confusion matrix and per-class report show which diseases confuse the model |
| **Works with limited RAM** | Compared to large CNNs, though 100 trees still produce a large `model.pkl` (~635 MB) |

### 5.3 Hyperparameters (`Merge-Project/config.py` + `train.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `N_ESTIMATORS` | **100** | Number of trees — more trees → stabler votes, larger file |
| `class_weight` | **`balanced`** | Reduces bias when some disease folders have fewer images |
| `random_state` | **42** | Reproducible train/val split and forest |
| `n_jobs` | **-1** | Use all CPU cores for faster training |
| `TEST_SIZE` | **0.2** | 20% held out for validation |
| `stratify` | **yes** | Same class proportions in train and val |

### 5.4 Preprocessing before the forest

| Step | Tool | Why |
|------|------|-----|
| **Label encoding** | `LabelEncoder` | Converts string class names → integers 0…37 |
| **Feature scaling** | `StandardScaler` | Puts HSV histogram, edge density, and channel stats on comparable scales so no single feature dominates |

**Important:** The same `StandardScaler` fitted on **training data only** must be saved and applied at prediction time (`scaler.transform`).

---

## 6. Feature Extraction (OpenCV) — Not End-to-End CNN

Training does **not** feed raw pixels directly into the forest. Each image is converted to **103 features** in `features.py`:

| Feature block | Count | Method | Why it helps |
|---------------|-------|--------|--------------|
| **HSV histograms** | 96 (32×3) | Hue, Saturation, Value bins | Captures yellowing, browning, and green vs diseased tones; HSV is more stable under lighting than RGB alone |
| **Edge density** | 1 | Canny edge map → fraction of edge pixels | Spots and lesions add edges; healthy leaves are smoother |
| **BGR mean & std** | 6 (mean+std × 3 channels) | Per-channel statistics | Overall colour and patchiness (uneven disease) |

**Image resize:** 128×128 before extraction — balances detail and speed.

**Rule:** `features.py` must stay **identical** after training. Prediction code must use the same logic (see `plant-disease-backend/predict.py` → `extract_features`).

---

## 7. Training Workflow (Step by Step)

```
┌─────────────────┐
│ resized_merged/ │  38 class folders, JPG/PNG leaf images
└────────┬────────┘
         ▼
┌─────────────────┐
│  dataset.py     │  cv2.imread → resize 128×128 → extract_features()
└────────┬────────┘
         ▼
   X (N × 103), y (N,) string labels
         ▼
┌─────────────────┐
│ LabelEncoder    │  y → integers
└────────┬────────┘
         ▼
┌─────────────────┐
│ train_test_split│  80/20 stratified
└────────┬────────┘
         ▼
┌─────────────────┐
│ StandardScaler  │  fit on train → transform train & val
└────────┬────────┘
         ▼
┌─────────────────┐
│ RandomForest    │  fit on X_train, y_train
└────────┬────────┘
         ▼
┌─────────────────┐
│ Evaluate        │  accuracy, classification_report, confusion_matrix
└────────┬────────┘
         ▼
┌─────────────────┐
│ joblib.dump     │  model.pkl, scaler.pkl, label_encoder.pkl
└─────────────────┘
```

---

## 8. Training Outputs

| File | Description |
|------|-------------|
| `output/model.pkl` | Trained `RandomForestClassifier` |
| `output/scaler.pkl` | Fitted `StandardScaler` |
| `output/label_encoder.pkl` | Maps index ↔ class name string |
| `output/results.txt` | Train/val accuracy + per-class precision/recall/F1 |
| `output/confusion_matrix.png` | Heatmap of true vs predicted labels |

After training, copy the three `.pkl` files to `plant-disease-backend/models/` for the API to use them.

---

## 9. Evaluation Metrics Used

| Metric | Tool | Purpose |
|--------|------|---------|
| **Train accuracy** | `model.score` | Check learning on training set |
| **Validation accuracy** | `accuracy_score` | Unseen 20% performance |
| **Classification report** | precision, recall, F1 per class | See weak diseases |
| **Confusion matrix** | `confusion_matrix` + seaborn heatmap | Visualize confusions (e.g. grape vs corn) |

If train accuracy is much higher than validation (>15% gap), `train.py` prints an **overfitting warning**.

---

## 10. Alternate Training Path — Deep Learning (Reference Only)

**Location:** `project/training/`

| Item | Detail |
|------|--------|
| **Algorithm** | **MobileNetV2** (transfer learning) + custom Dense head |
| **Framework** | **TensorFlow / Keras** |
| **Input** | Raw images **224×224×3**, normalized 0–1 |
| **Augmentation** | `ImageDataGenerator` (rotation, flip, zoom, shear, brightness) |
| **Optimizer** | **Adam** (lr 0.001, then 0.0001 for fine-tune) |
| **Loss** | `categorical_crossentropy` |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **Output** | `plant_model.keras` (under `project/models/`) |

**Why it exists:** README and coursework often describe a CNN (PlantVillage + MobileNetV2).  
**Why production uses Random Forest instead:** The deployed backend loads `model.pkl` from Merge-Project training; no `.keras` file is required to run `python app.py` today.

To train the CNN path:

```powershell
cd "d:\ai data\Final\project\training"
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
# Place dataset under project/dataset/<class_name>/...
python train.py
```

---

## 11. 38 Disease Classes (Training Labels)

Class names match folder names under `resized_merged/` (from `Merge-Project/config.py`):

- **Apple:** Cedar apple rust, healthy, scab, black rot  
- **Blueberry:** healthy  
- **Cherry:** healthy, powdery mildew  
- **Corn (maize):** Cercospora/Gray leaf spot, common rust, healthy, northern leaf blight  
- **Grape:** Black rot, Esca, healthy, leaf blight  
- **Orange:** Citrus greening  
- **Peach:** Bacterial spot, healthy  
- **Pepper (bell):** Bacterial spot, healthy  
- **Potato:** Early blight, healthy, late blight  
- **Raspberry, Soybean, Squash, Strawberry:** healthy / mildew / scorch as applicable  
- **Tomato:** Bacterial spot, early/late blight, leaf mold, septoria, spider mites, target spot, viruses, healthy  

*(Full exact strings are in `config.py` → `CLASSES`.)*

---

## 12. Summary Table — Training vs Inference

| Stage | Tools | Algorithm / method |
|-------|-------|-------------------|
| **Load images** | OpenCV | Resize 128×128 |
| **Features** | OpenCV (HSV, Canny, stats) | 103-D vector |
| **Train** | scikit-learn | Random Forest (100 trees, balanced) |
| **Scale** | scikit-learn | StandardScaler |
| **Encode labels** | scikit-learn | LabelEncoder |
| **Save** | joblib | `.pkl` files |
| **Report** | matplotlib, seaborn | Accuracy + confusion matrix |
| **Inference** (separate doc) | Same features + scaler + forest | `predict_proba` → top class + confidence |

---

## 13. Related Files (Quick Reference)

| Path | Purpose |
|------|---------|
| `Merge-Project/train.py` | **Main training script (production model)** |
| `Merge-Project/features.py` | Feature extraction definition |
| `Merge-Project/dataset.py` | Build `X`, `y` from folders |
| `Merge-Project/config.py` | Hyperparameters and paths |
| `project/training/train.py` | Optional MobileNetV2 training |
| `plant-disease-backend/predict.py` | Inference (uses saved `.pkl` files) |
| `plant-disease-backend/BACKEND_REPORT.md` | API, auth, deployment (not training) |

---

*End of model training report.*
