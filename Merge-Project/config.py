# config.py
# ─────────────────────────────────────────────
# All project settings in one place.
# Change something here → it updates everywhere.
# ─────────────────────────────────────────────

# ── Paths ──────────────────────────────────────
DATASET_PATH  = "dataset_sri_lanka"   # banana + SL crops; or "resized_merged" for full PlantVillage
OUTPUT_DIR    = "output"

# If True, train every subfolder under DATASET_PATH (ignore missing CLASSES entries)
AUTO_DISCOVER_CLASSES = True

# ── Image settings ─────────────────────────────
IMG_SIZE      = (128, 128)         # width x height in pixels

# ── Model settings ─────────────────────────────
TEST_SIZE     = 0.2                # 20% for validation
RANDOM_STATE  = 42
N_ESTIMATORS  = 80                 # number of trees in Random Forest
N_JOBS        = 2                  # limit parallel jobs (avoids MemoryError on large SL set)

# ── Output file paths ──────────────────────────
MODEL_PATH    = "output/model.pkl"
SCALER_PATH   = "output/scaler.pkl"
ENCODER_PATH  = "output/label_encoder.pkl"
REPORT_PATH   = "output/results.txt"
CM_PATH       = "output/confusion_matrix.png"

# ── Your 39 disease classes (exact folder names) ─
CLASSES = [
    "Apple_Cedar_apple_rust",
    "Apple_healthy",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Blueberry_healthy",
    "Cherry_(including_sour)_healthy",
    "Cherry_(including_sour)_Powdery_mildew",
    "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)_Common_rust_",
    "Corn_(maize)_healthy",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Grape_Black_rot",
    "Grape_Esca_(Black_Measles)",
    "Grape_healthy",
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange_Haunglongbing_(Citrus_greening)",
    "Peach_Bacterial_spot",
    "Peach_healthy",
    "Pepper,_bell_Bacterial_spot",
    "Pepper,_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_Powdery_mildew",
    "Strawberry_healthy",
    "Strawberry_Leaf_scorch",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two-spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
]
# NOTE: 'test' folder is intentionally excluded — it has no disease label