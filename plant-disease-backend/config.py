import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Resolve model paths relative to this config file so they work regardless
# of the current working directory when the app is started.
BASE_DIR = Path(__file__).resolve().parent
# RF models are trained to Merge-Project/output/ by train_rf_pipeline.py
OUTPUT_DIR = BASE_DIR.parent / "Merge-Project" / "output"
MODEL_PATH = str(OUTPUT_DIR / "model.pkl")
SCALER_PATH = str(OUTPUT_DIR / "scaler.pkl")
LABEL_ENCODER_PATH = str(OUTPUT_DIR / "label_encoder.pkl")
IMG_SIZE = 128


def validate_config():
    """Validate that all required configuration is present."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise RuntimeError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")


def log_startup_config() -> None:
    print("Plant disease API: auth disabled; history stored locally in prediction_history.json")


try:
    validate_config()
    log_startup_config()
except RuntimeError as e:
    print(f"Configuration Error: {e}")
