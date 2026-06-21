# predict.py
# ─────────────────────────────────────────────
# Test any leaf image after training is done.
#
# HOW TO RUN:
#   python predict.py path\to\your\leaf.jpg
#
# Example:
#   python predict.py C:\Users\MIM.ASHRIF\Pictures\leaf.jpg
# ─────────────────────────────────────────────

import sys
import cv2
import numpy as np
import joblib

from features import extract_features
from config   import (IMG_SIZE, MODEL_PATH,
                      SCALER_PATH, ENCODER_PATH)


def predict_disease(image_path):
    # ── Load saved model files ───────────────────────────────────────
    print(f"\nLoading model files ...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le     = joblib.load(ENCODER_PATH)

    # ── Load image ───────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image → {image_path}")
        print("Check that the file path is correct.")
        return

    # ── Prepare image ────────────────────────────────────────────────
    img      = cv2.resize(img, IMG_SIZE)
    features = extract_features(img)
    features = scaler.transform([features])      # scale same as training

    # ── Predict ──────────────────────────────────────────────────────
    prediction    = model.predict(features)
    probabilities = model.predict_proba(features)[0]

    disease    = le.inverse_transform(prediction)[0]
    confidence = probabilities.max() * 100

    # ── Print result ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Image     : {image_path}")
    print(f"  Disease   : {disease}")
    print(f"  Confidence: {confidence:.1f}%")
    print("=" * 50)

    # Show top 5 most likely diseases
    print("\nTop 5 predictions:")
    top5 = sorted(zip(le.classes_, probabilities),
                  key=lambda x: x[1], reverse=True)[:5]
    for rank, (cls, prob) in enumerate(top5, 1):
        bar = '█' * int(prob * 40)
        print(f"  {rank}. {cls[:45]:45s} {prob*100:5.1f}%  {bar}")

    # Confidence warning
    if confidence < 60:
        print("\n[WARNING] Confidence is low — the image may be:")
        print("  - Poor lighting or blurry")
        print("  - A disease not in the training set")
        print("  - Not a leaf image")

    return disease, confidence


# ── Run from command line ────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_leaf_image>")
        print("Example: python predict.py leaf.jpg")
    else:
        predict_disease(sys.argv[1])