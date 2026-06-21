# features.py
# ─────────────────────────────────────────────
# OpenCV feature extraction.
# Converts a leaf image into numbers for Random Forest.
# IMPORTANT: Never change this file after training —
# prediction must use the exact same function.
# ─────────────────────────────────────────────

import cv2
import numpy as np


def extract_features(img):
    """
    Input : img  — colour leaf image (BGR), already resized to IMG_SIZE
    Output: 1D numpy array of 103 numbers describing the leaf

    Feature breakdown:
      [0  – 95 ] → HSV colour histograms  (32 bins × 3 channels)
      [96       ] → Edge density           (how spotted/rough the leaf is)
      [97 – 102 ] → Mean + Std per BGR channel
    """
    features = []

    # ── 1. HSV Colour Histogram ─────────────────────────────────────────
    # HSV separates colour (Hue) from brightness so lighting changes
    # affect the result less than plain BGR histograms.
    #
    # Healthy leaf   → H ≈ 80–120  (green)
    # Yellow disease → H ≈ 20–45   (yellow)
    # Brown spot     → H ≈  5–20   (orange-brown)
    # Late blight    → H ≈  0–10   (very dark)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):                          # H, S, V
        hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()    # 32 numbers
        features.extend(hist)                         # → 96 total

    # ── 2. Edge Density ─────────────────────────────────────────────────
    # Counts how many edges (spots, lesions) exist on the leaf surface.
    # Healthy leaf  → few edges  (smooth, uniform surface)
    # Brown spot    → many edges (dark spots create sharp edges)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size    # 1 number
    features.append(edge_density)

    # ── 3. Mean and Std per BGR Channel ─────────────────────────────────
    # Mean  = average colour of the whole leaf
    # Std   = how uneven the colour is (high std = patchy = diseased)
    for channel in range(3):                          # B, G, R
        features.append(np.mean(img[:, :, channel])) # 3 numbers
        features.append(np.std(img[:, :, channel]))  # 3 numbers

    return np.array(features, dtype=np.float32)       # total = 103