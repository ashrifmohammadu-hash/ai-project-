"""
leaf_classifier.py – Binary leaf vs non-leaf classifier.

Two modes:
  1. Trained model: loads leaf_classifier/leaf_classifier.pth
  2. Fallback: uses ResNet50 confidence as leaf-proxy

Usage:
    from leaf_classifier import is_leaf
    ok, conf, source = is_leaf(image_bytes)
"""

import os, sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

MODEL_PATH = Path(__file__).parent / "leaf_classifier" / "leaf_classifier.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Lazy-loaded model
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        return None
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        _model = model
        return model
    except Exception as e:
        print(f"[leaf_classifier] Model load failed: {e}")
        return None


def classify_leaf(image_bytes):
    """
    Binary leaf classification.
    Returns (is_leaf: bool, confidence: float, source: str).
    confidence = probability of leaf class (0-100).
    """
    model = _load_model()

    if model is not None:
        # ── Trained model mode ──
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = _transform(pil_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                leaf_conf = round(float(probs[0][1].item()) * 100, 1)
            is_leaf = leaf_conf >= 50
            return is_leaf, leaf_conf, "leaf_classifier_model"
        except Exception as e:
            print(f"[leaf_classifier] Inference failed: {e}")

    # ── Fallback: use existing plant disease ResNet50 ──
    try:
        from predict import _predict_with_pytorch
        result = _predict_with_pytorch(image_bytes)
        if result:
            disease_name, conf, plant_type = result
            # ResNet50 was trained on leaf images only
            # High confidence → likely a leaf
            # Low confidence → likely not a leaf
            is_leaf = conf >= 15
            return is_leaf, conf, "resnet50_proxy"
    except Exception as e:
        print(f"[leaf_classifier] Fallback failed: {e}")

    return False, 0.0, "none"
