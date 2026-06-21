"""
Minimal Flask + PyTorch prediction endpoint example.
Drop into a Flask backend; adapt paths and class_map to your project.

Usage (dev):
  pip install -r requirements-flask-pytorch.txt
  python scripts/flask_pytorch_example.py

Then POST an image:
  curl -F "file=@leaf.jpg" -F "force_clarify=true" http://127.0.0.1:5000/predict

This is an example skeleton — replace model loading and transforms
with whatever you used during training.
"""
from io import BytesIO
import json
import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)

# CONFIG: adjust paths
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
CLASS_MAP_PATH = os.environ.get("CLASS_MAP_PATH", "class_map.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.5

# Load class map (expects dict: idx -> {"label": ..., "display_name": ..., ...})
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        CLASS_MAP = json.load(f)
else:
    # Minimal fallback
    CLASS_MAP = {"0": {"label": "unknown", "display_name": "Unknown"}}

# Load model
if os.path.exists(MODEL_PATH):
    try:
        model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    except Exception:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
else:
    model = None

# Example transforms - replace mean/std with training values
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(file_storage):
    img = Image.open(BytesIO(file_storage.read())).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    return tensor


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    force_clarify = (request.form.get("force_clarify") or "").lower() in ("1", "true", "yes", "on")

    try:
        x = preprocess_image(file)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]

        topk = torch.topk(probs, k=min(TOP_K, probs.numel()))
        indices = topk.indices.cpu().numpy().tolist()
        scores = topk.values.cpu().numpy().tolist()

        top_predictions = []
        for idx, score in zip(indices, scores):
            key = str(int(idx))
            meta = CLASS_MAP.get(key, {"label": key, "display_name": CLASS_MAP.get(key, {}).get("display_name", key)})
            top_predictions.append({
                "label": meta.get("label", key),
                "display_name": meta.get("display_name", meta.get("label", key)),
                "probability": float(score),
            })

        top_prob = top_predictions[0]["probability"] if top_predictions else 0.0
        is_confident = top_prob >= CONFIDENCE_THRESHOLD
        needs_clarification = force_clarify or (not is_confident)

        display_info = {
            "display_name": top_predictions[0]["display_name"] if top_predictions else "Unknown",
            "description": "",
        }

        response = {
            "prediction": top_predictions[0]["label"] if top_predictions else "unknown",
            "confidence": top_prob,
            "display_info": display_info,
            "top_predictions": top_predictions,
            "plant_type": None,
            "is_confident": is_confident,
            "needs_clarification": needs_clarification,
            "clarification_questions": [],
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
