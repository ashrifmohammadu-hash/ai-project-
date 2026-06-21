import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
from torchvision import datasets

# ---------- CONFIGURATION (change these paths) ----------
MODEL_PATH = r"D:\ai data\Final\pytorch_project\plant_disease_resnet50.pth"
# Folder that contains your class subfolders (used to get class names in correct order)
DATA_DIR = r"D:\ai data\Final\Merge-Project\resized_merged"
# --------------------------------------------------------

_model = None
_class_names = None
_device = None

def _load_model():
    global _model, _class_names, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Loading model on {_device}")

    # Get class names from the dataset folder (order must match training)
    dataset = datasets.ImageFolder(DATA_DIR)
    _class_names = dataset.classes
    num_classes = len(_class_names)

    # Build ResNet50 architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location=_device)
    model.load_state_dict(state_dict)
    model = model.to(_device)
    model.eval()
    _model = model
    print(f"[PyTorch] Loaded {num_classes} classes")

# Transform must match the one used during validation
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_pytorch(image_bytes):
    """
    image_bytes: bytes of an image file (JPEG, PNG, etc.)
    Returns:
        disease: str (class name)
        confidence: float (0-100)
        top_predictions: list of dicts [{"disease": ..., "confidence": ...}, ...]
    """
    _load_model()
    # Read image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Transform
    input_tensor = _transform(image).unsqueeze(0).to(_device)
    # Inference
    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=5, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    top_predictions = [
        {"disease": _class_names[idx], "confidence": float(prob * 100)}
        for idx, prob in zip(top_indices, top_probs)
    ]
    disease = top_predictions[0]["disease"]
    confidence = top_predictions[0]["confidence"]
    return disease, confidence, top_predictions