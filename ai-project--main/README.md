# Plant Disease Detection System

AI-powered plant disease detection using MobileNetV3-Large + OpenCV leaf gate.

## Quick Setup

```bash
# 1. Clone
git clone https://github.com/ashrifmohammadu-hash/ai-project-.git
cd ai-project-

# 2. Download trained model (required!)
# Place in: Merge-Project/output/plant_disease_mobilenet.pth
# Download from: https://github.com/ashrifmohammadu-hash/ai-project-/releases/download/v1.0.0/plant_disease_mobilenet.pth

# 3. Python deps
pip install -r plant-disease-backend/requirements.txt

# 4. Create .env in plant-disease-backend/ with your API keys:
#    GEMINI_API_KEY=your_key
#    GROQ_API_KEY=your_key
#    VISION_PROVIDER=gemini

# 5. Start backend
cd plant-disease-backend && py app.py

# 6. Start frontend (separate terminal)
cd web && npm install && npm run dev
```

## Model Download

The trained model (16.6 MB) is on GitHub Releases:  
**https://github.com/ashrifmohammadu-hash/ai-project-/releases/download/v1.0.0/plant_disease_mobilenet.pth**

Save it to `Merge-Project/output/plant_disease_mobilenet.pth`

## Pipeline

1. **OpenCV leaf gate** — rejects non-leaf images
2. **MobileNetV3-Large** — 69 classes, 95% accuracy
3. **Gemini/Groq** — fallback (when API available)

## Tech Stack

- **Backend**: Flask, PyTorch, OpenCV, scikit-learn
- **Frontend**: React, Vite
- **Training**: MobileNetV3-Large, 5 epochs, weighted sampling
