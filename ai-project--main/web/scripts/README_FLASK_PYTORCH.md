Quick Flask + PyTorch prediction example

Files:
- flask_pytorch_example.py : minimal Flask app demonstrating image preprocessing, model loading, and JSON response fields.
- requirements-flask-pytorch.txt : Python packages used in the example.

Usage:
1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r scripts/requirements-flask-pytorch.txt
```

2. Place your model and class map next to the script or set env vars:
- `MODEL_PATH` (default: model.pt)
- `CLASS_MAP_PATH` (default: class_map.json)

3. Run the example:

```bash
python scripts/flask_pytorch_example.py
```

4. Test with curl:

```bash
curl -F "file=@leaf.jpg" -F "force_clarify=true" http://127.0.0.1:5000/predict
```

Notes:
- Replace transforms `mean`/`std` and image sizing to match your training pipeline.
- For production, load a scripted or traced model and disable `debug`.
- If inference is slow, consider batching, using GPU, or moving long jobs to a background queue.
