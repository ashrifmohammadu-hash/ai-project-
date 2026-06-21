import os
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so local imports work when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict import predict_disease
from leaf_analysis import is_banana_leaf, resolve_banana_disease
UPLOADS = ROOT / "uploads"
OUTDIR = ROOT / "eval_output" / "corrections"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    files = sorted([p for p in UPLOADS.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    summary = []
    for f in files:
        try:
            with open(f, 'rb') as fh:
                image_bytes = fh.read()
            disease, confidence, info, top_predictions, plant_type, metrics = predict_disease(image_bytes)
        except Exception as e:
            print(f"Skipping {f.name}: predict error: {e}")
            continue

        corrected = None
        if disease and disease.startswith('Papaya_') and is_banana_leaf(metrics):
            # pick a banana class from the top_predictions if present, else default to Banana_healthy
            banana_choice = None
            for p in top_predictions:
                d = p.get('disease')
                if d and d.startswith('Banana_'):
                    banana_choice = d
                    break
            if banana_choice:
                corrected = {'from': disease, 'to': banana_choice, 'confidence': p.get('confidence')}
            else:
                corrected = {'from': disease, 'to': 'Banana_healthy', 'confidence': 48.0}

        if corrected:
            out = {
                'file': str(f.name),
                'original': disease,
                'corrected': corrected,
                'metrics': metrics,
            }
            with open(OUTDIR / (f.name + '.json'), 'w', encoding='utf8') as fh:
                json.dump(out, fh, indent=2)
            summary.append(out)

    print(f"Scanned {len(files)} uploads, corrections suggested: {len(summary)}")
    for s in summary[:50]:
        print(f"{s['file']}: {s['original']} -> {s['corrected']['to']} ({s['corrected']['confidence']})")

if __name__ == '__main__':
    main()
