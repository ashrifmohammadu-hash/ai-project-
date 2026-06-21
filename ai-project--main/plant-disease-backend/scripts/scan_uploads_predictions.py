import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict import predict_disease

UPLOADS = ROOT / "uploads"
OUT = ROOT / "eval_output" / "upload_predictions_summary.csv"

def main():
    files = sorted([p for p in UPLOADS.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    rows = []
    for f in files:
        try:
            with open(f, 'rb') as fh:
                image_bytes = fh.read()
            disease, confidence, info, top_predictions, plant_type, metrics = predict_disease(image_bytes)
            top5 = ";".join([p.get('disease','')+":"+str(p.get('confidence','')) for p in top_predictions[:5]])
            rows.append((str(f.name), disease or '', confidence or 0.0, plant_type or '', top5))
        except Exception as e:
            rows.append((str(f.name), 'ERROR', str(e), '', ''))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['file','disease','confidence','plant_type','top5'])
        writer.writerows(rows)

    print(f"Wrote summary to {OUT}")

if __name__ == '__main__':
    main()
