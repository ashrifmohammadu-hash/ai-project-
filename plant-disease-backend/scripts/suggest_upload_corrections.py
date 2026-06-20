import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "eval_output" / "upload_predictions_summary.csv"
OUT = ROOT / "eval_output" / "upload_correction_suggestions.json"

def parse_top5(top5str):
    items = []
    if not top5str:
        return items
    for part in top5str.split(';'):
        if ':' in part:
            d, p = part.split(':',1)
            try:
                items.append((d, float(p)))
            except Exception:
                continue
    return items

def main():
    if not CSV.exists():
        print('Summary CSV not found:', CSV)
        return
    suggestions = []
    with open(CSV, newline='', encoding='utf8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            file = r['file']
            disease = r['disease']
            top5 = parse_top5(r.get('top5',''))
            # Suggest correction when predicted Papaya but Banana appears in top5 >= 8
            if disease.startswith('Papaya_'):
                for d,p in top5:
                    if d.startswith('Banana_') and p >= 8.0:
                        suggestions.append({
                            'file': file,
                            'from': disease,
                            'to': d,
                            'score': p,
                        })
                        break
            # Also suggest when predicted Mango but Banana is top with moderate score
            if disease.startswith('Mango_'):
                for d,p in top5:
                    if d.startswith('Banana_') and p >= 10.0:
                        suggestions.append({
                            'file': file,
                            'from': disease,
                            'to': d,
                            'score': p,
                        })
                        break

    with open(OUT, 'w', encoding='utf8') as outfh:
        json.dump(suggestions, outfh, indent=2)
    print(f'Wrote {len(suggestions)} suggestions to {OUT}')

if __name__ == '__main__':
    main()
