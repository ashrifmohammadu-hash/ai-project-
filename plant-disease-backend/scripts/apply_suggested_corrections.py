import json
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
SUG = ROOT / "eval_output" / "upload_correction_suggestions.json"
CSV = ROOT / "eval_output" / "upload_predictions_summary.csv"
OUT_CSV = ROOT / "eval_output" / "upload_predictions_summary_corrected.csv"
LOG = ROOT / "eval_output" / "corrections_applied.json"

def main():
    if not SUG.exists():
        print('No suggestions file:', SUG)
        return
    if not CSV.exists():
        print('No summary CSV:', CSV)
        return
    with open(SUG, 'r', encoding='utf8') as fh:
        suggestions = json.load(fh)
    sugg_map = {s['file']: s for s in suggestions}

    applied = []
    rows = []
    with open(CSV, newline='', encoding='utf8') as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        for r in reader:
            fname = r.get('file')
            if fname in sugg_map:
                s = sugg_map[fname]
                old = {'disease': r.get('disease'), 'confidence': r.get('confidence'), 'top5': r.get('top5')}
                r['disease'] = s['to']
                # set confidence to suggested score (approx) and push into top5 front
                r['confidence'] = str(round(s.get('score', 48.0),1))
                # prepend corrected top5
                r['top5'] = s['to'] + ':' + str(s.get('score',48.0)) + ';' + (r.get('top5') or '')
                applied.append({'file': fname, 'from': old, 'to': s})
            rows.append(r)

    # write corrected CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf8') as outfh:
        writer = csv.DictWriter(outfh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    # write log
    with open(LOG, 'w', encoding='utf8') as logfh:
        json.dump(applied, logfh, indent=2)

    print(f'Applied {len(applied)} corrections, wrote {OUT_CSV} and {LOG}')

if __name__ == '__main__':
    main()
