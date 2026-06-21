"""
Batch-run predictions on `eval_output/mispredictions` and produce CSV/JSON/MD reports.

Usage:
  python scripts/generate_misprediction_report.py --root eval_output/mispredictions --limit 200

Outputs saved to `plant-disease-backend/eval_output/` with timestamped filenames.
"""
from __future__ import annotations
import json
import csv
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Ensure project root is importable when running from scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import predict_disease


def find_images(root: Path):
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for img in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
            yield folder.name, img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="eval_output/mispredictions")
    p.add_argument("--outdir", default="eval_output")
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = outdir / f"misprediction_report_{ts}.csv"
    md_path = outdir / f"misprediction_report_{ts}.md"
    json_path = outdir / f"misprediction_report_{ts}.jsonl"

    rows = []
    pair_counts = Counter()
    expected_counts = Counter()
    predicted_counts = Counter()

    i = 0
    for folder_name, img_path in find_images(root):
        if args.limit and i >= args.limit:
            break
        expected = folder_name.split("__to__")[0] if "__to__" in folder_name else folder_name
        try:
            data = img_path.read_bytes()
            disease, conf, info, top_predictions, plant_type, metrics = predict_disease(data)
        except Exception as e:
            rows.append({
                "input": str(img_path),
                "expected": expected,
                "error": str(e),
            })
            i += 1
            continue

        top1 = top_predictions[0]["disease"] if top_predictions else ""
        top1_conf = top_predictions[0].get("confidence") if top_predictions else None

        row = {
            "input": str(img_path),
            "expected": expected,
            "predicted": disease,
            "confidence": conf,
            "plant_type": plant_type,
            "top1": top1,
            "top1_conf": top1_conf,
        }
        # flatten metrics
        if metrics:
            for k, v in metrics.items():
                row[k] = v

        rows.append(row)
        pair_counts[(expected, disease)] += 1
        expected_counts[expected] += 1
        predicted_counts[disease] += 1
        i += 1

    # write CSV
    # collect all keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = ["input", "expected", "predicted", "confidence", "plant_type", "top1", "top1_conf"] + sorted(k for k in keys if k not in ("input","expected","predicted","confidence","plant_type","top1","top1_conf","error"))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in keys})

    # write jsonl
    with json_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # write markdown summary
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Misprediction report {ts}\n\n")
        f.write(f"Scanned: {len(rows)} images (limit={args.limit})\n\n")

        f.write("## Top expected classes\n\n")
        for exp, c in expected_counts.most_common(20):
            f.write(f"- {exp}: {c}\n")
        f.write("\n## Top predicted classes\n\n")
        for pred, c in predicted_counts.most_common(20):
            f.write(f"- {pred}: {c}\n")

        f.write("\n## Top expected→predicted pairs\n\n")
        for (exp, pred), c in pair_counts.most_common(50):
            f.write(f"- {exp} → {pred}: {c}\n")

        f.write("\nFiles:\n")
        f.write(f"- CSV: {csv_path}\n")
        f.write(f"- JSONL: {json_path}\n")
        f.write(f"- MD summary: {md_path}\n")

    print("Report generated:")
    print(f"  CSV: {csv_path}")
    print(f"  JSONL: {json_path}")
    print(f"  MD: {md_path}")


if __name__ == "__main__":
    main()
