"""
Count images per class in Merge-Project/dataset_sri_lanka and save CSV/MD summary.
Usage:
  python scripts/compute_dataset_counts.py --root ../Merge-Project/dataset_sri_lanka --outdir eval_output
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv

p = argparse.ArgumentParser()
p.add_argument("--root", default="../Merge-Project/dataset_sri_lanka")
p.add_argument("--outdir", default="../eval_output")
args = p.parse_args()

root = Path(args.root).resolve()
outdir = Path(args.outdir).resolve()
outdir.mkdir(parents=True, exist_ok=True)

counts = Counter()
file_types = ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG")
if not root.exists():
    print(f"Dataset root not found: {root}")
    raise SystemExit(2)

for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue
    total = 0
    for pat in file_types:
        total += len(list(folder.glob(pat)))
    counts[folder.name] = total

ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
csv_path = outdir / f"dataset_counts_{ts}.csv"
md_path = outdir / f"dataset_counts_{ts}.md"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class","count"])
    for cls, c in counts.most_common():
        writer.writerow([cls, c])

with md_path.open("w", encoding="utf-8") as f:
    f.write(f"# Dataset class counts {ts}\n\n")
    f.write(f"Root: {root}\n\n")
    f.write("## Top classes by count\n\n")
    for cls, c in counts.most_common(50):
        f.write(f"- {cls}: {c}\n")
    f.write("\nTotal classes: %d\n" % len(counts))
    f.write("\nFiles:\n")
    f.write(f"- CSV: {csv_path}\n")
    f.write(f"- MD: {md_path}\n")

print("Counts written:")
print(f"  CSV: {csv_path}")
print(f"  MD: {md_path}")
