"""
Quarantine likely wrong-folder images from dataset_mislabel_candidates.csv.

Run AFTER recheck_dataset_mislabels.py (uses output CSV):
  python quarantine_mislabeled_images.py

Moves only cross-crop disagreements (crop_match == no) with proba >= CONF_MIN.
Files moved to dataset_sri_lanka/<Class>/_quarantine/
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from config import DATASET_PATH

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / DATASET_PATH
CONF_MIN = 0.55  # only quarantine high-confidence disagreements
def main() -> None:
    csv_path = ROOT / "output" / "dataset_mislabel_candidates.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path} - run recheck_dataset_mislabels.py first.")

    moved = 0
    scanned = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scanned += 1
            if row.get("crop_match") != "no":
                continue
            conf = float(row.get("proba") or 0.0)
            if conf < CONF_MIN:
                continue

            rel_path = row.get("path", "")
            pred = row.get("predicted", "UNKNOWN")
            src = ROOT / rel_path
            if not src.exists():
                continue
            quarantine = src.parent / "_quarantine"
            quarantine.mkdir(exist_ok=True)
            dest = quarantine / f"{pred}_{conf:.2f}_{src.name}"
            if dest.exists():
                continue
            shutil.move(str(src), str(dest))
            moved += 1

    print(f"Scanned {scanned} CSV rows, quarantined {moved} cross-crop mislabels (conf >= {CONF_MIN}).")
    if moved:
        print("Retrain:")
        print("  python train.py")


if __name__ == "__main__":
    main()
