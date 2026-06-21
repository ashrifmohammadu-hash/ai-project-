"""
Scan dataset_sri_lanka, collect misclassified images, save CSV and confusion matrix.

Outputs:
 - eval_output/mispredictions/<true>__to__<pred>/image.jpg
 - eval_output/mispredictions_summary.csv
 - eval_output/confusion_matrix.png
"""
from __future__ import annotations

import csv
from pathlib import Path
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from predict import predict_disease


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "Merge-Project" / "dataset_sri_lanka"
OUT = ROOT / "eval_output" / "mispredictions"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = []
true_list = []
pred_list = []
rows = []

def main():
    if not DATASET.is_dir():
        print("Dataset not found", DATASET)
        return 1

    random.seed(42)
    for folder in sorted(DATASET.iterdir()):
        if not folder.is_dir():
            continue
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not images:
            continue
        LABELS.append(folder.name)
        sample = random.sample(images, min(12, len(images)))
        for img in sample:
            try:
                pred, conf, info, top_predictions, plant_type, metrics = predict_disease(img.read_bytes())
            except Exception as e:
                pred = None
                conf = 0.0
            true_list.append(folder.name)
            pred_list.append(pred or "?")
            if pred != folder.name:
                # save copy into folder
                sub = OUT / f"{folder.name}__to__{pred or 'UNKNOWN'}"
                sub.mkdir(parents=True, exist_ok=True)
                dest = sub / img.name
                if not dest.exists():
                    dest.write_bytes(img.read_bytes())
                rows.append({"true": folder.name, "pred": pred or "?", "image": str(dest), "confidence": conf})

    # write CSV
    csv_path = OUT.parent / "mispredictions_summary.csv"
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["true", "pred", "image", "confidence"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # confusion matrix for labels seen
    labels = sorted(set(true_list + pred_list))
    if labels:
        cm = confusion_matrix(true_list, pred_list, labels=labels)
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = OUT.parent / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        print("Saved confusion matrix ->", cm_path)

    print("Saved CSV ->", csv_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
