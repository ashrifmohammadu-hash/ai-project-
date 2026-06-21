"""
Check how well the deployed model predicts Sri Lanka crop folders.

Run from plant-disease-backend:
  python evaluate_sl_crops.py

Uses random samples from Merge-Project/dataset_sri_lanka (must exist).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

from predict import predict_disease

DATASET = Path(__file__).resolve().parent.parent / "Merge-Project" / "dataset_sri_lanka"
SL_PREFIXES = ("Banana", "Rice", "Coconut", "Tea", "Chili", "Mango", "Papaya")
SAMPLES_PER_CLASS = 8
EVAL_OUTPUT = Path(__file__).resolve().parent / "eval_output"


def main() -> int:
    if not DATASET.is_dir():
        print(f"Dataset not found: {DATASET}")
        return 1

    random.seed(42)
    summary: dict[str, tuple[int, int]] = {}
    wrong: list[tuple[str, str, float]] = []
    all_true: list[str] = []
    all_pred: list[str] = []

    for folder in sorted(DATASET.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.startswith(SL_PREFIXES):
            continue
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not images:
            continue
        sample = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
        ok = 0
        for img in sample:
            pred, conf, *_ = predict_disease(img.read_bytes())
            predicted_label = pred or "?"
            all_true.append(folder.name)
            all_pred.append(predicted_label)
            if predicted_label == folder.name:
                ok += 1
            else:
                wrong.append((folder.name, predicted_label, conf or 0.0))
        summary[folder.name] = (ok, len(sample))

    total_ok = sum(v[0] for v in summary.values())
    total = sum(v[1] for v in summary.values())
    pct = 100.0 * total_ok / total if total else 0.0

    print("=" * 60)
    print("Sri Lanka crop prediction check (API + rules)")
    print(f"Dataset: {DATASET}")
    print(f"Accuracy: {total_ok}/{total} ({pct:.1f}%)")
    print("=" * 60)

    weak = [k for k, v in summary.items() if v[0] < v[1]]
    if weak:
        print(f"\nClasses with errors ({len(weak)}):")
        for name in weak:
            o, t = summary[name]
            print(f"  {name}: {o}/{t}")

    if wrong:
        print("\nSample misclassifications (first 20):")
        for true_label, pred, conf in wrong[:20]:
            print(f"  {true_label} -> {pred} ({conf}%)")

    # ── Confusion matrix ──────────────────────────────────────────────
    class_names = sorted(set(all_true + all_pred))
    # Filter to only classes that were evaluated
    present_classes = sorted(set(all_true))
    y_true = [present_classes.index(t) for t in all_true]
    y_pred = [present_classes.index(p) for p in all_pred]

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=present_classes,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("Classification Report (SL crops):")
    print(report)

    EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    report_path = EVAL_OUTPUT / "classification_report_sl_crops.txt"
    with open(report_path, "w") as f:
        f.write("SRI LANKA CROPS EVALUATION — CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"Report saved -> {report_path}")

    if HAS_VIZ:
        cm_path = EVAL_OUTPUT / "confusion_matrix_sl_crops.png"
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=present_classes, yticklabels=present_classes,
            linewidths=0.3,
        )
        plt.title("Confusion Matrix — Sri Lanka Crops", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved -> {cm_path}")
    else:
        print("matplotlib/seaborn not available — install with: pip install matplotlib seaborn")

    print("\nNote: Coconut/Chili trained on proxy images — retrain with real")
    print("photos for best accuracy. See TRAINING_ACCURACY_REPORT.md")
    return 0 if pct >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
