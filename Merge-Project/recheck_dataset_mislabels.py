"""
Re-check every training image: folder label vs current Random Forest prediction.

Use this after training to find misfiled images (wrong subfolder). The model is not
ground truth - use the CSV + quarantine folder for manual review, then delete bad
copies or move files to the correct class folder.

Run (from Merge-Project):
  python recheck_dataset_mislabels.py
  python recheck_dataset_mislabels.py --quarantine 0.72   # move strong disagreements for review

After fixing folders / removing bad images:
  python train.py
  Copy output/*.pkl to plant-disease-backend/models/
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np

from config import DATASET_PATH, IMG_SIZE, MODEL_PATH, SCALER_PATH, ENCODER_PATH
from features import extract_features

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _class_prefix(label: str) -> str:
    """Crop prefix e.g. Mango from Mango_Anthracnose."""
    if "_" not in label:
        return label
    parts = label.split("_")
    if parts[0] in {"Corn", "Cherry", "Pepper"}:
        if label.startswith("Corn_(maize)_"):
            return "Corn_(maize)"
        if label.startswith("Cherry_(including_sour)_"):
            return "Cherry_(including_sour)"
        if label.startswith("Pepper,_bell_"):
            return "Pepper,_bell"
    return parts[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit training images vs model predictions")
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help="Dataset root (default: config DATASET_PATH)",
    )
    ap.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Only list rows where predicted class probability >= this (0-1)",
    )
    ap.add_argument(
        "--quarantine",
        type=float,
        default=None,
        metavar="MIN_PROBA",
        help="If set, copy disagreeing images (pred proba >= MIN_PROBA) to output/quarantine_review/",
    )
    args = ap.parse_args()

    ds_root = ROOT / args.dataset
    model_p = ROOT / MODEL_PATH
    scaler_p = ROOT / SCALER_PATH
    enc_p = ROOT / ENCODER_PATH

    if not model_p.is_file():
        raise SystemExit(f"Missing model: {model_p} - run train.py first.")
    if not scaler_p.is_file():
        raise SystemExit(f"Missing scaler: {scaler_p}")
    if not enc_p.is_file():
        raise SystemExit(f"Missing encoder: {enc_p}")
    if not ds_root.is_dir():
        raise SystemExit(f"Missing dataset: {ds_root}")

    model = joblib.load(model_p)
    scaler = joblib.load(scaler_p)
    le = joblib.load(enc_p)
    classes = le.classes_
    # Single-thread inference: per-image predict_proba with n_jobs>1 is very slow
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except (ValueError, TypeError):
            pass

    # Pass 1: load features (paths that fail stay as READ_ERROR)
    batch_paths: list[Path] = []
    batch_folder: list[str] = []
    batch_feats: list[np.ndarray] = []
    read_errors: list[dict] = []

    for folder in sorted(ds_root.iterdir()):
        if not folder.is_dir():
            continue
        folder_label = folder.name
        if folder_label.startswith(".") or folder_label == "quarantine_review":
            continue

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                read_errors.append(
                    {
                        "path": str(img_path.relative_to(ROOT)),
                        "folder_label": folder_label,
                        "predicted": "READ_ERROR",
                        "proba": 0.0,
                        "top2": "",
                        "top2_proba": 0.0,
                        "crop_match": "no",
                    }
                )
                continue
            img = cv2.resize(img, IMG_SIZE)
            feat = extract_features(img)
            batch_paths.append(img_path)
            batch_folder.append(folder_label)
            batch_feats.append(feat)

    CHUNK = 512
    rows: list[dict] = []
    by_folder: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    disagree_crop: dict[tuple[str, str], int] = defaultdict(int)

    for start in range(0, len(batch_feats), CHUNK):
        end = min(start + CHUNK, len(batch_feats))
        X = np.stack(batch_feats[start:end], axis=0)
        Xs = scaler.transform(X)
        probas = model.predict_proba(Xs)
        for j, row_idx in enumerate(range(start, end)):
            img_path = batch_paths[row_idx]
            folder_label = batch_folder[row_idx]
            proba = probas[j]
            order = np.argsort(proba)[::-1]
            top_i = int(order[0])
            top2_i = int(order[1]) if len(order) > 1 else top_i
            pred = str(classes[top_i])
            p1 = float(proba[top_i])
            p2 = float(proba[top2_i])
            crop_folder = _class_prefix(folder_label)
            crop_pred = _class_prefix(pred)
            crop_match = "yes" if crop_folder == crop_pred else "no"

            by_folder[folder_label][pred] += 1
            if pred != folder_label:
                disagree_crop[(folder_label, pred)] += 1

            rec = {
                "path": str(img_path.relative_to(ROOT)),
                "folder_label": folder_label,
                "predicted": pred,
                "proba": round(p1, 4),
                "top2": str(classes[top2_i]),
                "top2_proba": round(p2, 4),
                "crop_match": crop_match,
            }
            rows.append(rec)

            if (
                args.quarantine is not None
                and pred != folder_label
                and p1 >= float(args.quarantine)
            ):
                qdir = OUT_DIR / "quarantine_review" / folder_label / f"pred_{pred}"
                qdir.mkdir(parents=True, exist_ok=True)
                dest = qdir / img_path.name
                if not dest.exists():
                    shutil.copy2(img_path, dest)

    rows.extend(read_errors)
    disagree = [r for r in rows if r["predicted"] != r["folder_label"]]
    if args.min_conf > 0:
        disagree = [r for r in disagree if r["proba"] >= args.min_conf]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "dataset_mislabel_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "folder_label",
                "predicted",
                "proba",
                "top2",
                "top2_proba",
                "crop_match",
            ],
        )
        w.writeheader()
        for r in rows:
            if r["predicted"] == r["folder_label"]:
                continue
            if r["proba"] < args.min_conf:
                continue
            w.writerow(r)

    report_path = OUT_DIR / "dataset_recheck_report.txt"
    lines = [
        "DATASET RE-CHECK (folder label vs trained RF prediction)",
        "=" * 60,
        f"Dataset: {ds_root}",
        f"Model:   {model_p}",
        f"Total images scanned: {len(rows)}",
        f"Folder != predicted (any): {sum(1 for r in rows if r['predicted'] != r['folder_label'])}",
        f"Folder != predicted (proba >= {args.min_conf}): {len(disagree)}",
        "",
        "--- Top folder->predicted confusion counts (same crop?) ---",
    ]
    pairs = sorted(disagree_crop.items(), key=lambda x: -x[1])[:40]
    for (true_l, pred_l), n in pairs:
        same_crop = "same crop" if _class_prefix(true_l) == _class_prefix(pred_l) else "DIFF CROP"
        lines.append(f"  {n:5d}  {true_l}  ->  {pred_l}  ({same_crop})")

    lines.extend(
        [
            "",
            "--- Per-folder: most common predicted label (when pred != folder) ---",
        ]
    )
    for fl in sorted(by_folder.keys())[:80]:
        counts = by_folder[fl]
        if len(counts) == 1 and fl in counts:
            continue
        top_preds = sorted(counts.items(), key=lambda x: -x[1])[:5]
        lines.append(f"  {fl}:")
        for pred, n in top_preds:
            mark = "*" if pred != fl else ""
            lines.append(f"       {n:5d}  ->  {pred}  {mark}")

    lines.extend(
        [
            "",
            "NEXT STEPS",
            "  1. Open output/dataset_mislabel_candidates.csv in Excel.",
            "  2. Visually verify rows with high 'proba' - move image to correct folder or delete.",
            "  3. Run: python fix_dataset_mislabels.py  (fixes known proxy folders in resized_merged)",
            "  4. Run: python build_dataset_sri_lanka.py  (if you use SL build)",
            "  5. Run: python train.py  then copy output/*.pkl to plant-disease-backend/models/",
        ]
    )
    if args.quarantine is not None:
        lines.append(
            f"  Quarantine copies (pred proba >= {args.quarantine}): {OUT_DIR / 'quarantine_review'}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:25]))
    print(f"\n... full report -> {report_path}")
    print(f"CSV (disagreements only) -> {csv_path}")


if __name__ == "__main__":
    main()
