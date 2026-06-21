"""Augment weak papaya classes (flip + HSV jitter) to improve training balance."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent / "resized_merged"
TARGET_CLASSES = ("Papaya_Bacterial_spot", "Papaya_Leaf_curl", "Papaya_healthy")
GOAL = 400


def augment(img: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    if rng.random() > 0.5:
        out = cv2.flip(out, 1)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= float(rng.uniform(0.85, 1.15))
    hsv[:, :, 2] *= float(rng.uniform(0.9, 1.1))
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main() -> None:
    for cls in TARGET_CLASSES:
        folder = ROOT / cls
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.jpg"))
        idx = len(files)
        need = max(0, GOAL - idx)
        if need == 0:
            print(f"{cls}: already {idx}")
            continue
        added = 0
        fi = 0
        while added < need and files:
            src = files[fi % len(files)]
            img = cv2.imread(str(src))
            if img is None:
                fi += 1
                continue
            img = cv2.resize(img, (128, 128))
            out = augment(img, fi + added)
            cv2.imwrite(
                str(folder / f"papaya_aug_{idx + added:05d}.jpg"),
                out,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            added += 1
            fi += 1
        print(f"{cls}: {idx} -> {idx + added}")


if __name__ == "__main__":
    main()
