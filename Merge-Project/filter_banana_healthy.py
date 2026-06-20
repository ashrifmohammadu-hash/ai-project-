"""
Remove likely diseased images mislabeled as Banana_healthy (improves F1).

Heuristic: healthy banana leaves have more green and less brown/yellow damage.

Run:
  python filter_banana_healthy.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "resized_merged" / "Banana_healthy"
QUARANTINE = FOLDER / "_quarantine_diseased"


def damage_score(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = ((h >= 25) & (h <= 95) & (s >= 30) & (v >= 30)).astype(np.float32)
    brown = ((h >= 5) & (h <= 25) & (s >= 40) & (v >= 40)).astype(np.float32)
    yellow = ((h >= 15) & (h <= 35) & (s >= 50) & (v >= 80)).astype(np.float32)
    total = max(green.size, 1)
    return float(brown.sum() + yellow.sum()) / total


def main() -> None:
    if not FOLDER.is_dir():
        print(f"Missing {FOLDER}")
        return
    QUARANTINE.mkdir(exist_ok=True)
    moved = 0
    kept = 0
    for path in sorted(FOLDER.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        dmg = damage_score(img)
        green_ratio = 1.0 - dmg
        # Diseased: lots of brown/yellow streaks
        if dmg > 0.22 or green_ratio < 0.45:
            shutil.move(str(path), str(QUARANTINE / path.name))
            moved += 1
        else:
            kept += 1
    print(f"Banana_healthy: kept {kept}, quarantined {moved} (likely diseased)")
    print(f"  Review: {QUARANTINE}")


if __name__ == "__main__":
    main()
