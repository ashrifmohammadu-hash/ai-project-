"""
Build dataset_sri_lanka = banana (downloaded) + Sri Lanka–relevant PlantVillage crops.

Run after download_banana_and_build.py or standalone (downloads banana if needed).
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "resized_merged"
OUT = ROOT / "dataset_sri_lanka"

# Crops useful in Sri Lanka that exist in PlantVillage resized_merged
SL_CLASS_PREFIXES = (
    "Banana_",
    "Rice_",
    "Coconut_",
    "Tea_",
    "Chili_",
    "Mango_",
    "Papaya_",
    "Tomato_",
    # Pepper excluded — Chili_* already uses pepper images (avoid duplicate labels)
    "Potato_",
    "Corn_(maize)_",
    "Grape_",
    "Orange_",
    "Apple_",
    "Peach_",
    "Cherry_(including_sour)_",
    "Strawberry_",
    "Squash_",
    "Soybean_",
    "Blueberry_",
    "Raspberry_",
)


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"Missing {SRC}")

    # Ensure banana folders exist in resized_merged first
    banana_dirs = [p for p in SRC.iterdir() if p.is_dir() and p.name.startswith("Banana_")]
    if len(banana_dirs) < 2:
        print("Banana folders missing — running download_banana_and_build.py ...")
        import download_banana_and_build

        download_banana_and_build.main()
        banana_dirs = [p for p in SRC.iterdir() if p.is_dir() and p.name.startswith("Banana_")]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    copied = 0
    classes = 0
    for folder in sorted(SRC.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if not any(name.startswith(p) for p in SL_CLASS_PREFIXES):
            continue
        dest = OUT / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        n = 0
        for f in folder.iterdir():
            if f.is_dir() or f.name.startswith("."):
                continue
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            shutil.copy2(f, dest / f.name)
            n += 1
        copied += n
        classes += 1
        print(f"  {name}: {n}")

    print(f"\nBuilt {OUT}")
    print(f"  Classes: {classes}")
    print(f"  Images : {copied}")
    print("\nSet config.py:")
    print('  DATASET_PATH = "dataset_sri_lanka"')
    print("  AUTO_DISCOVER_CLASSES = True")


if __name__ == "__main__":
    main()
