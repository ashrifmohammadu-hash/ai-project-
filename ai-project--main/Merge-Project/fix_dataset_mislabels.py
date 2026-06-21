"""
Fix known proxy mislabels and duplicate classes in resized_merged / dataset_sri_lanka.

Problems fixed:
  - Coconut_* copied from Papaya_* (same images, two labels → low F1 for both)
  - Chili_* copied from Pepper,_bell_* while Pepper also in dataset
  - Coconut rebuilt from long-leaf Corn/Grape (distinct from papaya oval leaves)

Run:
  python fix_dataset_mislabels.py
  python build_dataset_sri_lanka.py
  python train.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "resized_merged"
IMG_SIZE = (128, 128)
MAX_PER_CLASS = 400

# Coconut from long-leaf crops (NOT papaya)
COCONUT_FROM_PLANTVILLAGE = {
    "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot": "Coconut_Gray_leaf_spot",
    "Corn_(maize)_Northern_Leaf_Blight": "Coconut_Leaf_rot",
    "Corn_(maize)_healthy": "Coconut_healthy",
}

# Chili: keep pepper-based but remove duplicate Pepper folders from SL build later
CHILI_FROM_PEPPER = {
    "Pepper,_bell_healthy": "Chili_healthy",
    "Pepper,_bell_Bacterial_spot": "Chili_Bacterial_spot",
}

COCONUT_CLASSES = tuple(COCONUT_FROM_PLANTVILLAGE.values())
CHILI_CLASSES = tuple(CHILI_FROM_PEPPER.values())


def _write_images(src_dir: Path, dest_class: Path, prefix: str, limit: int) -> int:
    dest_class.mkdir(parents=True, exist_ok=True)
    for old in dest_class.glob(f"{prefix}_*.jpg"):
        old.unlink()
    count = 0
    for path in sorted(src_dir.rglob("*")):
        if count >= limit:
            break
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        out = dest_class / f"{prefix}_{count:05d}.jpg"
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        count += 1
    return count


def _rebuild_coconut() -> dict[str, int]:
    print("\n=== Rebuild COCONUT (corn long-leaf proxy, NOT papaya) ===")
    stats: dict[str, int] = {}
    for src_name, dst_name in COCONUT_FROM_PLANTVILLAGE.items():
        src = TARGET / src_name
        dst = TARGET / dst_name
        if not src.is_dir():
            print(f"  SKIP missing source: {src_name}")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        n = _write_images(src, dst, "coconut", MAX_PER_CLASS)
        stats[dst_name] = n
        print(f"  {dst_name}: {n} <- {src_name}")
    return stats


def _rebuild_chili() -> dict[str, int]:
    print("\n=== Rebuild CHILI (pepper proxy — Pepper removed from SL build) ===")
    stats: dict[str, int] = {}
    for src_name, dst_name in CHILI_FROM_PEPPER.items():
        src = TARGET / src_name
        dst = TARGET / dst_name
        if not src.is_dir():
            print(f"  SKIP missing source: {src_name}")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        n = _write_images(src, dst, "chili", MAX_PER_CLASS)
        stats[dst_name] = n
        print(f"  {dst_name}: {n} <- {src_name}")
    return stats


def main() -> None:
    if not TARGET.is_dir():
        raise SystemExit(f"Missing {TARGET} — need PlantVillage resized_merged first.")

    coconut = _rebuild_coconut()
    chili = _rebuild_chili()

    print("\n" + "=" * 55)
    print("  Fixed proxy labels in resized_merged")
    print("=" * 55)
    for k, v in {**coconut, **chili}.items():
        print(f"  {k}: {v}")
    print("\nNext:")
    print("  python build_dataset_sri_lanka.py")
    print("  python train.py")


if __name__ == "__main__":
    main()
