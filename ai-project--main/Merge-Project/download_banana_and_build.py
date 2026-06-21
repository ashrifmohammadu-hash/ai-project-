"""
Download public banana leaf images and add them to the training folder.

Sources (CC / research use):
  - PurnaChandar26/Banana-Leaf-Disease (GitHub) — Health, Sigatoka, Xanthomonas

Run:
  cd Merge-Project
  python download_banana_and_build.py
"""

from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_datasets" / "banana_mendeley"
TRAIN_ZIP_URL = (
    "https://raw.githubusercontent.com/PurnaChandar26/"
    "Banana-Leaf-Disease/main/training.zip"
)
IMG_SIZE = (128, 128)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Folder names must match predict.py DISEASE_INFO keys
CLASS_MAP = {
    "healthy": "Banana_healthy",
    "segatoka": "Banana_Sigatoka",
    "xamthomonas": "Banana_Xanthomonas_wilt",
    "xanthomonas": "Banana_Xanthomonas_wilt",
}


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_training_zip() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    zp = RAW / "training.zip"
    if zp.exists() and zp.stat().st_size > 1_000_000:
        print(f"Using cached {zp}")
        return zp
    print("Downloading banana training.zip ...")
    req = urllib.request.Request(TRAIN_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=180) as resp:
        zp.write_bytes(resp.read())
    print(f"Saved {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    return zp


def extract_zip(zp: Path) -> Path:
    out = RAW / "extracted"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(out)
    return out


def write_class_images(src_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src_dir.rglob("*")):
        if path.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        out = dest_dir / f"ban_{count:05d}.jpg"
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        count += 1
    return count


def install_into(dataset_root: Path, extracted: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    train_root = extracted / "training"
    if not train_root.is_dir():
        # flat layout fallback
        train_root = extracted
    for src_name, dest_name in CLASS_MAP.items():
        src = train_root / src_name
        if not src.is_dir():
            continue
        dest = dataset_root / dest_name
        if dest.exists():
            shutil.rmtree(dest)
        n = write_class_images(src, dest)
        stats[dest_name] = n
        print(f"  {dest_name}: {n} images")
    return stats


def main() -> None:
    target = ROOT / "resized_merged"
    if not target.is_dir():
        raise SystemExit(f"Missing {target} — run PlantVillage resize first.")

    zp = download_training_zip()
    extracted = extract_zip(zp)
    print(f"\nInstalling banana classes into {target} ...")
    stats = install_into(target, extracted)
    if not stats:
        raise SystemExit("No banana images installed — check zip layout.")

    total = sum(stats.values())
    print(f"\nDone. Added {total} banana images in {len(stats)} classes.")
    print("Next:")
    print("  1. Edit config.py: AUTO_DISCOVER_CLASSES = True")
    print("  2. python train.py")
    print("  3. copy output/*.pkl to plant-disease-backend/models/")


if __name__ == "__main__":
    main()
