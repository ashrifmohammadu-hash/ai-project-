"""
Resize raw Sri Lanka leaf photos into the folder layout required for train.py.

Example:
  python prepare_sri_lanka_dataset.py ^
    --input "D:\Photos\raw_banana" ^
    --output dataset_sri_lanka ^
    --class-name Banana_Sigatoka
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

IMG_SIZE = (128, 128)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def resize_folder(src: Path, dest_class_dir: Path) -> int:
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        out = dest_class_dir / f"{src.name}_{count:05d}.jpg"
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Sri Lanka training folders")
    parser.add_argument("--input", required=True, help="Folder of images OR parent of class folders")
    parser.add_argument("--output", default="dataset_sri_lanka", help="Output dataset root")
    parser.add_argument(
        "--class-name",
        help="If input is one class only, set folder name e.g. Banana_Sigatoka",
    )
    args = parser.parse_args()

    src = Path(args.input)
    out_root = Path(args.output)
    if not src.is_dir():
        raise SystemExit(f"Input not found: {src}")

    total = 0
    if args.class_name:
        n = resize_folder(src, out_root / args.class_name)
        print(f"{args.class_name}: {n} images")
        total += n
    else:
        # Expect input/Crop_Disease/*.jpg OR input/Banana_Sigatoka/*.jpg
        subdirs = [p for p in src.iterdir() if p.is_dir()]
        if not subdirs:
            raise SystemExit("No subfolders in input. Use --class-name for a single folder.")
        for sub in subdirs:
            n = resize_folder(sub, out_root / sub.name)
            print(f"{sub.name}: {n} images")
            total += n

    print(f"Done. Total images: {total} -> {out_root.resolve()}")
    print("Next: set DATASET_PATH in config.py and run python train.py")


if __name__ == "__main__":
    main()
