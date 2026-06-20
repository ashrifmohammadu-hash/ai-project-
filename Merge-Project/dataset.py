
# dataset.py
# ─────────────────────────────────────────────
# Loads all images from resized_merged/ folder,
# extracts features, and returns X and y arrays
# ready for scikit-learn training.
# ─────────────────────────────────────────────

import os
import cv2
import numpy as np
from tqdm import tqdm

from features import extract_features
from config import DATASET_PATH, IMG_SIZE, CLASSES, AUTO_DISCOVER_CLASSES


def _discover_classes(root: str) -> list[str]:
    """All subfolder names that contain at least one image."""
    found = []
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        has_img = any(
            f.lower().endswith((".jpg", ".jpeg", ".png"))
            for f in os.listdir(path)
        )
        if has_img:
            found.append(name)
    return found


def load_dataset():
    """
    Returns:
        X : numpy array, shape (num_images, 103)  — feature vectors
        y : numpy array, shape (num_images,)       — disease label strings
    """
    X       = []
    y       = []
    skipped = 0

    print("=" * 55)
    print("  LOADING DATASET")
    print(f"  Source folder : {DATASET_PATH}")
    print(f"  Image size    : {IMG_SIZE[0]} x {IMG_SIZE[1]} pixels")
    if AUTO_DISCOVER_CLASSES:
        class_list = _discover_classes(DATASET_PATH)
        print(f"  Mode          : AUTO_DISCOVER_CLASSES")
    else:
        class_list = list(CLASSES)

    print(f"  Classes       : {len(class_list)}")
    print("=" * 55)

    if not class_list:
        raise RuntimeError(
            f"No classes found under {DATASET_PATH}. "
            "Create subfolders like Banana_Sigatoka/ with images, or set AUTO_DISCOVER_CLASSES."
        )

    for disease_name in class_list:
        folder_path = os.path.join(DATASET_PATH, disease_name)

        # Skip if folder does not exist
        if not os.path.isdir(folder_path):
            print(f"  [WARNING] Folder not found: {folder_path}")
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(image_files) == 0:
            print(f"  [WARNING] No images in: {disease_name}")
            continue

        # Progress bar per disease class
        for img_file in tqdm(image_files,
                             desc=f"{disease_name[:40]:40s}",
                             ncols=80):
            img_path = os.path.join(folder_path, img_file)
            img      = cv2.imread(img_path)

            if img is None:
                skipped += 1
                continue

            # Resize to standard size (already done but just in case)
            img = cv2.resize(img, IMG_SIZE)

            features = extract_features(img)
            X.append(features)
            y.append(disease_name)

    print()
    print(f"  Total images loaded : {len(X)}")
    print(f"  Total images skipped: {skipped}")
    print("=" * 55)

    return np.array(X, dtype=np.float32), np.array(y)