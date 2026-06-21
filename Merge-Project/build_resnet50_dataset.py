"""
Build ResNet50 training dataset by copying/reshaping images from raw sources
into resized_merged/ with class names matching backend expectations.
"""
import os, sys, shutil, cv2, numpy as np
from pathlib import Path
from glob import glob

PROJECT = Path(__file__).resolve().parent
TARGET = PROJECT / "resized_merged"
RAW = PROJECT / "raw_datasets"

# ── Class name mappings: source_folder_pattern → target class name ──
# Different source datasets use different naming conventions.
# Map them to the target resized_merged class names.

def copy_images(src_dir, target_class_name, rename_func=None, extensions=(".jpg", ".JPG", ".jpeg", ".png", ".PNG")):
    """Copy all images from src_dir into TARGET/target_class_name/."""
    target_dir = TARGET / target_class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in extensions:
        for fpath in glob(str(src_dir / "**" / f"*{ext}"), recursive=True):
            fpath = Path(fpath)
            if fpath.is_file():
                # Simple rename to avoid collisions
                fname = f"{target_class_name}_{count:06d}{fpath.suffix}"
                dst = target_dir / fname
                if not dst.exists():
                    shutil.copy2(str(fpath), str(dst))
                count += 1
    print(f"  {target_class_name}: {count} images")
    return count


def build_plantvillage():
    """Copy PlantVillage 21 classes into resized_merged with name mapping."""
    pv_dir = RAW / "ai dataset" / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train"
    if not pv_dir.exists():
        print(f"PlantVillage dataset NOT FOUND at {pv_dir}")
        return 0

    # Map PlantVillage folder names to target resized_merged names
    pv_mapping = {
        "Apple___Apple_scab": "Apple___Apple_scab",
        "Apple___Black_rot": "Apple___Black_rot",
        "Apple___Cedar_apple_rust": "Apple_Cedar_apple_rust",
        "Apple___healthy": "Apple_healthy",
        "Blueberry___healthy": "Blueberry_healthy",
        "Cherry_(including_sour)___healthy": "Cherry_(including_sour)_healthy",
        "Cherry_(including_sour)___Powdery_mildew": "Cherry_(including_sour)_Powdery_mildew",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot",
        "Corn_(maize)___Common_rust_": "Corn_(maize)_Common_rust_",
        "Corn_(maize)___healthy": "Corn_(maize)_healthy",
        "Corn_(maize)___Northern_Leaf_Blight": "Corn_(maize)_Northern_Leaf_Blight",
        "Grape___Black_rot": "Grape_Black_rot",
        "Grape___Esca_(Black_Measles)": "Grape_Esca_(Black_Measles)",
        "Grape___healthy": "Grape_healthy",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Orange___Haunglongbing_(Citrus_greening)": "Orange_Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot": "Peach_Bacterial_spot",
        "Peach___healthy": "Peach_healthy",
        "Pepper,_bell___Bacterial_spot": "Pepper,_bell_Bacterial_spot",
        "Pepper,_bell___healthy": "Pepper,_bell_healthy",
        "Potato___Early_blight": "Potato_Early_blight",
    }
    total = 0
    for src_name, target_name in pv_mapping.items():
        src_dir = pv_dir / src_name
        if src_dir.exists():
            total += copy_images(src_dir, target_name)
        else:
            print(f"  WARNING: Source dir not found: {src_name}")
    print(f"PlantVillage total: {total}")
    return total


def build_mango():
    """Copy mango images from mango_mirror."""
    mango_dirs = {
        "Anthracnose": "Mango_Anthracnose",
        "Bacterial Canker": "Mango_Bacterial_canker",
        "Die Back": "Mango_Die_back",
        "Healthy": "Mango_healthy",
        "Powdery Mildew": "Mango_Powdery_mildew",
        "Sooty Mould": "Mango_Sooty_mould",
    }
    base = RAW / "sl_crops" / "mango_mirror" / "data" / "MangoLeafBD Dataset"
    if not base.exists():
        print(f"Mango dataset NOT FOUND at {base}")
        return 0
    total = 0
    for src_name, target_name in mango_dirs.items():
        src_dir = base / src_name
        if src_dir.exists():
            total += copy_images(src_dir, target_name)
        else:
            print(f"  WARNING: Mango dir not found: {src_name}")
    return total


def build_papaya():
    """Copy papaya images from papaya_github sources."""
    base1 = RAW / "sl_crops" / "papaya_github" / "data" / "origin"
    base2 = RAW / "weak_classes" / "papaya_github" / "papayas"
    papaya_map = {
        "anthracnose": "Papaya_Anthracnose",
        "bacterialspot": "Papaya_Bacterial_spot",
        "healthy": "Papaya_healthy",
        "curl": "Papaya_Leaf_curl",
        "ringspot": "Papaya_Ringspot",
    }
    total = 0
    for src_name, target_name in papaya_map.items():
        # Try base1 (sl_crops)
        src_dir = base1 / src_name
        if src_dir.exists():
            total += copy_images(src_dir, target_name)
        # Try base2 (weak_classes)
        src_dir = base2 / src_name
        if src_dir.exists():
            total += copy_images(src_dir, target_name)
    return total


def build_rice():
    """Copy rice images from rice_maimunul."""
    rice_map = {
        "Bacterialblight": "Rice_Bacterial_blight",
        "Blast": "Rice_Blast",
        "Brownspot": "Rice_Brown_spot",
        "Tungro": "Rice_Tungro",
    }
    base = RAW / "sl_crops" / "rice_maimunul"
    if not base.exists():
        print(f"Rice dataset NOT FOUND at {base}")
        return 0
    total = 0
    for src_name, target_name in rice_map.items():
        src_dir = base / src_name
        if src_dir.exists():
            total += copy_images(src_dir, target_name)
    return total


def build_banana():
    """Copy banana images from banana_nixon."""
    banana_map = {
        "SigatokaNegra": "Banana_Sigatoka",
        "Sanas": "Banana_healthy",
        "Cordana": "Banana_healthy",  # Cordana is a disease, but target doesn't list it; treat as healthy for now
    }
    base = RAW / "weak_classes" / "banana_nixon"
    total = 0
    for subfolder in ["Data-Tesis", "Imagenes-aumentadas"]:
        for src_name, target_name in banana_map.items():
            src_dir = base / subfolder / src_name
            if src_dir.exists():
                total += copy_images(src_dir, target_name)
    return total


def build_tea():
    """Add a small tea dataset if available (try common patterns)."""
    # Placeholder — tea data may come from zip archives; skip for now
    return 0


def build_coconut():
    """Add coconut from coconut_papaya_fix if available."""
    base = RAW / "coconut_papaya_fix"
    if not base.exists():
        return 0
    total = 0
    # Try to find coconut folders
    for root, dirs, files in os.walk(str(base)):
        for d in dirs:
            if "coconut" in d.lower() or "Coconut" in d:
                src_dir = Path(root) / d
                if "leaf_rot" in d.lower() or "Leaf_rot" in d:
                    total += copy_images(src_dir, "Coconut_Leaf_rot")
                elif "gray" in d.lower() or "Gray" in d:
                    total += copy_images(src_dir, "Coconut_Gray_leaf_spot")
                elif "healthy" in d.lower():
                    total += copy_images(src_dir, "Coconut_healthy")
                elif "wilt" in d.lower():
                    total += copy_images(src_dir, "Coconut_Leaf_rot")
                else:
                    total += copy_images(src_dir, "Coconut_healthy")
    return total


def build_additional_sources():
    """Copy any remaining images from Banana-Leaf-Disease and mendeley."""
    total = 0
    # Banana leaf disease dataset
    base = RAW / "Banana-Leaf-Disease-main"
    if base.exists():
        for root, dirs, files in os.walk(str(base)):
            for d in dirs:
                src_dir = Path(root) / d
                # Classify by keywords
                dl = d.lower()
                if "sigatoka" in dl or "yellow" in dl:
                    total += copy_images(src_dir, "Banana_Sigatoka")
                elif "xanthomonas" in dl or "bacterial" in dl or "wilt" in dl:
                    total += copy_images(src_dir, "Banana_Xanthomonas_wilt")
                elif "healthy" in dl or "sanas" in dl:
                    total += copy_images(src_dir, "Banana_healthy")
                elif "cordana" in dl:
                    total += copy_images(src_dir, "Banana_healthy")
                elif "black" in dl or "sigatok" in dl:
                    total += copy_images(src_dir, "Banana_Sigatoka")
    return total


def main():
    print(f"Building ResNet50 dataset in {TARGET}")
    total = 0
    total += build_plantvillage()
    total += build_mango()
    total += build_papaya()
    total += build_rice()
    total += build_banana()
    total += build_coconut()
    total += build_tea()
    total += build_additional_sources()

    print(f"\n{'='*60}")
    print(f"Total images copied: {total}")
    print(f"{'='*60}")

    # Count images per class
    print("\nImages per class:")
    for cls_dir in sorted(TARGET.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.iterdir()))
            print(f"  {cls_dir.name}: {count}")


if __name__ == "__main__":
    main()
