import os
import random
import shutil
from PIL import Image, ImageEnhance
import numpy as np

SOURCE = r"D:\ai data\Final\Merge-Project\dataset_sri_lanka"
TARGET = r"D:\ai data\Final\Merge-Project\dataset_balanced"
CAP = 400
MIN_AUGMENT = 200
SKIP = {"Coconut_healthy"}  # 0 images, useless

os.makedirs(TARGET, exist_ok=True)

def augment_image(img):
    """Apply random augmentation and return new image."""
    # Random flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    # Random rotation
    angle = random.randint(-30, 30)
    img = img.rotate(angle)
    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img

classes = sorted(os.listdir(SOURCE))
total_classes = 0
total_images = 0

for cls in classes:
    src_cls = os.path.join(SOURCE, cls)
    if not os.path.isdir(src_cls):
        continue
    if cls in SKIP:
        print(f"  SKIP   {cls}")
        continue

    images = [f for f in os.listdir(src_cls)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    tgt_cls = os.path.join(TARGET, cls)
    os.makedirs(tgt_cls, exist_ok=True)

    # --- CAP large classes ---
    if len(images) > CAP:
        selected = random.sample(images, CAP)
        for f in selected:
            shutil.copy2(os.path.join(src_cls, f), os.path.join(tgt_cls, f))
        print(f"  CAP    {cls}: {len(images)} → {CAP}")
        total_images += CAP

    # --- AUGMENT small classes ---
    elif len(images) < MIN_AUGMENT:
        # Copy originals first
        for f in images:
            shutil.copy2(os.path.join(src_cls, f), os.path.join(tgt_cls, f))
        # Generate augmented images until MIN_AUGMENT
        needed = MIN_AUGMENT - len(images)
        aug_count = 0
        while aug_count < needed:
            src_file = random.choice(images)
            try:
                img = Image.open(os.path.join(src_cls, src_file)).convert("RGB")
                img = augment_image(img)
                aug_name = f"aug_{aug_count:04d}_{src_file}"
                img.save(os.path.join(tgt_cls, aug_name))
                aug_count += 1
            except Exception as e:
                print(f"    Warning: {e}")
        print(f"  AUG    {cls}: {len(images)} → {len(images)+aug_count}")
        total_images += len(images) + aug_count

    # --- KEEP as-is (already in range) ---
    else:
        for f in images:
            shutil.copy2(os.path.join(src_cls, f), os.path.join(tgt_cls, f))
        print(f"  OK     {cls}: {len(images)}")
        total_images += len(images)

    total_classes += 1

print()
print(f"Done! {total_classes} classes, {total_images} total images")
print(f"Saved to: {TARGET}")