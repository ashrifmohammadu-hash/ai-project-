"""
Train MobileNetV3-Large on the merged plant disease dataset.
Saves model state_dict to the path expected by the backend.

Usage:
    python train_mobilenet.py                        # full training
    python train_mobilenet.py --quick                # 5 epochs (test run)
    python train_mobilenet.py --epochs 10            # custom epochs
"""
import os
import sys
import argparse
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    _tqdm = None
    HAS_TQDM = False

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "resized_merged"
MODEL_DIR = BASE / "output"
MODEL_PATH = MODEL_DIR / "plant_disease_mobilenet.pth"
OUTPUT_DIR = BASE / "output"

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
INITIAL_LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 7
MIN_LR = 1e-6

LOW_DATA_THRESHOLD = 500


class CustomTrainDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, heavy_classes, heavy_transform, light_transform):
        self.base = base_dataset
        self.indices = indices
        self.heavy_classes = heavy_classes
        self.heavy_transform = heavy_transform
        self.light_transform = light_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, y = self.base.samples[self.indices[idx]]
        image = datasets.folder.default_loader(path)
        if self.base.classes[y] in self.heavy_classes:
            image = self.heavy_transform(image)
        else:
            image = self.light_transform(image)
        return image, y


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, y = self.base.samples[self.indices[idx]]
        image = datasets.folder.default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, y


def get_training_transforms(augment_heavy=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment_heavy:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def count_images_per_class(dataset):
    counts = {}
    for _, y in dataset.samples:
        name = dataset.classes[y]
        counts[name] = counts.get(name, 0) + 1
    return counts


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    if HAS_TQDM:
        pbar = _tqdm(loader, desc=f"Epoch {epoch}")
    else:
        pbar = loader

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if HAS_TQDM:
            pbar.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large on plant disease dataset")
    parser.add_argument("--quick", action="store_true", help="Quick test run (5 epochs, small subset)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if not DATA_DIR.is_dir():
        print(f"ERROR: Dataset directory not found: {DATA_DIR}")
        sys.exit(1)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading dataset from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(str(DATA_DIR))
    n_classes = len(full_dataset.classes)
    if "test" in full_dataset.classes:
        test_idx = full_dataset.class_to_idx["test"]
        full_dataset.samples = [s for s in full_dataset.samples if s[1] != test_idx]
        full_dataset.targets = [s[1] for s in full_dataset.samples]
        full_dataset.class_to_idx = {k: i for i, (k, v) in enumerate(
            [(c, i) for c, i in full_dataset.class_to_idx.items() if c != "test"]
        )}
        full_dataset.classes = [c for c in full_dataset.classes if c != "test"]
        print(f"Removed 'test' folder")
    n_classes = len(full_dataset.classes)
    print(f"Classes: {n_classes}")
    print(f"Total images: {len(full_dataset.samples)}")

    from sklearn.model_selection import train_test_split
    targets = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        range(len(full_dataset.samples)),
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )

    counts = count_images_per_class(full_dataset)
    low_data_classes = {c for c, n in counts.items() if n < LOW_DATA_THRESHOLD}
    print(f"Low-data classes (< {LOW_DATA_THRESHOLD} images): {len(low_data_classes)}")

    train_transform_heavy, val_transform = get_training_transforms()
    train_transform_light = get_training_transforms(augment_heavy=False)[0]

    train_dataset = CustomTrainDataset(full_dataset, train_idx, low_data_classes,
                                       train_transform_heavy, train_transform_light)
    val_dataset = ValDataset(full_dataset, val_idx, val_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_counts = {}
    for idx in train_idx:
        y = full_dataset.samples[idx][1]
        cls_name = full_dataset.classes[y]
        train_counts[cls_name] = train_counts.get(cls_name, 0) + 1

    weights = [0.0] * len(train_idx)
    for i, idx in enumerate(train_idx):
        y = full_dataset.samples[idx][1]
        cls_name = full_dataset.classes[y]
        weights[i] = 1.0 / train_counts[cls_name]

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\nBuilding MobileNetV3-Large with ImageNet pretrained weights...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)

    model = model.to(device)

    class_counts = [0.0] * n_classes
    for idx in train_idx:
        y = full_dataset.samples[idx][1]
        class_counts[y] += 1.0
    class_weights = [max(class_counts) / c if c > 0 else 1.0 for c in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=MIN_LR
    )

    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"Batch size: {BATCH_SIZE}, Initial LR: {INITIAL_LR}")
    print(f"{'='*60}\n")

    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("Using AMP (mixed precision) for faster training")

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  *** New best validation accuracy: {best_val_acc:.2f}% ***")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {PATIENCE})")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"\nModel saved to: {MODEL_PATH}")

    full_path = MODEL_DIR / "plant_disease_mobilenet_full.pth"
    torch.save(model, str(full_path))
    print(f"Full model saved to: {full_path}")

    print("\n--- Per-class validation accuracy ---")
    model.eval()
    class_correct = [0] * n_classes
    class_total = [0] * n_classes
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(labels.size(0)):
                class_total[labels[i]] += 1
                if predicted[i] == labels[i]:
                    class_correct[labels[i]] += 1

    for i, cls_name in enumerate(full_dataset.classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            marker = " *** LOW ***" if acc < 60 else ""
            print(f"  {cls_name:45s} {acc:5.1f}% ({class_correct[i]:3d}/{class_total[i]:3d}){marker}")

    print("\n--- Generating confusion matrix ---")
    all_true = []
    all_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_true, all_pred)
    class_names = full_dataset.classes

    report = classification_report(all_true, all_pred, target_names=class_names)
    print("\nClassification report (validation set):")
    print(report)

    report_path = OUTPUT_DIR / "classification_report_mobilenet.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("MOBILENET V3 LARGE — CLASSIFICATION REPORT (Validation Set)\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"Classification report saved -> {report_path}")

    if HAS_VIZ:
        cm_path = OUTPUT_DIR / "confusion_matrix_mobilenet.png"
        plt.figure(figsize=(24, 22))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.3,
        )
        plt.title("Confusion Matrix — MobileNetV3-Large (Validation Set)", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved -> {cm_path}")
    else:
        print("matplotlib/seaborn not available — skipping confusion matrix plot")

    print(f"\nDone. Model ready at: {MODEL_PATH}")
    print("Restart the Flask backend to load the new model.")


if __name__ == "__main__":
    main()
