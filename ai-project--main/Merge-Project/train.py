# train.py
# ─────────────────────────────────────────────
# Main training script.
# Run this once → saves model, scaler, encoder,
# confusion matrix, and results report.
#
# HOW TO RUN:
#   python train.py
# ─────────────────────────────────────────────

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')                          # no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (accuracy_score,
                                      classification_report,
                                      confusion_matrix)
from dataset import load_dataset
from config  import (OUTPUT_DIR, MODEL_PATH, SCALER_PATH,
                     ENCODER_PATH, REPORT_PATH, CM_PATH,
                     TEST_SIZE, RANDOM_STATE,
                     N_ESTIMATORS, N_JOBS)

# ── 0. Create output folder ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load dataset ─────────────────────────────────────────────────────
X, y = load_dataset()

# ── 2. Encode string labels → numbers ───────────────────────────────────
le        = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nClasses found : {len(le.classes_)}")
for i, cls in enumerate(le.classes_):
    print(f"  {i:2d}  {cls}")

# ── 3. Train / Validation / Test split (70/15/15) ──────────────────────
# First split: train (70%) vs temporary (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# Second split: split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,   # because 0.5 of 30% = 15% of total
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"\nTraining samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Test samples       : {len(X_test)}")

# ── 4. Scale features ───────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit ONLY on training data
X_val   = scaler.transform(X_val)         # apply same scale to val
X_test  = scaler.transform(X_test)        # apply same scale to test

# ── 5. Train Random Forest ──────────────────────────────────────────────
print("\nTraining Random Forest ...")
print(f"  Trees (n_estimators) : {N_ESTIMATORS}")
print(f"  CPU cores (n_jobs)   : {N_JOBS} (all cores)")

model = RandomForestClassifier(
    n_estimators  = N_ESTIMATORS,
    class_weight  = 'balanced',    # handles unequal class sizes
    random_state  = RANDOM_STATE,
    n_jobs        = N_JOBS
)
model.fit(X_train, y_train)
print("Training complete!")

# ── 6. Evaluate on all sets ─────────────────────────────────────────────
train_acc = model.score(X_train, y_train) * 100

y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred) * 100

y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred) * 100

# Classification report on test set (unseen data)
report = classification_report(y_test, y_test_pred, target_names=le.classes_)

print("\n" + "=" * 55)
print(f"  Train Accuracy      : {train_acc:.2f}%")
print(f"  Validation Accuracy : {val_acc:.2f}%")
print(f"  Test Accuracy       : {test_acc:.2f}%")
if train_acc - val_acc > 15:
    print("  [WARNING] Large gap — possible overfitting!")
print("=" * 55)
print("\nPer-class report (on TEST set):")
print(report)

# ── 7. Save results to text file ────────────────────────────────────────
with open(REPORT_PATH, "w") as f:
    f.write("PLANT DISEASE DETECTION — RESULTS\n")
    f.write("=" * 55 + "\n")
    f.write(f"Algorithm          : Random Forest\n")
    f.write(f"Feature extraction : OpenCV HSV histogram + edge + stats\n")
    f.write(f"Image size         : 128 x 128 pixels\n")
    f.write(f"Training samples   : {len(X_train)}\n")
    f.write(f"Validation samples : {len(X_val)}\n")
    f.write(f"Test samples       : {len(X_test)}\n")
    f.write(f"Number of classes  : {len(le.classes_)}\n")
    f.write(f"Train Accuracy     : {train_acc:.2f}%\n")
    f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
    f.write(f"Test Accuracy      : {test_acc:.2f}%\n")
    f.write("=" * 55 + "\n\n")
    f.write(report)
print(f"Results saved -> {REPORT_PATH}")

# ── 8. Save confusion matrix (using test set predictions) ───────────────
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(20, 18))
sns.heatmap(
    cm,
    annot       = True,
    fmt         = 'd',
    cmap        = 'Greens',
    xticklabels = le.classes_,
    yticklabels = le.classes_,
    linewidths  = 0.3
)
plt.title("Confusion Matrix — Plant Disease Detection (Test Set)", fontsize=14)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label",      fontsize=11)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.close()
print(f"Confusion matrix saved -> {CM_PATH}")

# ── 9. Save model files ─────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le,     ENCODER_PATH)
print(f"Model saved   -> {MODEL_PATH}")
print(f"Scaler saved  -> {SCALER_PATH}")
print(f"Encoder saved -> {ENCODER_PATH}")

print("\nAll done! Now run:  python predict.py <image_path>")