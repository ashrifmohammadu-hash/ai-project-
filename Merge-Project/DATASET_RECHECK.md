# Training dataset — re-check & fix mislabels

## 1. Automated scan (folder vs current model)

From `Merge-Project` (after `python train.py` so `output/model.pkl` exists):

```powershell
cd "D:\ai data\Final\Merge-Project"
..\Merge-Project\plant_env\Scripts\python.exe recheck_dataset_mislabels.py
```

Outputs:

- `output/dataset_mislabel_candidates.csv` — images where **folder name ≠ model prediction** (optional filter `--min-conf 0.55`)
- `output/dataset_recheck_report.txt` — summary + top confused pairs

**Optional:** copy strong disagreements for visual review (does not delete originals):

```powershell
..\Merge-Project\plant_env\Scripts\python.exe recheck_dataset_mislabels.py --quarantine 0.72
```

Review files under `output/quarantine_review/`.

## 2. Known structural fixes (proxy folders)

```powershell
..\Merge-Project\plant_env\Scripts\python.exe fix_dataset_mislabels.py
..\Merge-Project\plant_env\Scripts\python.exe build_dataset_sri_lanka.py
```

## 3. Per-class quality (F1 from last train)

```powershell
..\Merge-Project\plant_env\Scripts\python.exe audit_training_labels.py
```

## 4. Retrain & deploy model

```powershell
..\Merge-Project\plant_env\Scripts\python.exe train.py
```

Copy `output/model.pkl`, `scaler.pkl`, `label_encoder.pkl` to `plant-disease-backend/models/`.

---

**Note:** The RF model can be wrong on individual images. Use the CSV + your eyes to move/delete misfiled photos; do not blindly trust every high-confidence disagreement.
