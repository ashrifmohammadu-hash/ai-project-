# Training & Accuracy Report — Honest Status

**Date:** May 2026  
**Model:** `plant-disease-backend/models/*.pkl`  
**Training folder:** `Merge-Project/dataset_sri_lanka`

---

## 1. Were all images trained?

| Question | Answer |
|----------|--------|
| All **69 classes** in dataset? | **Yes** — 69 folders |
| Total training images? | **97,652** |
| Training run completed? | **Yes** — see `Merge-Project/output/results.txt` |
| Models copied to backend? | **Yes** — `models/model.pkl`, `scaler.pkl`, `label_encoder.pkl` |

**Conclusion:** Training ran on **all class folders**. Every class had at least 50 images.

---

## 2. Is overall accuracy good?

From last training (`output/results.txt`):

| Metric | Value |
|--------|-------|
| Validation accuracy (all 69 classes) | **94.19%** |
| Train accuracy | 99.93% |

That sounds high, but **many Sri Lanka crops are still wrong in real use**.

---

## 3. Per-crop quality (validation F1 from training)

| Crop | Validation F1 (approx.) | Problem |
|------|-------------------------|---------|
| Tomato, Potato, Corn, Grape | **0.95+** | Good (PlantVillage data) |
| Mango (most classes) | **0.96+** | Good |
| Papaya | **~0.85+** | Fair |
| Rice | **~0.85+** | Fair |
| Tea | **~0.85+** | Fair |
| **Coconut** | **0.08–0.21** | **Poor** — papaya proxy images |
| **Chili** | **0.53–0.73** | **Weak** — bell pepper proxy |
| **Banana healthy** | **0.16** | **Poor** — small / confused set |

**Coconut and chili were not trained on real field photos** when downloads failed; papaya/pepper images were copied with coconut/chili folder names. The model learned the wrong visual patterns.

---

## 4. Live API test (random photos from dataset)

Run:

```powershell
cd "d:\ai data\Final\plant-disease-backend"
python evaluate_sl_crops.py
```

Typical result: **~15–40%** correct on Sri Lanka-only classes with current rules, because:

1. ML gives **0%** to coconut classes on many real coconut photos.
2. Proxy training data does not match your phone photos.
3. Code rules fix **coconut vs grape** on upload, but cannot fix all 31 SL classes without better training data.

---

## 5. What was fixed in code (not retraining)

| Issue | Fix |
|-------|-----|
| Coconut photo → **Grape** | Palm-leaf detection + coconut disease labels |
| Grape forced at 55% | Removed for non-grapevine leaves |
| ML prefers another SL crop | Trust ML top Sri Lanka crop when confidence ≥ 10% |

**Your coconut upload should show Coconut — Gray leaf spot**, not Grape, after restarting `python app.py`.

---

## 6. What you must do for “all images correct”

Code alone **cannot** reach high accuracy on coconut/chili/banana without **new real photos** and **retrain**:

1. Collect **real** coconut, chili, banana leaf photos (150+ per class).
2. Put in `Merge-Project/dataset_sri_lanka/<ClassName>/`.
3. Run `python train.py` in Merge-Project.
4. Copy 3 `.pkl` files to `plant-disease-backend/models/`.
5. Restart `python app.py`.
6. Run `python evaluate_sl_crops.py` again — aim for **>70%** per class.

---

## 7. Quick checklist

- [x] All 69 classes trained (yes)
- [x] Overall val accuracy 94% (yes, but misleading for SL proxies)
- [ ] All crops correct on your phone photos (**no** — need real data + retrain)
- [x] Coconut vs Grape on upload (**fixed in code** — restart backend)

---

*For steps see `d:\ai data\Final\MANUAL_TRAIN_AND_TEST.md`*
