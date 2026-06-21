# Download guide — weak classes (F1 < 80%)

## Automatic (run on PC)

```powershell
cd "d:\ai data\Final\Merge-Project"
.\plant_env\Scripts\Activate.ps1
python download_weak_classes.py
python filter_banana_healthy.py
python build_dataset_sri_lanka.py
python train.py
copy output\*.pkl "..\plant-disease-backend\models\"
```

## What downloads automatically

| Class | Source | Status |
|-------|--------|--------|
| Banana_* | GitHub `PurnaChandar26/Banana-Leaf-Disease` training.zip | Usually works |
| Papaya_* | GitHub papaya_leaf_disease_classification | Works |
| Tea_Anthracnose | Hugging Face `yunusserhat/tea_sickness_dataset` | Works |
| Coconut_* | Mendeley `gh56wbsnj5` | Often **403** from script — manual ZIP needed |
| BananaLSD | Mendeley `9tb7k297ff` | Often **403** — manual or Kaggle mirror |

## Manual coconut (required for real palm photos)

1. Open: https://data.mendeley.com/datasets/gh56wbsnj5/1  
2. Click **Download All** (CC BY 4.0)  
3. Extract ZIP to: `Merge-Project\raw_datasets\weak_classes\coconut_manual\`  
4. Run:

```powershell
python -c "
from pathlib import Path
import download_sl_crops as sl
sl.TARGET = Path('resized_merged')
ext = Path('raw_datasets/weak_classes/coconut_manual')
for c in ('Coconut_Gray_leaf_spot','Coconut_Leaf_rot','Coconut_healthy'):
    import shutil
    p = sl.TARGET / c
    if p.exists(): shutil.rmtree(p)
sl._install_from_tree(ext, sl.COCONUT_MAP, 'coconut')
"
python build_dataset_sri_lanka.py
python train.py
```

## Kaggle BananaLSD (optional)

1. https://www.kaggle.com/datasets/shifatearman/bananalsd  
2. Extract under `raw_datasets\weak_classes\bananalsd_kaggle\`  
3. Re-run `python download_weak_classes.py` (or append with `_install_tree_append`)

## After training

Restart backend: `python app.py` in `plant-disease-backend`.
