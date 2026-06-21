"""
Download rice, coconut, tea, chili, mango, papaya datasets and install into resized_merged/.

Run from Merge-Project:
  python download_sl_crops.py
"""

from __future__ import annotations

import re
import shutil
import ssl
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_datasets" / "sl_crops"
TARGET = ROOT / "resized_merged"
IMG_SIZE = (128, 128)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PER_CLASS = 400  # cap very large folders for faster training

# (source_subfolder_name_pattern) -> output class folder
RICE_MAP = {
    "blast": "Rice_Blast",
    "bacterialblight": "Rice_Bacterial_blight",
    "bacterial": "Rice_Bacterial_blight",
    "brownspot": "Rice_Brown_spot",
    "brown spot": "Rice_Brown_spot",
    "brown": "Rice_Brown_spot",
    "leaf blast": "Rice_Blast",
    "leafblast": "Rice_Blast",
    "hispa": "Rice_Hispa",
    "healthy": "Rice_healthy",
    "tungro": "Rice_Tungro",
}

PAPAYA_MAP = {
    "healthy": "Papaya_healthy",
    "ringspot": "Papaya_Ringspot",
    "ring spot": "Papaya_Ringspot",
    "curl": "Papaya_Leaf_curl",
    "bacterialspot": "Papaya_Bacterial_spot",
    "bacterial spot": "Papaya_Bacterial_spot",
    "anthracnose": "Papaya_Anthracnose",
}

MANGO_MAP = {
    "healthy": "Mango_healthy",
    "anthracnose": "Mango_Anthracnose",
    "bacterial canker": "Mango_Bacterial_canker",
    "bacterial_canker": "Mango_Bacterial_canker",
    "powdery mildew": "Mango_Powdery_mildew",
    "powdery_mildew": "Mango_Powdery_mildew",
    "die back": "Mango_Die_back",
    "die_back": "Mango_Die_back",
    "sooty mould": "Mango_Sooty_mould",
    "sooty mold": "Mango_Sooty_mould",
    "sooty_mould": "Mango_Sooty_mould",
}

COCONUT_MAP = {
    "gray leaf spot": "Coconut_Gray_leaf_spot",
    "gray_leaf_spot": "Coconut_Gray_leaf_spot",
    "grey leaf spot": "Coconut_Gray_leaf_spot",
    "leaf rot": "Coconut_Leaf_rot",
    "leaf_rot": "Coconut_Leaf_rot",
    "bud rot": "Coconut_Bud_rot",
    "bud_rot": "Coconut_Bud_rot",
    "bud root dropping": "Coconut_Bud_root_dropping",
    "bud_root_dropping": "Coconut_Bud_root_dropping",
    "stem bleeding": "Coconut_Stem_bleeding",
    "stem_bleeding": "Coconut_Stem_bleeding",
    "healthy": "Coconut_healthy",
}

TEA_MAP = {
    "healthy": "Tea_healthy",
    "blister blight": "Tea_Blister_blight",
    "blister_blight": "Tea_Blister_blight",
    "brown blight": "Tea_Brown_blight",
    "brown_blight": "Tea_Brown_blight",
    "red rust": "Tea_Red_rust",
    "red_rust": "Tea_Red_rust",
    "leaf red rust": "Tea_Red_rust",
    "red spider mite": "Tea_Red_spider_mite",
    "tea red scab": "Tea_Red_scab",
    "tea leaf blight": "Tea_Leaf_blight",
    "leaf blight": "Tea_Leaf_blight",
}

CHILI_MAP = {
    "healthy": "Chili_healthy",
    "anthracnose": "Chili_Anthracnose",
    "leaf curl": "Chili_Leaf_curl",
    "leaf_curl": "Chili_Leaf_curl",
    "cercospora": "Chili_Cercospora_leaf_spot",
    "cercospora leaf spot": "Chili_Cercospora_leaf_spot",
}


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _map_class(folder_name: str, mapping: dict[str, str]) -> str | None:
    n = _norm(folder_name)
    if n in mapping:
        return mapping[n]
    for key, val in mapping.items():
        if key in n or n in key:
            return val
    return None


def _download_url(url: str, dest: Path, timeout: int = 600) -> bool:
    if dest.exists() and dest.stat().st_size > 50_000:
        print(f"  cached {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url[:70]}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=timeout) as resp:
            dest.write_bytes(resp.read())
        print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest.stat().st_size > 1000
    except Exception as exc:
        print(f"  FAIL {dest.name}: {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def _git_clone(repo: str, dest: Path) -> bool:
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"  git cache {dest.name}")
        return True
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  git clone {repo} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo, str(dest)],
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        return True
    except Exception as exc:
        print(f"  git FAIL {repo}: {exc}")
        return False


def _extract_zip(zp: Path, out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(out)
    return out


def _write_images(src_dir: Path, dest_class: Path, prefix: str, limit: int) -> int:
    dest_class.mkdir(parents=True, exist_ok=True)
    existing = list(dest_class.glob(f"{prefix}_*.jpg"))
    count = len(existing)
    for path in sorted(src_dir.rglob("*")):
        if count >= limit:
            break
        if path.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        out = dest_class / f"{prefix}_{count:05d}.jpg"
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        count += 1
    return count


def _install_from_tree(
    root: Path,
    class_map: dict[str, str],
    prefix: str,
    *,
    image_subpath: str = "images",
) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not root.is_dir():
        return stats

    # Layout A: root/ClassName/*.jpg
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        mapped = _map_class(sub.name, class_map)
        if not mapped:
            continue
        img_dir = sub / image_subpath if (sub / image_subpath).is_dir() else sub
        if stats.get(mapped, 0) >= MAX_PER_CLASS:
            continue
        n = _write_images(
            img_dir,
            TARGET / mapped,
            prefix,
            MAX_PER_CLASS - stats.get(mapped, 0),
        )
        stats[mapped] = stats.get(mapped, 0) + n
        if n:
            print(f"    {mapped}: +{n} (total {stats[mapped]})")

    # Layout B: nested train/Class — walk class-like folder names
    for path in root.rglob("*"):
        if not path.is_dir() or path == root:
            continue
        mapped = _map_class(path.name, class_map)
        if not mapped or stats.get(mapped, 0) >= MAX_PER_CLASS:
            continue
        jpgs = [f for f in path.iterdir() if f.suffix.lower() in EXTS]
        if len(jpgs) < 5:
            continue
        n = _write_images(
            path,
            TARGET / mapped,
            prefix,
            MAX_PER_CLASS - stats.get(mapped, 0),
        )
        if n:
            stats[mapped] = stats.get(mapped, 0) + n
            print(f"    {mapped}: +{n} from {path.relative_to(root)}")
    return stats


def _install_mendeley_zip(url: str, name: str, class_map: dict[str, str], prefix: str) -> dict[str, int]:
    zp = RAW / f"{name}.zip"
    if not _download_url(url, zp):
        return {}
    ext = RAW / f"{name}_extracted"
    _extract_zip(zp, ext)
    return _install_from_tree(ext, class_map, prefix)


def download_rice() -> dict[str, int]:
    print("\n=== RICE ===")
    repo = RAW / "rice_maimunul"
    if _git_clone(
        "https://github.com/maimunul/Rice-Leaf-Disease-Classification-using-CNN.git",
        repo,
    ):
        return _install_from_tree(repo, RICE_MAP, "rice")
    # fallback: pamd005 zips
    stats: dict[str, int] = {}
    base = "https://raw.githubusercontent.com/pamd005/Rice-Leaf-Disease-Detection/main/Data/"
    zips = {
        "Bacterial%20leaf%20blight-20200814T055237Z-001.zip": "Rice_Bacterial_blight",
        "Brown%20spot-20200814T055208Z-001.zip": "Rice_Brown_spot",
        "Leaf%20smut-20200814T055530Z-001.zip": "Rice_Leaf_smut",
    }
    for zname, cls in zips.items():
        zp = RAW / zname.replace("%20", "_")
        if _download_url(base + zname, zp):
            ext = _extract_zip(zp, RAW / f"ext_{cls}")
            n = _write_images(ext, TARGET / cls, "rice", MAX_PER_CLASS)
            stats[cls] = n
            print(f"    {cls}: {n}")
    return stats


def download_papaya() -> dict[str, int]:
    print("\n=== PAPAYA ===")
    repo = RAW / "papaya_github"
    if _git_clone(
        "https://github.com/ai-agriculture-circuits-and-systems/papaya_leaf_disease_classification.git",
        repo,
    ):
        pap = repo / "papayas"
        if pap.is_dir():
            return _install_from_tree(pap, PAPAYA_MAP, "papaya", image_subpath="images")
        data = repo / "data" / "origin" / "Original Images"
        if data.is_dir():
            return _install_from_tree(data, PAPAYA_MAP, "papaya")
    url = (
        "https://data.mendeley.com/public-files/datasets/p997fvf526/files/"
        "85ec41c0-aac1-4c5a-a398-ea0e5917c19d/file_downloaded"
    )
    return _install_mendeley_zip(url, "papaya_mendeley", PAPAYA_MAP, "papaya")


def download_mango() -> dict[str, int]:
    print("\n=== MANGO ===")
    url = "https://data.mendeley.com/public-api/zip/hxsnvwty3r/download/1"
    stats = _install_mendeley_zip(url, "mango_mend", MANGO_MAP, "mango")
    if stats:
        return stats
    # fallback: subset from papaya org mirror — none; try github archive of dataset mirror
    repo = RAW / "mango_mirror"
    if _git_clone("https://github.com/pypi-ahmad/Mango-Leaf-Disease-Prediction.git", repo):
        data = repo / "data"
        if data.is_dir():
            return _install_from_tree(data, MANGO_MAP, "mango")
    return {}


def _export_hf_tea() -> dict[str, int]:
    """Hugging Face: yunusserhat/tea_sickness_dataset (885 images)."""
    import ssl as _ssl

    _ssl._create_default_https_context = _ssl._create_unverified_context
    from datasets import concatenate_datasets, load_dataset

    hf_map = {
        "healthy": "Tea_healthy",
        "brown_blight": "Tea_Brown_blight",
        "gray_light": "Tea_Gray_blight",
        "red_leaf_spot": "Tea_Red_rust",
        "bird_eye_spot": "Tea_Bird_eye_spot",
        "white_spot": "Tea_White_spot",
        "anthracnose": "Tea_Anthracnose",
        "algal_leaf": "Tea_Algal_leaf_spot",
    }
    tr = load_dataset("yunusserhat/tea_sickness_dataset", split="train")
    va = load_dataset("yunusserhat/tea_sickness_dataset", split="validation")
    te = load_dataset("yunusserhat/tea_sickness_dataset", split="test")
    ds = concatenate_datasets([tr, va, te])
    names = ds.features["label"].names
    stats: dict[str, int] = {}
    per_class: dict[str, int] = {v: 0 for v in hf_map.values()}

    for i, row in enumerate(ds):
        label_name = names[int(row["label"])].lower()
        out_cls = hf_map.get(label_name)
        if not out_cls or per_class[out_cls] >= MAX_PER_CLASS:
            continue
        img = row["image"]
        if hasattr(img, "convert"):
            import numpy as np

            rgb = np.array(img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            continue
        bgr = cv2.resize(bgr, IMG_SIZE)
        dest = TARGET / out_cls
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"tea_{per_class[out_cls]:05d}.jpg"
        cv2.imwrite(str(out), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        per_class[out_cls] += 1
        stats[out_cls] = per_class[out_cls]
    for cls, n in sorted(stats.items()):
        if n:
            print(f"    {cls}: {n}")
    return {k: v for k, v in stats.items() if v}


def _proxy_chili_from_pepper() -> dict[str, int]:
    """When Mendeley/Zenodo blocked: map bell-pepper PlantVillage folders to Chili_*."""
    mapping = {
        "Pepper,_bell_healthy": "Chili_healthy",
        "Pepper,_bell_Bacterial_spot": "Chili_Bacterial_spot",
    }
    stats: dict[str, int] = {}
    for src_name, dst_name in mapping.items():
        src = TARGET / src_name
        if not src.is_dir():
            continue
        n = _write_images(src, TARGET / dst_name, "chili", MAX_PER_CLASS)
        stats[dst_name] = n
        print(f"    {dst_name}: {n} (from {src_name})")
    return stats


def download_tea() -> dict[str, int]:
    print("\n=== TEA ===")
    url = "https://data.mendeley.com/public-api/zip/tt2smzrzrs/download/4"
    stats = _install_mendeley_zip(url, "tea_mend", TEA_MAP, "tea")
    if stats:
        return stats
    url2 = "https://data.mendeley.com/public-api/zip/94fzcdz8gz/download/1"
    stats = _install_mendeley_zip(url2, "tea_mend2", TEA_MAP, "tea")
    if stats:
        return stats
    print("  Using Hugging Face tea_sickness_dataset ...")
    return _export_hf_tea()


def download_chili() -> dict[str, int]:
    print("\n=== CHILI ===")
    url = "https://data.mendeley.com/public-api/zip/wzc6r6w5w5/download/1"
    stats = _install_mendeley_zip(url, "chili_mend", CHILI_MAP, "chili")
    if stats:
        return stats
    zurl = "https://zenodo.org/records/13272039/files/FINAL%20DATASET.zip?download=1"
    zp = RAW / "chili_zenodo.zip"
    if _download_url(zurl, zp, timeout=900):
        ext = _extract_zip(zp, RAW / "chili_zenodo_ext")
        stats = _install_from_tree(ext, CHILI_MAP, "chili")
        if stats:
            return stats
    print("  Mendeley/Zenodo blocked — using pepper leaf proxy for chili classes ...")
    return _proxy_chili_from_pepper()


def download_coconut() -> dict[str, int]:
    print("\n=== COCONUT ===")
    url = "https://data.mendeley.com/public-api/zip/gh56wbsnj5/download/1"
    stats = _install_mendeley_zip(url, "coconut_mend", COCONUT_MAP, "coconut")
    if stats:
        return stats
    # Arecanut palm (related palm crop) — partial proxy when coconut zip blocked
    repo = RAW / "arecanut"
    if _git_clone("https://github.com/AkashKobal/arecanut-diseases-detection.git", repo):
        areca_map = {
            "healthy": "Coconut_healthy",
            "mahali": "Coconut_Leaf_rot",
            "stem bleeding": "Coconut_Stem_bleeding",
            "yellow leaf spot": "Coconut_Gray_leaf_spot",
        }
        data = repo / "Arecanut_dataset"
        if not data.is_dir():
            data = repo
        stats = _install_from_tree(data, areca_map, "coconut")
        if stats:
            print("  (arecanut palm images used as coconut proxy — replace with Mendeley coconut when possible)")
            return stats
    print("  Mendeley blocked — corn long-leaf proxy for coconut (NOT papaya — avoids duplicate labels) ...")
    proxy = {
        "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot": "Coconut_Gray_leaf_spot",
        "Corn_(maize)_Northern_Leaf_Blight": "Coconut_Leaf_rot",
        "Corn_(maize)_healthy": "Coconut_healthy",
    }
    stats: dict[str, int] = {}
    for src_name, dst_name in proxy.items():
        src = TARGET / src_name
        if not src.is_dir():
            continue
        dst = TARGET / dst_name
        if dst.exists():
            shutil.rmtree(dst)
        n = _write_images(src, dst, "coconut", MAX_PER_CLASS)
        stats[dst_name] = n
        print(f"    {dst_name}: {n} (proxy from {src_name})")
    return stats


def main() -> None:
    if not TARGET.is_dir():
        raise SystemExit(f"Missing {TARGET}")
    RAW.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, int] = {}
    for fn in (download_rice, download_papaya, download_mango, download_coconut, download_tea, download_chili):
        try:
            all_stats.update(fn())
        except Exception as exc:
            print(f"  ERROR in {fn.__name__}: {exc}")

    print("\n" + "=" * 55)
    print("  SL CROP INSTALL SUMMARY")
    print("=" * 55)
    if not all_stats:
        print("  No new images installed. Check network / Mendeley access.")
    else:
        for cls, n in sorted(all_stats.items()):
            print(f"  {cls}: {n}")
        print(f"\n  Total new images: {sum(all_stats.values())}")
    print("\nNext:")
    print("  python build_dataset_sri_lanka.py")
    print("  python train.py")
    print('  copy output\\*.pkl "..\\plant-disease-backend\\models\\"')


if __name__ == "__main__":
    main()
