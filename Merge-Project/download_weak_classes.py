"""
Download / refresh datasets for classes with F1 < 80%.

Sources:
  - Banana: GitHub training.zip, NixonJimenez02 thesis data, Mendeley BananaLSD
  - Coconut: Mendeley gh56wbsnj5 (real coconut tree diseases)
  - Papaya: GitHub papaya_leaf_disease_classification
  - Tea: Hugging Face yunusserhat/tea_sickness_dataset (more Anthracnose)

Run:
  cd Merge-Project
  python download_weak_classes.py
  python build_dataset_sri_lanka.py
  python train.py
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
RAW = ROOT / "raw_datasets" / "weak_classes"
TARGET = ROOT / "resized_merged"
IMG_SIZE = (128, 128)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PER_CLASS = 400

BANANA_MAP = {
    "healthy": "Banana_healthy",
    "health": "Banana_healthy",
    "segatoka": "Banana_Sigatoka",
    "sigatoka": "Banana_Sigatoka",
    "black sigatoka": "Banana_Sigatoka",
    "black_sigatoka": "Banana_Sigatoka",
    "xamthomonas": "Banana_Xanthomonas_wilt",
    "xanthomonas": "Banana_Xanthomonas_wilt",
    "fusarium": "Banana_Xanthomonas_wilt",
    "wilt": "Banana_Xanthomonas_wilt",
    "cordana": "Banana_Sigatoka",
}

BANANALSD_MAP = {
    "healthy": "Banana_healthy",
    "sigatoka": "Banana_Sigatoka",
    "pestalotiopsis": "Banana_Sigatoka",
    "cordana": "Banana_Sigatoka",
}

COCONUT_MAP = {
    "gray leaf spot": "Coconut_Gray_leaf_spot",
    "gray_leaf_spot": "Coconut_Gray_leaf_spot",
    "grey leaf spot": "Coconut_Gray_leaf_spot",
    "leaf rot": "Coconut_Leaf_rot",
    "leaf_rot": "Coconut_Leaf_rot",
    "bud rot": "Coconut_Leaf_rot",
    "bud_rot": "Coconut_Leaf_rot",
    "healthy": "Coconut_healthy",
    "bud root dropping": "Coconut_healthy",
}

PAPAYA_MAP = {
    "healthy": "Papaya_healthy",
    "ringspot": "Papaya_Ringspot",
    "curl": "Papaya_Leaf_curl",
    "bacterialspot": "Papaya_Bacterial_spot",
    "bacterial spot": "Papaya_Bacterial_spot",
    "anthracnose": "Papaya_Anthracnose",
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


def _download_url(url: str, dest: Path, timeout: int = 900) -> bool:
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  cached {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url[:75]}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=timeout) as resp:
            dest.write_bytes(resp.read())
        return dest.stat().st_size > 5000
    except Exception as exc:
        print(f"  FAIL: {exc}")
        dest.unlink(missing_ok=True)
        return False


def _git_clone(repo: str, dest: Path) -> bool:
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"  git cache {dest.name}")
        return True
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"  git FAIL: {exc}")
        return False


def _extract_zip(zp: Path, out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(out)
    return out


def _next_index(dest: Path, prefix: str) -> int:
    existing = list(dest.glob(f"{prefix}_*.jpg"))
    if not existing:
        return 0
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums, default=-1) + 1


def _append_images(
    src_dir: Path,
    dest_class: Path,
    prefix: str,
    limit: int,
    class_map: dict[str, str] | None = None,
) -> int:
    """Add images to dest_class without deleting existing."""
    dest_class.mkdir(parents=True, exist_ok=True)
    start = _next_index(dest_class, prefix)
    count = 0
    for path in sorted(src_dir.rglob("*")):
        if count >= limit:
            break
        if path.suffix.lower() not in EXTS:
            continue
        if class_map and path.parent.is_dir():
            mapped = _map_class(path.parent.name, class_map)
            if mapped and mapped != dest_class.name:
                continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        out = dest_class / f"{prefix}_{start + count:05d}.jpg"
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        count += 1
    return count


def _install_tree_append(
    root: Path,
    class_map: dict[str, str],
    prefix: str,
    per_class_cap: int = MAX_PER_CLASS,
) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not root.is_dir():
        return stats
    for sub in sorted(root.rglob("*")):
        if not sub.is_dir():
            continue
        mapped = _map_class(sub.name, class_map)
        if not mapped:
            continue
        jpgs = [f for f in sub.iterdir() if f.suffix.lower() in EXTS]
        if len(jpgs) < 3 and not any(sub.rglob("*.jpg")):
            continue
        dest = TARGET / mapped
        current = len(list(dest.glob("*.jpg"))) if dest.is_dir() else 0
        room = max(0, per_class_cap - current)
        if room <= 0:
            continue
        n = _append_images(sub, dest, prefix, room)
        if n:
            stats[mapped] = stats.get(mapped, 0) + n
            print(f"    {mapped}: +{n} (now ~{current + n})")
    return stats


def download_banana_extra() -> dict[str, int]:
    print("\n=== BANANA (healthy + Sigatoka + wilt) ===")
    stats: dict[str, int] = {}

    # 1) PurnaChandar training.zip
    try:
        import download_banana_and_build as ban

        zp = ban.download_training_zip()
        ext = ban.extract_zip(zp)
        train_root = ext / "training" if (ext / "training").is_dir() else ext
        for src_name, dest_name in ban.CLASS_MAP.items():
            src = train_root / src_name
            if not src.is_dir():
                continue
            dest = TARGET / dest_name
            n = _append_images(src, dest, "ban", MAX_PER_CLASS)
            if n:
                stats[dest_name] = stats.get(dest_name, 0) + n
                print(f"    {dest_name}: +{n} from training.zip")
    except Exception as exc:
        print(f"  training.zip skip: {exc}")

    # 2) NixonJimenez thesis folders
    repo = RAW / "banana_nixon"
    if _git_clone("https://github.com/NixonJimenez02/deep-learning-banana-diseases.git", repo):
        for folder in ("Data-Tesis", "Imagenes-aumentadas"):
            root = repo / folder
            if root.is_dir():
                stats.update(_install_tree_append(root, BANANA_MAP, "ban"))

    # 3) Mendeley BananaLSD
    url = "https://data.mendeley.com/public-api/zip/9tb7k297ff/download/1"
    zp = RAW / "bananalsd.zip"
    if _download_url(url, zp):
        ext = _extract_zip(zp, RAW / "bananalsd_ext")
        stats.update(_install_tree_append(ext, BANANALSD_MAP, "banlsd"))

    return stats


def download_coconut_real() -> dict[str, int]:
    print("\n=== COCONUT (Mendeley real coconut tree dataset) ===")
    url = "https://data.mendeley.com/public-api/zip/gh56wbsnj5/download/1"
    zp = RAW / "coconut_mendeley.zip"
    if not _download_url(url, zp, timeout=1200):
        print("  Mendeley coconut blocked — try manual download:")
        print("  https://data.mendeley.com/datasets/gh56wbsnj5/1")
        return {}
    ext = _extract_zip(zp, RAW / "coconut_mendeley_ext")
    # Replace corn proxy folders with real coconut images
    stats: dict[str, int] = {}
    for cls in ("Coconut_Gray_leaf_spot", "Coconut_Leaf_rot", "Coconut_healthy"):
        folder = TARGET / cls
        if folder.is_dir():
            shutil.rmtree(folder)
    stats = _install_tree_append(ext, COCONUT_MAP, "coconut")
    return stats


def download_papaya_refresh() -> dict[str, int]:
    print("\n=== PAPAYA (bacterial spot + leaf curl) ===")
    repo = RAW / "papaya_github"
    if not _git_clone(
        "https://github.com/ai-agriculture-circuits-and-systems/papaya_leaf_disease_classification.git",
        repo,
    ):
        return {}
    pap = repo / "papayas"
    root = pap if pap.is_dir() else repo / "data" / "origin" / "Original Images"
    if not root.is_dir():
        root = repo
    return _install_tree_append(root, PAPAYA_MAP, "papaya")


def download_tea_anthracnose() -> dict[str, int]:
    print("\n=== TEA (Hugging Face — extra Anthracnose) ===")
    try:
        import ssl as _ssl

        _ssl._create_default_https_context = _ssl._create_unverified_context
        from datasets import concatenate_datasets, load_dataset
    except ImportError:
        print("  pip install datasets — skipping HF tea")
        return {}

    hf_map = {"anthracnose": "Tea_Anthracnose", "healthy": "Tea_healthy"}
    tr = load_dataset("yunusserhat/tea_sickness_dataset", split="train")
    va = load_dataset("yunusserhat/tea_sickness_dataset", split="validation")
    te = load_dataset("yunusserhat/tea_sickness_dataset", split="test")
    ds = concatenate_datasets([tr, va, te])
    names = ds.features["label"].names
    stats: dict[str, int] = {}
    import numpy as np

    for i, row in enumerate(ds):
        label_name = names[int(row["label"])].lower()
        out_cls = hf_map.get(label_name)
        if not out_cls:
            continue
        dest = TARGET / out_cls
        current = len(list(dest.glob("*.jpg"))) if dest.is_dir() else 0
        if current >= MAX_PER_CLASS:
            continue
        img = row["image"]
        if not hasattr(img, "convert"):
            continue
        rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, IMG_SIZE)
        dest.mkdir(parents=True, exist_ok=True)
        idx = _next_index(dest, "tea")
        out = dest / f"tea_{idx:05d}.jpg"
        cv2.imwrite(str(out), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        stats[out_cls] = stats.get(out_cls, 0) + 1
    for k, v in stats.items():
        print(f"    {k}: +{v}")
    return stats


def main() -> None:
    if not TARGET.is_dir():
        raise SystemExit(f"Missing {TARGET}")
    RAW.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, int] = {}
    for fn in (
        download_banana_extra,
        download_coconut_real,
        download_papaya_refresh,
        download_tea_anthracnose,
    ):
        try:
            all_stats.update(fn())
        except Exception as exc:
            print(f"  ERROR {fn.__name__}: {exc}")

    print("\n" + "=" * 55)
    print("  WEAK CLASS DOWNLOAD SUMMARY")
    print("=" * 55)
    if all_stats:
        for k, v in sorted(all_stats.items()):
            print(f"  {k}: +{v} images")
    else:
        print("  No new images (network/Mendeley may be blocked).")
        print("  Manual: download coconut from data.mendeley.com/datasets/gh56wbsnj5/1")
    print("\nNext:")
    print("  python build_dataset_sri_lanka.py")
    print("  python train.py")
    print('  copy output\\*.pkl "..\\plant-disease-backend\\models\\"')


if __name__ == "__main__":
    main()
