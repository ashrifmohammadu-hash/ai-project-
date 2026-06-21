"""
Download real coconut + papaya datasets and replace weak proxy folders.

Run:
  python download_coconut_papaya_fix.py
  python build_dataset_sri_lanka.py
  python train.py
"""
from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_datasets" / "coconut_papaya_fix"
TARGET = ROOT / "resized_merged"
IMG_SIZE = (128, 128)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PER_CLASS = 400

COCONUT_MAP = {
    "gray leaf spot": "Coconut_Gray_leaf_spot",
    "gray_leaf_spot": "Coconut_Gray_leaf_spot",
    "grey leaf spot": "Coconut_Gray_leaf_spot",
    "gray leafspot": "Coconut_Gray_leaf_spot",
    "leaf rot": "Coconut_Leaf_rot",
    "leaf_rot": "Coconut_Leaf_rot",
    "bud rot": "Coconut_Leaf_rot",
    "healthy": "Coconut_healthy",
    "bud root dropping": "Coconut_healthy",
}

PAPAYA_BD_MAP = {
    "anthracnose": "Papaya_Anthracnose",
    "bacterial spot": "Papaya_Bacterial_spot",
    "bacterial_spot": "Papaya_Bacterial_spot",
    "bacterialspot": "Papaya_Bacterial_spot",
    "curl": "Papaya_Leaf_curl",
    "leaf curl": "Papaya_Leaf_curl",
    "ring spot": "Papaya_Ringspot",
    "ringspot": "Papaya_Ringspot",
    "ring_spot": "Papaya_Ringspot",
    "healthy": "Papaya_healthy",
    "healthy leaf": "Papaya_healthy",
}

DOWNLOAD_URLS = [
    (
        "coconut_zip",
        "https://data.mendeley.com/public-api/zip/gh56wbsnj5/download/1",
        "coconut",
        COCONUT_MAP,
        ("Coconut_Gray_leaf_spot", "Coconut_Leaf_rot", "Coconut_healthy"),
    ),
    (
        "coconut_file",
        "https://data.mendeley.com/public-files/datasets/gh56wbsnj5/files/"
        "file_downloaded",
        "coconut",
        COCONUT_MAP,
        ("Coconut_Gray_leaf_spot", "Coconut_Leaf_rot", "Coconut_healthy"),
    ),
    (
        "papaya_bd_zip",
        "https://data.mendeley.com/public-api/zip/p997fvf526/download/1",
        "papaya",
        PAPAYA_BD_MAP,
        (
            "Papaya_Bacterial_spot",
            "Papaya_Leaf_curl",
            "Papaya_Ringspot",
            "Papaya_healthy",
            "Papaya_Anthracnose",
        ),
    ),
    (
        "papaya_bd_file",
        "https://data.mendeley.com/public-files/datasets/p997fvf526/files/"
        "85ec41c0-aac1-4c5a-a398-ea0e5917c19d/file_downloaded",
        "papaya",
        PAPAYA_BD_MAP,
        (
            "Papaya_Bacterial_spot",
            "Papaya_Leaf_curl",
            "Papaya_Ringspot",
            "Papaya_healthy",
            "Papaya_Anthracnose",
        ),
    ),
    (
        "papaya_orchard_zip",
        "https://data.mendeley.com/public-api/zip/44p8v6ywsm/download/1",
        "papaya2",
        {
            "bacterial spot": "Papaya_Bacterial_spot",
            "leaf curl": "Papaya_Leaf_curl",
            "curl": "Papaya_Leaf_curl",
            "ring spot": "Papaya_Ringspot",
            "ringspot": "Papaya_Ringspot",
            "mosaic": "Papaya_Ringspot",
            "healthy": "Papaya_healthy",
            "healthy leaf": "Papaya_healthy",
        },
        ("Papaya_Bacterial_spot", "Papaya_Leaf_curl"),
    ),
]


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _norm(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _map_class(folder_name: str, mapping: dict[str, str]) -> str | None:
    n = _norm(folder_name)
    if n in mapping:
        return mapping[n]
    for key, val in mapping.items():
        if key in n or n in key:
            return val
    return None


def _download(url: str, dest: Path, timeout: int = 1800) -> bool:
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  cached {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  try {url[:85]}...")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 10_000:
            print(f"  too small ({len(data)} bytes)")
            return False
        dest.write_bytes(data)
        print(f"  saved {dest.name} ({len(data) / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  fail: {exc}")
        dest.unlink(missing_ok=True)
        return False


def _extract(zp: Path, out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        # Mendeley often ships nested zip
        if len(names) == 1 and names[0].lower().endswith(".zip"):
            nested = out / "nested.zip"
            zf.extract(names[0], out)
            inner = out / names[0]
            if not inner.exists():
                inner = next(out.rglob("*.zip"))
            shutil.move(str(inner), str(nested))
            with zipfile.ZipFile(nested) as zf2:
                zf2.extractall(out)
            nested.unlink(missing_ok=True)
        else:
            zf.extractall(out)
    return out


def _write_folder(src: Path, dest: Path, prefix: str, limit: int) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob(f"{prefix}_*.jpg"):
        old.unlink()
    n = 0
    for path in sorted(src.rglob("*")):
        if n >= limit:
            break
        if path.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(
            str(dest / f"{prefix}_{n:05d}.jpg"),
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        n += 1
    return n


def _install_tree(root: Path, class_map: dict[str, str], prefix: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not root.is_dir():
        return stats
    # Collect best source folder per class
    buckets: dict[str, list[Path]] = {}
    for sub in root.rglob("*"):
        if not sub.is_dir():
            continue
        mapped = _map_class(sub.name, class_map)
        if not mapped:
            continue
        jpgs = [f for f in sub.iterdir() if f.suffix.lower() in EXTS]
        if len(jpgs) >= 3:
            buckets.setdefault(mapped, []).append(sub)
        elif any(sub.rglob("*.jpg")):
            buckets.setdefault(mapped, []).append(sub)
    for mapped, folders in buckets.items():
        # Use folder with most images
        best = max(folders, key=lambda p: len(list(p.rglob("*.jpg"))))
        dest = TARGET / mapped
        if dest.exists():
            shutil.rmtree(dest)
        n = _write_folder(best, dest, prefix, MAX_PER_CLASS)
        if n:
            stats[mapped] = n
            print(f"    {mapped}: {n} <- {best.name}")
    return stats


def _try_source(name: str, url: str, prefix: str, class_map: dict, replace_classes: tuple) -> dict[str, int]:
    zp = RAW / f"{name}.zip"
    if not _download(url, zp):
        return {}
    try:
        ext = _extract(zp, RAW / f"{name}_ext")
    except Exception as exc:
        print(f"  extract fail: {exc}")
        return {}
    return _install_tree(ext, class_map, prefix)


def _fallback_coconut_arecanut() -> dict[str, int]:
    """Git arecanut palm dataset if Mendeley blocked."""
    import subprocess

    repo = RAW / "arecanut"
    if not (repo / ".git").exists():
        print("  git clone arecanut...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/AkashKobal/arecanut-diseases-detection.git",
                    str(repo),
                ],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except Exception as exc:
            print(f"  git fail: {exc}")
            return {}
    areca_map = {
        "healthy": "Coconut_healthy",
        "mahali": "Coconut_Leaf_rot",
        "yellow leaf spot": "Coconut_Gray_leaf_spot",
        "stem bleeding": "Coconut_Leaf_rot",
    }
    data = repo / "Arecanut_dataset"
    if not data.is_dir():
        data = repo
    stats: dict[str, int] = {}
    for cls in ("Coconut_Gray_leaf_spot", "Coconut_Leaf_rot", "Coconut_healthy"):
        p = TARGET / cls
        if p.is_dir():
            shutil.rmtree(p)
    return _install_tree(data, areca_map, "coconut")


def main() -> None:
    if not TARGET.is_dir():
        raise SystemExit(f"Missing {TARGET}")
    RAW.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, int] = {}
    coconut_ok = False
    papaya_ok = False

    for name, url, prefix, cmap, replace in DOWNLOAD_URLS:
        print(f"\n=== {name} ===")
        stats = _try_source(name, url, prefix, cmap, replace)
        if stats:
            all_stats.update(stats)
            if name.startswith("coconut"):
                coconut_ok = True
            if name.startswith("papaya"):
                papaya_ok = papaya_ok or bool(
                    stats.get("Papaya_Bacterial_spot") or stats.get("Papaya_Leaf_curl")
                )

    if not coconut_ok:
        print("\n=== Coconut fallback (arecanut palm) ===")
        stats = _fallback_coconut_arecanut()
        all_stats.update(stats)

    print("\n" + "=" * 55)
    if all_stats:
        for k, v in sorted(all_stats.items()):
            print(f"  {k}: {v} images")
    else:
        print("  No downloads succeeded.")
        print("  Manual coconut: https://data.mendeley.com/datasets/gh56wbsnj5/1")
        print("  Manual papaya:  https://data.mendeley.com/datasets/p997fvf526/2")
    print("\nNext: python build_dataset_sri_lanka.py && python train.py")


if __name__ == "__main__":
    main()
