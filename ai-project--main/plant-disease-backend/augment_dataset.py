"""
download_more_training_data.py

Downloads additional images for under-represented classes (< 500 images)
using icrawler Bing search. Places them into the correct class folder.
"""
import sys, time, os, random
from pathlib import Path
from icrawler.builtin import BingImageCrawler

DATA_DIR = Path(r"D:\ai data\Final\Merge-Project\resized_merged")

# Classes with < 500 images, target count per class
TARGET = 500
# Search queries for each class (plant disease + leaf)
SEARCH_QUERIES = {
    "Tea_Gray_blight": "tea gray blight disease leaf",
    "Tea_Bird_eye_spot": "tea bird eye spot disease leaf",
    "Tea_Brown_blight": "tea brown blight disease leaf",
    "Tea_Algal_leaf_spot": "tea algal leaf spot disease",
    "Tea_White_spot": "tea white spot disease leaf",
    "Tea_Red_rust": "tea red rust disease leaf",
    "Tea_healthy": "tea healthy green leaf",
    "Tea_Anthracnose": "tea anthracnose disease leaf",
    "Banana_healthy": "banana healthy green leaf",
    "Papaya_Anthracnose": "papaya anthracnose disease leaf",
    "Papaya_Ringspot": "papaya ringspot virus leaf",
    "Papaya_Bacterial_spot": "papaya bacterial spot leaf",
    "Papaya_Leaf_curl": "papaya leaf curl disease",
    "Papaya_healthy": "papaya healthy leaf",
    "Mango_Anthracnose": "mango anthracnose disease leaf",
    "Mango_Bacterial_canker": "mango bacterial canker leaf",
    "Mango_Die_back": "mango dieback disease leaf",
    "Mango_healthy": "mango healthy green leaf",
    "Mango_Powdery_mildew": "mango powdery mildew leaf",
    "Mango_Sooty_mould": "mango sooty mould leaf",
    "Rice_Bacterial_blight": "rice bacterial blight leaf",
    "Rice_Blast": "rice blast disease leaf",
    "Rice_Brown_spot": "rice brown spot leaf",
    "Rice_Tungro": "rice tungro disease leaf",
    "Coconut_Gray_leaf_spot": "coconut gray leaf spot",
    "Coconut_healthy": "coconut healthy frond leaf",
    "Coconut_Leaf_rot": "coconut leaf rot disease",
    "Chili_Bacterial_spot": "chili bacterial spot leaf",
    "Chili_healthy": "chili healthy green leaf",
}

def count_images(class_dir):
    return len([f for f in class_dir.glob("*.*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])

def download_for_class(class_name, query, max_download=100):
    class_dir = DATA_DIR / class_name
    if not class_dir.is_dir():
        print(f"  SKIP: {class_name} - directory not found")
        return 0
    
    existing = count_images(class_dir)
    if existing >= TARGET:
        print(f"  SKIP: {class_name} - already {existing} images")
        return 0
    
    need = min(max_download, TARGET - existing)
    print(f"  {class_name}: {existing} images, need {need} more...")
    
    crawler = BingImageCrawler(
        storage={"root_dir": str(class_dir)},
        downloader_threads=4,
    )
    # Use file_idx_offset to not overwrite existing files
    # icrawler names files 000001.jpg, 000002.jpg etc.
    offset = existing + 1
    try:
        crawler.crawl(
            keyword=query,
            max_num=need,
            min_size=(100, 100),
            file_idx_offset=offset,
        )
    except Exception as e:
        print(f"    Error: {e}")
        return 0
    
    after = count_images(class_dir)
    downloaded = after - existing
    print(f"    Downloaded: {downloaded} (total: {after})")
    return downloaded

# Sort by need (most under-represented first)
classes = [(cn, SEARCH_QUERIES[cn]) for cn in SEARCH_QUERIES]
class_counts = {}
for cn, _ in classes:
    class_dir = DATA_DIR / cn
    if class_dir.is_dir():
        class_counts[cn] = count_images(class_dir)
    else:
        class_counts[cn] = TARGET  # skip missing

classes.sort(key=lambda x: class_counts[x[0]])

total_downloaded = 0
for class_name, query in classes:
    count = class_counts[class_name]
    if count >= TARGET:
        continue
    print(f"\nDownloading: {class_name} ({count}/{TARGET})")
    d = download_for_class(class_name, query, max_download=100)
    total_downloaded += d
    time.sleep(2)  # Be polite between crawls

print(f"\n\nTotal downloaded: {total_downloaded} images")
