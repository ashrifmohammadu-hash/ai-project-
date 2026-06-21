"""
Download more training data with SSL verification disabled.
Targets only the worst classes (<300 images).
"""
import os, sys, ssl, time
from pathlib import Path

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context
# Also try disabling via environment
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

from icrawler.builtin import BingImageCrawler

DATA_DIR = Path(r"D:\ai data\Final\Merge-Project\resized_merged")
TARGET = 500
QUERIES = {
    "Tea_Gray_blight": "tea gray blight disease leaf plant",
    "Tea_Bird_eye_spot": "tea bird eye spot disease leaf",
    "Tea_Brown_blight": "tea brown blight disease leaf",
    "Tea_Algal_leaf_spot": "tea algal leaf spot disease",
    "Tea_White_spot": "tea white spot disease leaf",
    "Tea_Red_rust": "tea red rust disease leaf",
    "Tea_healthy": "tea plantation healthy green leaf",
    "Tea_Anthracnose": "tea anthracnose disease leaf",
    "Banana_healthy": "banana plant healthy green leaf",
}

def count_images(class_dir):
    return len([f for f in class_dir.glob("*.*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])

# Patch urllib3 to disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Monkey-patch the requests adapter to never verify SSL
import requests
from requests.adapters import HTTPAdapter
original_init = HTTPAdapter.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.config = {}
requests.adapters.HTTPAdapter.__init__ = patched_init

# Better: patch the session creation
original_request = requests.Session.request
def patched_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_request

total = 0
for class_name, query in QUERIES.items():
    class_dir = DATA_DIR / class_name
    if not class_dir.is_dir():
        continue
    existing = count_images(class_dir)
    if existing >= TARGET:
        print(f"SKIP {class_name}: already {existing}")
        continue
    need = min(150, TARGET - existing)
    print(f"\n{class_name}: {existing} images, downloading {need}...")
    
    try:
        crawler = BingImageCrawler(
            storage={"root_dir": str(class_dir)},
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=query,
            max_num=need,
            min_size=(100, 100),
            file_idx_offset=existing + 1,
        )
    except Exception as e:
        print(f"  Error: {e}")
    
    after = count_images(class_dir)
    d = after - existing
    total += d
    print(f"  Got {d} new images (total: {after})")
    time.sleep(3)

print(f"\nTotal new images: {total}")
