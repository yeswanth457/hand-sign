"""
HuggingFace Dataset API Client for vidit031/isl-isolated-40words.

Dynamically discovers config/split via /splits endpoint, then paginates
through /rows to collect metadata for all target classes. Downloads real
MP4 video files from the HF Git LFS repository.

NEVER hard-codes config/split without checking /splits first.
NEVER treats Dataset Viewer rows as actual video files.
NEVER generates synthetic data or hard-codes predictions.
"""

import os
import sys
import json
import shutil
import time
import urllib.request
import urllib.error
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

# Optional: use huggingface_hub for Git LFS downloads
try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("[WARNING] huggingface_hub not installed. Install with: pip install huggingface_hub")

REPO_ID = "vidit031/isl-isolated-40words"
HF_BASE = "https://datasets-server.huggingface.co"
USER_AGENT = "RT-STAMP-SLR/2.0 (ISL-Translator; Educational/Research)"

# Target classes for this project (from config.py)
TARGET_CLASSES = [
    "hello", "thank_you", "welcome", "goodbye", "yes", "no", "please",
    "sorry", "help", "stop", "water", "food", "school", "teacher",
    "mother", "father", "sister", "brother", "friend", "house", "work"
]

# Explicit label mapping from dataset word -> project class name
# "thank you" (with space) in dataset maps to "thank_you" (underscore) in project
# "home" in dataset maps to "house" in project (semantic equivalent in ISL)
LABEL_MAP = {
    "thank you": "thank_you",
    "home": "house",
}

# Classes that exist in the dataset with their exact dataset label
# We do NOT invent mappings for classes not present in the dataset
DATASET_LABELS_TO_TARGET = {}
for cls in TARGET_CLASSES:
    DATASET_LABELS_TO_TARGET[cls] = cls
# Add reverse mappings
for ds_label, proj_label in LABEL_MAP.items():
    DATASET_LABELS_TO_TARGET[ds_label] = proj_label


def _make_request(url, timeout=30):
    """Make an HTTP GET request with proper headers."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_splits():
    """
    Step 1: Call /splits to discover the actual config and split.
    Returns (config, split) tuple.
    """
    url = f"{HF_BASE}/splits?dataset={REPO_ID}"
    print(f"[HF API] Querying /splits: {url}")
    
    data = _make_request(url)
    splits = data.get("splits", [])
    
    if not splits:
        raise RuntimeError(f"No splits found for dataset {REPO_ID}")
    
    # Use the first available split
    config = splits[0]["config"]
    split = splits[0]["split"]
    
    print(f"[HF API] Discovered config='{config}', split='{split}'")
    print(f"[HF API] Total splits available: {len(splits)}")
    for s in splits:
        print(f"  - config='{s['config']}', split='{s['split']}'")
    
    return config, split


def fetch_all_rows(config, split, page_size=100):
    """
    Step 2: Paginate through /rows to collect metadata for all rows.
    Returns list of row dicts with 'word', 'video_path', 'signer', etc.
    """
    all_rows = []
    offset = 0
    total_rows = None
    
    print(f"\n[HF API] Fetching rows (config='{config}', split='{split}')...")
    
    while True:
        url = (
            f"{HF_BASE}/rows?dataset={REPO_ID}"
            f"&config={config}&split={split}"
            f"&offset={offset}&length={page_size}"
        )
        
        try:
            data = _make_request(url)
        except Exception as e:
            print(f"[HF API] Error fetching offset={offset}: {e}")
            break
        
        if total_rows is None:
            total_rows = data.get("num_rows_total", 0)
            print(f"[HF API] Total rows in dataset: {total_rows}")
        
        rows = data.get("rows", [])
        if not rows:
            break
        
        for entry in rows:
            row = entry.get("row", {})
            all_rows.append(row)
        
        offset += len(rows)
        print(f"  Fetched {offset}/{total_rows} rows...", end="\r")
        
        if offset >= total_rows:
            break
        
        # Be polite to the API
        time.sleep(0.3)
    
    print(f"\n[HF API] Total rows fetched: {len(all_rows)}")
    return all_rows


def filter_target_rows(all_rows):
    """
    Step 3: Filter rows to only those belonging to target classes.
    Maps dataset labels to project class names.
    Returns dict: project_class -> [list of row metadata dicts]
    """
    target_rows = defaultdict(list)
    unmapped_labels = set()
    
    for row in all_rows:
        word = row.get("word", "").strip().lower()
        normalized = row.get("normalized_word", "").strip().lower()
        
        # Try exact match first, then normalized, then via label map
        project_class = None
        if word in DATASET_LABELS_TO_TARGET:
            project_class = DATASET_LABELS_TO_TARGET[word]
        elif normalized in DATASET_LABELS_TO_TARGET:
            project_class = DATASET_LABELS_TO_TARGET[normalized]
        
        if project_class:
            target_rows[project_class].append(row)
        else:
            unmapped_labels.add(word)
    
    # Report findings
    print(f"\n[HF API] Target class filtering results:")
    found_classes = []
    missing_classes = []
    
    for cls in TARGET_CLASSES:
        count = len(target_rows.get(cls, []))
        if count > 0:
            found_classes.append(cls)
            print(f"  {cls:15s}: {count:3d} rows")
        else:
            missing_classes.append(cls)
            print(f"  {cls:15s}:   0 rows  *** MISSING ***")
    
    if missing_classes:
        print(f"\n[WARNING] Missing target classes: {missing_classes}")
        print(f"  These classes have NO videos in the '{REPO_ID}' dataset.")
        print(f"  They require additional real ISL video data (e.g., from CISLR or manual recording).")
    
    if unmapped_labels:
        print(f"\n[INFO] Non-target labels in dataset (ignored): {sorted(unmapped_labels)}")
    
    return dict(target_rows), missing_classes


def download_videos(target_rows, raw_dir, max_per_class=50):
    """
    Step 4: Download actual MP4 files from the HF Git LFS repository.
    Uses huggingface_hub for reliable Git LFS downloads.
    
    IMPORTANT: The video_path field in rows is a relative path in the repo,
    NOT a direct download URL. We use hf_hub_download to get the actual file.
    """
    if not HF_HUB_AVAILABLE:
        print("[ERROR] huggingface_hub required for Git LFS downloads.")
        print("  Install: pip install huggingface_hub")
        return {}
    
    import cv2
    
    os.makedirs(raw_dir, exist_ok=True)
    download_stats = {}
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING REAL MP4 VIDEOS FROM HF REPO")
    print(f"{'='*60}")
    
    for cls, rows in target_rows.items():
        cls_dir = os.path.join(raw_dir, cls, "hf_real")
        os.makedirs(cls_dir, exist_ok=True)
        
        # Count existing valid videos
        existing = 0
        if os.path.isdir(os.path.join(raw_dir, cls)):
            for root, dirs, files in os.walk(os.path.join(raw_dir, cls)):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        fp = os.path.join(root, f)
                        try:
                            cap = cv2.VideoCapture(fp)
                            if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                                existing += 1
                            cap.release()
                        except:
                            pass
        
        # Limit downloads if we already have enough
        needed = max(0, max_per_class - existing)
        to_download = rows[:needed] if needed > 0 else []
        
        downloaded = 0
        skipped = 0
        failed = 0
        
        print(f"\n--- {cls} ({existing} existing, {len(to_download)} to download) ---")
        
        for i, row in enumerate(to_download):
            video_path = row.get("video_path", "")
            if not video_path:
                failed += 1
                continue
            
            fname = os.path.basename(video_path)
            dest_path = os.path.join(cls_dir, fname)
            
            # Skip if already downloaded and valid
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
                try:
                    cap = cv2.VideoCapture(dest_path)
                    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                        cap.release()
                        skipped += 1
                        continue
                    cap.release()
                except:
                    pass
            
            try:
                # Download from HF repo using Git LFS
                downloaded_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=video_path,
                    repo_type="dataset",
                    local_dir=None  # Use default HF cache
                )
                
                # Copy to our raw directory
                shutil.copy2(downloaded_path, dest_path)
                
                # Validate the downloaded file
                cap = cv2.VideoCapture(dest_path)
                if cap.isOpened():
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if fc > 0:
                        downloaded += 1
                        if (i + 1) % 5 == 0 or i == 0:
                            fsize = os.path.getsize(dest_path)
                            print(f"  [{i+1}/{len(to_download)}] {fname} ({fsize:,} bytes, {fc} frames)")
                    else:
                        os.remove(dest_path)
                        failed += 1
                        print(f"  [{i+1}/{len(to_download)}] INVALID (0 frames): {fname}")
                else:
                    cap.release()
                    os.remove(dest_path)
                    failed += 1
                    print(f"  [{i+1}/{len(to_download)}] CORRUPT: {fname}")
                    
            except Exception as e:
                failed += 1
                print(f"  [{i+1}/{len(to_download)}] FAILED: {fname}: {e}")
        
        total_now = existing + downloaded
        download_stats[cls] = {
            "existing": existing,
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "total": total_now
        }
        print(f"  Result: {total_now} total ({existing} existing + {downloaded} new, {failed} failed)")
    
    return download_stats


def get_final_inventory(raw_dir, target_classes=None):
    """Count valid videos per class after downloads."""
    import cv2
    
    if target_classes is None:
        target_classes = TARGET_CLASSES
    
    inventory = {}
    total = 0
    
    for cls in target_classes:
        cls_dir = os.path.join(raw_dir, cls)
        count = 0
        if os.path.isdir(cls_dir):
            for root, dirs, files in os.walk(cls_dir):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        fp = os.path.join(root, f)
                        try:
                            cap = cv2.VideoCapture(fp)
                            if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                                count += 1
                            cap.release()
                        except:
                            pass
        inventory[cls] = count
        total += count
    
    return inventory, total


def run_full_api_pipeline(raw_dir):
    """
    Complete HF Dataset API pipeline:
    1. /splits -> discover config/split
    2. /rows -> paginate all metadata
    3. Filter to target classes
    4. Download actual MP4 files
    5. Report inventory
    """
    print("=" * 70)
    print("  HuggingFace Dataset API Pipeline")
    print(f"  Dataset: {REPO_ID}")
    print("=" * 70)
    
    # Step 1: Discover config/split
    config, split = discover_splits()
    
    # Step 2: Fetch all rows
    all_rows = fetch_all_rows(config, split, page_size=100)
    
    # Step 3: Filter to target classes
    target_rows, missing_classes = filter_target_rows(all_rows)
    
    # Step 4: Download real MP4 files
    download_stats = download_videos(target_rows, raw_dir, max_per_class=50)
    
    # Step 5: Final inventory
    print(f"\n{'='*60}")
    print("FINAL VIDEO INVENTORY")
    print(f"{'='*60}")
    
    inventory, total = get_final_inventory(raw_dir)
    for cls in TARGET_CLASSES:
        count = inventory.get(cls, 0)
        status = "OK" if count >= 3 else ("INSUFFICIENT" if count > 0 else "MISSING")
        print(f"  {cls:15s}: {count:3d} valid videos  [{status}]")
    print(f"\n  TOTAL: {total}")
    
    still_missing = [cls for cls in TARGET_CLASSES if inventory.get(cls, 0) == 0]
    low_count = [cls for cls in TARGET_CLASSES if 0 < inventory.get(cls, 0) < 3]
    
    if still_missing:
        print(f"\n[CRITICAL] Classes with ZERO real videos: {still_missing}")
        print("  These classes CANNOT be trained until real video data is obtained.")
    if low_count:
        print(f"\n[WARNING] Classes with fewer than 3 videos: {low_count}")
        print("  These may have poor model performance.")
    
    return {
        "config": config,
        "split": split,
        "total_api_rows": len(all_rows),
        "target_rows_found": {cls: len(rows) for cls, rows in target_rows.items()},
        "missing_classes": missing_classes,
        "download_stats": download_stats,
        "inventory": inventory,
        "total_videos": total,
        "still_missing": still_missing,
        "low_count": low_count,
    }


if __name__ == "__main__":
    from config import RAW_DATA_DIR
    result = run_full_api_pipeline(RAW_DATA_DIR)
    
    # Save API status report
    report_path = os.path.join(os.path.dirname(RAW_DATA_DIR), "metadata", "hf_api_status.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] API status report: {report_path}")
