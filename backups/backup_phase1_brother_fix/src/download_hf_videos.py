"""
Download real ISL videos from HuggingFace Git LFS repo vidit031/isl-isolated-40words.
Downloads actual .mp4 files from the repo tree using huggingface_hub.
Targets all 21 project classes, mapping 'home' -> 'house' and 'thank_you' -> 'thank_you'.
"""
import os
import sys
import shutil

sys.stdout.reconfigure(line_buffering=True)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, hf_hub_download
from config import CLASS_NAMES, RAW_DATA_DIR

# All 21 target classes with their HF repo directory mapping
# Most classes map 1:1, but 'house' maps to 'home' in the repo
TARGET_REPO_MAP = {}
for cls in CLASS_NAMES:
    TARGET_REPO_MAP[cls] = cls  # Default: same directory name

# Override: 'house' class uses videos from 'home' directory in HF repo
TARGET_REPO_MAP["house"] = "home"
# 'welcome' and 'work' don't exist in the HF repo
# They need local recordings or another dataset

REPO_ID = "vidit031/isl-isolated-40words"
MIN_VIDEOS = 15

api = HfApi()

def count_valid_videos(cls_dir):
    import cv2
    count = 0
    if not os.path.isdir(cls_dir):
        return 0
    for root, dirs, files in os.walk(cls_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                fp = os.path.join(root, f)
                try:
                    cap = cv2.VideoCapture(fp)
                    if cap.isOpened():
                        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        if fc > 0:
                            count += 1
                    else:
                        cap.release()
                except:
                    pass
    return count


# Check existing inventory
print("=" * 60)
print("CURRENT VALID VIDEO INVENTORY")
print("=" * 60)
classes_needed = []
for cls in CLASS_NAMES:
    cls_dir = os.path.join(RAW_DATA_DIR, cls)
    n = count_valid_videos(cls_dir)
    status = "OK" if n >= MIN_VIDEOS else "NEED MORE"
    print(f"  {cls:12s}: {n:3d} valid [{status}]")
    if n < MIN_VIDEOS:
        classes_needed.append(cls)

if not classes_needed:
    print("\nAll classes have sufficient videos.")
    sys.exit(0)

print(f"\nClasses needing download: {classes_needed}")

# For each class that needs data, list repo files and download .mp4s
for cls in classes_needed:
    repo_dir = TARGET_REPO_MAP.get(cls, cls)
    print(f"\n--- Downloading videos for '{cls}' (repo dir: '{repo_dir}') ---")
    cls_dir = os.path.join(RAW_DATA_DIR, cls, "hf_real")
    os.makedirs(cls_dir, exist_ok=True)
    
    try:
        items = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=repo_dir))
        mp4_files = [item for item in items if hasattr(item, "rfilename") and item.rfilename.endswith(".mp4")]
        print(f"  Found {len(mp4_files)} .mp4 files in repo/{repo_dir}/")
        
        for i, item in enumerate(mp4_files):
            fname = os.path.basename(item.rfilename)
            dest_path = os.path.join(cls_dir, fname)
            
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
                print(f"  [{i+1}/{len(mp4_files)}] Already exists: {fname}")
                continue
            
            try:
                # Download from HF repo using LFS
                downloaded_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=item.rfilename,
                    repo_type="dataset",
                    local_dir=None  # Use default cache
                )
                # Copy to our raw directory
                shutil.copy2(downloaded_path, dest_path)
                fsize = os.path.getsize(dest_path)
                print(f"  [{i+1}/{len(mp4_files)}] Downloaded: {fname} ({fsize:,} bytes)")
            except Exception as e:
                print(f"  [{i+1}/{len(mp4_files)}] FAILED: {fname}: {e}")
    except Exception as e:
        if "404" in str(e):
            print(f"  Directory '{repo_dir}' not found in HF repo. This class needs local recordings.")
        else:
            print(f"  ERROR listing repo for {repo_dir}: {e}")

# Final inventory
print(f"\n{'='*60}")
print("FINAL VALID VIDEO INVENTORY")
print("=" * 60)
total = 0
missing = []
for cls in CLASS_NAMES:
    cls_dir = os.path.join(RAW_DATA_DIR, cls)
    n = count_valid_videos(cls_dir)
    total += n
    status = "OK" if n >= MIN_VIDEOS else ("LOW" if n > 0 else "MISSING")
    print(f"  {cls:12s}: {n:3d} valid [{status}]")
    if n == 0:
        missing.append(cls)
print(f"\n  TOTAL: {total}")
if missing:
    print(f"\n  STILL MISSING: {missing}")
    print("  These classes require additional real video data.")
print("=" * 60)
