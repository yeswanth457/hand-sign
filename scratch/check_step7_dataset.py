import os
import sys
import glob
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, DATASET_DIR, RAW_DATA_DIR, TOKENS_DIR

classes_to_check = ["water", "hello", "thank_you", "please", "school", "no", "thalapathy"]

csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")
meta_rows = []
if os.path.exists(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

print(f"{'='*70}")
print("STEP 7 — DATASET VERIFICATION REPORT")
print(f"{'='*70}")

for c in classes_to_check:
    raw_dir = os.path.join(RAW_DATA_DIR, c)
    raw_videos = []
    if os.path.exists(raw_dir):
        raw_videos = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))]
    
    # Matching rows in dataset.csv
    c_rows = [r for r in meta_rows if r.get("sign_class") == c]
    valid_videos = [r for r in c_rows if r.get("status") == "success"]
    
    total_frames = 0
    one_hand_frames = 0
    two_hand_frames = 0
    left_hand_frames = 0
    right_hand_frames = 0
    
    for r in valid_videos:
        token_file = r.get("token_file")
        if token_file and os.path.exists(token_file):
            t_data = np.load(token_file)
            tokens = t_data["tokens"]
            n_f = len(tokens)
            total_frames += n_f
            for row in tokens:
                has_lh = np.any(row[:6] != 0)
                has_rh = np.any(row[6:12] != 0)
                if has_lh:
                    left_hand_frames += 1
                if has_rh:
                    right_hand_frames += 1
                if has_lh and has_rh:
                    two_hand_frames += 1
                elif has_lh or has_rh:
                    one_hand_frames += 1

    print(f"\nClass: {c.upper()}")
    print(f"  Raw videos       : {len(raw_videos)}")
    print(f"  Valid videos     : {len(valid_videos)}")
    print(f"  Valid frames     : {total_frames}")
    print(f"  1-hand frames    : {one_hand_frames}")
    print(f"  2-hand frames    : {two_hand_frames}")
    print(f"  Left-hand frames : {left_hand_frames}")
    print(f"  Right-hand frames: {right_hand_frames}")
