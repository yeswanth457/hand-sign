import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR, RAW_DATA_DIR, CLASS_TO_INDEX, NUM_CLASSES, TOKEN_DIM, MODEL_PATH

print("==================================================")
print("FORENSIC ANALYSIS: BROTHER vs WATER")
print("==================================================")

# Step 1: Inspect Current Brother and Water Class
print(f"NUM_CLASSES:           {NUM_CLASSES}")
print(f"TOKEN_DIM:             {TOKEN_DIM}")
print(f"MODEL_PATH:            {MODEL_PATH}")
print(f"Brother Class Index:   {CLASS_TO_INDEX.get('brother')}")
print(f"Water Class Index:     {CLASS_TO_INDEX.get('water')}")

csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    meta_rows = list(csv.DictReader(f))

# Step 2: Inspect Brother Training Data & Water Training Data
def inspect_class_data(cname):
    raw_dir = os.path.join(RAW_DATA_DIR, cname)
    raw_vids = []
    if os.path.exists(raw_dir):
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov", ".webm", ".mkv")):
                    raw_vids.append(os.path.join(root, file))

    rows = [r for r in meta_rows if r.get("sign_class") == cname]
    valid_rows = [r for r in rows if r.get("status") == "success"]
    invalid_rows = [r for r in rows if r.get("status") != "success"]
    
    total_frames = 0
    one_hand_frames = 0
    two_hand_frames = 0
    left_hand_frames = 0
    right_hand_frames = 0
    
    all_tokens = []
    
    for r in valid_rows:
        tf = r.get("token_file")
        if tf and os.path.exists(tf):
            data = np.load(tf)
            tokens = data["tokens"]
            all_tokens.append(tokens)
            total_frames += len(tokens)
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

    return {
        "raw_videos": len(raw_vids),
        "valid_videos": len(valid_rows),
        "invalid_videos": len(invalid_rows),
        "total_frames": total_frames,
        "one_hand_frames": one_hand_frames,
        "two_hand_frames": two_hand_frames,
        "left_hand_frames": left_hand_frames,
        "right_hand_frames": right_hand_frames,
        "tokens": all_tokens,
        "valid_rows": valid_rows
    }

b_stats = inspect_class_data("brother")
w_stats = inspect_class_data("water")

print("\n--- BROTHER TRAINING DATA (Step 2) ---")
print(f"videos:           {b_stats['raw_videos']}")
print(f"valid:            {b_stats['valid_videos']}")
print(f"invalid:          {b_stats['invalid_videos']}")
print(f"one-hand frames:  {b_stats['one_hand_frames']}")
print(f"two-hand frames:  {b_stats['two_hand_frames']}")
print(f"left-hand frames: {b_stats['left_hand_frames']}")
print(f"right-hand frames:{b_stats['right_hand_frames']}")

print("\nPer-sample Brother breakdown:")
for r in b_stats["valid_rows"]:
    tf = r["token_file"]
    data = np.load(tf)["tokens"]
    lh_cnt = sum(1 for row in data if np.any(row[:6] != 0))
    rh_cnt = sum(1 for row in data if np.any(row[6:12] != 0))
    both_cnt = sum(1 for row in data if np.any(row[:6] != 0) and np.any(row[6:12] != 0))
    print(f"  {r['video_id'][:35]:35s} | frames={len(data):3d} | LH={lh_cnt:3d} | RH={rh_cnt:3d} | Both={both_cnt:3d}")

print("\n--- WATER TRAINING DATA (Step 2) ---")
print(f"videos:           {w_stats['raw_videos']}")
print(f"valid:            {w_stats['valid_videos']}")
print(f"invalid:          {w_stats['invalid_videos']}")
print(f"one-hand frames:  {w_stats['one_hand_frames']}")
print(f"two-hand frames:  {w_stats['two_hand_frames']}")
print(f"left-hand frames: {w_stats['left_hand_frames']}")
print(f"right-hand frames:{w_stats['right_hand_frames']}")
