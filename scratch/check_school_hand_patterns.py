import os
import sys
import cv2
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.landmark_extractor import LandmarkExtractor

extractor = LandmarkExtractor()
school_vids_dir = os.path.join(BASE_DIR, "dataset", "raw", "school", "hf_real")
vids = [os.path.join(school_vids_dir, f) for f in os.listdir(school_vids_dir) if f.endswith(".mp4")]

print(f"Analyzing {len(vids)} school training videos for hand patterns...")
total_frames = 0
zero_hands = 0
one_hand = 0
two_hands = 0

sample_patterns = {}

for vpath in vids[:10]:
    vname = os.path.basename(vpath)
    cap = cv2.VideoCapture(vpath)
    counts = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = extractor.process_frame(frame, timestamp_ms=int(idx * 33.3))
        nh = len(res.get("hands", []))
        counts.append(nh)
        idx += 1
    cap.release()
    total_frames += len(counts)
    zero_hands += counts.count(0)
    one_hand += counts.count(1)
    two_hands += counts.count(2)
    sample_patterns[vname] = {
        "frames": len(counts),
        "0_hands": counts.count(0),
        "1_hand": counts.count(1),
        "2_hands": counts.count(2),
    }

print("\nSample Training School Hand Patterns:")
for k, v in sample_patterns.items():
    print(f"  {k:<30}: total={v['frames']}, 0-hand={v['0_hands']}, 1-hand={v['1_hand']}, 2-hands={v['2_hands']}")

print(f"\nOverall across 10 sample videos ({total_frames} frames):")
print(f"  0 hands: {zero_hands} ({zero_hands/total_frames*100:.1f}%)")
print(f"  1 hand:  {one_hand} ({one_hand/total_frames*100:.1f}%)")
print(f"  2 hands: {two_hands} ({two_hands/total_frames*100:.1f}%)")
