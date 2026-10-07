import os
import sys
import numpy as np
from pathlib import Path

lm_dir = Path("dataset/landmarks")
classes = sorted([d.name for d in lm_dir.iterdir() if d.is_dir()])

print("=" * 70)
print("CHECKING HAND OCCURRENCE IN LANDMARKS DATASET ACROSS ALL CLASSES:")
print("=" * 70)

class_summary = {}

for c in classes:
    files = list((lm_dir / c).glob("**/*.npz"))
    total_samples = len(files)
    if total_samples == 0:
        continue
        
    two_hand_samples = 0
    one_hand_samples = 0
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            lms = data["landmarks"]
            # check hand count across frames
            max_hands = 0
            for frame_lm in lms:
                if isinstance(frame_lm, dict):
                    h_list = frame_lm.get("hands", [])
                    max_hands = max(max_hands, len(h_list))
            if max_hands >= 2:
                two_hand_samples += 1
            else:
                one_hand_samples += 1
        except Exception:
            pass
            
    class_summary[c] = {
        "total": total_samples,
        "1_hand": one_hand_samples,
        "2_hand": two_hand_samples,
        "type": "2-hand" if two_hand_samples > total_samples * 0.4 else "1-hand"
    }
    print(f"{c:15s}: {class_summary[c]['type']:6s} (Total: {total_samples:3d} | 1-hand: {one_hand_samples:3d}, 2-hand: {two_hand_samples:3d})")
