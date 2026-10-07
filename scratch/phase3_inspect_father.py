"""
Phase 3: Forensic Inspection of all Father videos.
Calculates frame counts, hand detection %, left/right hand %, sequence lengths, and token quality.
"""

import os
import glob
import numpy as np

def inspect_all_father():
    print("=" * 70)
    print("PHASE 3: FORENSIC INSPECTION OF FATHER RECORDINGS")
    print("=" * 70)

    token_files = glob.glob("dataset/tokens/father/**/*.npz", recursive=True)
    print(f"Total Father token files found: {len(token_files)}")

    total_frames = 0
    total_valid = 0
    total_lh_frames = 0
    total_rh_frames = 0
    total_both_frames = 0
    total_nohand_frames = 0

    print(f"\n{'Filename':<40} | {'Frames':<6} | {'LH %':<6} | {'RH %':<6} | {'Both %':<6} | {'NoHand %':<8} | {'Quality'}")
    print("-" * 90)

    for f in sorted(token_files):
        fname = os.path.basename(f)
        d = np.load(f)
        tk = d["tokens"] # (N, 12)
        n = len(tk)
        total_frames += n

        lh_mask = np.any(tk[:, :6] != 0, axis=1)
        rh_mask = np.any(tk[:, 6:] != 0, axis=1)
        both_mask = lh_mask & rh_mask
        single_lh_mask = lh_mask & (~rh_mask)
        single_rh_mask = rh_mask & (~lh_mask)
        nohand_mask = (~lh_mask) & (~rh_mask)

        n_lh = int(np.sum(single_lh_mask))
        n_rh = int(np.sum(single_rh_mask))
        n_both = int(np.sum(both_mask))
        n_nohand = int(np.sum(nohand_mask))

        total_lh_frames += n_lh
        total_rh_frames += n_rh
        total_both_frames += n_both
        total_nohand_frames += n_nohand

        lh_pct = (n_lh / n) * 100
        rh_pct = (n_rh / n) * 100
        both_pct = (n_both / n) * 100
        nohand_pct = (n_nohand / n) * 100

        quality = "EXCELLENT" if np.isfinite(tk).all() and n >= 10 else "VALID"
        print(f"{fname[:40]:<40} | {n:<6d} | {lh_pct:5.1f}% | {rh_pct:5.1f}% | {both_pct:5.1f}% | {nohand_pct:7.1f}% | {quality}")

    print("-" * 90)
    print(f"TOTAL FRAMES ACROSS ALL FATHER VIDEOS: {total_frames}")
    print(f"  Single Left-Hand frames:  {total_lh_frames:4d} ({total_lh_frames/total_frames*100:5.1f}%)")
    print(f"  Single Right-Hand frames: {total_rh_frames:4d} ({total_rh_frames/total_frames*100:5.1f}%)")
    print(f"  Two-Hand frames:          {total_both_frames:4d} ({total_both_frames/total_frames*100:5.1f}%)")
    print(f"  No-Hand frames:           {total_nohand_frames:4d} ({total_nohand_frames/total_frames*100:5.1f}%)")
    print(f"  Overall Hand Detection:   {((total_frames - total_nohand_frames)/total_frames*100):5.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    inspect_all_father()
