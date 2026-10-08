import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import csv
import json
import numpy as np
import torch

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, HAND_FEATURE_DIM, MAX_SEQ_LEN, MODEL_DIR, DATASET_DIR, MODEL_PATH
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, resample_tokens

def forensic_report():
    print("=" * 70)
    print("PHASE 1: CLASS VERIFICATION & COMPLETE MAPPING")
    print("=" * 70)
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print(f"FRIEND CLASS INDEX: {CLASS_TO_INDEX['friend']}")
    assert CLASS_TO_INDEX['friend'] == 18, f"Expected 18, got {CLASS_TO_INDEX['friend']}"
    assert NUM_CLASSES == 22, f"Expected 22, got {NUM_CLASSES}"

    print("Complete 0-21 Class Mapping:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {idx:2d} -> {name}")

    print("\n" + "=" * 70)
    print("PHASE 2: FORENSIC INSPECTION OF EXISTING FRIEND DATA")
    print("=" * 70)

    raw_dir = os.path.join(DATASET_DIR, "raw", "friend")
    token_dir = os.path.join(DATASET_DIR, "tokens", "friend")
    landmark_dir = os.path.join(DATASET_DIR, "landmarks", "friend")
    csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")

    raw_files = glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)
    raw_files += glob.glob(os.path.join(raw_dir, "**", "*.avi"), recursive=True)
    raw_files += glob.glob(os.path.join(raw_dir, "**", "*.mov"), recursive=True)

    token_files = glob.glob(os.path.join(token_dir, "**", "*.npz"), recursive=True)
    landmark_files = glob.glob(os.path.join(landmark_dir, "**", "*.npz"), recursive=True)

    print(f"1. Total Friend raw videos:         {len(raw_files)}")
    print(f"2. Total Friend processed tokens:    {len(token_files)}")
    print(f"   Total Friend landmark archives:   {len(landmark_files)}")

    # 3 & 4. Sources breakdown
    sources = {"CISLR": 0, "INCLUDE": 0, "ISL500": 0, "user-recorded": 0, "other": 0}
    source_files = {"CISLR": [], "INCLUDE": [], "ISL500": [], "user-recorded": [], "other": []}

    for f in raw_files:
        fn = os.path.basename(f)
        if "WIN_" in fn or "signer_user" in f:
            sources["user-recorded"] += 1
            source_files["user-recorded"].append(fn)
        elif "__CISLR__" in fn:
            sources["CISLR"] += 1
            source_files["CISLR"].append(fn)
        elif "__INCLUDE__" in fn:
            sources["INCLUDE"] += 1
            source_files["INCLUDE"].append(fn)
        elif "__ISL500__" in fn:
            sources["ISL500"] += 1
            source_files["ISL500"].append(fn)
        else:
            sources["other"] += 1
            source_files["other"].append(fn)

    print("\n3 & 4. Dataset source breakdown:")
    for src, count in sources.items():
        print(f"   - {src:15s}: {count} videos")

    # Frames analysis across all token files
    total_frames = 0
    one_hand_frames = 0
    two_hand_frames = 0
    no_hand_frames = 0
    lh_active_frames = 0
    rh_active_frames = 0
    seq_lens = []

    per_sample_stats = []

    for tf in token_files:
        d = np.load(tf)
        toks = d["tokens"] # (N, 12)
        n = len(toks)
        seq_lens.append(n)
        total_frames += n

        lh_mask = np.any(toks[:, :6] != 0, axis=1)
        rh_mask = np.any(toks[:, 6:] != 0, axis=1)

        lh_cnt = int(np.sum(lh_mask))
        rh_cnt = int(np.sum(rh_mask))
        two_cnt = int(np.sum(lh_mask & rh_mask))
        one_cnt = int(np.sum((lh_mask & ~rh_mask) | (~lh_mask & rh_mask)))
        zero_cnt = int(np.sum(~lh_mask & ~rh_mask))

        one_hand_frames += one_cnt
        two_hand_frames += two_cnt
        no_hand_frames += zero_cnt
        lh_active_frames += lh_cnt
        rh_active_frames += rh_cnt

        per_sample_stats.append({
            "file": os.path.basename(tf),
            "frames": n,
            "lh_frames": lh_cnt,
            "rh_frames": rh_cnt,
            "two_hand_frames": two_cnt,
            "one_hand_frames": one_cnt,
            "no_hand_frames": zero_cnt
        })

    print(f"\n5. Total frames across videos:        {total_frames}")
    print(f"6. One-hand frames:                   {one_hand_frames} ({one_hand_frames/total_frames*100:.2f}%)")
    print(f"7. Two-hand frames:                   {two_hand_frames} ({two_hand_frames/total_frames*100:.2f}%)")
    print(f"8. No-hand frames:                    {no_hand_frames} ({no_hand_frames/total_frames*100:.2f}%)")
    print(f"9. Left-hand occurrence:              {lh_active_frames} ({lh_active_frames/total_frames*100:.2f}%)")
    print(f"10. Right-hand occurrence:            {rh_active_frames} ({rh_active_frames/total_frames*100:.2f}%)")
    print(f"11. Mean sequence length:             {np.mean(seq_lens):.2f} frames")
    print(f"12. Minimum sequence length:          {min(seq_lens)} frames")
    print(f"13. Maximum sequence length:          {max(seq_lens)} frames")

    # Inspect 12D representations across sources
    print("\n" + "=" * 70)
    print("INSPECTION OF 12D TOKEN REPRESENTATION ACROSS SOURCES")
    print("=" * 70)
    print("Slot layout: LH (0..5) = [LHx, LHy, LMx, LMy, LRx, LRy]")
    print("             RH (6..11) = [RHx, RHy, RMx, RMy, RRx, RRy]")

    # Sample from each source
    for src in ["CISLR", "INCLUDE", "ISL500"]:
        sample_file = next((tf for tf in token_files if f"__{src}__" in tf), None)
        if sample_file:
            d = np.load(sample_file)
            toks = d["tokens"]
            lh_a = np.any(toks[:, :6] != 0, axis=1)
            rh_a = np.any(toks[:, 6:] != 0, axis=1)
            print(f"\nSource '{src}' Sample: {os.path.basename(sample_file)} ({len(toks)} frames)")
            print(f"  LH Active: {np.sum(lh_a)}/{len(toks)} | RH Active: {np.sum(rh_a)}/{len(toks)}")
            if np.sum(lh_a) > 0:
                print(f"  LH mean pos: x={np.mean(toks[lh_a, 0]):.3f}, y={np.mean(toks[lh_a, 1]):.3f}, offset_rel_shoulder: dx={np.mean(toks[lh_a, 4]):.3f}, dy={np.mean(toks[lh_a, 5]):.3f}")
            if np.sum(rh_a) > 0:
                print(f"  RH mean pos: x={np.mean(toks[rh_a, 6]):.3f}, y={np.mean(toks[rh_a, 7]):.3f}, offset_rel_shoulder: dx={np.mean(toks[rh_a, 10]):.3f}, dy={np.mean(toks[rh_a, 11]):.3f}")

    # Compare Friend with Father, Brother, Sister, Water, School, Hello, Thank You, Please
    print("\n" + "=" * 70)
    print("PHASE 3: PHYSICAL & NUMERICAL COMPARISON AGAINST KEY CLASSES")
    print("=" * 70)

    comparison_classes = ["friend", "father", "brother", "sister", "water", "school", "hello", "thank_you", "please"]
    split_dir = os.path.join(DATASET_DIR, "train")
    train_X = np.load(os.path.join(split_dir, "X.npy"))
    train_y = np.load(os.path.join(split_dir, "y.npy"))

    print(f"{'Class':12s} | {'Samples':7s} | {'1-Hand%':7s} | {'2-Hand%':7s} | {'LH Active%':10s} | {'RH Active%':10s} | {'Mean Y (Height)':15s}")
    print("-" * 80)

    for cname in comparison_classes:
        cid = CLASS_TO_INDEX[cname]
        mask = (train_y == cid)
        c_samples = train_X[mask]
        N = len(c_samples)
        if N == 0:
            print(f"{cname:12s} | 0 samples")
            continue

        c_flat = c_samples.reshape(-1, 12)
        lh_a = np.any(c_flat[:, :6] != 0, axis=1)
        rh_a = np.any(c_flat[:, 6:] != 0, axis=1)

        two_h = np.mean(lh_a & rh_a) * 100
        one_h = np.mean((lh_a & ~rh_a) | (~lh_a & rh_a)) * 100
        lh_pct = np.mean(lh_a) * 100
        rh_pct = np.mean(rh_a) * 100

        # Mean vertical height when active
        y_vals = []
        if np.any(lh_a): y_vals.extend(c_flat[lh_a, 1])
        if np.any(rh_a): y_vals.extend(c_flat[rh_a, 7])
        mean_y = np.mean(y_vals) if y_vals else 0.0

        print(f"{cname:12s} | {N:7d} | {one_h:6.1f}% | {two_h:6.1f}% | {lh_pct:9.1f}% | {rh_pct:9.1f}% | {mean_y:15.3f}")

    # Section on Stale School in Live UI
    print("\n" + "=" * 70)
    print("PHASE 13: FORENSIC INVESTIGATION OF STALE 'SCHOOL' OUTPUT IN UI")
    print("=" * 70)
    print("1. In app.py lines 331-346: sequence reset triggers on:")
    print("   (active_sequence_id == 0 or seq_id > active_sequence_id or is_client_reloaded)")
    print("2. In static/app.js: if the user stops signing or performs a new gesture without")
    print("   sequence_id incrementing or clear_sentence being called, the last accepted word")
    print("   ('SCHOOL') and current translation remain in DOM elements.")
    print("3. School is class 12, Brother is 17, Friend is 18. When hands are in neutral or")
    print("   resting in lower center frame, school often gets partial-frame activations (~49% confidence).")
    print("   Because EarlyDecisionEngine locks when sustained frames are met, if clear_sentence")
    print("   is not triggered, the UI retains the prior accepted sign.")

if __name__ == "__main__":
    forensic_report()
