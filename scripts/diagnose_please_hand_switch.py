"""
PLEASE Hand Switch & Live Sequence Diagnostic Script
====================================================
Performs:
  1. Direct CNN inference on the captured (25, 6) live sequence.
  2. Top 5 predictions with exact probabilities.
  3. Detailed comparison with training token distributions for:
     - PLEASE
     - WATER
     - SCHOOL
     - THANK_YOU
  4. Per-feature distances and similarity metrics for Hx, Hy, Mx, My, Rx, Ry.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import CLASS_NAMES, NUM_CLASSES, MODEL_DIR, DATASET_DIR
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.build_real_split import resample_tokens

TOKENS_DIR = os.path.join(DATASET_DIR, "tokens")
LIVE_SEQ_NPY = os.path.join(MODEL_DIR, "debug_failed_please_live_sequence.npy")
FAILED_PLEASE_NPY = os.path.join(MODEL_DIR, "debug_failed_please_as_water.npy")

def load_class_tokens(class_name):
    token_dir = os.path.join(TOKENS_DIR, class_name)
    sequences = []
    if not os.path.isdir(token_dir):
        return sequences

    for root_dir, _, files in os.walk(token_dir):
        for fname in sorted(files):
            if not fname.endswith(".npz"):
                continue
            fpath = os.path.join(root_dir, fname)
            try:
                npz = np.load(fpath, allow_pickle=True)
                tokens = None
                for key in ["tokens", "token_sequence", "arr_0", "data"]:
                    if key in npz:
                        tokens = npz[key].astype(np.float32)
                        break
                if tokens is None and len(npz.files) > 0:
                    tokens = npz[npz.files[0]].astype(np.float32)

                if tokens is not None and tokens.ndim == 2 and tokens.shape[1] == 6:
                    if tokens.shape[0] != 25:
                        tokens = resample_tokens(tokens, target_t=25)
                    sequences.append(tokens)
            except Exception:
                pass
    return sequences

def run_diagnostic():
    print("=" * 60)
    print("STEP 7 & 8: LIVE PLEASE SEQUENCE DIAGNOSTIC & COMPARISON")
    print("=" * 60)

    target_npy = None
    if os.path.exists(LIVE_SEQ_NPY):
        target_npy = LIVE_SEQ_NPY
    elif os.path.exists(FAILED_PLEASE_NPY):
        target_npy = FAILED_PLEASE_NPY

    if target_npy is None:
        print("No live sequence file found in models/ directory.")
        print("Please perform the PLEASE gesture in the browser first.")
        return

    live_tokens = np.load(target_npy)
    print(f"Loaded Live Sequence: {target_npy}")
    print(f"Sequence Shape: {live_tokens.shape}")

    if live_tokens.shape != (25, 6):
        print(f"ERROR: Expected (25, 6), got {live_tokens.shape}")
        return

    # Direct CNN-GRU Inference
    engine = CNNGRUInferenceEngine()
    pred = engine.predict_sequence(live_tokens)
    top_class = pred.get("word", "--")
    top_conf = pred.get("confidence", 0.0)
    probs = pred.get("probabilities", {})
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\n" + "-" * 50)
    print("STEP 7: DIRECT CNN-GRU TOP 5 PREDICTIONS")
    print("-" * 50)
    print(f"DIRECT_CNN_PREDICTED_CLASS={top_class}")
    print(f"DIRECT_CNN_PREDICTED_CONFIDENCE={top_conf:.4f}")
    for i, (c, p) in enumerate(sorted_probs, 1):
        print(f"TOP{i}={c} (prob={p:.4f})")

    # Step 8: Compare with training distributions
    print("\n" + "-" * 50)
    print("STEP 8: DISTRIBUTION COMPARISON (PLEASE, WATER, SCHOOL, THANK_YOU)")
    print("-" * 50)

    classes_to_compare = ["please", "water", "school", "thank_you"]
    live_flat = live_tokens.flatten()
    feat_names = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]

    live_mean = live_tokens.mean(axis=0)
    live_std = live_tokens.std(axis=0)
    print(f"LIVE_GESTURE_MEAN = {[round(float(v), 4) for v in live_mean]}")
    print(f"LIVE_GESTURE_STD  = {[round(float(v), 4) for v in live_std]}")

    for cname in classes_to_compare:
        seqs = load_class_tokens(cname)
        if not seqs:
            print(f"\n[CLASS: {cname.upper()}] No training tokens found.")
            continue

        stack = np.stack(seqs, axis=0) # (N, 25, 6)
        c_mean = stack.mean(axis=(0, 1))
        c_std = stack.std(axis=(0, 1))
        centroid_25x6 = stack.mean(axis=0)

        euclidean_dist = float(np.linalg.norm(live_tokens - centroid_25x6))
        
        sims = []
        for s in seqs:
            s_flat = s.flatten()
            sim = float(np.dot(live_flat, s_flat) / (np.linalg.norm(live_flat) * np.linalg.norm(s_flat) + 1e-8))
            sims.append(sim)
        avg_cos_sim = float(np.mean(sims)) if sims else 0.0

        print(f"\n[CLASS: {cname.upper()}] (Samples={len(seqs)})")
        print(f"  Training Mean: {[round(float(v), 4) for v in c_mean]}")
        print(f"  Training Std : {[round(float(v), 4) for v in c_std]}")
        print(f"  Euclidean Distance to Live Sequence: {euclidean_dist:.4f}")
        print(f"  Average Cosine Similarity: {avg_cos_sim:.4f}")
        print(f"  Per-feature absolute difference (|live_mean - train_mean|):")
        for fi, fn in enumerate(feat_names):
            diff = abs(float(live_mean[fi]) - float(c_mean[fi]))
            print(f"    {fn}: live={live_mean[fi]:.4f}, train={c_mean[fi]:.4f}, diff={diff:.4f}")

if __name__ == "__main__":
    run_diagnostic()
