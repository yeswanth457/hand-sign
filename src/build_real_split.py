"""
Build clean REAL-ONLY train/validation splits from extracted MediaPipe 6D tokens.
Filters to ACTIVE_VOCABULARY classes only, remaps class IDs to [0..N-1],
and creates a proper 80/20 per-class stratified split.
"""

import os
import sys
import csv
import shutil
import numpy as np
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ACTIVE_VOCABULARY, ACTIVE_WORD_TO_ID, DATASET_DIR


def resample_tokens(tokens, target_t=25):
    """Resamples or pads token sequence (N, 6) to exact target length T=25."""
    n, c = tokens.shape
    if n >= target_t:
        indices = np.linspace(0, n - 1, target_t, dtype=int)
        return tokens[indices]
    else:
        return np.pad(tokens, ((0, target_t - n), (0, 0)), mode="edge")


def build_real_splits():
    # 1. Quarantine old synthetic/stale data
    backup_dir = os.path.join(DATASET_DIR, "synthetic_legacy_backup")
    os.makedirs(backup_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATASET_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        for fname in ["X.npy", "y.npy"]:
            src = os.path.join(split_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(backup_dir, f"{split}_{fname}")
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                print(f"[Backup] Quarantined {src} -> {dst}")

    # 2. Load extracted real video token metadata
    csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        meta_rows = [r for r in csv.DictReader(f) if r.get("status") == "success"]

    # 3. Filter to ONLY active classes and group by class
    active_set = set(ACTIVE_VOCABULARY)
    class_samples = defaultdict(list)  # class_name -> [(tokens_25, record_info), ...]

    skipped_classes = set()
    for r in meta_rows:
        sign_class = r["sign_class"]
        token_file = r["token_file"]

        if sign_class not in active_set:
            skipped_classes.add(sign_class)
            continue

        if not os.path.exists(token_file):
            print(f"[Warning] Token file missing: {token_file}")
            continue

        data = np.load(token_file)
        raw_tokens = data["tokens"].astype(np.float32)
        tokens_25 = resample_tokens(raw_tokens, target_t=25)

        # Remap to active class ID (0, 1, 2)
        class_id = ACTIVE_WORD_TO_ID[sign_class]

        record_info = {
            "video_id": r["video_id"],
            "sign_class": sign_class,
            "class_id": class_id,
            "signer_id": r["signer_id"],
            "original_frames": raw_tokens.shape[0],
            "resampled_frames": 25,
        }

        class_samples[sign_class].append((tokens_25, class_id, record_info))

    if skipped_classes:
        print(f"[Filter] Skipped non-active classes: {sorted(skipped_classes)}")

    # 4. Stratified 80/20 split per class
    train_x, train_y = [], []
    val_x, val_y = [], []
    train_meta, val_meta = [], []

    np.random.seed(42)

    for class_name in ACTIVE_VOCABULARY:
        samples = class_samples.get(class_name, [])
        if len(samples) == 0:
            print(f"[WARNING] No samples found for class '{class_name}'!")
            continue

        # Shuffle samples for this class
        indices = np.arange(len(samples))
        np.random.shuffle(indices)

        # 80/20 split
        n_val = max(1, int(len(samples) * 0.20))
        val_indices = set(indices[:n_val])

        for i, (tokens, class_id, info) in enumerate(samples):
            if i in val_indices:
                val_x.append(tokens)
                val_y.append(class_id)
                val_meta.append(info)
            else:
                train_x.append(tokens)
                train_y.append(class_id)
                train_meta.append(info)

        print(f"  {class_name}: {len(samples)} total -> {len(samples) - n_val} train / {n_val} val")

    # 5. Convert to numpy arrays
    arr_train_x = np.array(train_x, dtype=np.float32) if train_x else np.empty((0, 25, 6), dtype=np.float32)
    arr_train_y = np.array(train_y, dtype=np.int64) if train_y else np.empty((0,), dtype=np.int64)

    arr_val_x = np.array(val_x, dtype=np.float32) if val_x else np.empty((0, 25, 6), dtype=np.float32)
    arr_val_y = np.array(val_y, dtype=np.int64) if val_y else np.empty((0,), dtype=np.int64)

    # 6. Save to active split directories
    np.save(os.path.join(DATASET_DIR, "train", "X.npy"), arr_train_x)
    np.save(os.path.join(DATASET_DIR, "train", "y.npy"), arr_train_y)

    np.save(os.path.join(DATASET_DIR, "val", "X.npy"), arr_val_x)
    np.save(os.path.join(DATASET_DIR, "val", "y.npy"), arr_val_y)

    # 7. Print summary report
    print("\n=== REAL ISL 3-CLASS DATASET SPLIT SUMMARY ===")
    print(f"Active Classes : {ACTIVE_VOCABULARY}")
    print(f"TRAIN Set      : X={arr_train_x.shape}, y={arr_train_y.shape}")
    print(f"VAL Set        : X={arr_val_x.shape}, y={arr_val_y.shape}")

    # Per-class breakdown
    print("\nPer-Class Distribution:")
    for cid, cname in enumerate(ACTIVE_VOCABULARY):
        n_train = int((arr_train_y == cid).sum())
        n_val = int((arr_val_y == cid).sum())
        print(f"  [{cid}] {cname:12s}: {n_train} train / {n_val} val")

    total_real = len(train_x) + len(val_x)
    print(f"\nTotal Real Sequences: {total_real}")

    nan_count = int(np.isnan(arr_train_x).sum() + np.isnan(arr_val_x).sum())
    inf_count = int(np.isinf(arr_train_x).sum() + np.isinf(arr_val_x).sum())
    print(f"NaN Count : {nan_count}")
    print(f"Inf Count : {inf_count}")

    # Data leakage check
    train_vids = {r["video_id"] for r in train_meta}
    val_vids = {r["video_id"] for r in val_meta}
    overlap = train_vids.intersection(val_vids)
    print(f"Video Leakage  : {'PASS (0 overlap)' if len(overlap) == 0 else f'FAIL ({len(overlap)} overlapping)'}")
    print("=" * 50)


if __name__ == "__main__":
    build_real_splits()
