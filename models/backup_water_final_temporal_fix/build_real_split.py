"""
Build clean REAL-ONLY train/validation/test splits from extracted MediaPipe 6D tokens.
Filters to the 21 active classes, remaps class IDs to [0..20],
and creates a proper stratified split with signer-level separation where possible.
"""

import os
import sys
import csv
import shutil
import numpy as np
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CLASS_NAMES, CLASS_TO_INDEX, DATASET_DIR, NUM_CLASSES, TOKEN_DIM


def resample_tokens(tokens, target_t=25):
    """Resamples token sequence (N, C) to exact target length T=25 using smooth temporal interpolation."""
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    n, c = tokens.shape
    if n == target_t:
        return tokens.astype(np.float32)
    old_t = np.linspace(0.0, 1.0, max(1, n))
    new_t = np.linspace(0.0, 1.0, target_t)
    resampled = np.zeros((target_t, c), dtype=np.float32)
    for j in range(c):
        resampled[:, j] = np.interp(new_t, old_t, tokens[:, j])
    return resampled


def build_real_splits():
    """Build train/val/test splits from real extracted tokens."""
    # 1. Quarantine old data
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

    # 3. Filter to ONLY the 22 active classes and group by class
    active_set = set(CLASS_NAMES)
    class_samples = defaultdict(list)  # class_name -> [(tokens_25, class_id, record_info), ...]
    
    # Track signers per class for signer-level split
    class_signers = defaultdict(lambda: defaultdict(list))

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
        
        # Validate tokens
        if raw_tokens.ndim != 2 or raw_tokens.shape[1] != TOKEN_DIM:
            print(f"[Warning] Invalid token shape {raw_tokens.shape} in {token_file} (expected {TOKEN_DIM} features)")
            continue
        if not np.isfinite(raw_tokens).all():
            print(f"[Warning] Non-finite values in {token_file}")
            continue
        
        tokens_25 = resample_tokens(raw_tokens, target_t=25)
        class_id = CLASS_TO_INDEX[sign_class]
        signer_id = r.get("signer_id", "signer_01")

        record_info = {
            "video_id": r["video_id"],
            "sign_class": sign_class,
            "class_id": class_id,
            "signer_id": signer_id,
            "original_frames": raw_tokens.shape[0],
            "resampled_frames": 25,
        }

        class_samples[sign_class].append((tokens_25, class_id, record_info))
        class_signers[sign_class][signer_id].append(len(class_samples[sign_class]) - 1)

    if skipped_classes:
        print(f"[Filter] Skipped non-active classes: {sorted(skipped_classes)}")

    # 4. Stratified split: 70/15/15 train/val/test with signer separation where possible
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []
    train_meta, val_meta, test_meta = [], [], []

    np.random.seed(42)

    for class_name in CLASS_NAMES:
        samples = class_samples.get(class_name, [])
        if len(samples) == 0:
            print(f"[WARNING] No samples found for class '{class_name}'!")
            continue

        # Try signer-level split
        signers = list(class_signers[class_name].keys())
        
        if len(signers) >= 3 and len(samples) >= 5:
            # Signer-level split: assign signers to train/val/test
            np.random.shuffle(signers)
            n_test_signers = max(1, len(signers) // 5)
            n_val_signers = max(1, len(signers) // 5)
            
            test_signers = set(signers[:n_test_signers])
            val_signers = set(signers[n_test_signers:n_test_signers + n_val_signers])
            train_signers = set(signers[n_test_signers + n_val_signers:])
            
            for tokens, class_id, info in samples:
                sid = info["signer_id"]
                if sid in test_signers:
                    test_x.append(tokens); test_y.append(class_id); test_meta.append(info)
                elif sid in val_signers:
                    val_x.append(tokens); val_y.append(class_id); val_meta.append(info)
                else:
                    train_x.append(tokens); train_y.append(class_id); train_meta.append(info)
        else:
            # Random split for small datasets
            indices = np.arange(len(samples))
            np.random.shuffle(indices)

            if len(samples) >= 5:
                n_test = max(1, int(len(samples) * 0.15))
                n_val = max(1, int(len(samples) * 0.15))
            elif len(samples) >= 3:
                n_test = 1
                n_val = 1
            elif len(samples) >= 2:
                n_test = 0
                n_val = 1
            else:
                n_test = 0
                n_val = 0

            test_indices = set(indices[:n_test])
            val_indices = set(indices[n_test:n_test + n_val])

            for i, (tokens, class_id, info) in enumerate(samples):
                if i in test_indices:
                    test_x.append(tokens); test_y.append(class_id); test_meta.append(info)
                elif i in val_indices:
                    val_x.append(tokens); val_y.append(class_id); val_meta.append(info)
                else:
                    train_x.append(tokens); train_y.append(class_id); train_meta.append(info)
                    # For single-sample classes, also put in val for coverage
                    if len(samples) == 1:
                        val_x.append(tokens); val_y.append(class_id); val_meta.append(info)

        n_tr = sum(1 for t, c, i in samples if i not in [m for m in test_meta + val_meta])
        print(f"  {class_name}: {len(samples)} total")

    # 5. Convert to numpy arrays
    arr_train_x = np.array(train_x, dtype=np.float32) if train_x else np.empty((0, 25, TOKEN_DIM), dtype=np.float32)
    arr_train_y = np.array(train_y, dtype=np.int64) if train_y else np.empty((0,), dtype=np.int64)
    arr_val_x = np.array(val_x, dtype=np.float32) if val_x else np.empty((0, 25, TOKEN_DIM), dtype=np.float32)
    arr_val_y = np.array(val_y, dtype=np.int64) if val_y else np.empty((0,), dtype=np.int64)
    arr_test_x = np.array(test_x, dtype=np.float32) if test_x else np.empty((0, 25, TOKEN_DIM), dtype=np.float32)
    arr_test_y = np.array(test_y, dtype=np.int64) if test_y else np.empty((0,), dtype=np.int64)

    # 6. Save feature normalization stats from training set
    if len(arr_train_x) > 0:
        all_features = arr_train_x.reshape(-1, TOKEN_DIM)
        feature_mean = np.mean(all_features, axis=0).astype(np.float32)
        feature_std = np.std(all_features, axis=0).astype(np.float32)
        feature_std = np.where(feature_std < 1e-7, 1.0, feature_std)
        
        np.save(os.path.join(os.path.dirname(DATASET_DIR), "models", "feature_mean.npy"), feature_mean)
        np.save(os.path.join(os.path.dirname(DATASET_DIR), "models", "feature_std.npy"), feature_std)
        print(f"\n[Stats] Feature mean: {feature_mean}")
        print(f"[Stats] Feature std:  {feature_std}")

    # 7. Save to active split directories
    np.save(os.path.join(DATASET_DIR, "train", "X.npy"), arr_train_x)
    np.save(os.path.join(DATASET_DIR, "train", "y.npy"), arr_train_y)
    np.save(os.path.join(DATASET_DIR, "val", "X.npy"), arr_val_x)
    np.save(os.path.join(DATASET_DIR, "val", "y.npy"), arr_val_y)
    np.save(os.path.join(DATASET_DIR, "test", "X.npy"), arr_test_x)
    np.save(os.path.join(DATASET_DIR, "test", "y.npy"), arr_test_y)

    # 8. Print summary report
    print(f"\n{'='*60}")
    print(f"REAL ISL 21-CLASS DATASET SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Active Classes : {CLASS_NAMES}")
    print(f"TRAIN Set      : X={arr_train_x.shape}, y={arr_train_y.shape}")
    print(f"VAL Set        : X={arr_val_x.shape}, y={arr_val_y.shape}")
    print(f"TEST Set       : X={arr_test_x.shape}, y={arr_test_y.shape}")

    # Per-class breakdown
    print(f"\nPer-Class Distribution:")
    for cid, cname in enumerate(CLASS_NAMES):
        n_train = int((arr_train_y == cid).sum())
        n_val = int((arr_val_y == cid).sum())
        n_test = int((arr_test_y == cid).sum())
        print(f"  [{cid}] {cname:12s}: {n_train} train / {n_val} val / {n_test} test")

    total_real = len(train_x) + len(val_x) + len(test_x)
    print(f"\nTotal Real Sequences: {total_real}")

    nan_count = int(np.isnan(arr_train_x).sum() + np.isnan(arr_val_x).sum() + np.isnan(arr_test_x).sum())
    inf_count = int(np.isinf(arr_train_x).sum() + np.isinf(arr_val_x).sum() + np.isinf(arr_test_x).sum())
    print(f"NaN Count : {nan_count}")
    print(f"Inf Count : {inf_count}")

    # Data leakage check
    train_vids = {r["video_id"] for r in train_meta}
    val_vids = {r["video_id"] for r in val_meta}
    test_vids = {r["video_id"] for r in test_meta}
    tv_overlap = train_vids.intersection(val_vids)
    tt_overlap = train_vids.intersection(test_vids)
    print(f"Train-Val Video Leakage  : {'PASS (0 overlap)' if len(tv_overlap) == 0 else f'FAIL ({len(tv_overlap)} overlapping)'}")
    print(f"Train-Test Video Leakage : {'PASS (0 overlap)' if len(tt_overlap) == 0 else f'FAIL ({len(tt_overlap)} overlapping)'}")
    
    # Signer leakage check
    train_signers = {r["signer_id"] for r in train_meta}
    val_signers = {r["signer_id"] for r in val_meta}
    test_signers = {r["signer_id"] for r in test_meta}
    signer_tv = train_signers.intersection(val_signers)
    signer_tt = train_signers.intersection(test_signers)
    print(f"Train Signers: {sorted(train_signers)}")
    print(f"Val Signers:   {sorted(val_signers)}")
    print(f"Test Signers:  {sorted(test_signers)}")
    print(f"Signer Train-Val Overlap : {len(signer_tv)} signers")
    print(f"Signer Train-Test Overlap: {len(signer_tt)} signers")
    print("=" * 60)


if __name__ == "__main__":
    build_real_splits()
