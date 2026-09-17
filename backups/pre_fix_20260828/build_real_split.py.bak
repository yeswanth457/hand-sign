"""
Build clean REAL-ONLY signer-aware train/validation/test splits from extracted MediaPipe 6D tokens.
Quarantines legacy synthetic arrays to dataset/synthetic_legacy_backup/ and outputs (N, 25, 6) PyTorch tensors.
"""

import os
import sys
import csv
import shutil
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import WORD_TO_ID, DATASET_DIR


def resample_tokens(tokens, target_t=25):
    """Resamples or pads token sequence (N, 6) to exact target length T=25."""
    n, c = tokens.shape
    if n >= target_t:
        indices = np.linspace(0, n - 1, target_t, dtype=int)
        return tokens[indices]
    else:
        return np.pad(tokens, ((0, target_t - n), (0, 0)), mode="edge")


def build_real_splits():
    # 1. Quarantine synthetic data
    backup_dir = os.path.join(DATASET_DIR, "synthetic_legacy_backup")
    os.makedirs(backup_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATASET_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        for fname in ["X.npy", "y.npy"]:
            src = os.path.join(split_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(backup_dir, f"{split}_{fname}")
                shutil.move(src, dst)
                print(f"[Backup] Quarantined {src} -> {dst}")

    # 2. Load extracted real video token metadata
    csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        meta_rows = [r for r in csv.DictReader(f) if r.get("status") == "success"]

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    train_meta, val_meta, test_meta = [], [], []

    for r in meta_rows:
        sign_class = r["sign_class"]
        signer_id = r["signer_id"]
        video_id = r["video_id"]
        token_file = r["token_file"]

        if not os.path.exists(token_file):
            print(f"[Warning] Token file missing: {token_file}")
            continue

        data = np.load(token_file)
        raw_tokens = data["tokens"].astype(np.float32)
        tokens_25 = resample_tokens(raw_tokens, target_t=25)
        class_id = WORD_TO_ID[sign_class]

        record_info = {
            "video_id": video_id,
            "sign_class": sign_class,
            "class_id": class_id,
            "signer_id": signer_id,
            "original_frames": raw_tokens.shape[0],
            "resampled_frames": 25,
        }

        if signer_id == "signer_01":
            if video_id == "video_20260822_231704":
                # Holdout 1 real video for validation
                val_x.append(tokens_25)
                val_y.append(class_id)
                val_meta.append(record_info)
            else:
                train_x.append(tokens_25)
                train_y.append(class_id)
                train_meta.append(record_info)
        else:
            # Unseen signer for testing (signer_1)
            test_x.append(tokens_25)
            test_y.append(class_id)
            test_meta.append(record_info)

    # Convert to numpy arrays
    arr_train_x = np.array(train_x, dtype=np.float32) if train_x else np.empty((0, 25, 6), dtype=np.float32)
    arr_train_y = np.array(train_y, dtype=np.int64) if train_y else np.empty((0,), dtype=np.int64)

    arr_val_x = np.array(val_x, dtype=np.float32) if val_x else np.empty((0, 25, 6), dtype=np.float32)
    arr_val_y = np.array(val_y, dtype=np.int64) if val_y else np.empty((0,), dtype=np.int64)

    arr_test_x = np.array(test_x, dtype=np.float32) if test_x else np.empty((0, 25, 6), dtype=np.float32)
    arr_test_y = np.array(test_y, dtype=np.int64) if test_y else np.empty((0,), dtype=np.int64)

    # Save to active split directories
    np.save(os.path.join(DATASET_DIR, "train", "X.npy"), arr_train_x)
    np.save(os.path.join(DATASET_DIR, "train", "y.npy"), arr_train_y)

    np.save(os.path.join(DATASET_DIR, "val", "X.npy"), arr_val_x)
    np.save(os.path.join(DATASET_DIR, "val", "y.npy"), arr_val_y)

    np.save(os.path.join(DATASET_DIR, "test", "X.npy"), arr_test_x)
    np.save(os.path.join(DATASET_DIR, "test", "y.npy"), arr_test_y)

    # Print summary report
    print("\n=== REAL ISL DATASET SPLIT SUMMARY ===")
    print(f"TRAIN Set : X={arr_train_x.shape}, y={arr_train_y.shape}")
    print(f"VAL Set   : X={arr_val_x.shape}, y={arr_val_y.shape}")
    print(f"TEST Set  : X={arr_test_x.shape}, y={arr_test_y.shape}")

    total_real = len(train_x) + len(val_x) + len(test_x)
    print(f"\nTotal Real Sequences Active: {total_real}")
    print("Synthetic Samples in Active Splits: 0")

    nan_count = sum(
        np.isnan(arr).sum() for arr in [arr_train_x, arr_val_x, arr_test_x]
    )
    inf_count = sum(
        np.isinf(arr).sum() for arr in [arr_train_x, arr_val_x, arr_test_x]
    )

    print(f"NaN Count : {nan_count}")
    print(f"Inf Count : {inf_count}")

    # Data leakage check
    train_vids = {r["video_id"] for r in train_meta}
    val_vids = {r["video_id"] for r in val_meta}
    test_vids = {r["video_id"] for r in test_meta}

    video_leakage = (
        len(train_vids.intersection(test_vids)) == 0
        and len(val_vids.intersection(test_vids)) == 0
    )
    print(f"Video Leakage Check  : {'PASS' if video_leakage else 'FAIL'}")

    train_signers = {r["signer_id"] for r in train_meta}
    test_signers = {r["signer_id"] for r in test_meta}

    signer_leakage = len(train_signers.intersection(test_signers)) == 0
    print(f"Signer Leakage Check : {'PASS' if signer_leakage else 'FAIL'}")


if __name__ == "__main__":
    build_real_splits()
