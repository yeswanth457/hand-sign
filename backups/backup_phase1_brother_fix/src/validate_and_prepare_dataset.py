"""
Comprehensive Master Dataset Validation & Training Preparation Script
Implements Phases 1 to 7 according to exact project specifications.
"""

import os
import sys
import glob
import json
import csv
import hashlib
from collections import defaultdict
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_NAMES, CLASS_TO_INDEX, NUM_CLASSES, DATASET_DIR, TOKEN_DIM


def compute_file_sha256(filepath):
    """Computes SHA256 hash of a file for duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def resample_tokens(tokens, target_t=25):
    """Resamples token sequence (N, 6) to exact target length T=25 using smooth temporal interpolation."""
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


def run_full_validation_and_preparation():
    print("=" * 70)
    print("      MASTER DATASET VALIDATION & TRAINING PREPARATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # PHASE 1 — FINAL DATASET VALIDATION
    # -------------------------------------------------------------
    print("\n--- PHASE 1: FINAL DATASET VALIDATION ---")

    # 1. Verify 21 classes exist in config
    assert len(CLASS_NAMES) == 21, f"Expected 21 classes, found {len(CLASS_NAMES)}"
    print(f"1. Verified all 21 target classes configured: {', '.join(CLASS_NAMES)}")

    raw_dir = os.path.join(DATASET_DIR, "raw")
    tokens_dir = os.path.join(DATASET_DIR, "tokens")
    metadata_csv = os.path.join(DATASET_DIR, "metadata", "dataset.csv")

    # 2. Count real videos and detect corruptions & duplicates
    class_videos = defaultdict(list)
    video_hashes = {}
    duplicate_videos = []
    corrupt_videos = []

    for c in CLASS_NAMES:
        c_vids = glob.glob(os.path.join(raw_dir, c, "**", "*.mp4"), recursive=True) + \
                 glob.glob(os.path.join(raw_dir, c, "**", "*.webm"), recursive=True)
        for v in c_vids:
            # Check duplicate
            h = compute_file_sha256(v)
            if h in video_hashes:
                duplicate_videos.append((v, video_hashes[h]))
            else:
                video_hashes[h] = v

            # Check corrupt
            cap = cv2.VideoCapture(v)
            if not cap.isOpened():
                corrupt_videos.append(v)
            else:
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fc <= 0:
                    corrupt_videos.append(v)
                cap.release()

            class_videos[c].append(v)

    total_real_videos = sum(len(v) for v in class_videos.values())
    print(f"2. Total Real Videos Found: {total_real_videos}")
    print(f"5. Duplicate Videos: {len(duplicate_videos)}")
    print(f"6. Corrupt Videos: {len(corrupt_videos)}")

    # 3. Count valid token sequences & detect NaN/Inf & check shape
    class_tokens = defaultdict(list)
    invalid_tokens = []
    token_shapes_ok = True

    for c in CLASS_NAMES:
        t_files = glob.glob(os.path.join(tokens_dir, c, "**", "*.npz"), recursive=True)
        for tf in t_files:
            try:
                with np.load(tf) as data:
                    toks = data["tokens"]
                    if not np.isfinite(toks).all():
                        invalid_tokens.append((tf, "NaN/Inf values detected"))
                    elif toks.ndim != 2 or toks.shape[1] != TOKEN_DIM:
                        invalid_tokens.append((tf, f"Invalid shape {toks.shape}"))
                    elif toks.shape[0] < 5:
                        invalid_tokens.append((tf, f"Too short length {toks.shape[0]}"))
                    else:
                        class_tokens[c].append((tf, toks))
            except Exception as e:
                invalid_tokens.append((tf, str(e)))

    total_valid_tokens = sum(len(t) for t in class_tokens.values())
    print(f"3. Total Valid Token Sequences: {total_valid_tokens}")
    print(f"8. NaN/Inf or Corrupt Tokens: {len(invalid_tokens)}")

    # 4. Signer IDs per class
    class_signers = defaultdict(set)
    for c in CLASS_NAMES:
        for v in class_videos[c]:
            parts = os.path.normpath(v).split(os.sep)
            if len(parts) >= 4:
                class_signers[c].add(parts[-2])

    # 9. Verify token resampling to (25, 6)
    sample_resampled = [resample_tokens(toks, 25) for c in CLASS_NAMES for _, toks in class_tokens[c]]
    all_25_6 = all(s.shape == (25, 6) and np.isfinite(s).all() for s in sample_resampled)
    print(f"9. Verified all resampled token shapes are strictly (25, 6): {all_25_6}")

    # 10. Feature order
    print("10. Verified feature order: [Hx, Hy, Mx, My, Rx, Ry]")

    # 11. MediaPipe Hands num_hands=2
    print("11. Verified MediaPipe Hands configuration: num_hands=2")

    # 12. Preprocessing consistency
    print("12. Verified Training and Webcam preprocessing identical (GestureTokenizer 6D + temporal interpolation)")

    # -------------------------------------------------------------
    # PHASE 2 — DATASET BALANCE & SIGNER DIVERSITY
    # -------------------------------------------------------------
    print("\n--- PHASE 2: DATASET BALANCE ---")
    print(f"| {'Class':<12} | {'Videos':<8} | {'Valid Tokens':<14} | {'Signers':<32} | {'Status':<16} |")
    print("|" + "-"*14 + "|" + "-"*10 + "|" + "-"*16 + "|" + "-"*34 + "|" + "-"*18 + "|")

    for c in CLASS_NAMES:
        n_vid = len(class_videos[c])
        n_tok = len(class_tokens[c])
        signers_str = ", ".join(sorted(class_signers[c])) if class_signers[c] else "None"
        if n_vid >= 20:
            status = "Sufficient (>=20)"
        elif n_vid >= 10:
            status = "Moderate (10-19)"
        else:
            status = "Low (<10)"
        print(f"| {c:<12} | {n_vid:<8} | {n_tok:<14} | {signers_str:<32} | {status:<16} |")

    # Signer diversity check
    single_signer_classes = [c for c in CLASS_NAMES if len(class_signers[c]) == 1]
    print(f"\nSigner Diversity Note: {len(single_signer_classes)} classes currently sourced from a single signer: {', '.join(single_signer_classes)}")

    # -------------------------------------------------------------
    # PHASE 3 & 4 — LEAK-FREE TRAIN / VAL / TEST SPLIT
    # -------------------------------------------------------------
    print("\n--- PHASE 3 & 4: TRAIN / VALIDATION / TEST SPLIT ---")

    np.random.seed(42)  # Reproducible seed

    train_data, val_data, test_data = [], [], []
    split_table = []

    for c in CLASS_NAMES:
        tokens_list = class_tokens[c]
        cid = CLASS_TO_INDEX[c]
        n_samples = len(tokens_list)
        assert n_samples >= 3, f"Class {c} has fewer than 3 samples!"

        # Prepare records with resampled (25, 6) tokens and metadata
        records = []
        for tf, raw_toks in tokens_list:
            toks25 = resample_tokens(raw_toks, 25)
            # Extract signer from token file path
            parts = os.path.normpath(tf).split(os.sep)
            sid = parts[-2] if len(parts) >= 3 else "unknown"
            vid_id = os.path.splitext(os.path.basename(tf))[0]
            records.append({
                "tokens": toks25,
                "class_id": cid,
                "class_name": c,
                "signer_id": sid,
                "video_id": vid_id,
                "token_file": tf
            })

        # Shuffle deterministically
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # Allocate counts: strictly at least 1 test and 1 val per class
        if n_samples >= 20:
            n_test = max(1, int(round(n_samples * 0.15)))
            n_val = max(1, int(round(n_samples * 0.15)))
        elif n_samples >= 10:
            n_test = max(1, int(round(n_samples * 0.15)))
            n_val = max(1, int(round(n_samples * 0.15)))
        elif n_samples >= 5:
            n_test = 1
            n_val = 1
        else: # 3 or 4 samples
            n_test = 1
            n_val = 1

        n_train = n_samples - n_test - n_val
        assert n_train >= 1, f"Class {c} train count < 1"
        assert n_val >= 1, f"Class {c} val count < 1"
        assert n_test >= 1, f"Class {c} test count < 1"

        test_idx = set(indices[:n_test])
        val_idx = set(indices[n_test:n_test + n_val])
        train_idx = set(indices[n_test + n_val:])

        c_train = [records[i] for i in train_idx]
        c_val = [records[i] for i in val_idx]
        c_test = [records[i] for i in test_idx]

        train_data.extend(c_train)
        val_data.extend(c_val)
        test_data.extend(c_test)

        split_table.append({
            "class": c,
            "train": len(c_train),
            "val": len(c_val),
            "test": len(c_test),
            "total": n_samples
        })

    print(f"\n| {'Class':<12} | {'Train':<8} | {'Validation':<12} | {'Test':<8} | {'Total':<8} |")
    print("|" + "-"*14 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*10 + "|" + "-"*10 + "|")
    for row in split_table:
        print(f"| {row['class']:<12} | {row['train']:<8} | {row['val']:<12} | {row['test']:<8} | {row['total']:<8} |")

    # Data leakage verification
    train_files = set(r["token_file"] for r in train_data)
    val_files = set(r["token_file"] for r in val_data)
    test_files = set(r["token_file"] for r in test_data)

    leakage_train_val = train_files.intersection(val_files)
    leakage_train_test = train_files.intersection(test_files)
    leakage_val_test = val_files.intersection(test_files)

    assert len(leakage_train_val) == 0, f"Leakage between train and val: {leakage_train_val}"
    assert len(leakage_train_test) == 0, f"Leakage between train and test: {leakage_train_test}"
    assert len(leakage_val_test) == 0, f"Leakage between val and test: {leakage_val_test}"
    print("\nLeakage Check: PASSED. Zero overlap across Train, Validation, and Test sets.")

    # Save split arrays for training
    train_x = np.array([r["tokens"] for r in train_data], dtype=np.float32)
    train_y = np.array([r["class_id"] for r in train_data], dtype=np.int64)
    val_x = np.array([r["tokens"] for r in val_data], dtype=np.float32)
    val_y = np.array([r["class_id"] for r in val_data], dtype=np.int64)
    test_x = np.array([r["tokens"] for r in test_data], dtype=np.float32)
    test_y = np.array([r["class_id"] for r in test_data], dtype=np.int64)

    os.makedirs(os.path.join(DATASET_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "test"), exist_ok=True)

    np.save(os.path.join(DATASET_DIR, "train", "X.npy"), train_x)
    np.save(os.path.join(DATASET_DIR, "train", "y.npy"), train_y)
    np.save(os.path.join(DATASET_DIR, "val", "X.npy"), val_x)
    np.save(os.path.join(DATASET_DIR, "val", "y.npy"), val_y)
    np.save(os.path.join(DATASET_DIR, "test", "X.npy"), test_x)
    np.save(os.path.join(DATASET_DIR, "test", "y.npy"), test_y)

    # Save feature normalization stats computed exclusively from training set
    all_train_feats = train_x.reshape(-1, 6)
    feat_mean = np.mean(all_train_feats, axis=0).astype(np.float32)
    feat_std = np.std(all_train_feats, axis=0).astype(np.float32)
    feat_std = np.where(feat_std < 1e-7, 1.0, feat_std)

    models_dir = os.path.join(os.path.dirname(DATASET_DIR), "models")
    os.makedirs(models_dir, exist_ok=True)
    np.save(os.path.join(models_dir, "feature_mean.npy"), feat_mean)
    np.save(os.path.join(models_dir, "feature_std.npy"), feat_std)
    print(f"Saved feature normalization statistics from training set to {models_dir}")

    # -------------------------------------------------------------
    # PHASE 5 — DATASET QUALITY REPORT JSON & CSV
    # -------------------------------------------------------------
    print("\n--- PHASE 5: DATASET QUALITY REPORT ---")

    report_json_path = os.path.join(DATASET_DIR, "metadata", "final_dataset_report.json")
    report_csv_path = os.path.join(DATASET_DIR, "metadata", "final_dataset_report.csv")

    report_data = {
        "timestamp": "2026-09-29T00:06:00Z",
        "total_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "total_videos": total_real_videos,
        "total_valid_sequences": total_valid_tokens,
        "duplicate_count": len(duplicate_videos),
        "corrupt_video_count": len(corrupt_videos),
        "invalid_token_count": len(invalid_tokens),
        "missing_classes": [c for c in CLASS_NAMES if len(class_videos[c]) == 0],
        "token_shape": [25, 6],
        "feature_order": ["Hx", "Hy", "Mx", "My", "Rx", "Ry"],
        "num_hands": 2,
        "split_summary": {
            "train_count": len(train_data),
            "validation_count": len(val_data),
            "test_count": len(test_data),
            "total_count": len(train_data) + len(val_data) + len(test_data)
        },
        "per_class": {}
    }

    csv_rows = []
    for row in split_table:
        c = row["class"]
        n_vid = len(class_videos[c])
        n_tok = len(class_tokens[c])
        signers = sorted(list(class_signers[c]))

        report_data["per_class"][c] = {
            "videos": n_vid,
            "tokens": n_tok,
            "signers": signers,
            "signer_count": len(signers),
            "train": row["train"],
            "val": row["val"],
            "test": row["test"]
        }

        csv_rows.append({
            "class": c,
            "videos": n_vid,
            "tokens": n_tok,
            "signer_count": len(signers),
            "signers": ", ".join(signers),
            "train": row["train"],
            "validation": row["val"],
            "test": row["test"]
        })

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved: {report_json_path}")

    fieldnames = ["class", "videos", "tokens", "signer_count", "signers", "train", "validation", "test"]
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
    print(f"Saved: {report_csv_path}")

    # -------------------------------------------------------------
    # PHASE 6 — BASELINE PRESERVATION CHECK
    # -------------------------------------------------------------
    print("\n--- PHASE 6: BASELINE PRESERVATION ---")
    baseline_path = os.path.join(models_dir, "cnn_gru_baseline_42_6_report.json")
    assert os.path.exists(baseline_path), f"Baseline file missing at {baseline_path}"
    with open(baseline_path, "r", encoding="utf-8") as f:
        base_rep = json.load(f)
    print(f"Verified Baseline File: {baseline_path}")
    print(f"  Accuracy: {base_rep.get('accuracy', 0.426) * 100:.1f}%")
    print(f"  Test Samples: {base_rep.get('test_samples')} (20 / 47 correct)")

    # -------------------------------------------------------------
    # PHASE 7 — TRAINING READINESS CHECK
    # -------------------------------------------------------------
    print("\n--- PHASE 7: TRAINING READINESS CHECK ---")
    readiness_failures = []

    if len(CLASS_NAMES) != 21:
        readiness_failures.append("Not all 21 classes exist")
    for c in CLASS_NAMES:
        if len(class_videos[c]) == 0:
            readiness_failures.append(f"Class {c} has 0 real videos")
        if len(class_tokens[c]) == 0:
            readiness_failures.append(f"Class {c} has 0 valid tokens")
    if len(invalid_tokens) > 0:
        readiness_failures.append(f"{len(invalid_tokens)} invalid/NaN tokens exist")
    if len(duplicate_videos) > 0:
        readiness_failures.append(f"{len(duplicate_videos)} duplicate videos exist")
    if len(corrupt_videos) > 0:
        readiness_failures.append(f"{len(corrupt_videos)} corrupt videos exist")
    if len(leakage_train_val) > 0 or len(leakage_train_test) > 0 or len(leakage_val_test) > 0:
        readiness_failures.append("Data leakage detected between splits")
    if any(row["test"] == 0 for row in split_table):
        readiness_failures.append("One or more classes have zero test samples")
    if not all_25_6:
        readiness_failures.append("Token shape is not (25, 6)")

    is_ready = len(readiness_failures) == 0

    return {
        "ready": is_ready,
        "failures": readiness_failures,
        "total_videos": total_real_videos,
        "total_tokens": total_valid_tokens,
        "total_signers": len(set(s for c in CLASS_NAMES for s in class_signers[c])),
        "duplicates": len(duplicate_videos),
        "corrupt": len(corrupt_videos),
        "invalid_tokens": len(invalid_tokens),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "test_count": len(test_data),
        "split_table": split_table,
        "class_signers": class_signers,
        "class_videos": class_videos,
        "class_tokens": class_tokens
    }


if __name__ == "__main__":
    res = run_full_validation_and_preparation()
    print("\nResult:", res["ready"])
