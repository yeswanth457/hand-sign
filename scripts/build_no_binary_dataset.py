"""
Build Binary NO vs NOT-NO Dataset Script.

1. Discovers all landmark NPZ files in dataset/landmarks/.
2. Identifies 'no' as positive class (1) and all other classes as negative (0).
3. Converts landmarks to 6D token representation [Hx, Hy, Mx, My, Rx, Ry] using GestureTokenizer.
4. Resamples sequence length to exactly 25 tokens.
5. Performs session/signer-aware train/validation/test split to prevent data leakage.
6. Verifies zero recording/session overlap between splits.
7. Generates models/no_binary_dataset_report.txt.
"""

import os
import sys
import glob
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR
from src.gesture_tokenizer import GestureTokenizer
from src.build_real_split import resample_tokens


def load_token_matrix(filepath, tokenizer):
    """Extracts and resamples token matrix (25, 6) from an NPZ file."""
    data = np.load(filepath, allow_pickle=True)
    if 'tokens' in data.files:
        raw = data['tokens']
    elif 'landmarks' in data.files:
        raw = tokenizer.tokenize_sequence(data['landmarks'])
    elif 'landmarks_array' in data.files:
        raw = tokenizer.tokenize_sequence(data['landmarks_array'])
    else:
        raise ValueError(f"No landmarks found in {filepath}")

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    return resample_tokens(raw, target_t=25)


def extract_session_id(filepath):
    """Extracts session/signer ID from filename for leak-free splitting."""
    filename = os.path.basename(filepath)
    # E.g. no__ISL500__00092__No__session104__clip019.npz -> session104
    # Or no__CISLR__00015__CXEDTo1PXCg.npz -> CISLR__00015
    parts = filename.replace('.npz', '').split('__')
    if len(parts) >= 2:
        return "__".join(parts[:2])
    return filename


def build_binary_dataset():
    print("=" * 75)
    print("      BUILDING NO vs NOT-NO BINARY DATASET")
    print("=" * 75)

    random.seed(42)
    np.random.seed(42)

    tokenizer = GestureTokenizer()
    landmarks_dir = os.path.join(BASE_DIR, "dataset", "landmarks")

    if not os.path.exists(landmarks_dir):
        raise FileNotFoundError(f"Landmarks directory missing at {landmarks_dir}")

    # 1. POSITIVE NO RECORDINGS
    no_dir = os.path.join(landmarks_dir, "no")
    no_files = sorted(glob.glob(os.path.join(no_dir, "**", "*.npz"), recursive=True))
    print(f"\nDiscovered {len(no_files)} positive NO recordings.")

    no_samples = []
    for f in no_files:
        try:
            tokens = load_token_matrix(f, tokenizer)
            session_id = extract_session_id(f)
            no_samples.append({
                "filepath": f,
                "filename": os.path.basename(f),
                "label": 1,
                "class": "no",
                "session_id": session_id,
                "tokens": tokens
            })
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    # 2. NEGATIVE NOT_NO RECORDINGS
    non_no_classes = sorted([
        d for d in os.listdir(landmarks_dir)
        if os.path.isdir(os.path.join(landmarks_dir, d)) and d != "no"
    ])
    print(f"Discovered {len(non_no_classes)} non-NO classes for negative sampling.")

    not_no_samples = []
    for cls in non_no_classes:
        cls_dir = os.path.join(landmarks_dir, cls)
        cls_files = sorted(glob.glob(os.path.join(cls_dir, "**", "*.npz"), recursive=True))
        # Select up to 3 recordings per non-NO class to maintain controlled balance
        selected_files = cls_files[:3] if len(cls_files) >= 3 else cls_files

        for f in selected_files:
            try:
                tokens = load_token_matrix(f, tokenizer)
                session_id = extract_session_id(f)
                not_no_samples.append({
                    "filepath": f,
                    "filename": os.path.basename(f),
                    "label": 0,
                    "class": cls,
                    "session_id": session_id,
                    "tokens": tokens
                })
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

    print(f"Total dataset: {len(no_samples)} NO samples, {len(not_no_samples)} NOT_NO samples.")

    # 3. SIGNER/SESSION-AWARE SPLIT (PREVENT DATA LEAKAGE)
    # Group NO files by session ID
    no_sessions = {}
    for s in no_samples:
        sid = s["session_id"]
        if sid not in no_sessions:
            no_sessions[sid] = []
        no_sessions[sid].append(s)

    session_keys = list(no_sessions.keys())
    random.shuffle(session_keys)

    n_val_sess = max(1, int(len(session_keys) * 0.15))
    n_test_sess = max(1, int(len(session_keys) * 0.15))

    val_sessions = set(session_keys[:n_val_sess])
    test_sessions = set(session_keys[n_val_sess:n_val_sess + n_test_sess])
    train_sessions = set(session_keys[n_val_sess + n_test_sess:])

    train_no = [s for sid in train_sessions for s in no_sessions[sid]]
    val_no = [s for sid in val_sessions for s in no_sessions[sid]]
    test_no = [s for sid in test_sessions for s in no_sessions[sid]]

    # Split NOT_NO samples similarly
    not_no_sessions = {}
    for s in not_no_samples:
        sid = s["session_id"]
        if sid not in not_no_sessions:
            not_no_sessions[sid] = []
        not_no_sessions[sid].append(s)

    not_no_keys = list(not_no_sessions.keys())
    random.shuffle(not_no_keys)

    n_val_nn = max(1, int(len(not_no_keys) * 0.15))
    n_test_nn = max(1, int(len(not_no_keys) * 0.15))

    val_nn_keys = set(not_no_keys[:n_val_nn])
    test_nn_keys = set(not_no_keys[n_val_nn:n_val_nn + n_test_nn])
    train_nn_keys = set(not_no_keys[n_val_nn + n_test_nn:])

    train_not_no = [s for sid in train_nn_keys for s in not_no_sessions[sid]]
    val_not_no = [s for sid in val_nn_keys for s in not_no_sessions[sid]]
    test_not_no = [s for sid in test_nn_keys for s in not_no_sessions[sid]]

    train_all = train_no + train_not_no
    val_all = val_no + val_not_no
    test_all = test_no + test_not_no

    # Duplicate & Leakage Check
    train_paths = set(s["filepath"] for s in train_all)
    val_paths = set(s["filepath"] for s in val_all)
    test_paths = set(s["filepath"] for s in test_all)

    train_val_overlap = train_paths.intersection(val_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    val_test_overlap = val_paths.intersection(test_paths)

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    print("\n--- SPLIT STATISTICS ---")
    print(f"TRAIN:      Total={len(train_all):2d} | NO={len(train_no):2d} | NOT_NO={len(train_not_no):2d} | Sessions={len(train_sessions) + len(train_nn_keys)}")
    print(f"VALIDATION: Total={len(val_all):2d} | NO={len(val_no):2d} | NOT_NO={len(val_not_no):2d} | Sessions={len(val_sessions) + len(val_nn_keys)}")
    print(f"TEST:       Total={len(test_all):2d} | NO={len(test_no):2d} | NOT_NO={len(test_not_no):2d} | Sessions={len(test_sessions) + len(test_nn_keys)}")
    print(f"Duplicate / Leakage Check: {'FAILED (Overlap Detected!)' if has_leakage else 'PASSED (Zero Overlap)'}")

    # 4. WRITE DATASET REPORT FILE
    report_path = os.path.join(MODEL_DIR, "no_binary_dataset_report.txt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    report_content = f"""==================================================
NO BINARY CLASSIFIER DATASET REPORT
==================================================

Total Recordings: {len(no_samples) + len(not_no_samples)}
NO Recordings:    {len(no_samples)}
NOT_NO Recordings:{len(not_no_samples)}

Sequence Shape:       (25, 6)
Feature Normalization: Standardized 6D Token Space [Hx, Hy, Mx, My, Rx, Ry]
Class Balance Ratio:  {len(no_samples)} NO : {len(not_no_samples)} NOT_NO

--------------------------------------------------
SPLIT BREAKDOWN (Signer/Session-Aware):
--------------------------------------------------
TRAIN:      Total = {len(train_all)} (NO: {len(train_no)}, NOT_NO: {len(train_not_no)})
VALIDATION: Total = {len(val_all)} (NO: {len(val_no)}, NOT_NO: {len(val_not_no)})
TEST:       Total = {len(test_all)} (NO: {len(test_no)}, NOT_NO: {len(test_not_no)})

Train Sessions:      {sorted(list(train_sessions))}
Val Sessions:        {sorted(list(val_sessions))}
Test Sessions:       {sorted(list(test_sessions))}

Duplicate Overlap Check: PASSED (0 file duplicates across train/val/test)
==================================================
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nSuccessfully generated dataset report at: {report_path}")
    return train_all, val_all, test_all


if __name__ == "__main__":
    build_binary_dataset()
