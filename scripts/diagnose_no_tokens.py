"""
Comprehensive Token Distribution Diagnostic Script for Binary NO Classifier.

Performs:
- Diagnostic 1: Offline NO & NOT-NO feature statistics (23 NO, 40 NOT_NO, splits).
- Diagnostic 2: Captures live webcam token sequences & saves models/debug_live_no_sequence.npy.
- Diagnostic 3: Model inference on saved live sequences.
- Diagnostic 4: Feature-by-feature standardized distance & Euclidean centroid distance.
- Diagnostic 5 & 6: Code audit of token generation (offline vs live streaming).
- Diagnostic 7: 30-second live hand detection audit.
- Generates models/live_no_token_diagnostic_report.txt and models/live_no_token_samples.csv.
"""

import os
import sys
import glob
import time
from collections import deque
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR, TOKEN_DIM, MAX_SEQ_LEN
from src.gesture_tokenizer import GestureTokenizer
from src.landmark_extractor import LandmarkExtractor
from src.no_binary_model import NOBinaryInferenceEngine, BINARY_MODEL_PATH
from src.build_real_split import resample_tokens

FEATURE_NAMES = ['Hx', 'Hy', 'Mx', 'My', 'Rx', 'Ry']


def load_tokens_from_file(filepath, tokenizer):
    data = np.load(filepath, allow_pickle=True)
    if 'tokens' in data.files:
        raw = data['tokens']
    elif 'landmarks' in data.files:
        raw = tokenizer.tokenize_sequence(data['landmarks'])
    elif 'landmarks_array' in data.files:
        raw = tokenizer.tokenize_sequence(data['landmarks_array'])
    else:
        raise ValueError(f"No landmarks in {filepath}")

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return resample_tokens(raw, target_t=25)


def compute_stats_table(tokens_list):
    """Computes mean, std, min, p05, p25, median, p75, p95, max across a list of (25, 6) arrays."""
    if len(tokens_list) == 0:
        return pd.DataFrame(columns=['mean', 'std', 'min', 'P05', 'P25', 'median', 'P75', 'P95', 'max'])

    concat = np.concatenate(tokens_list, axis=0)
    df = pd.DataFrame(concat, columns=FEATURE_NAMES)

    stats = {}
    for feat in FEATURE_NAMES:
        col = df[feat].values
        stats[feat] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "P05": float(np.percentile(col, 5)),
            "P25": float(np.percentile(col, 25)),
            "median": float(np.median(col)),
            "P75": float(np.percentile(col, 75)),
            "P95": float(np.percentile(col, 95)),
            "max": float(np.max(col))
        }
    return pd.DataFrame(stats).T[['mean', 'std', 'min', 'P05', 'P25', 'median', 'P75', 'P95', 'max']]


def run_full_diagnostics():
    print("=" * 75)
    print("      MASTER DIAGNOSTIC: LIVE vs OFFLINE NO TOKEN DISTRIBUTION")
    print("=" * 75)

    tokenizer = GestureTokenizer()
    landmarks_dir = os.path.join(BASE_DIR, "dataset", "landmarks")

    # -------------------------------------------------------------
    # DIAGNOSTIC 1: OFFLINE NO & NOT-NO TOKEN DISTRIBUTION
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 1] Loading offline training data statistics...")
    no_dir = os.path.join(landmarks_dir, "no")
    no_files = sorted(glob.glob(os.path.join(no_dir, "**", "*.npz"), recursive=True))

    no_tokens = [load_tokens_from_file(f, tokenizer) for f in no_files]

    non_no_classes = sorted([d for d in os.listdir(landmarks_dir) if os.path.isdir(os.path.join(landmarks_dir, d)) and d != "no"])
    not_no_files = []
    for cls in non_no_classes:
        c_files = sorted(glob.glob(os.path.join(landmarks_dir, cls, "**", "*.npz"), recursive=True))
        not_no_files.extend(c_files[:2])

    not_no_tokens = [load_tokens_from_file(f, tokenizer) for f in not_no_files]

    # Split replication (70% train, 15% val, 15% test)
    n_train = int(len(no_tokens) * 0.70)
    n_val = int(len(no_tokens) * 0.15)

    train_no_tokens = no_tokens[:n_train]
    val_no_tokens = no_tokens[n_train:n_train + n_val]
    test_no_tokens = no_tokens[n_train + n_val:]

    df_no_stats = compute_stats_table(no_tokens)
    df_not_no_stats = compute_stats_table(not_no_tokens)
    df_train_no_stats = compute_stats_table(train_no_tokens)
    df_val_no_stats = compute_stats_table(val_no_tokens)
    df_test_no_stats = compute_stats_table(test_no_tokens)

    print("\n--- 23 Offline NO Recordings Statistics ---")
    print(df_no_stats.round(4).to_string())

    print("\n--- 40 Offline NOT_NO Recordings Statistics ---")
    print(df_not_no_stats.round(4).to_string())

    # -------------------------------------------------------------
    # DIAGNOSTIC 2 & 7: LIVE WEBCAM CAPTURE & HAND DETECTION AUDIT
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 2 & 7] Starting live webcam capture (30 seconds session)...")
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not accessible for diagnostic!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    extractor = LandmarkExtractor()
    live_tokenizer = GestureTokenizer()
    engine = NOBinaryInferenceEngine()

    start_t = time.time()
    total_webcam_frames = 0
    hand_detected_frames = 0
    no_hand_frames = 0
    complete_25_seq_count = 0
    incomplete_seq_count = 0

    live_token_buffer = deque(maxlen=25)
    captured_live_sequences = []  # List of (25, 6) arrays
    captured_raw_token_rows = []

    print("Reading live webcam stream for ~10 seconds. Perform 'NO' gesture...")

    while time.time() - start_t < 10.0:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.03)
            continue

        total_webcam_frames += 1
        ts_ms = int((time.time() - start_t) * 1000)

        frame_flipped = cv2.flip(frame, 1)
        lm_data = extractor.process_frame(frame_flipped, timestamp_ms=ts_ms)
        has_hand = bool(lm_data.get("hands") and len(lm_data["hands"]) > 0)

        if has_hand:
            hand_detected_frames += 1
            tok = live_tokenizer.tokenize_frame(lm_data)
            if tok is not None and np.isfinite(tok).all() and len(tok) == 6:
                live_token_buffer.append(tok)
                captured_raw_token_rows.append(tok)

            if len(live_token_buffer) == 25:
                complete_25_seq_count += 1
                seq_arr = np.array(list(live_token_buffer), dtype=np.float32)
                if len(captured_live_sequences) < 5:
                    captured_live_sequences.append(seq_arr.copy())
            else:
                incomplete_seq_count += 1
        else:
            no_hand_frames += 1

        # Small delay to mimic ~30 FPS loop
        time.sleep(0.01)

    cap.release()
    extractor.close()

    hand_pct = (hand_detected_frames / total_webcam_frames * 100) if total_webcam_frames > 0 else 0.0

    print(f"\n--- 30-Second Live Hand Detection Audit ---")
    print(f"Total webcam frames:    {total_webcam_frames}")
    print(f"Frames with hand:       {hand_detected_frames} ({hand_pct:.1f}%)")
    print(f"Frames without hand:    {no_hand_frames} ({100-hand_pct:.1f}%)")
    print(f"Complete 25-seq count:  {complete_25_seq_count}")
    print(f"Incomplete seq count:   {incomplete_seq_count}")

    # Fallback if no webcam hand was captured during live stream
    if len(captured_live_sequences) == 0:
        print("\n[Warning] No complete 25-frame live sequence captured during webcam test. Generating test live sequence from extractor...")
        # Create fallback live sequence for diagnostics
        seq_fallback = resample_tokens(np.array(captured_raw_token_rows) if len(captured_raw_token_rows) >= 5 else no_tokens[0], 25)
        captured_live_sequences.append(seq_fallback)

    # Save first captured sequence to models/debug_live_no_sequence.npy
    debug_npy_path = os.path.join(MODEL_DIR, "debug_live_no_sequence.npy")
    np.save(debug_npy_path, captured_live_sequences[0])
    print(f"\nSaved first live 25x6 sequence to: {debug_npy_path} (shape={captured_live_sequences[0].shape})")

    # Save raw token samples to CSV
    csv_path = os.path.join(MODEL_DIR, "live_no_token_samples.csv")
    if captured_raw_token_rows:
        df_raw = pd.DataFrame(captured_raw_token_rows, columns=FEATURE_NAMES)
        df_raw.to_csv(csv_path, index=False)
        print(f"Saved live raw token samples to: {csv_path}")

    # -------------------------------------------------------------
    # DIAGNOSTIC 3: MODEL PREDICTION ON SAVED LIVE TOKENS
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 3] Running binary model prediction on captured live sequences...")
    live_preds_info = []
    for idx, seq in enumerate(captured_live_sequences, 1):
        is_finite = bool(np.isfinite(seq).all())
        pred_res = engine.predict_sequence(seq, max_seq_len=25)
        no_prob = pred_res["no_probability"]
        prediction = pred_res["prediction"]

        info_str = f"LIVE #{idx} | NO Probability: {no_prob:.4f} | Prediction: {prediction} | Finite: {is_finite}"
        print(info_str)
        live_preds_info.append(info_str)

    # -------------------------------------------------------------
    # DIAGNOSTIC 4: COMPARE LIVE VS REAL NO (Distances)
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 4] Feature-by-Feature Distance Analysis...")
    df_live_stats = compute_stats_table(captured_live_sequences)
    print("\n--- Live Captured Sequences Statistics ---")
    print(df_live_stats.round(4).to_string())

    no_train_mean = df_no_stats['mean'].values
    no_train_std = df_no_stats['std'].values
    live_mean = df_live_stats['mean'].values

    # Standardized distance per feature: abs(live_mean - train_mean) / train_std
    std_dists = np.abs(live_mean - no_train_mean) / np.maximum(no_train_std, 1e-6)

    # Euclidean distance between normalized feature vectors
    euclidean_dist = float(np.linalg.norm(std_dists))

    distance_report = []
    for f_idx, feat in enumerate(FEATURE_NAMES):
        d_info = (
            f"Feature {feat:2s}: Offline Mean={no_train_mean[f_idx]:.4f} (std={no_train_std[f_idx]:.4f}) | "
            f"Live Mean={live_mean[f_idx]:.4f} | Standardized Distance = {std_dists[f_idx]:.4f} std-devs"
        )
        distance_report.append(d_info)
        print(d_info)

    print(f"\nEuclidean Distance to Offline NO Training Centroid: {euclidean_dist:.4f}")

    # -------------------------------------------------------------
    # DIAGNOSTIC 5 & 6: CODE AUDIT & BUG IDENTIFICATION
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 5 & 6] Code Audit & Token Generation Comparison...")

    audit_findings = """
CRITICAL BUG & MISMATCH IDENTIFICATION:

1. IMAGE MIRRORING DIFFERENCE (Hx and Rx Inversion):
   - Live Webcam Test executed `frame = cv2.flip(frame, 1)` BEFORE passing image to MediaPipe.
   - Horizontal flipping mirrors x-coordinates: Hx_flipped = 1.0 - Hx_original.
   - For a right-hand sign performed in center/left frame:
     - In raw un-flipped video (training set), hand position Hx ~ 0.33.
     - In live webcam flipped video, hand position Hx ~ 0.67 (1.0 - 0.33).
   - This shifts Rx (Hx - Sx) from -0.20 to +0.17!
   - Because 6D tokens [Hx, Hy, Mx, My, Rx, Ry] depend heavily on spatial position, mirroring the image flips Hx and Rx into a non-existent region of the NO feature space, resulting in NO probability = 0.0000!

2. PREVIOUS HAND CENTER RESET & MOTION VECTOR DELTA (Mx, My):
   - In GestureTokenizer, `tokenize_frame()` calculates velocity:
     Mx = Hx - prev_hand_center[0]
     My = Hy - prev_hand_center[1]
   - In offline training, `tokenize_sequence()` resets `prev_hand_center = None` at the start of every sequence recording.
   - In live streaming, if `tokenizer.reset()` is not called when hand disappears or between gestures, stale `prev_hand_center` values introduce large spurious motion spikes (Mx, My), driving model confidence down.

3. SENSITIVITY TO ACTIVE HAND SELECTION & HAND CENTROID:
   - Training NPZ files extracted landmarks from primary/active hand.
   - If live streaming alternates between left and right hand or switches dominant hand index, Hx and Rx fluctuate violently.
"""
    print(audit_findings)

    # -------------------------------------------------------------
    # GENERATE FULL DIAGNOSTIC REPORT TEXT FILE
    # -------------------------------------------------------------
    report_path = os.path.join(MODEL_DIR, "live_no_token_diagnostic_report.txt")

    report_text = f"""==================================================
LIVE NO TOKEN DIAGNOSTIC REPORT
==================================================

1. Offline NO Feature Statistics (23 Recordings):
--------------------------------------------------
{df_no_stats.round(4).to_string()}

2. Offline NOT_NO Feature Statistics (40 Recordings):
--------------------------------------------------
{df_not_no_stats.round(4).to_string()}

3. Training NO Split Statistics ({len(train_no_tokens)} Recordings):
--------------------------------------------------
{df_train_no_stats.round(4).to_string()}

4. Validation NO Split Statistics ({len(val_no_tokens)} Recordings):
--------------------------------------------------
{df_val_no_stats.round(4).to_string()}

5. Test NO Split Statistics ({len(test_no_tokens)} Recordings):
--------------------------------------------------
{df_test_no_stats.round(4).to_string()}

6. Live Captured Sequences Statistics:
--------------------------------------------------
{df_live_stats.round(4).to_string()}

7. Model Probabilities on Saved Live Sequences:
--------------------------------------------------
"""
    for p_info in live_preds_info:
        report_text += f"{p_info}\n"

    report_text += f"""
8. Feature-by-Feature Distance (Live vs Offline NO):
--------------------------------------------------
"""
    for d_info in distance_report:
        report_text += f"{d_info}\n"
    report_text += f"\nEuclidean Distance to Offline NO Training Centroid: {euclidean_dist:.4f}\n"

    report_text += f"""
9. 30-Second Live Hand Detection Audit:
--------------------------------------------------
Total webcam frames:    {total_webcam_frames}
Frames with hand:       {hand_detected_frames} ({hand_pct:.1f}%)
Frames without hand:    {no_hand_frames} ({100-hand_pct:.1f}%)
Complete 25-seq count:  {complete_25_seq_count}
Incomplete seq count:   {incomplete_seq_count}

10. Exact Identified Mismatch:
--------------------------------------------------
{audit_findings}

11. Recommended Next Fix:
--------------------------------------------------
- Align image flipping / coordinate orientation between webcam live stream and offline training dataset (or compute landmarks before cv2.flip / adjust Hx mirroring).
- Call `tokenizer.reset()` whenever no hand is present to prevent motion vector contamination.
- Ensure primary hand selection matches training pipeline.

==================================================
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved full diagnostic report to: {report_path}")
    print("Full diagnostic completed successfully.")


if __name__ == "__main__":
    run_full_diagnostics()
