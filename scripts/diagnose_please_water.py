"""
PLEASE -> WATER MISCLASSIFICATION DIAGNOSIS SCRIPT
===================================================
Steps:
  1. Backup critical files
  2. Load training data for please & water -- compute statistics
  3. Load the failed live sequence (if captured) and run direct CNN inference
  4. Compare live please against training please and water distributions
  5. Check vertical position (Hy/Ry)
  6. Check token validity
  7. Determine root cause
  8. Recommend fix (retraining if distribution mismatch proven)

Does NOT:
  - Hard-code any gesture override
  - Change global thresholds
  - Modify School / NO / Water logic
"""

import os
import sys
import shutil
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    CLASS_NAMES, NUM_CLASSES, INDEX_TO_CLASS, CLASS_TO_INDEX,
    MODEL_DIR, DATASET_DIR
)
from src.cnn_gru_model import CNNGRUInferenceEngine

# Paths
BACKUP_DIR        = os.path.join(MODEL_DIR, "backup_real_please_water_failure")
FAILED_PLEASE_NPY = os.path.join(MODEL_DIR, "debug_failed_please_as_water.npy")
FAILED_PLEASE_TXT = os.path.join(MODEL_DIR, "debug_failed_please_as_water.txt")
TOKENS_DIR        = os.path.join(DATASET_DIR, "tokens")
PLEASE_TOKEN_DIR  = os.path.join(TOKENS_DIR, "please")
WATER_TOKEN_DIR   = os.path.join(TOKENS_DIR, "water")
RAW_PLEASE_DIR    = os.path.join(DATASET_DIR, "raw", "please")


# ==============================================================================
# STEP 1 - BACKUP
# ==============================================================================
def step1_backup():
    print("\n" + "="*60)
    print("STEP 1 - BACKUP")
    print("="*60)

    if os.path.exists(BACKUP_DIR):
        print(f"[Backup] Directory already exists: {BACKUP_DIR}")
        print("[Backup] Skipping -- do not overwrite existing backup.")
        print("REAL_PLEASE_WATER_BACKUP=SKIP (already exists)")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)

    files_to_backup = [
        os.path.join(MODEL_DIR, "isl_cnn_gru_21class_best.pt"),
        os.path.join(MODEL_DIR, "isl_cnn_gru.pt"),
        os.path.join(MODEL_DIR, "no_binary_classifier.pt"),
        os.path.join(MODEL_DIR, "no_binary_threshold.json"),
        os.path.join(PROJECT_ROOT, "config.py"),
        os.path.join(PROJECT_ROOT, "app.py"),
        os.path.join(PROJECT_ROOT, "static", "app.js"),
        os.path.join(PROJECT_ROOT, "static", "index.html"),
        os.path.join(PROJECT_ROOT, "static", "style.css"),
    ]

    backed_up = 0
    for src in files_to_backup:
        if os.path.exists(src):
            dst = os.path.join(BACKUP_DIR, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"  [OK] {os.path.basename(src)}")
            backed_up += 1
        else:
            print(f"  [MISSING] {src}")

    print(f"\nBacked up {backed_up}/{len(files_to_backup)} files.")
    print("REAL_PLEASE_WATER_BACKUP=PASS")


# ==============================================================================
# STEP 2 - CAPTURE INSTRUCTION
# ==============================================================================
def step2_capture_instruction():
    print("\n" + "="*60)
    print("STEP 2 - CAPTURE REAL FAILED PLEASE SEQUENCE")
    print("="*60)
    print("""
MANUAL ACTION REQUIRED:
  1. Open http://127.0.0.1:8000 in your browser (Ctrl+F5 to hard-refresh)
  2. Perform your REAL PLEASE gesture in front of the webcam
  3. When the UI shows 'Water' instead of 'Please.' -- the backend will
     automatically save the sequence since app.py captures any water prediction.
     The files will appear at:
       models/debug_failed_please_as_water.npy
       models/debug_failed_please_as_water.txt
  4. Re-run this script to get the full diagnosis.
""")

    if os.path.exists(FAILED_PLEASE_NPY):
        arr = np.load(FAILED_PLEASE_NPY)
        print(f"FAILED_PLEASE_SEQUENCE_CAPTURED=YES")
        print(f"FAILED_PLEASE_SEQUENCE_SHAPE={arr.shape}")
    else:
        print("FAILED_PLEASE_SEQUENCE_CAPTURED=NO")
        print("  -> Run the app, perform PLEASE, wait for Water misclassification.")
        print("  -> Then re-run this script.")


# ==============================================================================
# HELPERS - Load all token files for a class
# ==============================================================================
def load_class_tokens(class_name):
    """Load all (25,6) token sequences for a given class from dataset/tokens/<class>."""
    token_dir = os.path.join(TOKENS_DIR, class_name)
    sequences = []

    if not os.path.isdir(token_dir):
        return sequences

    for root_dir, _, files in os.walk(token_dir):
        for fname in files:
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
                if tokens is None:
                    keys = list(npz.files)
                    if keys:
                        tokens = npz[keys[0]].astype(np.float32)

                if tokens is not None and tokens.ndim == 2 and tokens.shape[1] == 6:
                    if tokens.shape[0] != 25:
                        from src.build_real_split import resample_tokens
                        tokens = resample_tokens(tokens, target_t=25)
                    sequences.append(tokens)
            except Exception:
                pass

    return sequences


def count_raw_videos(class_name):
    raw_dir = os.path.join(DATASET_DIR, "raw", class_name)
    if not os.path.isdir(raw_dir):
        return 0
    count = 0
    for root_dir, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                count += 1
    return count


# ==============================================================================
# STEP 3 - DIRECT CNN INFERENCE ON FAILED SEQUENCE
# ==============================================================================
def step3_direct_cnn_inference():
    print("\n" + "="*60)
    print("STEP 3 - DIRECT CNN INFERENCE ON FAILED PLEASE SEQUENCE")
    print("="*60)

    if not os.path.exists(FAILED_PLEASE_NPY):
        print("  [SKIP] No failed sequence captured yet.")
        print("  Run app.py, perform PLEASE, wait for Water misclassification.")
        return None, None, None

    tokens_np = np.load(FAILED_PLEASE_NPY)
    print(f"Loaded: {FAILED_PLEASE_NPY}")
    print(f"Shape: {tokens_np.shape}")

    if tokens_np.shape != (25, 6):
        print(f"  [ERROR] Expected (25,6), got {tokens_np.shape}")
        return None, None, None

    has_nan = bool(np.any(np.isnan(tokens_np)))
    has_inf = bool(np.any(np.isinf(tokens_np)))
    print(f"\nTOKEN_FINITE={'NO' if (has_nan or has_inf) else 'YES'}")

    hx = tokens_np[:, 0]
    hy = tokens_np[:, 1]
    range_valid = (hx.min() >= 0.0 and hx.max() <= 1.0 and
                   hy.min() >= 0.0 and hy.max() <= 1.0)
    print(f"TOKEN_RANGE_VALID={'YES' if range_valid else 'NO'}")

    diffs = np.diff(tokens_np[:, :2], axis=0)
    max_jump = float(np.max(np.abs(diffs)))
    stable = max_jump < 0.15
    print(f"TOKEN_TEMPORALLY_STABLE={'YES' if stable else 'NO (max_jump=' + str(round(max_jump,4)) + ')'}")

    # Direct CNN inference -- no NO model, no early decision, no frontend
    engine = CNNGRUInferenceEngine()
    if not engine.model_loaded:
        print("  [ERROR] CNN-GRU model not loaded!")
        return None, None, None

    result = engine.predict_sequence(tokens_np)
    direct_class = result.get("word", "--")
    direct_conf  = result.get("confidence", 0.0)
    probs        = result.get("probabilities", {})

    print(f"\nDIRECT_FAILED_PLEASE_CNN_CLASS={direct_class}")
    print(f"DIRECT_FAILED_PLEASE_CNN_CONFIDENCE={direct_conf:.4f}")
    print("\nTOP 5:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (c, p) in enumerate(sorted_probs, 1):
        print(f"  TOP{i}={c} ({p:.4f})")

    if direct_class == "water":
        print("\n>>> ROOT CAUSE: CNN ITSELF outputs 'water' for the live PLEASE sequence.")
        print(">>> This is a model/representation problem, NOT a routing/frontend issue.")
        print(">>> Frontend logic is NOT to blame.")
    elif direct_class == "please":
        print("\n>>> CNN outputs 'please' correctly.")
        print(">>> Root cause is in the routing/decision/UI pipeline, NOT the model.")
    else:
        print(f"\n>>> CNN outputs '{direct_class}' -- neither please nor water.")
        print(">>> Model is confused. Check token representation vs training distribution.")

    return tokens_np, direct_class, probs


# ==============================================================================
# STEP 4 - COMPARE PLEASE WITH TRAINING DATA
# ==============================================================================
def step4_compare_with_training(live_tokens):
    print("\n" + "="*60)
    print("STEP 4 - COMPARE LIVE PLEASE vs TRAINING PLEASE")
    print("="*60)

    please_seqs = load_class_tokens("please")
    print(f"PLEASE_VALID_TOKEN_COUNT={len(please_seqs)}")
    print(f"PLEASE_RAW_VIDEO_COUNT={count_raw_videos('please')}")

    if not please_seqs:
        print("  [ERROR] No PLEASE training tokens found!")
        return None, None

    please_stack = np.stack(please_seqs, axis=0)  # (N, 25, 6)
    train_mean = please_stack.mean(axis=(0, 1))    # (6,)
    train_std  = please_stack.std(axis=(0, 1))     # (6,)

    feat_names = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]
    print(f"\nTRAINING_PLEASE_MEAN={[round(float(v),4) for v in train_mean]}")
    print(f"TRAINING_PLEASE_STD ={[round(float(v),4) for v in train_std]}")

    training_hy_all = please_stack[:, :, 1].flatten()
    training_ry_all = please_stack[:, :, 5].flatten()
    print(f"\nTRAINING_PLEASE_MEAN_Hy={float(training_hy_all.mean()):.4f}")
    print(f"TRAINING_PLEASE_MEAN_Ry={float(training_ry_all.mean()):.4f}")
    print(f"PLEASE_Hy_RANGE=[{float(training_hy_all.min()):.4f}, {float(training_hy_all.max()):.4f}]")
    print(f"PLEASE_Ry_RANGE=[{float(training_ry_all.min()):.4f}, {float(training_ry_all.max()):.4f}]")

    hx_var = please_stack[:, :, 0].var(axis=1)
    single_hand_pct = float((hx_var < 0.02).mean() * 100)
    print(f"PLEASE_SINGLE_HAND_PERCENT={single_hand_pct:.1f}%")
    print(f"PLEASE_TWO_HAND_PERCENT={100-single_hand_pct:.1f}%")

    if live_tokens is None:
        print("\n  [SKIP] No live sequence to compare against training.")
        return train_mean, train_std

    live_mean = live_tokens.mean(axis=0)
    live_std  = live_tokens.std(axis=0)

    print(f"\nFAILED_LIVE_PLEASE_MEAN={[round(float(v),4) for v in live_mean]}")
    print(f"FAILED_LIVE_PLEASE_STD ={[round(float(v),4) for v in live_std]}")

    print("\nPer-feature distance (|live_mean - training_mean|):")
    for i, name in enumerate(feat_names):
        dist = abs(float(live_mean[i]) - float(train_mean[i]))
        print(f"  PLEASE_{name}_DISTANCE={dist:.4f}")

    return train_mean, train_std


# ==============================================================================
# STEP 5 - COMPARE LIVE PLEASE AGAINST WATER
# ==============================================================================
def step5_compare_against_water(live_tokens):
    print("\n" + "="*60)
    print("STEP 5 - COMPARE LIVE PLEASE vs WATER TRAINING DATA")
    print("="*60)

    please_seqs = load_class_tokens("please")
    water_seqs  = load_class_tokens("water")

    if not please_seqs or not water_seqs:
        print("  [ERROR] Insufficient training data for comparison.")
        return

    please_stack = np.stack(please_seqs, axis=0)
    water_stack  = np.stack(water_seqs,  axis=0)
    please_centroid = please_stack.mean(axis=0)  # (25,6)
    water_centroid  = water_stack.mean(axis=0)   # (25,6)

    if live_tokens is None:
        print("  [SKIP] No live sequence to compare.")
        return

    please_dist = float(np.linalg.norm(live_tokens - please_centroid))
    water_dist  = float(np.linalg.norm(live_tokens - water_centroid))

    print(f"FAILED_PLEASE_TO_PLEASE_DISTANCE={please_dist:.4f}")
    print(f"FAILED_PLEASE_TO_WATER_DISTANCE ={water_dist:.4f}")

    if water_dist < please_dist:
        print("\n  >>> LIVE PLEASE IS CLOSER TO WATER CENTROID THAN PLEASE CENTROID")
        print("  >>> This confirms a distribution mismatch.")
    else:
        print("\n  >>> LIVE PLEASE IS CLOSER TO PLEASE CENTROID -- model confusion is subtle.")

    live_flat   = live_tokens.flatten()
    please_sims = []
    water_sims  = []

    for seq in please_seqs:
        s_flat = seq.flatten()
        sim = float(np.dot(live_flat, s_flat) /
                    (np.linalg.norm(live_flat) * np.linalg.norm(s_flat) + 1e-8))
        please_sims.append(sim)

    for seq in water_seqs:
        s_flat = seq.flatten()
        sim = float(np.dot(live_flat, s_flat) /
                    (np.linalg.norm(live_flat) * np.linalg.norm(s_flat) + 1e-8))
        water_sims.append(sim)

    avg_please_sim = float(np.mean(please_sims)) if please_sims else 0.0
    avg_water_sim  = float(np.mean(water_sims))  if water_sims  else 0.0
    max_please_sim = float(np.max(please_sims))  if please_sims else 0.0
    max_water_sim  = float(np.max(water_sims))   if water_sims  else 0.0

    print(f"\nAVG_LIVE_PLEASE_TO_PLEASE={avg_please_sim:.4f}")
    print(f"AVG_LIVE_PLEASE_TO_WATER ={avg_water_sim:.4f}")
    print(f"LIVE_PLEASE_PLEASE_SIMILARITY (max)={max_please_sim:.4f}")
    print(f"LIVE_PLEASE_WATER_SIMILARITY (max) ={max_water_sim:.4f}")

    if avg_water_sim > avg_please_sim:
        print("\n  >>> Live PLEASE is MORE SIMILAR TO WATER than to PLEASE training samples.")
        print("  >>> CONCLUSION: LIVE_PLEASE_OUTSIDE_TRAINING_DISTRIBUTION=YES")
    else:
        print("\n  >>> Live PLEASE is more similar to training PLEASE samples.")
        print("  >>> CONCLUSION: LIVE_PLEASE_OUTSIDE_TRAINING_DISTRIBUTION=BORDERLINE")


# ==============================================================================
# STEP 6 - VERTICAL POSITION CHECK
# ==============================================================================
def step6_vertical_check(live_tokens):
    print("\n" + "="*60)
    print("STEP 6 - VERTICAL POSITION (Hy/Ry) CHECK")
    print("="*60)

    please_seqs = load_class_tokens("please")
    water_seqs  = load_class_tokens("water")

    t_p_hy = t_p_ry = t_w_hy = t_w_ry = None

    if please_seqs:
        please_stack = np.stack(please_seqs, axis=0)
        t_p_hy = float(please_stack[:, :, 1].mean())
        t_p_ry = float(please_stack[:, :, 5].mean())
        print(f"TRAINING_PLEASE_MEAN_Hy={t_p_hy:.4f}")
        print(f"TRAINING_PLEASE_MEAN_Ry={t_p_ry:.4f}")

    if water_seqs:
        water_stack = np.stack(water_seqs, axis=0)
        t_w_hy = float(water_stack[:, :, 1].mean())
        t_w_ry = float(water_stack[:, :, 5].mean())
        print(f"TRAINING_WATER_MEAN_Hy={t_w_hy:.4f}")
        print(f"TRAINING_WATER_MEAN_Ry={t_w_ry:.4f}")

    if live_tokens is not None:
        live_hy = float(live_tokens[:, 1].mean())
        live_ry = float(live_tokens[:, 5].mean())
        print(f"\nLIVE_PLEASE_MEAN_Hy={live_hy:.4f}")
        print(f"LIVE_PLEASE_MEAN_Ry={live_ry:.4f}")

        if t_p_hy is not None and t_w_hy is not None:
            dist_to_please_hy = abs(live_hy - t_p_hy)
            dist_to_water_hy  = abs(live_hy - t_w_hy)
            print(f"\nLive Hy distance to PLEASE training mean: {dist_to_please_hy:.4f}")
            print(f"Live Hy distance to WATER training mean:  {dist_to_water_hy:.4f}")

            if dist_to_water_hy < dist_to_please_hy:
                print("  >>> Hy indicates live PLEASE is closer to WATER distribution vertically.")
            else:
                print("  >>> Hy indicates live PLEASE is closer to PLEASE distribution vertically.")


# ==============================================================================
# STEP 11 - TRAINING DATA INSPECTION
# ==============================================================================
def step11_training_data_check():
    print("\n" + "="*60)
    print("STEP 11 - TRAINING PLEASE DATA INSPECTION")
    print("="*60)

    please_seqs = load_class_tokens("please")
    print(f"PLEASE_VALID_TOKEN_COUNT={len(please_seqs)}")
    print(f"PLEASE_RAW_VIDEO_COUNT={count_raw_videos('please')}")

    if not please_seqs:
        return

    please_stack = np.stack(please_seqs, axis=0)
    hy_all = please_stack[:, :, 1].flatten()
    ry_all = please_stack[:, :, 5].flatten()
    print(f"PLEASE_Hy_RANGE=[{hy_all.min():.4f}, {hy_all.max():.4f}]")
    print(f"PLEASE_Ry_RANGE=[{ry_all.min():.4f}, {ry_all.max():.4f}]")


# ==============================================================================
# STEP 12 - WATER SEPARATION CHECK
# ==============================================================================
def step12_water_data_check():
    print("\n" + "="*60)
    print("STEP 12 - WATER DATA CHECK (training distribution)")
    print("="*60)

    water_seqs  = load_class_tokens("water")
    please_seqs = load_class_tokens("please")

    if water_seqs and please_seqs:
        w_stack = np.stack(water_seqs,  axis=0)
        p_stack = np.stack(please_seqs, axis=0)

        w_hy = float(w_stack[:, :, 1].mean())
        p_hy = float(p_stack[:, :, 1].mean())
        w_ry = float(w_stack[:, :, 5].mean())
        p_ry = float(p_stack[:, :, 5].mean())

        hy_sep = abs(w_hy - p_hy)
        ry_sep = abs(w_ry - p_ry)

        print(f"Water mean Hy={w_hy:.4f}, Please mean Hy={p_hy:.4f}, separation={hy_sep:.4f}")
        print(f"Water mean Ry={w_ry:.4f}, Please mean Ry={p_ry:.4f}, separation={ry_sep:.4f}")
        print(f"WATER_TOKEN_COUNT={len(water_seqs)}")
        print(f"PLEASE_TOKEN_COUNT={len(please_seqs)}")

        if hy_sep < 0.05 and ry_sep < 0.05:
            print("  >>> PLEASE and WATER have VERY SIMILAR vertical positions in training data.")
            print("  >>> Small variation in live gesture -> easy misclassification.")
        else:
            print(f"  >>> Hy sep={hy_sep:.4f}, Ry sep={ry_sep:.4f} -- gestures are vertically separated.")


# ==============================================================================
# FINAL REPORT
# ==============================================================================
def final_report(direct_class, direct_conf, probs, live_tokens):
    print("\n" + "="*60)
    print("REAL PLEASE -> WATER FAILURE REPORT")
    print("="*60)

    captured = os.path.exists(FAILED_PLEASE_NPY)
    shape_str = str(np.load(FAILED_PLEASE_NPY).shape) if captured else "N/A"

    if captured:
        print("REAL_FAILURE_REPRODUCED=YES -- sequence captured")
    else:
        print("REAL_FAILURE_REPRODUCED=PENDING -- perform gesture in browser")
    print(f"FAILED_SEQUENCE_SHAPE={shape_str}")
    print(f"DIRECT_CNN_CLASS={direct_class or 'N/A'}")
    if direct_conf is not None:
        print(f"DIRECT_CNN_CONFIDENCE={round(direct_conf, 4)}")
    else:
        print("DIRECT_CNN_CONFIDENCE=N/A")

    if probs:
        sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"TOP_5={sorted_p}")

    please_seqs = load_class_tokens("please")
    water_seqs  = load_class_tokens("water")

    if please_seqs:
        ps = np.stack(please_seqs, axis=0)
        print(f"PLEASE_RAW_VIDEO_COUNT={count_raw_videos('please')}")
        print(f"PLEASE_VALID_TOKEN_COUNT={len(please_seqs)}")
        print(f"TRAINING_PLEASE_MEAN={[round(float(v),4) for v in ps.mean(axis=(0,1))]}")

    if live_tokens is not None and please_seqs and water_seqs:
        ps = np.stack(please_seqs, axis=0)
        ws = np.stack(water_seqs, axis=0)
        live_mean = live_tokens.mean(axis=0)
        print(f"LIVE_PLEASE_MEAN={[round(float(v),4) for v in live_mean]}")
        pd_ = float(np.linalg.norm(live_tokens - ps.mean(axis=0)))
        wd_ = float(np.linalg.norm(live_tokens - ws.mean(axis=0)))
        print(f"PLEASE_DISTANCE={pd_:.4f}")
        print(f"WATER_DISTANCE={wd_:.4f}")
        outside = wd_ < pd_
        print(f"LIVE_PLEASE_OUTSIDE_TRAINING_DISTRIBUTION={'YES' if outside else 'NO'}")
        print(f"LIVE_PLEASE_MEAN_Hy={live_tokens[:,1].mean():.4f}")
        print(f"LIVE_PLEASE_MEAN_Ry={live_tokens[:,5].mean():.4f}")
        print(f"TRAINING_PLEASE_MEAN_Hy={ps[:,:,1].mean():.4f}")
        print(f"TRAINING_PLEASE_MEAN_Ry={ps[:,:,5].mean():.4f}")
        print(f"TRAINING_WATER_MEAN_Hy={ws[:,:,1].mean():.4f}")
        print(f"TRAINING_WATER_MEAN_Ry={ws[:,:,5].mean():.4f}")

    print("\nNO_MODEL_CHANGED=NO")
    print("NO_THRESHOLD_CHANGED=NO")
    print("WATER_LOGIC_CHANGED=NO")
    print("SCHOOL_LOGIC_CHANGED=NO")

    print("\n--- ROOT CAUSE ANALYSIS ---")
    if direct_class == "water":
        print("ROOT_CAUSE=CNN MODEL outputs 'water' for the live PLEASE gesture.")
        print("  The model learned a feature representation where YOUR live PLEASE")
        print("  gesture maps into the water class space.")
        print("  This is a training distribution mismatch.")
        print("  YOUR real PLEASE gesture is NOT adequately represented in training data.")
        print("FIX_APPLIED=NONE YET -- RETRAINING REQUIRED")
        print("\nRECOMMENDED ACTION:")
        print("  1. Record 15-20 PLEASE gesture videos using your actual webcam:")
        print("       python src/record_webcam_dataset.py")
        print("  2. Re-run the token extraction pipeline:")
        print("       python src/run_real_pipeline.py")
        print("  3. Rebuild train/val/test split:")
        print("       python -c \"from src.build_real_split import build_real_splits; build_real_splits()\"")
        print("  4. Retrain the 21-class CNN-GRU:")
        print("       python src/train_21class_cnn_gru.py")
        print("  5. Validate PLEASE, WATER, SCHOOL, NO after retraining.")
        print("FINAL_STATUS=PENDING_RETRAINING")

    elif direct_class == "please":
        print("ROOT_CAUSE=CNN correctly outputs 'please' but routing/pipeline changed it to 'water'.")
        print("  Check EarlyDecisionEngine, sentence_processor, and final_class logic in app.py.")
        print("FIX_APPLIED=INVESTIGATE_ROUTING_PIPELINE")
        print("FINAL_STATUS=PENDING_ROUTING_FIX")

    else:
        if direct_class is not None:
            print(f"ROOT_CAUSE=CNN outputs '{direct_class}' -- model is confused by the gesture.")
        else:
            print("ROOT_CAUSE=No failed sequence captured yet.")
        print("  Likely distribution mismatch. Retraining with real PLEASE data recommended.")
        print("FINAL_STATUS=PENDING_DIAGNOSIS")

    if not captured:
        print("\nFINAL_STATUS=PENDING -- Perform PLEASE gesture in browser first, then re-run.")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PLEASE -> WATER MISCLASSIFICATION DIAGNOSIS")
    print("=" * 60)

    step1_backup()
    step2_capture_instruction()

    live_tokens, direct_class, probs = step3_direct_cnn_inference()
    if probs and direct_class and direct_class in probs:
        direct_conf = probs[direct_class]
    else:
        direct_conf = None

    step4_compare_with_training(live_tokens)
    step5_compare_against_water(live_tokens)
    step6_vertical_check(live_tokens)
    step11_training_data_check()
    step12_water_data_check()
    final_report(direct_class, direct_conf, probs, live_tokens)

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
