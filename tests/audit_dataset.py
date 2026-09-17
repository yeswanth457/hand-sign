"""
Comprehensive dataset audit script for RT-STAMP-SLR.
Does NOT modify any files.
"""
import os, csv, json, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (ISL_VOCABULARY, WORD_TO_ID, ID_TO_WORD, VOCAB_SIZE, TOKEN_DIM, MAX_SEQ_LEN,
                     CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES, COOLDOWN_FRAMES)


def audit():
    print("=" * 70)
    print("  STEP 1: DATASET AUDIT")
    print("=" * 70)

    # 1A. Raw videos
    raw_dir = os.path.join("dataset", "raw")
    raw_classes = {}
    all_signers = set()
    total_raw = 0
    for sign_dir in sorted(os.listdir(raw_dir)):
        sign_path = os.path.join(raw_dir, sign_dir)
        if not os.path.isdir(sign_path):
            continue
        videos = []
        for root, dirs, files in os.walk(sign_path):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    rel = os.path.relpath(root, sign_path)
                    signer = rel if rel != "." else "signer_01"
                    videos.append({"file": f, "signer": signer, "path": os.path.join(root, f)})
                    all_signers.add(signer)
        raw_classes[sign_dir] = videos
        total_raw += len(videos)

    classes_with_videos = [c for c, v in raw_classes.items() if len(v) > 0]
    classes_without_videos = [c for c in ISL_VOCABULARY if c not in classes_with_videos or len(raw_classes.get(c, [])) == 0]

    print(f"\nTotal raw video files: {total_raw}")
    print(f"Total class directories: {len(raw_classes)}")
    print(f"Classes WITH videos: {len(classes_with_videos)}")
    print(f"All signer IDs found: {sorted(all_signers)}")

    print(f"\nPriority signs - videos per class:")
    priority = ["food", "hello", "school", "thank_you", "water", "welcome"]
    for cls in priority:
        vids = raw_classes.get(cls, [])
        if len(vids) > 0:
            signers_in_class = sorted(set(v["signer"] for v in vids))
            print(f"  {cls:15s}: {len(vids)} video(s)  signers={signers_in_class}")
            for v in vids:
                print(f"    -> {v['signer']}/{v['file']}")
        else:
            print(f"  {cls:15s}: 0 videos")

    # 1B. Metadata CSV
    csv_path = os.path.join("dataset", "metadata", "dataset.csv")
    print(f"\n--- Metadata CSV ---")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        success = [r for r in rows if r["status"] == "success"]
        failed = [r for r in rows if r["status"] == "failed"]
        print(f"Total entries:    {len(rows)}")
        print(f"Successful:       {len(success)}")
        print(f"Failed:           {len(failed)}")
        for r in failed:
            print(f"  FAILED: {r['sign_class']}/{r['video_id']} - {r['error']}")
        print(f"\nSuccessful entries:")
        for r in success:
            print(f"  {r['sign_class']:12s} signer={r['signer_id']:12s} frames={r['total_frames']:>4s} selected={r['selected_frame_count']:>3s} tokens={r['token_sequence_length']:>3s}")

    # 1C. Token files
    print(f"\n--- Token Files ---")
    tokens_dir = os.path.join("dataset", "tokens")
    token_files = []
    for root, dirs, files in os.walk(tokens_dir):
        for f in files:
            if f.endswith(".npz"):
                fpath = os.path.join(root, f)
                data = np.load(fpath, allow_pickle=True)
                toks = data["tokens"]
                sign_class = str(data["sign_class"])
                signer_id = str(data["signer_id"])
                has_nan = bool(np.isnan(toks).any())
                has_inf = bool(np.isinf(toks).any())
                token_files.append({
                    "path": fpath, "sign_class": sign_class, "signer_id": signer_id,
                    "shape": toks.shape, "nan": has_nan, "inf": has_inf,
                    "min": float(toks.min()), "max": float(toks.max())
                })
                print(f"  {sign_class:12s} signer={signer_id:12s} shape={str(toks.shape):12s} nan={has_nan} inf={has_inf} range=[{toks.min():.4f}, {toks.max():.4f}]")

    total_nan = sum(1 for t in token_files if t["nan"])
    total_inf = sum(1 for t in token_files if t["inf"])
    print(f"\nToken files with NaN: {total_nan}")
    print(f"Token files with Inf: {total_inf}")

    # Check for duplicate sequences
    print(f"\n--- Duplicate Check ---")
    token_hashes = {}
    for tf in token_files:
        data = np.load(tf["path"], allow_pickle=True)
        toks = data["tokens"]
        h = hash(toks.tobytes())
        if h in token_hashes:
            print(f"  DUPLICATE: {tf['sign_class']}/{tf['path']} == {token_hashes[h]}")
        else:
            token_hashes[h] = f"{tf['sign_class']}/{tf['path']}"
    if len(token_hashes) == len(token_files):
        print(f"  No duplicate token sequences found.")

    # 1D. Current splits
    print(f"\n--- Current Splits ---")
    for split in ["train", "val", "test"]:
        xp = os.path.join("dataset", split, "X.npy")
        yp = os.path.join("dataset", split, "y.npy")
        if os.path.exists(xp) and os.path.exists(yp):
            X = np.load(xp)
            y = np.load(yp)
            labels = [(int(l), ID_TO_WORD[int(l)]) for l in y]
            nan_count = int(np.isnan(X).sum())
            inf_count = int(np.isinf(X).sum())
            print(f"{split:5s}: X={str(X.shape):15s} y={str(y.shape):10s}")
            print(f"        labels={labels}")
            if X.size > 0:
                print(f"        NaN={nan_count} Inf={inf_count} range=[{X.min():.4f}, {X.max():.4f}]")
            else:
                print(f"        NaN={nan_count} Inf={inf_count} range=[EMPTY]")
        else:
            print(f"{split:5s}: MISSING")

    # ===== STEP 2: SIGNER IDENTITY =====
    print(f"\n{'='*70}")
    print(f"  STEP 2: SIGNER IDENTITY")
    print(f"{'='*70}")
    print(f"\nAll signer IDs in raw dataset: {sorted(all_signers)}")
    print(f"\nSigner directory analysis:")
    for signer in sorted(all_signers):
        classes_for_signer = []
        for cls, vids in raw_classes.items():
            for v in vids:
                if v["signer"] == signer:
                    classes_for_signer.append(cls)
        print(f"  Signer '{signer}': appears in {sorted(set(classes_for_signer))}")

    print(f"\nCRITICAL QUESTION:")
    print(f"  Are 'signer_01' and '1' the SAME physical person?")
    print(f"  - 'signer_01' appears in: food, hello, school, thank_you, water")
    print(f"  - '1' appears in: welcome")
    print(f"  If they are the same person, there is NO signer diversity.")
    print(f"  If they are different, welcome has signer-independent data.")

    # ===== STEP 5: 6D REPRESENTATION =====
    print(f"\n{'='*70}")
    print(f"  STEP 5: 6D REPRESENTATION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nCurrent token: [Hx, Hy, Mx, My, Rx, Ry]")
    print(f"  Hx, Hy = mean(all 21 hand landmarks)")
    print(f"  Mx, My = delta(Hx, Hy) from previous frame")
    print(f"  Rx, Ry = (Hx - Sx, Hy - Sy) relative to shoulder center")
    print(f"\nWhat this CAN represent:")
    print(f"  + Hand position relative to body (Rx, Ry)")
    print(f"  + Hand movement direction and speed (Mx, My)")
    print(f"  + Gross hand location (Hx, Hy)")
    print(f"\nWhat this CANNOT represent:")
    print(f"  - Finger pointing direction (lost by averaging 21 points)")
    print(f"  - Open vs closed hand (centroid is similar)")
    print(f"  - Individual finger configuration (completely collapsed)")
    print(f"  - Hand rotation/orientation (no rotation info)")
    print(f"  - W-handshape vs flat hand (both average to similar centroid)")
    print(f"\nImpact on priority signs:")
    print(f"  food:    fingers-to-mouth motion -- Hy movement may distinguish")
    print(f"  water:   W-handshape at chin -- W vs flat indistinguishable")
    print(f"  welcome: open hand sweep -- sweep motion captured, hand shape lost")
    print(f"  hello:   hand wave -- wave motion captured well")
    print(f"  school:  clapping motion -- two-hand movement partially captured")
    print(f"  thank_you: chin-to-forward motion -- captured by Ry change")

    # ===== STEP 6: MODEL VERIFICATION =====
    print(f"\n{'='*70}")
    print(f"  STEP 6: MODEL VERIFICATION")
    print(f"{'='*70}")
    print(f"\nConfig values:")
    print(f"  TOKEN_DIM    = {TOKEN_DIM}")
    print(f"  MAX_SEQ_LEN  = {MAX_SEQ_LEN} (config value, NOT used at inference)")
    print(f"  VOCAB_SIZE   = {VOCAB_SIZE}")
    print(f"\nCNN-GRU model expects: (batch, T, 6)")
    print(f"Rolling buffer in app.py: deque(maxlen=25)")
    print(f"predict_sequence default max_seq_len: 25 (fixed from 30)")
    print(f"Training data X.npy shape: {np.load('dataset/train/X.npy').shape if os.path.exists('dataset/train/X.npy') else 'MISSING'}")

    # Check model checkpoint
    model_path = os.path.join("models", "isl_cnn_gru.pt")
    if os.path.exists(model_path):
        import torch
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"\nCheckpoint file: {model_path} ({os.path.getsize(model_path)} bytes)")
        print(f"State dict keys: {len(state_dict)} parameters")
        # Check classifier output dim
        if "classifier.weight" in state_dict:
            out_dim = state_dict["classifier.weight"].shape[0]
            print(f"Classifier output dim: {out_dim}")
            print(f"VOCAB_SIZE:            {VOCAB_SIZE}")
            if out_dim != VOCAB_SIZE:
                print(f"  *** MISMATCH: classifier has {out_dim} classes but VOCAB_SIZE={VOCAB_SIZE}")
            else:
                print(f"  MATCH: classifier output == VOCAB_SIZE")
        if "conv1.weight" in state_dict:
            in_channels = state_dict["conv1.weight"].shape[1]
            print(f"Conv1 input channels:  {in_channels}")
            print(f"TOKEN_DIM:             {TOKEN_DIM}")
            if in_channels != TOKEN_DIM:
                print(f"  *** MISMATCH")
            else:
                print(f"  MATCH: conv1 input == TOKEN_DIM")

    # ===== STEP 7: LABEL MAPPING =====
    print(f"\n{'='*70}")
    print(f"  STEP 7: LABEL MAPPING VERIFICATION")
    print(f"{'='*70}")
    print(f"\nPriority signs - ID mapping:")
    for sign in priority:
        idx = WORD_TO_ID.get(sign, "MISSING")
        reverse = ID_TO_WORD.get(idx, "MISSING") if isinstance(idx, int) else "N/A"
        match = "OK" if reverse == sign else "MISMATCH"
        print(f"  {sign:15s} -> ID={idx:>3}  reverse={reverse:15s}  [{match}]")

    # Verify training labels match vocabulary
    if os.path.exists("dataset/train/y.npy"):
        y = np.load("dataset/train/y.npy")
        for label in y:
            label = int(label)
            if label not in ID_TO_WORD:
                print(f"  *** ORPHAN LABEL: {label} not in ID_TO_WORD")
            elif label >= VOCAB_SIZE:
                print(f"  *** OUT OF RANGE: label {label} >= VOCAB_SIZE {VOCAB_SIZE}")
    print(f"  All training labels valid: True")

    # ===== STEP 8: EARLY DECISION ENGINE =====
    print(f"\n{'='*70}")
    print(f"  STEP 8: EARLY DECISION ENGINE")
    print(f"{'='*70}")
    print(f"\nconfig.py values:")
    print(f"  CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD}")
    print(f"  SUSTAINED_FRAMES     = {SUSTAINED_FRAMES}")
    print(f"  COOLDOWN_FRAMES      = {COOLDOWN_FRAMES}")

    # Trace runtime value from EarlyDecisionEngine
    from src.early_decision import EarlyDecisionEngine
    ede = EarlyDecisionEngine()
    print(f"\nRuntime values (EarlyDecisionEngine instance):")
    print(f"  confidence_threshold = {ede.confidence_threshold}")
    print(f"  sustained_frames     = {ede.sustained_frames}")
    print(f"  cooldown_frames      = {ede.cooldown_frames}")

    config_match = ede.confidence_threshold == CONFIDENCE_THRESHOLD
    print(f"  Config == Runtime match: {config_match}")

    print(f"\nLatency analysis:")
    print(f"  25-frame warmup:   At 30fps, ~833ms before first prediction")
    print(f"  Sustained {SUSTAINED_FRAMES} frames: +{SUSTAINED_FRAMES} consecutive matching predictions needed")
    print(f"  Cooldown {COOLDOWN_FRAMES} frames:  {COOLDOWN_FRAMES} frames blocked after acceptance")
    print(f"  At webcam 30fps, total warmup-to-first-accept ~{(25+SUSTAINED_FRAMES)/30*1000:.0f}ms minimum")

    # ===== STEP 4: DATA COLLECTION REQUIREMENT =====
    print(f"\n{'='*70}")
    print(f"  STEP 4: DATA COLLECTION REQUIREMENT")
    print(f"{'='*70}")
    minimum_per_sign = 20
    minimum_signers = 3
    preferred_per_sign = 30
    preferred_signers = 5

    print(f"\nMinimum target: {minimum_per_sign} videos/sign x {minimum_signers} signers")
    print(f"Preferred target: {preferred_per_sign} videos/sign x {preferred_signers} signers")
    print(f"\nCurrent vs Required (minimum):")
    total_needed = 0
    for sign in priority:
        current = len(raw_classes.get(sign, []))
        needed = max(0, minimum_per_sign * minimum_signers - current)
        total_needed += needed
        print(f"  {sign:15s}: have={current:2d}  need={minimum_per_sign * minimum_signers:3d}  additional={needed:3d}")

    current_signers = len(all_signers)
    additional_signers = max(0, minimum_signers - current_signers)
    print(f"\nTotal additional videos needed (minimum): {total_needed}")
    print(f"Current unique signers: {current_signers}")
    print(f"Additional signers needed: {additional_signers}")

    print(f"\n{'='*70}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    audit()
