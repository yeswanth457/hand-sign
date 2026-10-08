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
    TOKEN_DIM, MAX_SEQ_LEN, MODEL_DIR, DATASET_DIR, MODEL_PATH
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, resample_tokens

def inspect_friend():
    raw_dir = os.path.join(DATASET_DIR, "raw", "friend")
    token_dir = os.path.join(DATASET_DIR, "tokens", "friend")
    csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")

    raw_files = glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)
    raw_files += glob.glob(os.path.join(raw_dir, "**", "*.avi"), recursive=True)
    raw_files += glob.glob(os.path.join(raw_dir, "**", "*.mov"), recursive=True)

    token_files = glob.glob(os.path.join(token_dir, "**", "*.npz"), recursive=True)

    print("=" * 60)
    print("PHASE 2: INSPECT EXISTING FRIEND DATA")
    print("=" * 60)
    print(f"Number of raw video files: {len(raw_files)}")
    print(f"Number of token files:     {len(token_files)}")

    # Check metadata CSV
    csv_friend_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("sign_class", "").lower() == "friend":
                    csv_friend_rows.append(r)
    print(f"Number of Friend rows in dataset.csv: {len(csv_friend_rows)}")

    # Sources of friend videos
    sources = set()
    signers = set()
    total_frames = 0
    valid_token_frames = 0
    one_hand_frames = 0
    two_hand_frames = 0
    no_hand_frames = 0
    lh_present_frames = 0
    rh_present_frames = 0
    seq_lengths = []

    for tf in token_files:
        try:
            data = np.load(tf)
            toks = data["tokens"] # shape (N, 12)
            N = len(toks)
            seq_lengths.append(N)
            total_frames += N

            for row in toks:
                lh_active = np.any(row[:6] != 0)
                rh_active = np.any(row[6:] != 0)

                if lh_active: lh_present_frames += 1
                if rh_active: rh_present_frames += 1

                if lh_active and rh_active:
                    two_hand_frames += 1
                elif lh_active or rh_active:
                    one_hand_frames += 1
                else:
                    no_hand_frames += 1

                if lh_active or rh_active:
                    valid_token_frames += 1

            fname = os.path.basename(tf)
            if "__" in fname:
                parts = fname.split("__")
                sources.add(parts[1])
        except Exception as e:
            print(f"Error reading {tf}: {e}")

    for r in csv_friend_rows:
        signers.add(r.get("signer_id", "unknown"))

    print(f"Total frames across all tokens: {total_frames}")
    if total_frames > 0:
        print(f"One-hand frames: {one_hand_frames} ({one_hand_frames/total_frames*100:.2f}%)")
        print(f"Two-hand frames: {two_hand_frames} ({two_hand_frames/total_frames*100:.2f}%)")
        print(f"No-hand frames:  {no_hand_frames} ({no_hand_frames/total_frames*100:.2f}%)")
        print(f"Left hand active frames:  {lh_present_frames} ({lh_present_frames/total_frames*100:.2f}%)")
        print(f"Right hand active frames: {rh_present_frames} ({rh_present_frames/total_frames*100:.2f}%)")
        print(f"Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.1f}")
    print(f"Dataset sources represented: {sources}")
    print(f"Signers in metadata: {signers} (Count: {len(signers)})")

    # Evaluate current model on test set and specifically on Friend
    mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
    std_path = os.path.join(MODEL_DIR, "feature_std.npy")
    feat_mean = np.load(mean_path).astype(np.float32)
    feat_std = np.load(std_path).astype(np.float32)

    device = torch.device("cpu")
    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    test_X = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
    test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))

    norm_test_X = (test_X - feat_mean) / feat_std
    with torch.no_grad():
        logits = model(torch.tensor(norm_test_X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=1)

    print("\n" + "=" * 60)
    print("CURRENT MODEL TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"Current Checkpoint: {MODEL_PATH}")
    overall_acc = np.mean(preds == test_y) * 100
    print(f"Overall Test Accuracy: {overall_acc:.2f}% ({np.sum(preds == test_y)}/{len(test_y)})")

    for cid, cname in enumerate(CLASS_NAMES):
        mask = (test_y == cid)
        if np.sum(mask) > 0:
            c_acc = np.mean(preds[mask] == test_y[mask]) * 100
            print(f"  [{cid:2d}] {cname:12s}: {c_acc:5.1f}% ({np.sum(preds[mask] == test_y[mask])}/{np.sum(mask)})")

    # Deep dive into Friend predictions on test set
    f_mask = (test_y == 18)
    print(f"\nFriend Test Set Breakdown (N={np.sum(f_mask)}):")
    f_preds = preds[f_mask]
    f_probs = probs[f_mask]
    for idx, (p, prob) in enumerate(zip(f_preds, f_probs)):
        conf = prob[p] * 100
        p_name = INDEX_TO_CLASS[p]
        top3_indices = np.argsort(prob)[::-1][:3]
        top3_str = ", ".join([f"{INDEX_TO_CLASS[i]}: {prob[i]*100:.1f}%" for i in top3_indices])
        print(f"  Sample {idx+1}: Predicted={p_name} ({conf:.1f}%) | Top 3: [{top3_str}]")

    # Evaluate ALL friend tokens in dataset/tokens/friend/hf_real/
    print("\n" + "=" * 60)
    print("CURRENT MODEL ON ALL FRIEND TOKENS IN REPOSITORY")
    print("=" * 60)
    all_f_preds = []
    all_f_confs = []
    for tf in token_files:
        try:
            d = np.load(tf)
            toks_25 = resample_tokens(d["tokens"], 25)
            norm_t = (toks_25 - feat_mean) / feat_std
            with torch.no_grad():
                out = model(torch.tensor(norm_t, dtype=torch.float32).unsqueeze(0))
                pr = torch.softmax(out, dim=-1).numpy()[0]
                pred_c = np.argmax(pr)
                all_f_preds.append(pred_c)
                all_f_confs.append(pr[pred_c])
        except Exception:
            pass

    all_f_preds = np.array(all_f_preds)
    friend_correct = np.sum(all_f_preds == 18)
    print(f"Total Friend Tokens: {len(all_f_preds)}")
    print(f"Correct Friend Predictions: {friend_correct}/{len(all_f_preds)} ({friend_correct/len(all_f_preds)*100:.1f}%)")
    print(f"Mean Confidence on Friend Tokens: {np.mean(all_f_confs)*100:.1f}%")
    from collections import Counter
    pred_counts = Counter([INDEX_TO_CLASS[p] for p in all_f_preds])
    print(f"Predicted class distribution across all friend tokens: {dict(pred_counts)}")

if __name__ == "__main__":
    inspect_friend()
