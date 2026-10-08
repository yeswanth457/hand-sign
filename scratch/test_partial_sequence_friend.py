import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MODEL_DIR, MODEL_PATH
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, resample_tokens

def test_partial_sequences():
    print("=" * 70)
    print("PHASE 4: PARTIAL-SEQUENCE PREDICTION INSPECTION")
    print("=" * 70)

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

    token_files = glob.glob(os.path.join("dataset", "tokens", "friend", "**", "*.npz"), recursive=True)
    if not token_files:
        print("No Friend token files found.")
        return

    sample_npz = token_files[0]
    tokens_full = np.load(sample_npz)["tokens"].astype(np.float32)
    print(f"Testing on sample: {os.path.basename(sample_npz)} (total {len(tokens_full)} frames)")

    lengths_to_test = [5, 8, 10, 12, 15, 18, 20, 22, 25]
    print(f"{'Length':<8} | {'Top 1 Class':<12} | {'Top 1 Conf':<10} | {'Friend Prob':<12} | {'Top 2 Class':<12} | {'Top 2 Conf':<10} | {'Decision State':<12}")
    print("-" * 85)

    for l in lengths_to_test:
        sub_tokens = tokens_full[:min(l, len(tokens_full))]
        resampled_25 = resample_tokens(sub_tokens, 25)
        norm_t = (resampled_25 - feat_mean) / feat_std
        with torch.no_grad():
            logits = model(torch.tensor(norm_t, dtype=torch.float32).unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        sorted_idx = np.argsort(probs)[::-1]
        top1_cls = INDEX_TO_CLASS[sorted_idx[0]]
        top1_conf = probs[sorted_idx[0]]
        f_prob = probs[18]
        top2_cls = INDEX_TO_CLASS[sorted_idx[1]]
        top2_conf = probs[sorted_idx[1]]

        decision_state = "COLLECTING" if l < 20 else ("ACCEPTED" if top1_cls == "friend" and top1_conf >= 0.40 else "SIGNING")

        print(f"{l:>2d}/25    | {top1_cls:<12} | {top1_conf*100:>6.1f}%    | {f_prob*100:>6.1f}%      | {top2_cls:<12} | {top2_conf*100:>6.1f}%    | {decision_state:<12}")

if __name__ == "__main__":
    test_partial_sequences()
