import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import torch

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MODEL_DIR, DATASET_DIR, MODEL_PATH
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, resample_tokens

def swap_slots(tokens_25):
    swapped = np.zeros_like(tokens_25)
    swapped[:, :6] = tokens_25[:, 6:]
    swapped[:, 6:] = tokens_25[:, :6]
    return swapped

def test_friend_slots():
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

    token_files = glob.glob(os.path.join(DATASET_DIR, "tokens", "friend", "**", "*.npz"), recursive=True)
    print(f"Loaded {len(token_files)} Friend token files from repository.")

    orig_correct = 0
    swapped_correct = 0
    orig_preds = []
    swapped_preds = []

    for tf in token_files:
        d = np.load(tf)
        toks_25 = resample_tokens(d["tokens"], 25)

        # 1. Original slot
        norm_orig = (toks_25 - feat_mean) / feat_std
        with torch.no_grad():
            out_orig = model(torch.tensor(norm_orig, dtype=torch.float32).unsqueeze(0))
            p_orig = torch.softmax(out_orig, dim=-1).numpy()[0]
            pred_orig = np.argmax(p_orig)
            orig_preds.append(INDEX_TO_CLASS[pred_orig])
            if pred_orig == 18:
                orig_correct += 1

        # 2. Swapped slot (LH <-> RH)
        toks_swapped = swap_slots(toks_25)
        norm_swapped = (toks_swapped - feat_mean) / feat_std
        with torch.no_grad():
            out_swapped = model(torch.tensor(norm_swapped, dtype=torch.float32).unsqueeze(0))
            p_swapped = torch.softmax(out_swapped, dim=-1).numpy()[0]
            pred_swapped = np.argmax(p_swapped)
            swapped_preds.append(INDEX_TO_CLASS[pred_swapped])
            if pred_swapped == 18:
                swapped_correct += 1

    print("\nSLOT ASYMMETRY DIAGNOSTIC FOR FRIEND:")
    print(f"Original Repository Tokens: {orig_correct}/{len(token_files)} correct Friend ({orig_correct/len(token_files)*100:.1f}%)")
    print(f"Swapped (Right/Dominant Slot): {swapped_correct}/{len(token_files)} correct Friend ({swapped_correct/len(token_files)*100:.1f}%)")

    from collections import Counter
    print("\nWhen swapped into the other hand slot, Friend is misclassified as:")
    print(dict(Counter(swapped_preds)))

if __name__ == "__main__":
    test_friend_slots()
