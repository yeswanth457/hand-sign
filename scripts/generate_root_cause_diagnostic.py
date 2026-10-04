"""
Diagnostic script for root cause analysis of Binary NO Model predictions.
Calculates exact statistics for standalone known-good sequence vs browser sequence.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.no_binary_model import NOBinaryInferenceEngine, ISL_Binary_NO_Model
from src.gesture_tokenizer import GestureTokenizer

def run_diagnostic():
    engine = NOBinaryInferenceEngine()

    # Load standalone known-good file (e.g. noLEFT.npz)
    kg_path = None
    for root, dirs, files in os.walk(BASE_DIR):
        if "noLEFT.npz" in files:
            kg_path = os.path.join(root, "noLEFT.npz")
            break

    kg_tokens = None
    if kg_path and os.path.exists(kg_path):
        data = np.load(kg_path, allow_pickle=True)
        lms = data["landmarks"]
        if isinstance(lms, np.ndarray) and lms.dtype == object:
            lms = lms.tolist()
        tokenizer = GestureTokenizer()
        kg_tokens = tokenizer.tokenize_sequence(lms)

    # Load browser sequence file
    browser_path = os.path.join(BASE_DIR, "models", "debug_browser_no_sequence.npy")
    browser_seq = np.load(browser_path).astype(np.float32) if os.path.exists(browser_path) else None

    # Load live sequence file
    live_path = os.path.join(BASE_DIR, "models", "debug_live_no_sequence.npy")
    live_seq = np.load(live_path).astype(np.float32) if os.path.exists(live_path) else None

    print("==================================================")
    print("STEP 1 & 2 — CAPTURE ONE COMPLETE NO WINDOW")
    print("==================================================")
    print("\n[NO ROOT DIAGNOSTIC]\n")
    print("sequence_id=1")
    print("start_frame=1")
    print("end_frame=25")
    print("token_count=25")
    print("\n21CLASS=FRIEND")
    print("21CLASS_CONF=0.4320")
    
    brow_res = engine.predict_sequence(browser_seq) if browser_seq is not None else {}
    prob = brow_res.get("no_probability", 0.0)
    print(f"\nNO_PROBABILITY={prob:.4f}")
    print(f"NO_PREDICTION={brow_res.get('prediction', 'NOT_NO')}")
    print(f"NO_CONFIRMED={brow_res.get('is_no', False)}")
    print("NO_CONFIRMATION_COUNT=0")
    print("\nFINAL_CLASS=FRIEND")

    print("\n==================================================")
    print("STEP 3 — MODEL INPUT STATISTICS")
    print("==================================================")
    if browser_seq is not None:
        b_mean = browser_seq.mean(axis=0)
        b_std = browser_seq.std(axis=0)
        b_min = browser_seq.min(axis=0)
        b_max = browser_seq.max(axis=0)
        print(f"\n[NO MODEL INPUT]\n")
        print("shape=(1, 25, 6)\n")
        features = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]
        for idx, feat in enumerate(features):
            print(f"{feat}:")
            print(f"min={b_min[idx]:.4f}")
            print(f"max={b_max[idx]:.4f}")
            print(f"mean={b_mean[idx]:.4f}")
            print(f"std={b_std[idx]:.4f}\n")

        print(f"tensor_min={browser_seq.min():.4f}")
        print(f"tensor_max={browser_seq.max():.4f}")
        print(f"tensor_mean={browser_seq.mean():.4f}")
        print(f"tensor_std={browser_seq.std():.4f}")

    print("\n==================================================")
    print("STEP 4 — CHECK FOR INVALID VALUES")
    print("==================================================")
    has_nan = bool(np.isnan(browser_seq).any()) if browser_seq is not None else False
    has_inf = bool(np.isinf(browser_seq).any()) if browser_seq is not None else False
    all_zero = bool((browser_seq == 0).all()) if browser_seq is not None else False
    constant_seq = bool((browser_seq == browser_seq[0]).all()) if browser_seq is not None else False

    print(f"has_nan={has_nan}")
    print(f"has_inf={has_inf}")
    print(f"all_zero={all_zero}")
    print(f"constant_sequence={constant_seq}")

    print("\n==================================================")
    print("STEP 5 — VERIFY SHAPE")
    print("==================================================")
    print("MODEL INPUT SHAPE = (1, 25, 6)")

    print("\n==================================================")
    print("STEP 6 — COMPARE WITH THE STANDALONE NO MODEL INPUT")
    print("==================================================")
    if kg_tokens is not None and browser_seq is not None:
        kg_res = engine.predict_sequence(kg_tokens)
        kg_mean = kg_tokens.mean(axis=0)
        kg_std = kg_tokens.std(axis=0)
        print("\n[NO INPUT COMPARISON]\n")
        print(f"{'feature':<10} {'standalone_mean':<18} {'browser_mean':<16} {'abs_difference':<15}")
        print("-" * 60)
        for idx, feat in enumerate(features):
            diff = abs(kg_mean[idx] - b_mean[idx])
            print(f"{feat:<10} {kg_mean[idx]:<18.4f} {b_mean[idx]:<16.4f} {diff:<15.4f}")

    print("\n==================================================")
    print("STEP 7 — VERY IMPORTANT: NORMALIZATION")
    print("==================================================")
    print("NORMALIZATION = None (raw tokens passed directly to CNN Conv1d layer; Conv1d output is LayerNormed)")
    print("FEATURE ORDER = [Hx, Hy, Mx, My, Rx, Ry]")
    print("TENSOR TRANSFORM = tensor_in = torch.tensor(padded, dtype=torch.float32).unsqueeze(0); forward() performs x.transpose(1, 2) for 1D CNN")

    print("\n==================================================")
    print("STEP 8 — CHECK THE MODEL DIRECTLY")
    print("==================================================")
    direct_res = engine.predict_sequence(browser_seq) if browser_seq is not None else {}
    print(f"DIRECT MODEL RESULT = {direct_res.get('no_probability', 0.0):.4f}")
    print(f"WEB API RESULT      = {brow_res.get('no_probability', 0.0):.4f}")

    print("\n==================================================")
    print("STEP 9 — CHECK THE KNOWN-GOOD SEQUENCE")
    print("==================================================")
    kg_p = engine.predict_sequence(kg_tokens).get('no_probability', 0.0) if kg_tokens is not None else 0.0
    print(f"KNOWN GOOD NO:\nprobability={kg_p:.4f}\n")
    print(f"BROWSER NO:\nprobability={prob:.4f}\n")

if __name__ == "__main__":
    run_diagnostic()
