import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR, TOKEN_DIM, MODEL_DIR, CLASS_TO_INDEX
from src.cnn_gru_model import CNNGRUInferenceEngine, resample_tokens

print("==================================================")
print("STEP 6: COMPARE BROTHER AND WATER TRAINING DATA")
print("==================================================")

csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

b_rows = [r for r in rows if r.get("sign_class") == "brother" and r.get("status") == "success"]
w_rows = [r for r in rows if r.get("sign_class") == "water" and r.get("status") == "success"]

def load_and_resample(rows_list):
    resampled_list = []
    for r in rows_list:
        tf = r["token_file"]
        if os.path.exists(tf):
            tokens = np.load(tf)["tokens"]
            r_tokens = resample_tokens(tokens, target_t=25)
            resampled_list.append(r_tokens)
    return np.array(resampled_list) if resampled_list else np.empty((0, 25, 12))

b_arr = load_and_resample(b_rows) # (7, 25, 12)
w_arr = load_and_resample(w_rows) # (45, 25, 12)

print(f"Loaded Brother sequences: {b_arr.shape}")
print(f"Loaded Water sequences:   {w_arr.shape}")

b_flat = b_arr.reshape(-1, 12)
w_flat = w_arr.reshape(-1, 12)

feature_names = [
    "0:LHx", "1:LHy", "2:LMx", "3:LMy", "4:LRx", "5:LRy",
    "6:RHx", "7:RHy", "8:RMx", "9:RMy", "10:RRx", "11:RRy"
]

print("\n--- BROTHER FEATURE DISTRIBUTION (12D) ---")
print(f"{'Feature':8s} | {'Mean':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}")
print("-" * 50)
for j in range(12):
    col = b_flat[:, j]
    print(f"{feature_names[j]:8s} | {col.mean():8.4f} | {col.std():8.4f} | {col.min():8.4f} | {col.max():8.4f}")

print("\n--- WATER FEATURE DISTRIBUTION (12D) ---")
print(f"{'Feature':8s} | {'Mean':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}")
print("-" * 50)
for j in range(12):
    col = w_flat[:, j]
    print(f"{feature_names[j]:8s} | {col.mean():8.4f} | {col.std():8.4f} | {col.min():8.4f} | {col.max():8.4f}")

# Hand presence
def hand_presence(arr):
    # arr: (N, 25, 12)
    lh = np.any(arr[:, :, :6] != 0, axis=2) # (N, 25)
    rh = np.any(arr[:, :, 6:12] != 0, axis=2) # (N, 25)
    both = lh & rh
    lh_only = lh & (~rh)
    rh_only = rh & (~lh)
    total_frames = arr.shape[0] * arr.shape[1]
    return {
        "lh_presence": lh.sum() / total_frames,
        "rh_presence": rh.sum() / total_frames,
        "both_presence": both.sum() / total_frames,
        "lh_only": lh_only.sum() / total_frames,
        "rh_only": rh_only.sum() / total_frames
    }

b_pres = hand_presence(b_arr)
w_pres = hand_presence(w_arr)

print("\n--- HAND PRESENCE COMPARISON ---")
print("BROTHER:")
for k, v in b_pres.items():
    print(f"  {k:15s}: {v*100:5.1f}%")

print("WATER:")
for k, v in w_pres.items():
    print(f"  {k:15s}: {v*100:5.1f}%")

# Step 7: Check model predictions on Brother training and test samples
engine = CNNGRUInferenceEngine()
print("\n--- STEP 7: MODEL PREDICTIONS ON ALL 7 BROTHER SAMPLES ---")
for i, r in enumerate(b_rows):
    tf = r["token_file"]
    tokens = np.load(tf)["tokens"]
    r_tokens = resample_tokens(tokens, target_t=25)
    pred = engine.predict_sequence(r_tokens)
    top5 = list(pred["probabilities"].items())[:5]
    print(f"\nSample {i+1}: {r['video_id']}")
    print(f"  top1: {pred['word']} (conf: {pred['confidence']:.4f})")
    print(f"  top5: {top5}")
