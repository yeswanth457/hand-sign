import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR, TOKEN_DIM, MODEL_DIR, CLASS_TO_INDEX
from src.cnn_gru_model import CNNGRUInferenceEngine, resample_tokens

# Load Brother training samples and Water training samples
csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

b_rows = [r for r in rows if r.get("sign_class") == "brother" and r.get("status") == "success"]
w_rows = [r for r in rows if r.get("sign_class") == "water" and r.get("status") == "success"]

def load_resampled_all(r_list):
    arr = []
    for r in r_list:
        tf = r["token_file"]
        if os.path.exists(tf):
            tokens = np.load(tf)["tokens"]
            arr.append(resample_tokens(tokens, target_t=25))
    return np.array(arr)

b_samples = load_resampled_all(b_rows) # (7, 25, 12)
w_samples = load_resampled_all(w_rows) # (45, 25, 12)

# Compute mean trajectory for Brother and Water
b_mean_trajectory = np.mean(b_samples, axis=0) # (25, 12)
w_mean_trajectory = np.mean(w_samples, axis=0) # (25, 12)

# Check distance between Brother and Water themselves
bw_dist = np.linalg.norm(b_mean_trajectory - w_mean_trajectory)
print(f"Euclidean distance between Brother mean and Water mean trajectory: {bw_dist:.4f}")

# Look at two-handed Brother sample (video_bridgeconn_2650_brother)
b_2hand = None
for r in b_rows:
    if "2650" in r["video_id"]:
        b_2hand = resample_tokens(np.load(r["token_file"])["tokens"], target_t=25)
        break

if b_2hand is not None:
    dist_2h_to_b = np.linalg.norm(b_2hand - b_mean_trajectory)
    dist_2h_to_w = np.linalg.norm(b_2hand - w_mean_trajectory)
    print(f"\nTwo-handed Brother sample (video_bridgeconn_2650_brother):")
    print(f"  Distance to Brother mean trajectory: {dist_2h_to_b:.4f}")
    print(f"  Distance to Water mean trajectory:   {dist_2h_to_w:.4f}")
    
    # What if only right hand of two-handed brother is kept (simulating left hand drop)?
    b_2h_dropped_lh = b_2hand.copy()
    b_2h_dropped_lh[:, :6] = 0.0 # LH dropped by MediaPipe
    dist_drop_to_b = np.linalg.norm(b_2h_dropped_lh - b_mean_trajectory)
    dist_drop_to_w = np.linalg.norm(b_2h_dropped_lh - w_mean_trajectory)
    print(f"\nTwo-handed Brother with LH dropped (simulating occlusion):")
    print(f"  Distance to Brother mean trajectory: {dist_drop_to_b:.4f}")
    print(f"  Distance to Water mean trajectory:   {dist_drop_to_w:.4f}")

    # Predict with model
    engine = CNNGRUInferenceEngine()
    pred_full = engine.predict_sequence(b_2hand)
    pred_drop = engine.predict_sequence(b_2h_dropped_lh)
    print(f"\nModel prediction on full 2-hand Brother:  {pred_full['word']} ({pred_full['confidence']:.4f})")
    print(f"  top5: {list(pred_full['probabilities'].items())[:5]}")
    print(f"Model prediction with LH dropped:         {pred_drop['word']} ({pred_drop['confidence']:.4f})")
    print(f"  top5: {list(pred_drop['probabilities'].items())[:5]}")
