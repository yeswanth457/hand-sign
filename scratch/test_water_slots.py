import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.build_real_split import resample_tokens
from config import DATASET_DIR

engine = CNNGRUInferenceEngine()
csv_path = os.path.join(DATASET_DIR, "metadata", "dataset.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r.get("sign_class") == "water" and r.get("status") == "success"]

print("--- TESTING WATER SAMPLES DIRECTLY ---")
for i, r in enumerate(rows[:10]):
    tf = r["token_file"]
    if os.path.exists(tf):
        tokens = np.load(tf)["tokens"]
        resampled = resample_tokens(tokens, target_t=25)
        pred = engine.predict_sequence(resampled)
        lh = int(np.sum(np.any(resampled[:, :6] != 0, axis=1)))
        rh = int(np.sum(np.any(resampled[:, 6:12] != 0, axis=1)))
        print(f"Sample {i+1}: LH_frames={lh:2d}, RH_frames={rh:2d} -> top1={pred['word']:12s} conf={pred['confidence']:.4f}")
