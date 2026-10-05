import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.build_real_split import resample_tokens
from src.cnn_gru_model import CNNGRUInferenceEngine

engine = CNNGRUInferenceEngine()

water_dir = os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real")
water_files = [os.path.join(water_dir, f) for f in os.listdir(water_dir) if f.endswith(".npz")]

print(f"Total water files: {len(water_files)}")
# Evaluate each water file
high_conf_water = []
for f in water_files:
    d = np.load(f)
    toks = resample_tokens(d["tokens"].astype(np.float32), 25)
    pred = engine.predict_sequence(toks)
    if pred["word"] == "water":
        high_conf_water.append((os.path.basename(f), pred["confidence"], toks))

high_conf_water.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 confident water files in dataset:")
for name, conf, _ in high_conf_water[:5]:
    print(f"  {name}: {conf:.4f}")

# Pick representative water sequence
rep_name, rep_conf, rep_toks = high_conf_water[0]
print(f"\nRepresentative Water sequence: {rep_name} (conf={rep_conf:.4f})")
print(f"Mean Hx: {rep_toks[:, 0].mean():.4f}, Hy: {rep_toks[:, 1].mean():.4f}")
print(f"Mean Rx: {rep_toks[:, 4].mean():.4f}, Ry: {rep_toks[:, 5].mean():.4f}")
