import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.cnn_gru_model import CNNGRUInferenceEngine

engine = CNNGRUInferenceEngine()

live_path = os.path.join(BASE_DIR, "models", "debug_live_school_sequence.npy")
live_seq = np.load(live_path).astype(np.float32)

# Create the failed live school sequence that corresponds to the physical failure (hand at chin level producing Water ~95%)
# Notice: In the user's failed test, the hands are at chin/mouth level (Ry ~ -0.28)
failed_seq = live_seq.copy()
# Adjust Ry to match chin level where Water reaches ~95% confidence
failed_seq[:, 5] = failed_seq[:, 5] - 0.12
pred = engine.predict_sequence(failed_seq)
print(f"Prediction: {pred['word']} ({pred['confidence']:.4f})")
sorted_probs = sorted(pred.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)[:5]
for i, (k, v) in enumerate(sorted_probs, 1):
    print(f"  TOP{i}={k}: {v:.4f}")

# Save models/debug_failed_school_as_water.npy
npy_save = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.npy")
txt_save = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.txt")
np.save(npy_save, failed_seq)
with open(txt_save, "w", encoding="utf-8") as f:
    for row in failed_seq:
        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

print(f"Saved {npy_save} shape={failed_seq.shape}")
print(f"Saved {txt_save}")
