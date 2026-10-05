import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.cnn_gru_model import CNNGRUInferenceEngine

npy_path = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.npy")
seq = np.load(npy_path).astype(np.float32)
assert seq.shape == (25, 6), f"Expected (25, 6), got {seq.shape}"

engine = CNNGRUInferenceEngine()
res = engine.predict_sequence(seq)

direct_class = res.get("word")
direct_conf = float(res.get("confidence", 0.0))
probs = res.get("probabilities", {})
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

print(f"DIRECT_FAILED_SCHOOL_CNN_CLASS={direct_class}")
print(f"DIRECT_FAILED_SCHOOL_CNN_CONFIDENCE={direct_conf:.4f}")
for i, (k, v) in enumerate(sorted_probs, 1):
    print(f"TOP{i}={k}: {v:.4f}")
