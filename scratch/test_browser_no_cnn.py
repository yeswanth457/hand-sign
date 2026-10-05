import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.cnn_gru_model import CNNGRUInferenceEngine

engine = CNNGRUInferenceEngine()
s = np.load('models/debug_browser_no_sequence.npy')
res = engine.predict_sequence(s)
print("debug_browser_no_sequence RAW CNN-GRU:")
print("  Word:", res["word"], "Conf:", res["confidence"])
probs = sorted(res.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)[:5]
for i, (k, v) in enumerate(probs, 1):
    print(f"  TOP{i}={k}: {v:.4f}")
