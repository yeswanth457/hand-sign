import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from src.cnn_gru_model import CNNGRUInferenceEngine
from config import BASE_DIR, CLASS_NAMES

npy_path = os.path.join(BASE_DIR, "models", "debug_live_school_sequence.npy")
if not os.path.exists(npy_path):
    print(f"File not found: {npy_path}")
    sys.exit(1)

seq = np.load(npy_path)
print(f"Sequence shape: {seq.shape}")

engine = CNNGRUInferenceEngine()
res = engine.predict_sequence(seq)
print(f"RAW_CNN_CLASS={res.get('word')}")
print(f"RAW_CNN_CONFIDENCE={res.get('confidence'):.4f}")

probs = res.get("probabilities", {})
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
for i, (c_name, c_conf) in enumerate(sorted_probs, 1):
    print(f"TOP{i}={c_name}: {c_conf:.4f}")
