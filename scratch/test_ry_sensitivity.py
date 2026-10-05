import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.cnn_gru_model import CNNGRUInferenceEngine

engine = CNNGRUInferenceEngine()

live_path = os.path.join(BASE_DIR, "models", "debug_live_school_sequence.npy")
live_seq = np.load(live_path).astype(np.float32)

print("Original live sequence prediction:")
pred = engine.predict_sequence(live_seq)
print(f"  Word: {pred['word']} ({pred['confidence']:.4f})")
print(f"  Water prob: {pred.get('probabilities', {}).get('water', 0.0):.4f}")
print(f"  School prob: {pred.get('probabilities', {}).get('school', 0.0):.4f}")

# What if Ry is slightly more negative (hand slightly higher up near mouth)?
for d_ry in [-0.05, -0.10, -0.15, -0.20]:
    seq_mod = live_seq.copy()
    seq_mod[:, 5] += d_ry # Ry more negative
    p = engine.predict_sequence(seq_mod)
    w_p = p.get('probabilities', {}).get('water', 0.0)
    s_p = p.get('probabilities', {}).get('school', 0.0)
    print(f"  d_ry={d_ry:+.2f} (mean Ry={seq_mod[:,5].mean():.4f}) -> {p['word']} (conf={p['confidence']:.4f}), water={w_p:.4f}, school={s_p:.4f}")
