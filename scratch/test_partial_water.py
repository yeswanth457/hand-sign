import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.cnn_gru_model import CNNGRUInferenceEngine
from config import DATASET_DIR, CLASS_TO_INDEX

engine = CNNGRUInferenceEngine()
test_x = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))
water_idx = CLASS_TO_INDEX["water"]
sample = test_x[np.where(test_y == water_idx)[0][0]] # full 25 frames

print("--- PARTIAL SEQUENCE ZERO-PAD TEST ON TRUE WATER SAMPLE ---")
for L in [8, 10, 11, 12, 13, 14, 15, 18, 20, 22, 25]:
    partial = sample[:L]
    pred = engine.predict_sequence(partial)
    print(f"L={L:2d}/25: top1={pred['word']:25s} conf={pred['confidence']:.4f} top3={list(pred['probabilities'].items())[:3]}")
