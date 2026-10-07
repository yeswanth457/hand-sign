import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.cnn_gru_model import CNNGRUInferenceEngine
from config import DATASET_DIR, CLASS_TO_INDEX, INDEX_TO_CLASS

engine = CNNGRUInferenceEngine()
print(f"Engine loaded: {engine.model_loaded}, model_path: {engine.model_path}")

test_x = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))

water_idx = CLASS_TO_INDEX["water"]
water_indices = np.where(test_y == water_idx)[0]
print(f"Found {len(water_indices)} water test samples: {water_indices}")

print("\nWATER OFFLINE TEST")
for i, idx in enumerate(water_indices):
    sample = test_x[idx]
    pred = engine.predict_sequence(sample)
    top3 = list(pred["probabilities"].items())[:3]
    print(f"sample: test_{idx} (water #{i+1})")
    print(f"shape: {sample.shape}")
    print(f"top1: {pred['word']}")
    print(f"confidence: {pred['confidence']:.4f}")
    print(f"top3: {top3}")
    print("-" * 40)
