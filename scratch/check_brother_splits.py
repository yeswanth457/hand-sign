import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR
from src.cnn_gru_model import CNNGRUInferenceEngine

engine = CNNGRUInferenceEngine()
test_x = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
print("--- TEST SPLIT BROTHER SAMPLES ---")
for idx in [63, 64, 65]:
    pred = engine.predict_sequence(test_x[idx])
    top5 = list(pred["probabilities"].items())[:5]
    print(f"test_{idx}: top1={pred['word']} conf={pred['confidence']:.4f}")
    print(f"  top5: {top5}")

train_x = np.load(os.path.join(DATASET_DIR, "train", "X.npy"))
print("\n--- TRAIN SPLIT BROTHER SAMPLES ---")
for idx in [324, 325]:
    pred = engine.predict_sequence(train_x[idx])
    top5 = list(pred["probabilities"].items())[:5]
    print(f"train_{idx}: top1={pred['word']} conf={pred['confidence']:.4f}")
    print(f"  top5: {top5}")
