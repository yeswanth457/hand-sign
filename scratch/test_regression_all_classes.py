import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.build_real_split import resample_tokens
from config import DATASET_DIR, CLASS_TO_INDEX, CLASS_NAMES

engine = CNNGRUInferenceEngine()
test_x = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))

target_classes = ["no", "water", "please", "school", "thalapathy", "hello", "thank_you"]

print(f"{'='*70}")
print("REGRESSION TEST ON HELD-OUT TEST SPLIT (ALL TARGET CLASSES)")
print(f"{'='*70}")

for tc in target_classes:
    if tc not in CLASS_TO_INDEX:
        print(f"Class '{tc}' not in vocabulary!")
        continue
    cid = CLASS_TO_INDEX[tc]
    indices = np.where(test_y == cid)[0]
    print(f"\n--- Class: {tc.upper()} ({len(indices)} test samples) ---")
    correct = 0
    for idx in indices:
        sample = test_x[idx] # (25, 12)
        pred = engine.predict_sequence(sample)
        top1 = pred["word"]
        conf = pred["confidence"]
        is_match = (top1 == tc)
        if is_match:
            correct += 1
        status = "PASS" if is_match else "FAIL"
        print(f"  sample test_{idx:2d}: top1={top1:12s} conf={conf:.4f} [{status}]")
    acc = (correct / max(1, len(indices))) * 100
    print(f"  Accuracy for {tc}: {correct}/{len(indices)} ({acc:.1f}%)")
