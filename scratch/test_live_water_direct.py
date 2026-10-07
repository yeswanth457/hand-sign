import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cnn_gru_model import CNNGRUInferenceEngine
from config import CLASS_TO_INDEX

engine = CNNGRUInferenceEngine()

# Test 1: Real webcam live recorded sequence
live_file = os.path.join("models", "debug_failed_please_as_water.npy")
if os.path.exists(live_file):
    seq = np.load(live_file)
    pred = engine.predict_sequence(seq)
    top3 = list(pred["probabilities"].items())[:3]
    print(f"{'='*60}")
    print("LIVE WATER DIRECT MODEL TEST")
    print(f"{'='*60}")
    print(f"input_shape: {seq.shape}")
    print(f"sequence_length: {len(seq)}")
    print(f"top1: {pred['word']}")
    print(f"top1_index: {CLASS_TO_INDEX.get(pred['word'], -1)}")
    print(f"top1_confidence: {pred['confidence']:.4f}")
    print(f"top3: {top3}")

# Also test debug_failed_school_as_water.npy
school_fail = os.path.join("models", "debug_failed_school_as_water.npy")
if os.path.exists(school_fail):
    s_seq = np.load(school_fail)
    s_pred = engine.predict_sequence(s_seq)
    print(f"\n--- School sequence prediction ---")
    print(f"input_shape: {s_seq.shape}")
    print(f"top1: {s_pred['word']} conf: {s_pred['confidence']:.4f}")
    print(f"top3: {list(s_pred['probabilities'].items())[:3]}")
