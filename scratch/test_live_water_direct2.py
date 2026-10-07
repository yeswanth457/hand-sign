import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cnn_gru_model import CNNGRUInferenceEngine
from config import CLASS_TO_INDEX

engine = CNNGRUInferenceEngine()
seq = np.load(os.path.join("models", "debug_live_water_final.npy"))
pred = engine.predict_sequence(seq)
print("LIVE WATER DIRECT MODEL TEST")
print("input_shape:          ", seq.shape)
print("sequence_length:      ", len(seq))
print("top1:                 ", pred["word"])
print("top1_index:           ", CLASS_TO_INDEX.get(pred["word"], -1))
print(f"top1_confidence:       {pred['confidence']:.4f}")
print("top3:                 ", list(pred["probabilities"].items())[:3])
