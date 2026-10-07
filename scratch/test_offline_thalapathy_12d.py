import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import numpy as np
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.build_real_split import resample_tokens

engine = CNNGRUInferenceEngine()
files = sorted(glob.glob('dataset/tokens/thalapathy/**/*.npz', recursive=True))
print(f"=== THALAPATHY OFFLINE VALIDATION ({len(files)} samples) ===")

for idx, f in enumerate(files, 1):
    d = np.load(f)
    toks = d['tokens']
    toks_25 = resample_tokens(toks, target_t=25)
    res = engine.predict_sequence(toks_25)
    print(f"THALAPATHY_OFFLINE_{idx}:")
    print(f"  hand_count = {2 if np.any(toks[:, :6]) and np.any(toks[:, 6:]) else 1}")
    print(f"  raw_shape = {toks.shape} -> resampled {toks_25.shape}")
    print(f"  top1 = {res['word']}")
    print(f"  confidence = {res['confidence']*100:.2f}%")
    print(f"  top5 = {res['probabilities']}")
    print("-" * 50)
