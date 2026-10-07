import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath("."))
import urllib.request
import json
import numpy as np
from src.cnn_gru_model import CNNGRUInferenceEngine
from config import CLASS_NAMES, NUM_CLASSES

res = urllib.request.urlopen('http://127.0.0.1:8000/api/vocabulary')
data = json.loads(res.read().decode('utf-8'))
print(f"[VOCABULARY] Successfully loaded {len(data.get('vocabulary', []))} vocabulary items.")
print(f"Item 21: {data.get('vocabulary', [])[21]}")

engine = CNNGRUInferenceEngine()
print(f"MODEL_CLASS_COUNT = {engine.num_classes}")
print(f"MODEL_CLASS_21 = {CLASS_NAMES[21]}")

token_files = [
    'dataset/tokens/thalapathy/hf_real/WIN_20261006_22_05_30_Pro.npz',
    'dataset/tokens/thalapathy/hf_real/WIN_20261006_22_05_40_Pro.npz',
    'dataset/tokens/thalapathy/hf_real/WIN_20261006_22_05_51_Pro.npz',
    'dataset/tokens/thalapathy/hf_real/WIN_20261006_22_06_01_Pro.npz',
    'dataset/tokens/thalapathy/hf_real/WIN_20261006_22_06_10_Pro.npz',
    'dataset/tokens/thalapathy/hf_real/WIN_20261006_22_06_27_Pro.npz'
]

from src.build_real_split import resample_tokens

print("\n=== OFFLINE THALAPATHY CNN-GRU PREDICTIONS (RESAMPLED TO 25) ===")
for idx, tf in enumerate(token_files, 1):
    toks = np.load(tf)['tokens']
    toks_25 = resample_tokens(toks, target_t=25)
    res = engine.predict_sequence(toks_25)
    print(f"Sample {idx} (input shape {toks.shape} -> {toks_25.shape}): Word='{res['word']}' ({res['confidence']*100:.2f}%)")
    print(f"  Top 5 Probabilities: {res['probabilities']}")
