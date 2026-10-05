import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.cnn_gru_model import CNNGRUInferenceEngine
from src.build_real_split import resample_tokens

engine = CNNGRUInferenceEngine()

print("=" * 60)
print("NEW MODEL REGRESSION TEST")
print("=" * 60)

# Test 1: Failed Live School sequences (the user's actual webcam height)
failed_seq = np.load(os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.npy")).astype(np.float32)
live_seq = np.load(os.path.join(BASE_DIR, "models", "debug_live_school_sequence.npy")).astype(np.float32)

print("\n--- SCHOOL: User Real Webcam Sequences ---")
for i, seq in enumerate([failed_seq, live_seq], 1):
    res = engine.predict_sequence(seq)
    probs = res.get("probabilities", {})
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    hy = float(seq[:, 1].mean())
    ry = float(seq[:, 5].mean())
    status = "PASS" if res["word"] == "school" else "FAIL"
    print(f"  SCHOOL_TEST_{i}: {res['word']} ({res['confidence']:.4f}) | Hy={hy:.4f} Ry={ry:.4f} -> [{status}]")
    print(f"    Top3: {top5[:3]}")

# Also test the 25 generated user school tokens
print("\n--- SCHOOL: 25 Generated User School Tokens ---")
school_dir = os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real")
user_school_files = sorted([f for f in os.listdir(school_dir) if f.startswith("user_real_school_")])
school_results = []
for fname in user_school_files:
    d = np.load(os.path.join(school_dir, fname))
    toks = d["tokens"].astype(np.float32)
    res = engine.predict_sequence(toks)
    school_results.append(res["word"] == "school")
    status = "PASS" if res["word"] == "school" else "FAIL"
    print(f"  {fname}: {res['word']} ({res['confidence']:.4f}) [{status}]")

pass_rate = sum(school_results) / len(school_results) * 100
print(f"\n  School recall on user tokens: {sum(school_results)}/{len(school_results)} = {pass_rate:.1f}%")

# Test 2: Water dataset (must still predict water)
print("\n--- WATER: Dataset Tokens ---")
water_dir = os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real")
water_files = sorted([f for f in os.listdir(water_dir) if f.endswith(".npz")])
water_results = []
for fname in water_files[:10]:
    d = np.load(os.path.join(water_dir, fname))
    toks = resample_tokens(d["tokens"].astype(np.float32), 25)
    res = engine.predict_sequence(toks)
    water_results.append(res["word"] == "water")
    status = "PASS" if res["word"] == "water" else "FAIL"
    print(f"  {fname}: {res['word']} ({res['confidence']:.4f}) [{status}]")

water_rate = sum(water_results) / len(water_results) * 100
print(f"\n  Water recall on dataset: {sum(water_results)}/{len(water_results)} = {water_rate:.1f}%")

# Test 3: No dataset tokens
print("\n--- NO: Dataset Tokens ---")
no_dir = os.path.join(BASE_DIR, "dataset", "tokens", "no", "hf_real")
if os.path.exists(no_dir):
    no_files = sorted([f for f in os.listdir(no_dir) if f.endswith(".npz")])[:10]
    no_results = []
    for fname in no_files:
        d = np.load(os.path.join(no_dir, fname))
        toks = resample_tokens(d["tokens"].astype(np.float32), 25)
        res = engine.predict_sequence(toks)
        no_results.append(res["word"] == "no")
        status = "PASS" if res["word"] == "no" else "FAIL"
        print(f"  {fname}: {res['word']} ({res['confidence']:.4f}) [{status}]")
    no_rate = sum(no_results) / len(no_results) * 100 if no_results else 0
    print(f"\n  No recall on dataset: {sum(no_results)}/{len(no_results)} = {no_rate:.1f}%")
else:
    print("  No tokens dir not found.")

# Test 4: Please dataset tokens
print("\n--- PLEASE: Dataset Tokens ---")
please_dir = os.path.join(BASE_DIR, "dataset", "tokens", "please", "hf_real")
if os.path.exists(please_dir):
    please_files = sorted([f for f in os.listdir(please_dir) if f.endswith(".npz")])[:10]
    please_results = []
    for fname in please_files:
        d = np.load(os.path.join(please_dir, fname))
        toks = resample_tokens(d["tokens"].astype(np.float32), 25)
        res = engine.predict_sequence(toks)
        please_results.append(res["word"] == "please")
        status = "PASS" if res["word"] == "please" else "FAIL"
        print(f"  {fname}: {res['word']} ({res['confidence']:.4f}) [{status}]")
    please_rate = sum(please_results) / len(please_results) * 100 if please_results else 0
    print(f"\n  Please recall on dataset: {sum(please_results)}/{len(please_results)} = {please_rate:.1f}%")
else:
    print("  Please tokens dir not found.")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  SCHOOL (user webcam height) : {sum(school_results)}/{len(school_results)} ({pass_rate:.0f}%) {'PASS' if pass_rate >= 80 else 'FAIL'}")
print(f"  WATER (dataset)             : {sum(water_results)}/{len(water_results)} ({water_rate:.0f}%) {'PASS' if water_rate >= 80 else 'FAIL'}")
