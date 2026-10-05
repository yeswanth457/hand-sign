import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.build_real_split import resample_tokens

# 1. Failed Live Sequence
failed_seq_path = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.npy")
failed_seq = np.load(failed_seq_path).astype(np.float32)

failed_mean = failed_seq.mean(axis=0)
failed_std = failed_seq.std(axis=0)

# 2. Training School Tokens
school_dir = os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real")
school_files = [os.path.join(school_dir, f) for f in os.listdir(school_dir) if f.endswith(".npz")]
school_toks = [resample_tokens(np.load(f)["tokens"].astype(np.float32), 25) for f in school_files]
school_arr = np.array(school_toks) # (N, 25, 6)

train_school_mean = school_arr.mean(axis=(0, 1))
train_school_std = school_arr.std(axis=(0, 1))

# 3. Training Water Tokens
water_dir = os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real")
water_files = [os.path.join(water_dir, f) for f in os.listdir(water_dir) if f.endswith(".npz")]
water_toks = [resample_tokens(np.load(f)["tokens"].astype(np.float32), 25) for f in water_files]
water_arr = np.array(water_toks) # (M, 25, 6)

train_water_mean = water_arr.mean(axis=(0, 1))

# Distances
features = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]
feat_diff = np.abs(failed_mean - train_school_mean)

dist_to_school = float(np.linalg.norm(failed_mean - train_school_mean))
dist_to_water = float(np.linalg.norm(failed_mean - train_water_mean))

print(f"TRAINING_SCHOOL_MEAN={list(np.round(train_school_mean, 4))}")
print(f"TRAINING_SCHOOL_STD={list(np.round(train_school_std, 4))}")
print(f"FAILED_LIVE_SCHOOL_MEAN={list(np.round(failed_mean, 4))}")
print(f"FAILED_LIVE_SCHOOL_STD={list(np.round(failed_std, 4))}")

for i, feat in enumerate(features):
    print(f"{feat}_DISTANCE={feat_diff[i]:.4f}")

print(f"FAILED_SCHOOL_TO_SCHOOL_DISTANCE={dist_to_school:.4f}")
print(f"FAILED_SCHOOL_TO_WATER_DISTANCE={dist_to_water:.4f}")
