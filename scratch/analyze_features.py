import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.build_real_split import resample_tokens

# 1. Inspect Training School Tokens
school_dir = os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real")
school_files = [os.path.join(school_dir, f) for f in os.listdir(school_dir) if f.endswith(".npz")]

school_tokens = []
school_per_file = {}
for f in school_files:
    fname = os.path.basename(f)
    toks = resample_tokens(np.load(f)["tokens"].astype(np.float32), 25)
    school_tokens.append(toks)
    school_per_file[fname] = toks

school_arr = np.array(school_tokens) # (N, 25, 6)
train_school_mean = school_arr.mean(axis=(0, 1))
train_school_std = school_arr.std(axis=(0, 1))

# 2. Inspect Training Water Tokens
water_dir = os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real")
water_files = [os.path.join(water_dir, f) for f in os.listdir(water_dir) if f.endswith(".npz")]

water_tokens = []
for f in water_files:
    toks = resample_tokens(np.load(f)["tokens"].astype(np.float32), 25)
    water_tokens.append(toks)

water_arr = np.array(water_tokens) # (M, 25, 6)
train_water_mean = water_arr.mean(axis=(0, 1))
train_water_std = water_arr.std(axis=(0, 1))

# 3. Inspect Live Sequence
live_path = os.path.join(BASE_DIR, "models", "debug_live_school_sequence.npy")
if os.path.exists(live_path):
    live_seq = np.load(live_path)
    live_mean = live_seq.mean(axis=0)
    live_std = live_seq.std(axis=0)
else:
    live_seq = None
    live_mean = np.zeros(6)
    live_std = np.zeros(6)

print("=== FEATURE COMPARISON ===")
features = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]
print(f"{'Feature':<6} | {'Train School':<14} | {'Train Water':<14} | {'Live School':<14}")
for i, feat in enumerate(features):
    print(f"{feat:<6} | {train_school_mean[i]:>6.4f} +/- {train_school_std[i]:<5.4f} | {train_water_mean[i]:>6.4f} +/- {train_water_std[i]:<5.4f} | {live_mean[i]:>6.4f} +/- {live_std[i]:<5.4f}")

# Detailed file breakdown for seed recordings
print("\n=== SEED SCHOOL RECORDINGS BREAKDOWN ===")
seed_names = ["centerSchool.npz", "slowSchool.npz", "fastSchool.npz", "ZoomOutSchool.npz", "left_school_01.npz", "right_school_01.npz"]
for sname in seed_names:
    if sname in school_per_file:
        stoks = school_per_file[sname]
        s_mean = stoks.mean(axis=0)
        print(f"{sname:<20}: Hy={s_mean[1]:.4f}, Ry={s_mean[5]:.4f}, Hx={s_mean[0]:.4f}, Rx={s_mean[4]:.4f}")

if live_seq is not None:
    # Calculate Distances
    dist_live_to_school = float(np.linalg.norm(live_mean - train_school_mean))
    dist_live_to_water = float(np.linalg.norm(live_mean - train_water_mean))
    print(f"\nFAILED_SCHOOL_TO_SCHOOL_DISTANCE={dist_live_to_school:.4f}")
    print(f"FAILED_SCHOOL_TO_WATER_DISTANCE={dist_live_to_water:.4f}")
    
    # Feature-wise distance to school mean
    feat_diff = np.abs(live_mean - train_school_mean)
    for i, feat in enumerate(features):
        print(f"{feat}_DISTANCE={feat_diff[i]:.4f}")
