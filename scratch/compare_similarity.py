import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.build_real_split import resample_tokens

# 1. Load Live School
live_path = os.path.join(BASE_DIR, "models", "debug_live_school_sequence.npy")
live_seq = np.load(live_path).astype(np.float32) # (25, 6)

# 2. Load Representative Water
water_rep_path = os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real", "RightFastWater.npz")
water_rep = resample_tokens(np.load(water_rep_path)["tokens"].astype(np.float32), 25) # (25, 6)

# 3. Load Representative School
school_rep_path = os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real", "centerSchool.npz")
school_rep = resample_tokens(np.load(school_rep_path)["tokens"].astype(np.float32), 25) # (25, 6)

# Compute Cosine Similarity (flattened 150-dim vector)
def cosine_sim(a, b):
    a_f = a.flatten()
    b_f = b.flatten()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f)))

# Compute Sequence Cosine Similarity (per-frame average)
def per_frame_cosine_sim(a, b):
    sims = []
    for t in range(25):
        denom = np.linalg.norm(a[t]) * np.linalg.norm(b[t])
        if denom > 1e-6:
            sims.append(np.dot(a[t], b[t]) / denom)
    return float(np.mean(sims))

# Normalized Euclidean distance converted to similarity: 1 / (1 + distance)
def euclidean_similarity(a, b):
    dist = np.linalg.norm(a.flatten() - b.flatten()) / np.sqrt(150)
    return float(1.0 / (1.0 + dist))

print("=== SEQUENCE SIMILARITY CALCULATIONS ===")
cos_water = cosine_sim(live_seq, water_rep)
cos_school = cosine_sim(live_seq, school_rep)
print(f"Cosine Similarity (Flat) Live School vs Real Water: {cos_water:.4f}")
print(f"Cosine Similarity (Flat) Live School vs Real School: {cos_school:.4f}")

pf_cos_water = per_frame_cosine_sim(live_seq, water_rep)
pf_cos_school = per_frame_cosine_sim(live_seq, school_rep)
print(f"Per-Frame Cosine Sim Live School vs Real Water: {pf_cos_water:.4f}")
print(f"Per-Frame Cosine Sim Live School vs Real School: {pf_cos_school:.4f}")

euc_sim_water = euclidean_similarity(live_seq, water_rep)
euc_sim_school = euclidean_similarity(live_seq, school_rep)
print(f"Normalized Euclidean Sim Live School vs Real Water: {euc_sim_water:.4f}")
print(f"Normalized Euclidean Sim Live School vs Real School: {euc_sim_school:.4f}")

# Average similarity across all training Water vs all training School
water_dir = os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real")
water_files = [os.path.join(water_dir, f) for f in os.listdir(water_dir) if f.endswith(".npz")]
water_sims = [cosine_sim(live_seq, resample_tokens(np.load(f)["tokens"].astype(np.float32), 25)) for f in water_files]

school_dir = os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real")
school_files = [os.path.join(school_dir, f) for f in os.listdir(school_dir) if f.endswith(".npz")]
school_sims = [cosine_sim(live_seq, resample_tokens(np.load(f)["tokens"].astype(np.float32), 25)) for f in school_files]

print(f"\nMean Cosine Sim Live School vs ALL Training Water: {np.mean(water_sims):.4f}")
print(f"Mean Cosine Sim Live School vs ALL Training School: {np.mean(school_sims):.4f}")
