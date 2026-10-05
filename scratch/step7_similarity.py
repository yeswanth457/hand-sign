import os
import sys
import numpy as np

BASE_DIR = r"D:\isl-translator"
sys.path.insert(0, BASE_DIR)

from src.build_real_split import resample_tokens

# 1. Failed Live Sequence
failed_seq_path = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.npy")
failed_seq = np.load(failed_seq_path).astype(np.float32)

# 2. Representative Real Water Sequence
water_rep_path = os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real", "RightFastWater.npz")
water_rep = resample_tokens(np.load(water_rep_path)["tokens"].astype(np.float32), 25)

# 3. Representative Real School Sequence
school_rep_path = os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real", "centerSchool.npz")
school_rep = resample_tokens(np.load(school_rep_path)["tokens"].astype(np.float32), 25)

def cosine_similarity(a, b):
    a_f = a.flatten()
    b_f = b.flatten()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f)))

sim_water = cosine_similarity(failed_seq, water_rep)
sim_school = cosine_similarity(failed_seq, school_rep)

print(f"LIVE_SCHOOL_WATER_SIMILARITY={sim_water:.4f}")
print(f"LIVE_SCHOOL_SCHOOL_SIMILARITY={sim_school:.4f}")
