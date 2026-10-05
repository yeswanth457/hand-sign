import os
import cv2
import numpy as np

BASE_DIR = r"D:\isl-translator"
vid_path = os.path.join(BASE_DIR, "dataset", "raw", "school", "hf_real", "centerSchool.mp4")

# Let's extract landmark information from centerSchool.npz
npz_path = os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real", "centerSchool.npz")
if os.path.exists(npz_path):
    data = np.load(npz_path)
    print("centerSchool.npz keys:", list(data.keys()))
    toks = data["tokens"]
    print("Tokens shape:", toks.shape)
    print("Mean Hx:", toks[:, 0].mean(), "Hy:", toks[:, 1].mean())
    print("Mean Mx:", toks[:, 2].mean(), "My:", toks[:, 3].mean())
    print("Mean Rx:", toks[:, 4].mean(), "Ry:", toks[:, 5].mean())
    print("Min Hy:", toks[:, 1].min(), "Max Hy:", toks[:, 1].max())
    print("Min Ry:", toks[:, 5].min(), "Max Ry:", toks[:, 5].max())

# Let's also check other school npz files
all_school_npz = [os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real", f)
                  for f in os.listdir(os.path.join(BASE_DIR, "dataset", "tokens", "school", "hf_real"))
                  if f.endswith(".npz")]

all_hy = []
all_ry = []
for p in all_school_npz:
    d = np.load(p)
    t = d["tokens"]
    all_hy.append(t[:, 1].mean())
    all_ry.append(t[:, 5].mean())

print(f"\nAll {len(all_school_npz)} School tokens:")
print(f"Overall School Hy Mean: {np.mean(all_hy):.4f} (range: {np.min(all_hy):.4f} to {np.max(all_hy):.4f})")
print(f"Overall School Ry Mean: {np.mean(all_ry):.4f} (range: {np.min(all_ry):.4f} to {np.max(all_ry):.4f})")

# Check Water npz files
all_water_npz = [os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real", f)
                 for f in os.listdir(os.path.join(BASE_DIR, "dataset", "tokens", "water", "hf_real"))
                 if f.endswith(".npz")]

water_hy = []
water_ry = []
for p in all_water_npz:
    d = np.load(p)
    t = d["tokens"]
    water_hy.append(t[:, 1].mean())
    water_ry.append(t[:, 5].mean())

print(f"\nAll {len(all_water_npz)} Water tokens:")
print(f"Overall Water Hy Mean: {np.mean(water_hy):.4f} (range: {np.min(water_hy):.4f} to {np.max(water_hy):.4f})")
print(f"Overall Water Ry Mean: {np.mean(water_ry):.4f} (range: {np.min(water_ry):.4f} to {np.max(water_ry):.4f})")
