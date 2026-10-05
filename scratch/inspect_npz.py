import numpy as np

d = np.load('dataset/tokens/school/hf_real/centerSchool.npz')
print("Keys:", list(d.keys()))
for k in d.keys():
    v = d[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")
