import glob
import sys
sys.path.insert(0, ".")
import requests
import numpy as np
from src.build_real_split import resample_tokens

files = sorted(glob.glob("dataset/tokens/thalapathy/hf_real/*.npz"))
print(f"Testing {len(files)} Thalapathy files through live running API...")

import time
for idx, fpath in enumerate(files):
    raw = np.load(fpath)["tokens"]
    t25 = resample_tokens(raw, 25)
    res = None
    seq_id = int(time.time() * 1000) + idx * 100
    for i, row in enumerate(t25):
        payload = {
            "token": row.tolist(),
            "frame_id": i + 1,
            "sequence_id": seq_id,
            "has_hand": True,
            "raw_hands": 2,
            "left_detected": True,
            "right_detected": True,
            "active_hands": "Left+Right"
        }
        res = requests.post("http://127.0.0.1:8000/api/process_token", json=payload).json()
    
    p_class = res.get("primary_class")
    p_conf = res.get("primary_confidence", 0.0)
    f_class = res.get("final_class")
    print(f"THALAPATHY_API_{idx+1}: primary_class={p_class}, conf={p_conf*100:.2f}%, final_class={f_class}")
