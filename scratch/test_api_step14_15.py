import os
import sys
import json
import time
import urllib.request
import numpy as np

BASE_DIR = r"D:\isl-translator"
npy_path = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.npy")
seq = np.load(npy_path).astype(np.float32)

url = "http://127.0.0.1:8000/api/process_token"
seq_id = int(time.time() * 1000)

responses = []
for fid in range(1, 26):
    token = [float(v) for v in seq[fid - 1]]
    payload = {
        "token": token,
        "frame_id": fid,
        "sequence_id": seq_id,
        "timestamp_ms": float(fid * 50),
        "pose": {},
        "hand_center": [token[0], token[1]],
        "shoulder_center": [token[0] - token[4], token[1] - token[5]],
        "has_hand": True,
        "hand_count": 2,
        "primary_hand": "Right",
        "selected_hand": "Right",
        "selected_hand_index": 0,
        "selected_hand_confidence": 0.98,
        "camera_fps": 30.0,
        "token_fps": 20.0
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        responses.append(res_data)

last_res = responses[-1]
print("\n=== STEP 14: STALE WATER / LIVE ROUTING CHECK ===")
raw_cnn_class = last_res.get("primary_class")
raw_cnn_conf = last_res.get("primary_confidence")
early_decision = last_res.get("early_decision", {}).get("state", "--")
final_class = last_res.get("final_class")
ui_output = last_res.get("prediction", {}).get("word", "--")

print(f"RAW_CNN_CLASS={raw_cnn_class}")
print(f"RAW_CNN_CONFIDENCE={raw_cnn_conf}")
print(f"EARLY_DECISION={early_decision}")
print(f"FINAL_CLASS={final_class}")
print(f"UI_OUTPUT={ui_output}")

# Section 15: Check Sentence Processor transition
print("\n=== STEP 15: SENTENCE PROCESSOR CHECK ===")
old_class_before = final_class
new_seq_id = int(time.time() * 1000) + 10000

# Send a frame with new sequence ID
token_new = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
payload_new = {
    "token": token_new,
    "frame_id": 1,
    "sequence_id": new_seq_id,
    "timestamp_ms": 100.0,
    "has_hand": True,
    "hand_count": 1,
    "selected_hand": "Right"
}
data_bytes = json.dumps(payload_new).encode("utf-8")
req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    res_new = json.loads(resp.read().decode("utf-8"))

print(f"OLD_CLASS_BEFORE_SEQUENCE={old_class_before}")
print(f"NEW_SEQUENCE_ID={new_seq_id}")
print(f"FINAL_CLASS_AFTER_SEQUENCE={res_new.get('final_class')}")
