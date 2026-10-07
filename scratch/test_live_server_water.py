import os, sys, time, json
import numpy as np
import urllib.request

# Load the real live water sequence
seq_path = os.path.join("models", "debug_failed_please_as_water.npy")
seq = np.load(seq_path) # (22, 12)

url = "http://127.0.0.1:8000/api/process_token"
seq_id = int(time.time() * 1000)

print(f"--- Simulating real live water sequence ({len(seq)} frames) through /api/process_token ---")
responses = []
for fid, token in enumerate(seq, 1):
    rh_active = np.any(token[6:12] != 0)
    lh_active = np.any(token[:6] != 0)
    hc = [float(token[6]), float(token[7])] if rh_active else [float(token[0]), float(token[1])]
    
    payload = {
        "token": token.tolist(),
        "frame_id": fid,
        "sequence_id": seq_id,
        "timestamp_ms": fid * 50.0,
        "has_hand": True,
        "hand_count": 1,
        "raw_hands": 1,
        "left_detected": bool(lh_active),
        "right_detected": bool(rh_active),
        "selected_hand": "Right" if rh_active else "Left",
        "hand_center": hc,
        "shoulder_center": [0.5, 0.35]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        responses.append(res_data)
        
    early = res_data.get("early_decision", {})
    print(f"Frame {fid:2d}/{len(seq)}: primary={res_data.get('primary_class', '--'):12s} conf={res_data.get('primary_confidence', 0.0):.4f} final={res_data.get('final_class', '--'):12s} state={early.get('state', '--'):10s} accepted={early.get('accepted', False)}")

last_resp = responses[-1]
print("\nFinal response:")
print(f"  final_class: {last_resp.get('final_class')}")
print(f"  translation: {last_resp.get('translation')}")
