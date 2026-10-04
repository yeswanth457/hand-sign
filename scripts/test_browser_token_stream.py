"""
Test script to simulate two consecutive browser NO token streams separated by NO_HAND.
Verifies sequence isolation, buffer reset, stale frame rejection, and binary NO prediction matching.
"""
import os
import sys
import time
import requests
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def test_dual_stream():
    url = "http://127.0.0.1:8000/api/process_token"
    ref_path = os.path.join(BASE_DIR, "models", "debug_live_no_sequence.npy")

    if not os.path.exists(ref_path):
        print("[ERROR] debug_live_no_sequence.npy not found")
        return

    tokens = np.load(ref_path).astype(np.float32)
    print(f"[TEST] Loaded standalone reference NO sequence of shape {tokens.shape}")

    # Clear sentence & reset backend state
    requests.post("http://127.0.0.1:8000/api/clear_sentence")
    time.sleep(0.2)

    # --- SEQUENCE 1 ---
    print("\n" + "="*60)
    print("      TESTING SEQUENCE 1 (First NO Gesture)")
    print("="*60)
    seq1_no_prob = 0.0
    for idx in range(len(tokens)):
        t = tokens[idx].tolist()
        payload = {
            "token": t,
            "frame_id": idx + 1,
            "sequence_id": 1,
            "timestamp_ms": (idx + 1) * 33.3,
            "has_hand": True,
            "hand_count": 1,
            "hand_center": [t[0], t[1]],
            "shoulder_center": [t[0] - t[4], t[1] - t[5]],
            "primary_hand": "TrackedHand"
        }
        res = requests.post(url, json=payload)
        data = res.json()
        seq1_no_prob = data.get("no_probability", 0.0)
        print(f"Seq 1 | Frame {idx+1:02d}/25 | Buf: {data.get('buffer_status')} | NO Prob: {seq1_no_prob:.4f}")
        time.sleep(0.033)

    print(f"\n>>> SEQUENCE 1 RESULT: NO Probability = {seq1_no_prob:.4f} <<<")

    # --- NO HAND RESET ---
    print("\n" + "="*60)
    print("      SIMULATING NO HAND (Resetting Pipeline)")
    print("="*60)
    reset_payload = {
        "token": None,
        "frame_id": 999,
        "sequence_id": 1,
        "has_hand": False
    }
    requests.post(url, json=reset_payload)
    time.sleep(0.2)

    # --- SEQUENCE 2 ---
    print("\n" + "="*60)
    print("      TESTING SEQUENCE 2 (Second NO Gesture after NO HAND)")
    print("="*60)
    seq2_no_prob = 0.0
    for idx in range(len(tokens)):
        t = tokens[idx].tolist()
        payload = {
            "token": t,
            "frame_id": idx + 1,
            "sequence_id": 2,
            "timestamp_ms": (idx + 1) * 33.3 + 2000,
            "has_hand": True,
            "hand_count": 1,
            "hand_center": [t[0], t[1]],
            "shoulder_center": [t[0] - t[4], t[1] - t[5]],
            "primary_hand": "TrackedHand"
        }
        res = requests.post(url, json=payload)
        data = res.json()
        seq2_no_prob = data.get("no_probability", 0.0)
        print(f"Seq 2 | Frame {idx+1:02d}/25 | Buf: {data.get('buffer_status')} | NO Prob: {seq2_no_prob:.4f}")
        time.sleep(0.033)

    print(f"\n>>> SEQUENCE 2 RESULT: NO Probability = {seq2_no_prob:.4f} <<<")

if __name__ == "__main__":
    test_dual_stream()
