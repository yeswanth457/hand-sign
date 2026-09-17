"""
RT-STAMP-SLR Diagnostic Test Suite
Tests 1-12: Validates landmark rendering, CNN-GRU inference, and end-to-end pipeline.

Run: python tests/diagnostic_tests.py
"""

import sys
import os
import time
import json
import numpy as np
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ISL_VOCABULARY, ID_TO_WORD, CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES, COOLDOWN_FRAMES

API_BASE = "http://127.0.0.1:8000"


def send_token(token, frame_id=1):
    """Send a single 6D token to /api/process_token and return response."""
    r = requests.post(f"{API_BASE}/api/process_token", json={
        "token": token,
        "frame_id": frame_id,
        "timestamp_ms": time.time() * 1000
    })
    return r.json()


def send_sequence(tokens, label="test"):
    """Send a sequence of tokens and return the last response + timing."""
    # Clear state first
    requests.post(f"{API_BASE}/api/clear_sentence")
    
    results = []
    t_start = time.perf_counter()
    first_prediction_time = None
    threshold_time = None
    confirmed_time = None
    
    for i, tok in enumerate(tokens):
        r = send_token(tok, frame_id=i + 1)
        results.append(r)
        
        if r.get("prediction", {}).get("word", "BUFFERING") != "BUFFERING" and first_prediction_time is None:
            first_prediction_time = time.perf_counter() - t_start
        
        conf = r.get("prediction", {}).get("confidence", 0.0)
        if conf >= CONFIDENCE_THRESHOLD and threshold_time is None:
            threshold_time = time.perf_counter() - t_start
            
        if r.get("early_decision", {}).get("accepted", False) and confirmed_time is None:
            confirmed_time = time.perf_counter() - t_start
    
    t_total = time.perf_counter() - t_start
    last = results[-1]
    
    return {
        "label": label,
        "total_frames": len(tokens),
        "total_time_s": round(t_total, 3),
        "time_to_first_prediction": round(first_prediction_time, 3) if first_prediction_time else None,
        "time_to_threshold": round(threshold_time, 3) if threshold_time else None,
        "time_to_confirmed": round(confirmed_time, 3) if confirmed_time else None,
        "final_prediction": last.get("prediction", {}),
        "early_decision": last.get("early_decision", {}),
        "buffer_status": last.get("buffer_status"),
        "processing_time_ms": last.get("processing_time_ms"),
    }


def generate_static_hand_tokens(n=30):
    """TEST 1: Static open hand — hand stays at center."""
    return [[0.5, 0.5, 0.0, 0.0, 0.0, 0.15]] * n


def generate_slow_pointing_tokens(n=30):
    """TEST 2: Slow finger pointing — hand moves slowly right."""
    tokens = []
    for i in range(n):
        hx = 0.3 + (i / n) * 0.4
        hy = 0.5
        mx = 0.4 / n if i > 0 else 0.0
        my = 0.0
        rx = hx - 0.5
        ry = hy - 0.35
        tokens.append([hx, hy, mx, my, rx, ry])
    return tokens


def generate_fast_pointing_tokens(n=30):
    """TEST 3: Fast finger pointing — hand moves quickly right."""
    tokens = []
    for i in range(n):
        hx = 0.2 + (i / n) * 0.6
        hy = 0.4
        mx = 0.6 / n if i > 0 else 0.0
        my = 0.0
        rx = hx - 0.5
        ry = hy - 0.35
        tokens.append([hx, hy, mx, my, rx, ry])
    return tokens


def generate_left_right_tokens(n=30):
    """TEST 4: Move hand left → right."""
    tokens = []
    for i in range(n):
        hx = 0.2 + (i / n) * 0.6
        hy = 0.5
        mx = 0.6 / n if i > 0 else 0.0
        my = 0.0
        rx = hx - 0.5
        ry = hy - 0.35
        tokens.append([hx, hy, mx, my, rx, ry])
    return tokens


def generate_up_down_tokens(n=30):
    """TEST 5: Move hand up → down."""
    tokens = []
    for i in range(n):
        hx = 0.5
        hy = 0.2 + (i / n) * 0.6
        mx = 0.0
        my = 0.6 / n if i > 0 else 0.0
        rx = hx - 0.5
        ry = hy - 0.35
        tokens.append([hx, hy, mx, my, rx, ry])
    return tokens


def generate_rotation_tokens(n=30):
    """TEST 6: Rotate hand (circular motion)."""
    tokens = []
    for i in range(n):
        angle = (i / n) * 2 * np.pi
        hx = 0.5 + 0.15 * np.cos(angle)
        hy = 0.5 + 0.15 * np.sin(angle)
        mx = -0.15 * np.sin(angle) * (2 * np.pi / n) if i > 0 else 0.0
        my = 0.15 * np.cos(angle) * (2 * np.pi / n) if i > 0 else 0.0
        rx = hx - 0.5
        ry = hy - 0.35
        tokens.append([float(hx), float(hy), float(mx), float(my), float(rx), float(ry)])
    return tokens


def load_real_tokens_for_sign(sign_class):
    """Load real extracted tokens for a sign class if available."""
    tokens_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "tokens", sign_class)
    if not os.path.exists(tokens_dir):
        return None
    
    for root, dirs, files in os.walk(tokens_dir):
        for f in files:
            if f.endswith(".npz"):
                data = np.load(os.path.join(root, f))
                tokens = data["tokens"].astype(np.float32)
                # Resample to 25
                if len(tokens) >= 25:
                    indices = np.linspace(0, len(tokens) - 1, 25, dtype=int)
                    return tokens[indices].tolist()
                else:
                    padded = np.pad(tokens, ((0, 25 - len(tokens)), (0, 0)), mode="edge")
                    return padded.tolist()
    return None


def run_all_tests():
    print("\n" + "=" * 70)
    print("  RT-STAMP-SLR DIAGNOSTIC TEST SUITE")
    print("=" * 70)
    
    # Print configuration
    print(f"\nConfig:")
    print(f"  CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"  SUSTAINED_FRAMES:     {SUSTAINED_FRAMES}")
    print(f"  COOLDOWN_FRAMES:      {COOLDOWN_FRAMES}")
    print(f"  Vocabulary size:      {len(ISL_VOCABULARY)}")
    
    # Verify server is running
    try:
        r = requests.get(f"{API_BASE}/api/vocabulary", timeout=5)
        assert r.status_code == 200
        print(f"  Server:               ONLINE at {API_BASE}")
    except Exception as e:
        print(f"\n  ERROR: Server not reachable at {API_BASE}")
        print(f"  Start with: python -m uvicorn app:app --port 8000")
        return
    
    tests = [
        ("TEST 1: Static open hand", generate_static_hand_tokens(30)),
        ("TEST 2: Slow finger pointing", generate_slow_pointing_tokens(30)),
        ("TEST 3: Fast finger pointing", generate_fast_pointing_tokens(30)),
        ("TEST 4: Move hand left → right", generate_left_right_tokens(30)),
        ("TEST 5: Move hand up → down", generate_up_down_tokens(30)),
        ("TEST 6: Rotate hand", generate_rotation_tokens(30)),
    ]
    
    # Add real token tests for priority signs
    for sign in ["hello", "thank_you", "school", "food", "water", "welcome"]:
        real_tokens = load_real_tokens_for_sign(sign)
        if real_tokens:
            tests.append((f"TEST {len(tests)+1}: Perform {sign} (real tokens)", real_tokens))
        else:
            print(f"\n  [SKIP] No real tokens found for '{sign}'")
    
    results = []
    for name, tokens in tests:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        
        res = send_sequence(tokens, label=name)
        results.append(res)
        
        pred = res["final_prediction"]
        decision = res["early_decision"]
        
        print(f"  Frames sent:          {res['total_frames']}")
        print(f"  Total time:           {res['total_time_s']}s")
        print(f"  Backend proc/frame:   {res['processing_time_ms']}ms")
        print(f"  Buffer status:        {res['buffer_status']}")
        print(f"  Prediction:           {pred.get('word', 'N/A')}")
        print(f"  Confidence:           {pred.get('confidence', 0):.4f} ({pred.get('confidence', 0)*100:.1f}%)")
        print(f"  Decision state:       {decision.get('state', 'N/A')}")
        print(f"  Accepted:             {decision.get('accepted', False)}")
        print(f"  Time to 1st pred:     {res['time_to_first_prediction']}")
        print(f"  Time to threshold:    {res['time_to_threshold']}")
        print(f"  Time to confirmed:    {res['time_to_confirmed']}")
    
    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Test':<45} {'Prediction':<12} {'Conf%':<8} {'Accepted':<10}")
    print(f"{'─' * 75}")
    for r in results:
        pred = r["final_prediction"]
        decision = r["early_decision"]
        print(f"{r['label']:<45} {pred.get('word', 'N/A'):<12} {pred.get('confidence', 0)*100:>5.1f}%  {str(decision.get('accepted', False)):<10}")
    
    print(f"\n{'=' * 70}")
    print("  TEST SUITE COMPLETE")
    print(f"{'=' * 70}\n")
    
    return results


if __name__ == "__main__":
    run_all_tests()
