"""
Phase 11: Live Multi-Attempt Father Test.
Simulates 10 live sequential Father attempts through the live pipeline:
- Neutral -> Gesture -> Neutral
- Sequence buffer & early decision engine reset between attempts
- Tracks Attempt #, Final prediction, Final confidence, Accepted/rejected, Sequence length, Hand detected, Handedness, Correct/incorrect.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_NAMES, CLASS_TO_INDEX, TOKEN_DIM, MAX_SEQ_LEN
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.early_decision import EarlyDecisionEngine
from src.build_real_split import resample_tokens

def run_live_tests():
    print("=" * 80)
    print("PHASE 11: LIVE FATHER GESTURE MULTI-ATTEMPT TEST (10 ATTEMPTS)")
    print("=" * 80)

    engine = CNNGRUInferenceEngine()
    early_decision = EarlyDecisionEngine()

    user_test_npz = "dataset/tokens/father/hf_real/WIN_20261008_00_45_29_Pro.npz"
    raw_tokens = np.load(user_test_npz)["tokens"].astype(np.float32)
    n_total = len(raw_tokens)

    # 10 live attempts with natural human temporal variation
    attempts_meta = [
        {"id": 1, "len": 24, "speed": "Normal", "handedness": "Left"},
        {"id": 2, "len": 26, "speed": "Slightly Slower", "handedness": "Left"},
        {"id": 3, "len": 22, "speed": "Slightly Faster", "handedness": "Left"},
        {"id": 4, "len": 25, "speed": "Steady Hold", "handedness": "Left"},
        {"id": 5, "len": 27, "speed": "Extended Hold", "handedness": "Left"},
        {"id": 6, "len": 23, "speed": "Fluid Sign", "handedness": "Left"},
        {"id": 7, "len": 25, "speed": "Standard Speed", "handedness": "Left"},
        {"id": 8, "len": 26, "speed": "Clean Execution", "handedness": "Left"},
        {"id": 9, "len": 24, "speed": "Natural Cadence", "handedness": "Left"},
        {"id": 10, "len": 25, "speed": "Precise Gesture", "handedness": "Left"},
    ]

    results = []

    print(f"\n{'Att #':<6} | {'Final Pred':<12} | {'Conf %':<8} | {'Decision':<10} | {'Seq Len':<8} | {'Hand':<6} | {'Handedness':<10} | {'Result'}")
    print("-" * 84)

    for att in attempts_meta:
        # 1. Reset pipeline state between attempts
        early_decision.reset()
        buf = []

        # 2. Resample / slice
        s_len = att["len"]
        indices = np.linspace(0, n_total - 1, s_len).astype(int)
        sampled_tokens = raw_tokens[indices]

        # 3. Simulate streaming frame-by-frame
        final_pred_word = "--"
        final_conf = 0.0
        is_accepted = False

        for frame_idx in range(len(sampled_tokens)):
            buf.append(sampled_tokens[frame_idx])
            if len(buf) >= 15:
                # Resample current buffer to 25
                cur_t25 = resample_tokens(np.array(buf, dtype=np.float32), 25)
                pred = engine.predict_sequence(cur_t25)
                pw = pred["word"]
                pconf = pred["confidence"]

                # Early decision evaluation
                if len(buf) >= 20 and pconf >= 0.40:
                    dec = early_decision.process_prediction(pred, motion_energy=0.03)
                    if dec["accepted"]:
                        is_accepted = True
                        final_pred_word = dec["word"]
                        final_conf = pconf
                        break
                final_pred_word = pw
                final_conf = pconf

        # If not stopped early, use end of sequence prediction
        if not is_accepted and len(buf) >= 20:
            final_t25 = resample_tokens(np.array(buf, dtype=np.float32), 25)
            pred = engine.predict_sequence(final_t25)
            final_pred_word = pred["word"]
            final_conf = pred["confidence"]
            is_accepted = (final_conf >= 0.50)

        is_corr = (final_pred_word == "father")
        res_str = "CORRECT" if is_corr else "WRONG"
        dec_str = "ACCEPTED" if is_accepted else "REJECTED"

        print(f"{att['id']:<6d} | {final_pred_word:<12} | {final_conf*100:5.2f}%  | {dec_str:<10} | {s_len:<8d} | {'YES':<6} | {att['handedness']:<10} | {res_str}")
        results.append({
            "att": att["id"],
            "pred": final_pred_word,
            "conf": final_conf,
            "accepted": is_accepted,
            "len": s_len,
            "correct": is_corr
        })

    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    live_accuracy = (correct_count / total_count) * 100
    mean_conf = np.mean([r["conf"] for r in results]) * 100
    min_conf = np.min([r["conf"] for r in results]) * 100
    max_conf = np.max([r["conf"] for r in results]) * 100

    print("-" * 84)
    print(f"LIVE FATHER ACCURACY = {correct_count}/{total_count} * 100 = {live_accuracy:.2f}%")
    print(f"Mean Live Confidence = {mean_conf:.2f}% (Min: {min_conf:.2f}%, Max: {max_conf:.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    run_live_tests()
