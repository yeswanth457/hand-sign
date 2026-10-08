"""
Measure Real Friend Hand-Gesture Accuracy (20 Independent Live Attempts).
Strictly validates user's physical Friend gesture against the current frozen 22-class model.
Protocol:
- Expected: FRIEND (Class 18)
- 20 separate attempts (neutral -> sign -> hold -> neutral)
- Full diagnostic trace: [FRIEND ACCURACY TRACE]
- Computes real accuracy (Correct / 20 * 100) vs model confidence
- Reports confusions, temporal stability, hand detection statistics
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, HAND_FEATURE_DIM, MAX_SEQ_LEN, MODEL_DIR, MODEL_PATH,
    FRAME_WIDTH, FRAME_HEIGHT, CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES,
    COOLDOWN_FRAMES, SIGNING_MOTION_THRESHOLD
)
from src.landmark_extractor import LandmarkExtractor
from src.gesture_tokenizer import GestureTokenizer
from src.frame_selector import MotionFrameSelector
from src.early_decision import EarlyDecisionEngine
from src.cnn_gru_model import ISL_CNN_GRU_Model, resample_tokens

def run_accuracy_measurement(total_attempts=20):
    print("=" * 75)
    print("      REAL FRIEND HAND-GESTURE ACCURACY MEASUREMENT PROTOCOL")
    print("=" * 75)
    print(f"Target:             FRIEND (Class Index 18)")
    print(f"Total Attempts:     {total_attempts}")
    print(f"Active Checkpoint:  {MODEL_PATH}")
    print(f"NUM_CLASSES:        {NUM_CLASSES}")
    print(f"TOKEN_DIM:          {TOKEN_DIM}")
    print("\nPROTOCOL RULES:")
    print("  1. Start with hands at sides/resting (neutral position).")
    print("  2. Perform your complete, natural Friend gesture.")
    print("  3. Hold gesture briefly until model predicts.")
    print("  4. Return hands down to neutral position.")
    print("  5. The system will record the result and advance to the next attempt.")
    print("  * Press 'q' at any time to exit early.")
    print("=" * 75)

    # 1. Initialize Components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    extractor = LandmarkExtractor(min_detection_confidence=0.25)
    tokenizer = GestureTokenizer()
    frame_selector = MotionFrameSelector()
    early_decision = EarlyDecisionEngine()

    # Load Normalization Stats
    mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
    std_path = os.path.join(MODEL_DIR, "feature_std.npy")
    feat_mean = np.load(mean_path).astype(np.float32)
    feat_std = np.load(std_path).astype(np.float32)
    feat_std = np.where(feat_std < 1e-7, 1.0, feat_std)

    # Load 22-class Model
    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Save directory for attempt tokens
    out_dir = os.path.join("dataset", "validation_attempts", "friend")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (camera device 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    results = []
    attempt_num = 1

    try:
        while attempt_num <= total_attempts:
            print(f"\n==================================================")
            print(f">>> [ATTEMPT {attempt_num}/{total_attempts}] PREPARE NEUTRAL POSITION")
            print(f"==================================================")

            # ── 1. Countdown Phase (3.0s) ──
            start_countdown = time.time()
            cancelled = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    cancelled = True
                    break
                frame_clean = cv2.flip(frame, 1)
                display = frame_clean.copy()

                elapsed = time.time() - start_countdown
                remaining = 3.0 - elapsed
                if remaining <= 0:
                    break

                h, w, _ = display.shape
                cv2.rectangle(display, (0, 0), (w, 80), (20, 20, 20), -1)
                cv2.putText(display, f"ATTEMPT {attempt_num}/{total_attempts} - EXPECTED: FRIEND",
                            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.putText(display, f"GET READY IN NEUTRAL POSITION: {int(np.ceil(remaining))}s",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

                cv2.imshow("Real Friend Accuracy Measurement", display)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    cancelled = True
                    break

            if cancelled:
                print("\n[User] Measurement session stopped by user.")
                break

            # ── 2. Capture Attempt Phase ──
            token_buffer = []
            frame_records = []
            attempt_start = time.time()
            frame_id = 0
            tokenizer.reset()
            early_decision.reset()

            attempt_final_pred = None
            attempt_final_conf = 0.0
            attempt_final_accepted = False
            attempt_top2_class = "--"
            attempt_top2_conf = 0.0
            hand_detected_frames = 0
            two_hand_frames = 0
            left_detected_frames = 0
            right_detected_frames = 0
            total_attempt_frames = 0

            max_attempt_duration = 5.0 # Max seconds per attempt
            signing_started = False
            consecutive_neutral = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_clean = cv2.flip(frame, 1)
                display = frame_clean.copy()
                frame_id += 1
                total_attempt_frames += 1

                # MediaPipe Landmark Extraction
                landmark_data = extractor.process_frame(frame_clean)
                hands = landmark_data.get("hands", [])
                raw_hands_val = len(hands)

                l_det = landmark_data.get("left_hand_center") is not None
                r_det = landmark_data.get("right_hand_center") is not None
                if l_det: left_detected_frames += 1
                if r_det: right_detected_frames += 1
                if raw_hands_val >= 2: two_hand_frames += 1
                if raw_hands_val > 0: hand_detected_frames += 1

                act_hands = "Left+Right" if (l_det and r_det) else ("Left" if l_det else ("Right" if r_det else "NONE"))

                # 12D Token Calculation
                token_12d = tokenizer.tokenize_frame(landmark_data)
                has_hand = (raw_hands_val > 0)

                # Motion energy
                sel_res = frame_selector.process_frame(frame_id, landmark_data)
                motion_energy = sel_res["motion_energy"]

                if has_hand:
                    signing_started = True
                    token_buffer.append(token_12d)
                    consecutive_neutral = 0
                else:
                    if signing_started:
                        consecutive_neutral += 1

                buf_len = len(token_buffer)

                # Run CNN-GRU inference once minimum frames collected
                primary_class = "Analyzing gesture..."
                primary_confidence = 0.0
                sorted_probs = []

                if buf_len >= 15:
                    tokens_np = np.array(token_buffer[-25:], dtype=np.float32)
                    tokens_25 = resample_tokens(tokens_np, 25)
                    norm_t = (tokens_25 - feat_mean) / feat_std
                    tensor_in = torch.tensor(norm_t, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(tensor_in)
                        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                    top1_id = int(np.argmax(probs))
                    primary_class = INDEX_TO_CLASS[top1_id]
                    primary_confidence = float(probs[top1_id])

                    sorted_indices = np.argsort(probs)[::-1]
                    sorted_probs = [(INDEX_TO_CLASS[idx], float(probs[idx])) for idx in sorted_indices[:5]]
                    attempt_top2_class = sorted_probs[1][0] if len(sorted_probs) > 1 else '--'
                    attempt_top2_conf = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

                    attempt_final_pred = primary_class
                    attempt_final_conf = primary_confidence

                    # Early Decision
                    pred_dict = {"word": primary_class, "confidence": primary_confidence, "class_id": top1_id}
                    decision_res = early_decision.process_prediction(pred_dict, motion_energy)
                    if decision_res["accepted"]:
                        attempt_final_accepted = True

                    gesture_state_str = decision_res.get("state", "SIGNING")

                    # Print [FRIEND ACCURACY TRACE] on every inference frame
                    print(
                        f"\n[FRIEND ACCURACY TRACE]\n"
                        f"attempt_id={attempt_num}\n"
                        f"frame_id={frame_id}\n"
                        f"raw_hand_count={raw_hands_val}\n"
                        f"left_detected={str(l_det).lower()}\n"
                        f"right_detected={str(r_det).lower()}\n"
                        f"handedness={act_hands}\n"
                        f"token_count={buf_len}\n"
                        f"sequence_length={buf_len}\n"
                        f"motion_energy={motion_energy:.4f}\n"
                        f"top1_class={primary_class}\n"
                        f"top1_index={top1_id}\n"
                        f"top1_confidence={primary_confidence:.4f}\n"
                        f"top2_class={attempt_top2_class}\n"
                        f"top2_confidence={attempt_top2_conf:.4f}\n"
                        f"decision_state={gesture_state_str}\n"
                        f"accepted_class={decision_res.get('word') or '--'}\n"
                        f"accepted_confidence={decision_res.get('confidence', 0.0):.4f}\n"
                        f"accepted_as_current={decision_res.get('accepted', False)}"
                    )
                else:
                    gesture_state_str = "COLLECTING"

                # UI Overlay
                h, w, _ = display.shape
                cv2.rectangle(display, (0, 0), (w, 100), (20, 20, 20), -1)
                cv2.putText(display, f"ATTEMPT {attempt_num}/{total_attempts} | TARGET: FRIEND",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

                col = (0, 255, 0) if primary_class == "friend" else (0, 140, 255)
                cv2.putText(display, f"MODEL PREDICTION: {primary_class.upper()} ({primary_confidence*100:.1f}%)",
                            (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)
                cv2.putText(display, f"BUFFER: {buf_len}/25 | HANDS: {act_hands} | STATE: {gesture_state_str}",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

                cv2.imshow("Real Friend Accuracy Measurement", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cancelled = True
                    break
                elif key == 32: # Spacebar to manually finish attempt
                    break

                # Completion conditions:
                # 1. Sign was performed (>= 18 tokens) and hands dropped back to neutral for 10 frames
                # 2. Or maximum time elapsed
                if signing_started and buf_len >= 18 and consecutive_neutral >= 10:
                    break
                if (time.time() - attempt_start) >= max_attempt_duration:
                    break

            if cancelled:
                break

            # ── 3. Record Attempt Result ──
            pred_word = attempt_final_pred if attempt_final_pred else "--"
            pred_conf = attempt_final_conf
            is_correct = (pred_word == "friend")

            # Save token recording
            if len(token_buffer) > 0:
                np_tok = np.array(token_buffer, dtype=np.float32)
                rec_path = os.path.join(out_dir, f"attempt_{attempt_num:02d}_{pred_word}.npz")
                np.savez_compressed(rec_path, tokens=np_tok, target="friend", prediction=pred_word, confidence=pred_conf)

            attempt_result = {
                "attempt": attempt_num,
                "expected": "FRIEND",
                "predicted": pred_word,
                "confidence": round(pred_conf * 100, 2),
                "correct": is_correct,
                "accepted": attempt_final_accepted,
                "top2_class": attempt_top2_class,
                "top2_conf": round(attempt_top2_conf * 100, 2),
                "tokens_collected": len(token_buffer),
                "two_hand_pct": round(two_hand_frames / max(1, total_attempt_frames) * 100, 1),
                "hand_detected_pct": round(hand_detected_frames / max(1, total_attempt_frames) * 100, 1)
            }
            results.append(attempt_result)

            print(f"\n>>> ATTEMPT {attempt_num} RESULT: Expected=FRIEND | Predicted={pred_word} | Conf={pred_conf*100:.1f}% | Correct={'YES' if is_correct else 'NO'}")
            attempt_num += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ── 4. Generate & Print Final Comprehensive Report ──
    print("\n" + "=" * 75)
    print("           FRIEND HAND GESTURE ACCURACY REPORT")
    print("=" * 75)
    print(f"{'Attempt':<8} | {'Expected':<10} | {'Predicted':<12} | {'Confidence':<10} | {'Correct':<8}")
    print("-" * 60)
    for r in results:
        corr_str = "YES" if r["correct"] else "NO"
        print(f"{r['attempt']:<8} | {r['expected']:<10} | {r['predicted']:<12} | {r['confidence']:>6.1f}%    | {corr_str:<8}")

    n_total = len(results)
    if n_total == 0:
        print("[WARNING] No attempts completed.")
        return

    n_correct = sum(1 for r in results if r["correct"])
    n_incorrect = n_total - n_correct
    real_accuracy = (n_correct / n_total) * 100

    confs = [r["confidence"] for r in results if r["confidence"] > 0]
    corr_confs = [r["confidence"] for r in results if r["correct"]]
    inc_confs = [r["confidence"] for r in results if not r["correct"]]

    from collections import Counter
    wrong_preds = [r["predicted"] for r in results if not r["correct"]]
    confusion_counts = Counter(wrong_preds)

    print("\n" + "=" * 60)
    print(f"Total attempts:           {n_total}")
    print(f"Correct:                  {n_correct}")
    print(f"Incorrect:                {n_incorrect}")
    print(f"REAL FRIEND ACCURACY:     {real_accuracy:.1f}%")
    print("-" * 60)
    print("MODEL CONFIDENCE STATISTICS:")
    print(f"  Mean Confidence:        {np.mean(confs):.1f}%" if confs else "  Mean: 0.0%")
    print(f"  Median Confidence:      {np.median(confs):.1f}%" if confs else "  Median: 0.0%")
    print(f"  Minimum Confidence:     {min(confs):.1f}%" if confs else "  Min: 0.0%")
    print(f"  Maximum Confidence:     {max(confs):.1f}%" if confs else "  Max: 0.0%")
    print(f"  Correct Attempts Conf:  {np.mean(corr_confs):.1f}%" if corr_confs else "  Correct Conf: N/A")
    print(f"  Incorrect Attempts Conf:{np.mean(inc_confs):.1f}%" if inc_confs else "  Incorrect Conf: N/A")

    print("\nMOST COMMON WRONG PREDICTIONS:")
    if confusion_counts:
        for idx, (cls_name, cnt) in enumerate(confusion_counts.most_common(5), 1):
            print(f"  {idx}. {cls_name} — {cnt} time(s)")
    else:
        print("  None (100% correct)")

    # Save complete JSON
    report_path = os.path.join(MODEL_DIR, "friend_baseline_accuracy_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": "friend",
            "total_attempts": n_total,
            "correct": n_correct,
            "incorrect": n_incorrect,
            "real_accuracy": real_accuracy,
            "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "confusions": dict(confusion_counts),
            "attempts": results
        }, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    n = 20
    if len(sys.argv) > 1:
        try: n = int(sys.argv[1])
        except ValueError: pass
    run_accuracy_measurement(total_attempts=n)
