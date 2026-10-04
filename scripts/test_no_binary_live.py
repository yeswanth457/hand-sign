"""
Live Webcam Test Script for Binary NO Classifier.

Captures real-time webcam frames, extracts MediaPipe landmarks on RAW un-flipped frames
(matching the offline dataset coordinate convention), calculates 6D tokens [Hx, Hy, Mx, My, Rx, Ry],
maintains a 25-token rolling buffer, evaluates NOBinaryInferenceEngine, and displays NO on-screen
when confidence and temporal conditions are met.

Flips frame horizontally ONLY AFTER landmark extraction for display purposes.
Resets tokenizer state when no hand is detected to prevent motion vector contamination.

Press 'q' or 'ESC' in the video window to exit.
"""

import os
import sys
import time
from collections import deque
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.landmark_extractor import LandmarkExtractor
from src.gesture_tokenizer import GestureTokenizer
from src.no_binary_model import NOBinaryInferenceEngine, BINARY_MODEL_PATH


def run_live_binary_test():
    print("=" * 75)
    print("      LIVE WEBCAM TEST — TEMPORARY BINARY NO CLASSIFIER")
    print("=" * 75)

    if not os.path.exists(BINARY_MODEL_PATH):
        print(f"[ERROR] Binary model file missing at {BINARY_MODEL_PATH}")
        return

    print("[NO MODEL] Initializing binary inference engine...")
    engine = NOBinaryInferenceEngine(threshold=0.60)
    if not engine.model_loaded:
        print("[ERROR] Binary model could not be loaded!")
        return
    print(f"[NO MODEL] Binary model loaded successfully (threshold={engine.threshold})")

    print("[MEDIAPIPE] Initializing landmark extractor and tokenizer...")
    extractor = LandmarkExtractor()
    tokenizer = GestureTokenizer()

    print("[WEBCAM] Opening camera index 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WEBCAM] ERROR: Could not access webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[WEBCAM] Camera opened successfully (640x480)")

    token_buffer = deque(maxlen=25)
    consecutive_no_count = 0
    REQUIRED_CONSECUTIVE = 3

    start_time = time.time()
    last_log_time = 0.0
    frame_counter = 0

    print("\nStarting live webcam stream. Perform 'NO' gesture or other gestures...")
    print("Press 'q' or ESC in the video window to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("[WEBCAM] Warning: Empty or unreadable frame received.")
                time.sleep(0.03)
                continue

            frame_counter += 1
            # Strictly monotonic elapsed timestamp in milliseconds
            elapsed_ms = int((time.time() - start_time) * 1000)

            # 1. PROCESS RAW UN-FLIPPED FRAME TO PRESERVE TRAINING COORDINATES
            try:
                lm_data = extractor.process_frame(frame, timestamp_ms=elapsed_ms)
            except Exception as e:
                print(f"[MEDIAPIPE] Exception during landmark extraction: {e}")
                lm_data = {"hands": [], "hand_center": (0.5, 0.5), "shoulder_center": (0.5, 0.35)}

            has_hand = bool(lm_data.get("hands") and len(lm_data["hands"]) > 0)

            no_prob = 0.0
            is_no = False
            status_text = "Waiting for hand gesture..."
            pred_label = "NOT_NO"

            if has_hand:
                # 2. Tokenize frame if hand present
                token = tokenizer.tokenize_frame(lm_data)

                # Defensive finite token check
                if token is not None and np.isfinite(token).all() and len(token) == 6:
                    token_buffer.append(token)

                # 3. Predict sequence if buffer has at least 5 frames
                if len(token_buffer) >= 5:
                    seq_matrix = list(token_buffer)
                    pred_res = engine.predict_sequence(seq_matrix, max_seq_len=25)
                    no_prob = pred_res.get("no_probability", 0.0)

                    if no_prob >= engine.threshold:
                        consecutive_no_count += 1
                    else:
                        consecutive_no_count = 0

                    if consecutive_no_count >= REQUIRED_CONSECUTIVE:
                        is_no = True
                        pred_label = "NO"
                        status_text = "NO"
                    else:
                        pred_label = "NOT_NO"
            else:
                # Reset tokenizer state when hand is absent to prevent motion vector contamination
                tokenizer.reset()
                consecutive_no_count = 0
                if len(token_buffer) > 0 and frame_counter % 5 == 0:
                    token_buffer.popleft()

            # 4. Diagnostic & Throttled Console Logging (once per second)
            now_sec = time.time()
            if now_sec - last_log_time >= 1.0:
                last_log_time = now_sec
                if has_hand:
                    if len(token_buffer) == 25:
                        seq_arr = np.array(list(token_buffer))
                        m = np.mean(seq_arr, axis=0)
                        print(f"[LIVE TOKEN STATS] Hx={m[0]:.4f} (Offline ~0.337) | Hy={m[1]:.4f} | Mx={m[2]:.4f} | My={m[3]:.4f} | Rx={m[4]:.4f} (Offline ~-0.204) | Ry={m[5]:.4f}", flush=True)
                    print(f"[MEDIAPIPE] Frame processed | [BUFFER] Tokens: {len(token_buffer)}/25 | [NO MODEL] Probability: {no_prob:.4f} | Prediction: {pred_label}", flush=True)
                else:
                    print(f"[MEDIAPIPE] No hand detected | [BUFFER] Tokens: {len(token_buffer)}/25", flush=True)

            # 5. RENDER DISPLAY FRAME (FLIP ONLY FOR DISPLAY / SELFIE MODE AFTER MEDIAPIPE)
            display_frame = cv2.flip(frame, 1)
            h, w, c = display_frame.shape

            # Top Banner
            banner_color = (0, 180, 0) if is_no else (30, 30, 30)
            cv2.rectangle(display_frame, (0, 0), (w, 80), banner_color, -1)

            if is_no:
                cv2.putText(display_frame, "NO", (w // 2 - 40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
            else:
                cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

            # Buffer count overlay
            cv2.putText(display_frame, f"Buffer: {len(token_buffer)}/25", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Probability Bar Overlay
            bar_w = int(no_prob * 200)
            cv2.rectangle(display_frame, (w - 230, 25), (w - 30, 45), (70, 70, 70), -1)
            cv2.rectangle(display_frame, (w - 230, 25), (w - 230 + bar_w, 45), (0, 255, 0) if is_no else (0, 165, 255), -1)
            cv2.rectangle(display_frame, (w - 230, 25), (w - 30, 45), (255, 255, 255), 1)
            cv2.putText(display_frame, f"Prob: {no_prob:.2f}", (w - 230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Binary NO Gesture Live Detector", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # ESC or q
                print("\n[USER] Stop command received ('q' or ESC pressed). Exiting...")
                break
    except KeyboardInterrupt:
        print("\n[USER] KeyboardInterrupt received. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("[WEBCAM] Camera released. Live webcam test finished.")


if __name__ == "__main__":
    run_live_binary_test()
