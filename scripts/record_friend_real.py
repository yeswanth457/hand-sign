"""
Real ISL Friend Recording Script (Standardized Protocol).
Records 20-30 real two-handed Friend video clips with real-time feedback and validation.
Protocol:
- Start neutral (hands at sides/resting)
- Perform real two-handed Friend gesture in front of chest
- Hold briefly
- Return to neutral
Resolution: 640x480 at 30 FPS, ~3.5s per clip.
Automatically processes recorded video through MediaPipe into landmarks, 12D tokens, and dataset.csv.
"""

import os
import sys
import time
import uuid
from datetime import datetime
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RAW_DATA_DIR, FRAME_WIDTH, FRAME_HEIGHT
from src.landmark_extractor import LandmarkExtractor
from src.real_dataset_pipeline import RealISLDatasetPipeline


def record_friend_dataset(target_count=20, signer_id="signer_user", record_seconds=3.5):
    output_dir = os.path.join(RAW_DATA_DIR, "friend", signer_id)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam (camera device 0).")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    extractor = LandmarkExtractor(min_detection_confidence=0.25)
    pipeline = RealISLDatasetPipeline()
    session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 70)
    print("      REAL FRIEND GESTURE RECORDER (ISL PROTOCOL)")
    print("=" * 70)
    print(f"Target count: {target_count} clips")
    print(f"Signer ID:    {signer_id}")
    print(f"Save dir:     {output_dir}")
    print("\nGESTURE INSTRUCTION:")
    print("  1. Start with hands down in neutral position.")
    print("  2. Bring hands up and perform your genuine FRIEND gesture.")
    print("  3. Hold briefly in front of chest.")
    print("  4. Return hands down to neutral.")
    print("  * IMPORTANT: Keep hands clearly visible in camera frame!")
    print("\nPress 'q' at any time in the video window to finish / quit.")
    print("=" * 70)

    saved_clips = []

    try:
        while len(saved_clips) < target_count:
            clip_num = len(saved_clips) + 1
            print(f"\n>>> [Clip {clip_num}/{target_count}] Get ready...")

            # 1. Countdown Phase (3.0 seconds)
            start_countdown = time.time()
            cancelled = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Error] Failed to read frame from webcam.")
                    cancelled = True
                    break

                raw_clean = cv2.flip(frame, 1)
                display_frame = raw_clean.copy()

                elapsed = time.time() - start_countdown
                remaining = 3.0 - elapsed

                if remaining <= 0:
                    break

                h, w, _ = display_frame.shape
                count_str = str(int(np.ceil(remaining)))

                # Live hand check during countdown for visual feedback
                ld_check = extractor.process_frame(raw_clean)
                h_cnt = len(ld_check.get("hands", []))
                h_text = f"Live Hands: {h_cnt}"
                h_col = (0, 255, 0) if h_cnt >= 2 else ((0, 255, 255) if h_cnt == 1 else (150, 150, 150))

                # Overlay guidance
                cv2.rectangle(display_frame, (0, 0), (w, 100), (25, 25, 25), -1)
                cv2.putText(display_frame, f"SIGN: FRIEND ({clip_num}/{target_count}) - REAL WEBCAM GESTURE",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
                cv2.putText(display_frame, "Neutral -> Perform Friend Gesture -> Hold -> Neutral",
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
                cv2.putText(display_frame, h_text, (w - 180, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, h_col, 2)

                # Center countdown
                cv2.putText(display_frame, f"GET READY: {count_str}", (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 140, 255), 4)
                cv2.putText(display_frame, "Press 'q' to stop recording session", (15, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                cv2.imshow("FRIEND ISL Recorder", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cancelled = True
                    break

            if cancelled:
                print("\n[User] Recording session stopped by user.")
                break

            # 2. Recording Phase (record_seconds)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_suffix = uuid.uuid4().hex[:6]
            video_filename = f"video_friend_{signer_id}_{session_id}_{timestamp}_{unique_suffix}.mp4"
            video_path = os.path.join(output_dir, video_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

            start_rec = time.time()
            recorded_clean_frames = []

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    raw_clean = cv2.flip(frame, 1)
                    writer.write(raw_clean)
                    recorded_clean_frames.append(raw_clean)

                    display_frame = raw_clean.copy()
                    rec_elapsed = time.time() - start_rec
                    if rec_elapsed >= record_seconds:
                        break

                    h, w, _ = display_frame.shape
                    # Red recording bar
                    cv2.rectangle(display_frame, (0, 0), (w, 80), (0, 0, 180), -1)
                    cv2.circle(display_frame, (35, 40), 12, (255, 255, 255), -1)
                    cv2.putText(display_frame, f"RECORDING FRIEND ({rec_elapsed:.1f}s / {record_seconds}s)",
                                (60, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)
                    cv2.putText(display_frame, "PERFORM FRIEND GESTURE AND HOLD",
                                (60, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 200), 1)

                    cv2.imshow("FRIEND ISL Recorder", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cancelled = True
                        break
            finally:
                writer.release()

            if cancelled:
                if os.path.exists(video_path):
                    os.remove(video_path)
                print("\n[User] Recording cancelled.")
                break

            # 3. Quality Validation Phase
            print(f"  Validating clip ({len(recorded_clean_frames)} frames)...", end="", flush=True)
            valid_hand_frames = 0
            for frm in recorded_clean_frames:
                ld = extractor.process_frame(frm)
                if ld.get("hands") and len(ld["hands"]) > 0:
                    valid_hand_frames += 1

            if valid_hand_frames < 8:
                print(f" REJECTED! (Only {valid_hand_frames}/{len(recorded_clean_frames)} frames had hands). Retrying clip.")
                if os.path.exists(video_path):
                    os.remove(video_path)
                time.sleep(1.0)
                continue

            print(f" ACCEPTED! ({valid_hand_frames}/{len(recorded_clean_frames)} hand frames)")

            # 4. Process through Pipeline into 12D Tokens & dataset.csv
            entry = {
                "video_path": video_path,
                "sign_class": "friend",
                "signer_id": signer_id,
                "video_id": os.path.splitext(video_filename)[0]
            }
            res = pipeline.process_video_file(entry)
            print(f"  -> Extracted {res.get('token_sequence_length', 0)} tokens. Saved to tokens/friend/{signer_id}/")

            saved_clips.append(video_path)
            time.sleep(0.5)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    print("\n" + "=" * 70)
    print(f"RECORDING SUMMARY: Successfully recorded & processed {len(saved_clips)} real Friend clips!")
    print("=" * 70)
    return saved_clips


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    signer = sys.argv[2] if len(sys.argv) > 2 else "signer_user"
    record_friend_dataset(target_count=count, signer_id=signer)
