"""
Phase 2 Helper: Interactive Real ISL Webcam Video Recorder.
Records real human ISL sign language video clips from webcam directly into
the structured dataset directory: dataset/raw/<sign_class>/<signer_id>/video_<timestamp>.mp4
"""

import os
import sys
import time
from datetime import datetime
import cv2
import math
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, ISL_VOCABULARY, FRAME_WIDTH, FRAME_HEIGHT


def record_sign_video(sign_class="hello", signer_id="signer_01", record_seconds=3, camera_index=0):
    """
    Opens webcam, displays 3-second countdown overlay, records a real human ISL gesture video clip,
    and saves it to dataset/raw/<sign_class>/<signer_id>/video_<timestamp>.mp4.
    """
    if sign_class.lower() not in [v.lower() for v in ISL_VOCABULARY]:
        print(f"[Warning] '{sign_class}' is not in standard 50 ISL vocabulary list.")

    output_dir = os.path.join(RAW_DATA_DIR, sign_class.lower(), signer_id)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Error] Cannot open webcam (Camera index {camera_index}).")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # 1. Countdown Phase (3 seconds)
    print(f"\n[Record] Preparing to record '{sign_class}' for {signer_id}. Get ready in 3 seconds...")
    start_countdown = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read frame from webcam.")
            cap.release()
            return None

        frame = cv2.flip(frame, 1) # Mirror preview
        elapsed = time.time() - start_countdown
        remaining = 3.0 - elapsed

        if remaining <= 0:
            break

        # Render Countdown Text Overlay
        h, w, _ = frame.shape
        count_str = str(int(np.ceil(remaining))) if remaining > 0 else "GO!"
        cv2.putText(frame, f"SIGN: {sign_class.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"GET READY: {count_str}", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        cv2.imshow("Real ISL Video Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Record] Recording cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    # 2. Recording Phase
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"video_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    writer = cv2.VideoWriter(video_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"[Record] RECORDING NOW... (Perform gesture '{sign_class}')")
    start_rec = time.time()
    frames_recorded = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mirrored = cv2.flip(frame, 1)
        writer.write(frame_mirrored)
        frames_recorded += 1

        rec_elapsed = time.time() - start_rec
        if rec_elapsed >= record_seconds:
            break

        # Render Recording Indicator (Red dot)
        cv2.circle(frame_mirrored, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(frame_mirrored, f"RECORDING '{sign_class.upper()}' ({rec_elapsed:.1f}s)", (60, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Real ISL Video Recorder", frame_mirrored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"[Record] Saved real ISL video: {video_path} ({frames_recorded} frames)")
    return video_path


def record_batch_videos(sign_class="hello", signer_id="signer_01", num_reps=30, record_seconds=3, camera_index=0):
    """
    Opens webcam once and records a batch of N real human ISL gesture clips with 3-second countdowns.
    Saves clips to: dataset/raw/<sign_class>/<signer_id>/video_<timestamp>.mp4
    """
    sign_class = sign_class.lower().strip()
    signer_id = signer_id.lower().strip()

    if sign_class not in [v.lower() for v in ISL_VOCABULARY]:
        print(f"[Warning] '{sign_class}' is not in standard 50 ISL vocabulary list.")

    output_dir = os.path.join(RAW_DATA_DIR, sign_class, signer_id)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Error] Cannot open webcam (Camera index {camera_index}).")
        return 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    saved_count = 0

    print("=" * 60)
    print("      REAL ISL DATASET BATCH RECORDER")
    print("=" * 60)
    print(f"  Sign Class:   {sign_class.upper()}")
    print(f"  Signer ID:    {signer_id}")
    print(f"  Target Clips: {num_reps}")
    print(f"  Output Dir:   {output_dir}")
    print("=" * 60)

    for rep in range(1, num_reps + 1):
        print(f"\n[Record] Preparing clip {rep}/{num_reps} for '{sign_class.upper()}' ({signer_id})...")

        # 1. Countdown Phase (3 seconds)
        start_countdown = time.time()
        cancelled = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read frame from webcam.")
                cap.release()
                cv2.destroyAllWindows()
                return saved_count

            frame_mirrored = cv2.flip(frame, 1)
            elapsed = time.time() - start_countdown
            remaining = 3.0 - elapsed

            if remaining <= 0:
                break

            h, w, _ = frame_mirrored.shape
            count_str = str(int(np.ceil(remaining)))
            cv2.putText(frame_mirrored, f"SIGN: {sign_class.upper()} ({rep}/{num_reps})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame_mirrored, f"GET READY: {count_str}", (w // 2 - 140, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 165, 255), 4)
            cv2.putText(frame_mirrored, "Press 'q' to quit recording batch", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Real ISL Video Recorder", frame_mirrored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Record] Recording session cancelled by user.")
                cancelled = True
                break

        if cancelled:
            break

        # 2. Recording Phase (record_seconds)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"video_{timestamp}.mp4"
        video_path = os.path.join(output_dir, video_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        writer = cv2.VideoWriter(video_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

        start_rec = time.time()
        frames_recorded = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_mirrored = cv2.flip(frame, 1)
            writer.write(frame_mirrored)
            frames_recorded += 1

            rec_elapsed = time.time() - start_rec
            if rec_elapsed >= record_seconds:
                break

            h, w, _ = frame_mirrored.shape
            # Red recording indicator dot
            cv2.circle(frame_mirrored, (30, 35), 12, (0, 0, 255), -1)
            cv2.putText(frame_mirrored, f"RECORDING '{sign_class.upper()}' ({rep}/{num_reps}) - {rec_elapsed:.1f}s", (55, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Real ISL Video Recorder", frame_mirrored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cancelled = True
                break

        writer.release()

        if cancelled:
            print("[Record] Recording cancelled during clip writing.")
            if os.path.exists(video_path):
                os.remove(video_path)
            break

        saved_count += 1
        print(f"[Record] Saved [{saved_count}/{num_reps}]: {video_filename} ({frames_recorded} frames)")

        # Short pause between repetitions (1.0 sec)
        pause_start = time.time()
        while time.time() - pause_start < 1.0:
            ret, frame = cap.read()
            if ret:
                frame_mirrored = cv2.flip(frame, 1)
                cv2.putText(frame_mirrored, "REST / NEXT CLIP IN 1s...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Real ISL Video Recorder", frame_mirrored)
                cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"   RECORDING COMPLETE: {saved_count} / {num_reps} clips saved for '{sign_class.upper()}' ({signer_id})")
    print("=" * 60)
    return saved_count


def interactive_recorder_menu():
    """Interactive command-line menu for recording real ISL sign samples."""
    print("=" * 60)
    print("      REAL ISL WEBCAM VIDEO RECORDER TOOL")
    print("=" * 60)
    print(f"Target Directory: {RAW_DATA_DIR}")
    print(f"Vocabulary Size:  {len(ISL_VOCABULARY)} signs")
    print("-" * 60)

    signer_id = input("Enter Signer ID (default: signer_01): ").strip() or "signer_01"
    
    print("\nSample Vocabulary:")
    for i, word in enumerate(ISL_VOCABULARY[:10]):
        print(f"  {i+1}. {word}")
    print("  ... (type any of the 50 ISL words)")

    while True:
        sign_class = input("\nEnter ISL sign word to record (or 'exit' to quit): ").strip().lower()
        if sign_class == "exit" or not sign_class:
            print("Exiting recorder tool.")
            break

        reps_input = input(f"How many video clips to record for '{sign_class}'? (default 30): ").strip()
        num_reps = int(reps_input) if reps_input.isdigit() else 30

        record_batch_videos(sign_class=sign_class, signer_id=signer_id, num_reps=num_reps, record_seconds=3)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        sign = sys.argv[1]
        signer = sys.argv[2] if len(sys.argv) >= 3 else "signer_01"
        count = int(sys.argv[3]) if len(sys.argv) >= 4 and sys.argv[3].isdigit() else 30
        record_batch_videos(sign_class=sign, signer_id=signer, num_reps=count, record_seconds=3)
    else:
        interactive_recorder_menu()
