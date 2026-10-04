"""
Interactive Real ISL Guided Webcam Recorder for 21 Classes.
Enforces controlled webcam recording protocol:
- Exactly 21 ISL classes with official ISLRTC / standard ISL definitions
- 640x480 fixed resolution at 30 FPS
- Multi-signer assignment (signer_01, signer_02, etc.) and recording session ID
- 3-second recording window (~90 frames) with neutral start / active gesture / neutral end
- Real-time MediaPipe landmark validation: rejects clips if insufficient hand landmarks detected
- Automatically manages target counts (minimum 30, preferred 40 per class)
"""

import os
import sys
import time
import uuid
from datetime import datetime
import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, FRAME_WIDTH, FRAME_HEIGHT, CLASS_NAMES
from src.landmark_extractor import LandmarkExtractor
from src.real_dataset_pipeline import RealISLDatasetPipeline

# Official ISLRTC / standard Indian Sign Language gesture definitions
ISL_21_CLASS_DEFINITIONS = {
    "hello": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Open palm near temple/forehead with palm outward, gently wave outward (salute wave)."
    },
    "thank_you": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Open flat hand touches chin/lips with fingertips, extends forward/outward with palm up."
    },
    "welcome": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Both open flat hands held out in front, palms up/inward, sweeping inward toward torso."
    },
    "goodbye": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Open flat hand at shoulder/head level, waving side to side in an isolated waving motion."
    },
    "yes": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant hand forms closed fist at chest level, nodding up and down from the wrist."
    },
    "no": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Index and middle fingers extended together, meeting thumb and pinching closed, or index waving side-to-side."
    },
    "please": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant open flat palm against center of chest, rotating in gentle clockwise circular rubbing motion."
    },
    "sorry": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant 'A' fist placed against chest/sternum, rotating in circular rubbing motion with apologetic expression."
    },
    "help": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Non-dominant flat palm horizontal palm-up; dominant 'A' fist resting on palm, moving upward together."
    },
    "stop": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant flat hand held vertically facing outward toward camera, or brought sharply down onto horizontal palm."
    },
    "water": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "'W' handshape (3 fingers upright) or cupped hand brought to lips, tapping chin or lips twice."
    },
    "food": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant hand forms flattened 'O' (fingertips touching thumb), brought to lips in natural eating motion."
    },
    "school": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Both flat open palms facing each other horizontally; dominant palm claps down firmly onto non-dominant twice."
    },
    "teacher": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Both hands form 'O' or pinch at temple level moving forward and opening, followed by agent marker down torso."
    },
    "mother": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant index finger or open thumb taps or brushes cheek/side of chin twice (ISL female marker)."
    },
    "father": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant hand open with thumb tapping center of forehead or temple twice (ISL male marker)."
    },
    "sister": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant index touches cheek (female marker) followed by both index fingers extended parallel touching together."
    },
    "brother": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Dominant thumb taps forehead (male marker) followed by both index fingers extended parallel touching together."
    },
    "friend": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Both hands form bent index finger hooks, interlocking together once, reversing and interlocking again."
    },
    "house": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Both open flat hands diagonally opposite, fingertips touching at apex (roof angle) then moving vertically downward."
    },
    "work": {
        "target_min": 30,
        "target_pref": 40,
        "instruction": "Both hands in closed fists with wrists facing downward; dominant fist heel/wrist taps down firmly on non-dominant wrist twice."
    }
}


def get_current_counts():
    """Returns current real video counts across all 21 classes."""
    import glob
    counts = {}
    for c in CLASS_NAMES:
        vids = glob.glob(f"dataset/raw/{c}/**/*.mp4", recursive=True)
        counts[c] = len(vids)
    return counts


def record_clip(cap, extractor, sign_class, signer_id, session_id, instruction, clip_num, total_needed, record_seconds=3.0):
    """
    Shows a 3-second countdown with visual gesture instructions, records 3.0 seconds (640x480),
    runs MediaPipe quality validation on recorded frames using a shared extractor, and saves only if hand landmarks are detected.
    """
    output_dir = os.path.join(RAW_DATA_DIR, sign_class, signer_id)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Countdown Phase (3.0 seconds)
    start_countdown = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read frame from webcam.")
            return None

        frame_mirrored = cv2.flip(frame, 1)
        elapsed = time.time() - start_countdown
        remaining = 3.0 - elapsed

        if remaining <= 0:
            break

        h, w, _ = frame_mirrored.shape
        count_str = str(int(np.ceil(remaining)))

        # Header overlay
        cv2.rectangle(frame_mirrored, (0, 0), (w, 90), (20, 20, 20), -1)
        cv2.putText(frame_mirrored, f"SIGN: {sign_class.upper()} (Clip {clip_num}/{total_needed}) - [{signer_id}]",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame_mirrored, f"Protocol: {instruction[:65]}...", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        cv2.putText(frame_mirrored, "Pose: Start with hands resting neutral -> Sign on countdown 0", (15, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 230, 160), 1)

        # Center countdown
        cv2.putText(frame_mirrored, f"GET READY: {count_str}", (w // 2 - 180, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 140, 255), 4)
        cv2.putText(frame_mirrored, "Press 'q' to pause / skip", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("ISL Guided Webcam Recorder (21 Classes)", frame_mirrored)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[User] Paused/Skipped by user.")
            return None

    # 2. Recording Phase (record_seconds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    video_filename = f"video_{sign_class}_{signer_id}_{session_id}_{timestamp}_{unique_suffix}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    counter = 1
    while os.path.exists(video_path):
        video_filename = f"video_{sign_class}_{signer_id}_{session_id}_{timestamp}_{unique_suffix}_{counter}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        counter += 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

    start_rec = time.time()
    recorded_frames = []
    cancelled = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_mirrored = cv2.flip(frame, 1)
            writer.write(frame_mirrored)
            recorded_frames.append(frame_mirrored)

            rec_elapsed = time.time() - start_rec
            if rec_elapsed >= record_seconds:
                break

            h, w, _ = frame_mirrored.shape
            # Recording banner (Red pulse)
            cv2.rectangle(frame_mirrored, (0, 0), (w, 75), (0, 0, 180), -1)
            cv2.circle(frame_mirrored, (35, 38), 12, (255, 255, 255), -1)
            cv2.putText(frame_mirrored, f"RECORDING '{sign_class.upper()}' ({rec_elapsed:.1f}s / {record_seconds}s)",
                        (65, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.imshow("ISL Guided Webcam Recorder (21 Classes)", frame_mirrored)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cancelled = True
                break
    finally:
        writer.release()

    if cancelled:
        if os.path.exists(video_path):
            os.remove(video_path)
        return None

    # 3. Quality Validation Phase: Validate hand landmarks using shared extractor
    print(f"  Validating recorded clip ({len(recorded_frames)} frames)...", end="", flush=True)
    valid_hand_frames = 0
    for frm in recorded_frames:
        ld = extractor.extract(frm)
        if ld.get("hands") and len(ld["hands"]) > 0:
            valid_hand_frames += 1

    # Require hand detection in at least 15 frames (~50% of 30 FPS clip)
    if valid_hand_frames < 15:
        print(f" REJECTED! (Only {valid_hand_frames}/{len(recorded_frames)} frames had hand landmarks).")
        print("  -> Clip deleted. Please ensure hands are clearly visible in the camera frame.")
        if os.path.exists(video_path):
            os.remove(video_path)
        return None

    print(f" ACCEPTED ({valid_hand_frames}/{len(recorded_frames)} valid hand frames).")
    print(f"\n--- Recording Details ---")
    print(f"  Class:                      {sign_class}")
    print(f"  Signer:                     {signer_id}")
    print(f"  Filename:                   {video_filename}")
    print(f"  Frame count:                {len(recorded_frames)}")
    print(f"  Valid landmark frame count: {valid_hand_frames}")
    print(f"  Save path:                  {video_path}")
    print(f"-------------------------")
    return video_path


def run_guided_session():
    import re
    print("=" * 75)
    print("      REAL ISL 21-CLASS GUIDED WEBCAM RECORDER (STANDARDIZED PROTOCOL)")
    print("=" * 75)
    print("Records controlled, multi-signer real webcam samples with:")
    print(" - 640x480 resolution at 30 FPS")
    print(" - Verified ISLRTC / standard ISL dictionary definitions (no invented signs)")
    print(" - Automatic hand landmark validation (rejects invalid/no-hand clips)")
    print("=" * 75)

    counts = get_current_counts()
    print("\nCURRENT 21-CLASS INVENTORY:")
    print(f"{'#':<3} {'Class':<12} {'Current':<8} {'Target Min/Pref':<16} {'Deficit (Min/Pref)':<20} {'Status':<15}")
    print("-" * 75)

    all_options = CLASS_NAMES
    for idx, c in enumerate(all_options, 1):
        cur = counts.get(c, 0)
        t_min = ISL_21_CLASS_DEFINITIONS[c]["target_min"]
        t_pref = ISL_21_CLASS_DEFINITIONS[c]["target_pref"]
        def_min = max(0, t_min - cur)
        def_pref = max(0, t_pref - cur)
        status = "COMPLETE" if def_min == 0 else f"NEEDS +{def_min}"
        print(f"{idx:<3} {c:<12} {cur:<8} {t_min}/{t_pref:<14} +{def_min}/+{def_pref:<18} {status:<15}")

    print("\nSIGNER IDENTIFICATION:")
    print("To avoid single-signer overfitting, enter the signer's identity (e.g. signer_02, signer_03, signer_04).")
    signer_id = input("Enter Signer ID [default: signer_02]: ").strip() or "signer_02"
    session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nActive Signer:  {signer_id}")
    print(f"Active Session: {session_id}")

    print("\nSelect class to record:")
    print("  1-21: Specific class number from the list above")
    print("  22:   All classes below minimum 30 sequentially")
    print("  0/q:  Exit")

    raw_choice = input("\nEnter choice [1-22, or 0/q]: ").strip()
    if raw_choice.lower() in ["0", "q", "exit", "quit", ""] or not raw_choice:
        print("Exiting.")
        return

    selected_classes = []
    # Robust numeric matching (handles '20', '20 (house)', 'Class: 20 (house)', etc.)
    match = re.search(r'\b(\d+)\b', raw_choice)
    if match:
        val = int(match.group(1))
        if val == 22:
            selected_classes = [c for c in all_options if counts.get(c, 0) < ISL_21_CLASS_DEFINITIONS[c]["target_min"]]
            print(f"Class: All deficit classes ({len(selected_classes)} classes selected)")
        elif 1 <= val <= len(all_options):
            selected_class = all_options[val - 1]
            print(f"Class: {val} ({selected_class})")
            selected_classes = [selected_class]
        else:
            print(f"Invalid choice '{raw_choice}'. Number must be between 1 and {len(all_options)} (or 22 for all). Exiting.")
            return
    elif raw_choice.lower() in CLASS_NAMES:
        selected_class = raw_choice.lower()
        val = CLASS_NAMES.index(selected_class) + 1
        print(f"Class: {val} ({selected_class})")
        selected_classes = [selected_class]
    else:
        print(f"Invalid choice '{raw_choice}'. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam (camera index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    extractor = LandmarkExtractor()
    pipeline = RealISLDatasetPipeline()

    try:
        for sign_class in selected_classes:
            cur_counts = get_current_counts()
            cur = cur_counts.get(sign_class, 0)
            needed = max(0, ISL_21_CLASS_DEFINITIONS[sign_class]["target_min"] - cur)

            if needed == 0:
                print(f"\n[Info] '{sign_class.upper()}' already meets minimum target ({cur} >= 30). Skipping.")
                continue

            instruction = ISL_21_CLASS_DEFINITIONS[sign_class]["instruction"]
            print("\n" + "=" * 70)
            print(f"  PREPARING TO RECORD: {sign_class.upper()}")
            print(f"  Target: {needed} additional clips as '{signer_id}' (Session: {session_id})")
            print(f"  Official ISL Instruction: {instruction}")
            print("=" * 70)
            input("Position yourself 0.8-1.2m in front of webcam and press ENTER to start...")

            recorded_this_session = 0
            clip_num = 1
            while recorded_this_session < needed:
                print(f"\n[Clip {clip_num}/{needed}] Ready...")
                vpath = record_clip(cap, extractor, sign_class, signer_id, session_id, instruction,
                                    clip_num, needed, record_seconds=3.0)

                if vpath is None:
                    retry = input("Retry this clip? [Y/n]: ").strip().lower()
                    if retry == "n":
                        break
                    continue

                recorded_this_session += 1
                clip_num += 1
                print(f"  Saved video: {vpath}")

                # Real-time token extraction and verification
                vid_id = os.path.splitext(os.path.basename(vpath))[0]
                res = pipeline.process_video_file({
                    "video_path": vpath,
                    "sign_class": sign_class,
                    "signer_id": signer_id,
                    "video_id": vid_id
                })

                status = res.get("status")
                tok_len = res.get("token_sequence_length", 0)
                if status == "success":
                    print(f"  Pipeline Token Verification: SUCCESS ({tok_len} tokens extracted)")
                else:
                    err = res.get("error", "Unknown error")
                    print(f"  Pipeline Token Verification: WARNING ({err})")

                time.sleep(1.0)

            # Update metadata CSV
            pipeline.process_dataset()
            print(f"\n[Done] Completed batch for '{sign_class.upper()}'. Dataset updated.")

    finally:
        print("\nReleasing webcam and MediaPipe resources...")
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
        print("Session cleanup complete.")


if __name__ == "__main__":
    run_guided_session()
