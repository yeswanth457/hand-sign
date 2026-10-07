import os
import sys
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

pose_model_path = "models/pose_landmarker.task"
hand_model_path = "models/hand_landmarker.task"

pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=pose_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_options = vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

thalapathy_dir = Path("dataset/raw/thalapathy/hf_real")
video_files = sorted(list(thalapathy_dir.glob("*.mp4")))

print(f"Found {len(video_files)} Thalapathy raw videos.")
print("=" * 70)
print("INSPECTING THALAPATHY RAW VIDEOS FOR 1-HAND VS 2-HAND OCCURRENCE:")
print("=" * 70)

for v_path in video_files:
    cap = cv2.VideoCapture(str(v_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ts_ms = 0
    frame_idx = 0
    
    count_0_hand = 0
    count_1_hand = 0
    count_2_hand = 0
    left_hand_frames = 0
    right_hand_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        ts_ms += int(1000.0 / fps)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        try:
            h_res = hand_landmarker.detect_for_video(mp_image, ts_ms)
        except Exception:
            h_res = None
            
        if h_res and h_res.hand_landmarks:
            n_hands = len(h_res.hand_landmarks)
            if n_hands == 1:
                count_1_hand += 1
            elif n_hands >= 2:
                count_2_hand += 1
                
            labels = []
            if h_res.handedness:
                for h_meta in h_res.handedness:
                    lbl = h_meta[0].category_name if hasattr(h_meta[0], 'category_name') else h_meta[0].display_name
                    labels.append(lbl)
            if "Left" in labels:
                left_hand_frames += 1
            if "Right" in labels:
                right_hand_frames += 1
        else:
            count_0_hand += 1

    cap.release()
    valid_frames = count_1_hand + count_2_hand
    pct_1h = (count_1_hand / max(1, valid_frames)) * 100
    pct_2h = (count_2_hand / max(1, valid_frames)) * 100
    print(f"Video: {v_path.name}")
    print(f"  Total frames: {total_frames}, Valid hand frames: {valid_frames} (0-hand: {count_0_hand})")
    print(f"  1-Hand frames: {count_1_hand} ({pct_1h:.1f}%), 2-Hand frames: {count_2_hand} ({pct_2h:.1f}%)")
    print(f"  Left hand present: {left_hand_frames} frames, Right hand present: {right_hand_frames} frames")
    print("-" * 70)

pose_landmarker.close()
hand_landmarker.close()
