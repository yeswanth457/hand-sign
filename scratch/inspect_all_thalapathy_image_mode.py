import os
import sys
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

hand_model_path = "models/hand_landmarker.task"
hand_options = vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

thalapathy_dir = Path("dataset/raw/thalapathy/hf_real")
video_files = sorted(list(thalapathy_dir.glob("*.mp4")))

print("=" * 70)
print("INSPECTING ALL 6 THALAPATHY VIDEOS (IMAGE MODE, min_conf=0.3):")
print("=" * 70)

for v_path in video_files:
    cap = cv2.VideoCapture(str(v_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    c0 = 0
    c1 = 0
    c2 = 0
    left_cnt = 0
    right_cnt = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = hand_landmarker.detect(mp_image)
        if res and res.hand_landmarks:
            nh = len(res.hand_landmarks)
            if nh == 1:
                c1 += 1
            else:
                c2 += 1
            labels = [h[0].category_name for h in res.handedness]
            if "Left" in labels:
                left_cnt += 1
            if "Right" in labels:
                right_cnt += 1
        else:
            c0 += 1
            
    cap.release()
    valid = c1 + c2
    print(f"Video: {v_path.name}")
    print(f"  Total frames: {total_frames}, Valid hand frames: {valid} (0-hand: {c0})")
    print(f"  1-Hand: {c1} ({(c1/max(1, valid))*100:.1f}%), 2-Hand: {c2} ({(c2/max(1, valid))*100:.1f}%)")
    print(f"  Left hand present: {left_cnt}, Right hand present: {right_cnt}")
    print("-" * 70)

hand_landmarker.close()
