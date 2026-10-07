import os
import sys
import cv2
import numpy as np
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

cap = cv2.VideoCapture("dataset/raw/thalapathy/hf_real/WIN_20261006_22_05_40_Pro.mp4")
frame_idx = 0
found_hands = []

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame_idx += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = hand_landmarker.detect(mp_image)
    if res and res.hand_landmarks:
        found_hands.append((frame_idx, len(res.hand_landmarks), [h[0].category_name for h in res.handedness]))

cap.release()
print(f"Total frames: {frame_idx}, Frames with hands (IMAGE mode min_conf=0.3): {len(found_hands)}")
if len(found_hands) > 0:
    print("Sample detections:", found_hands[:10])
hand_landmarker.close()
