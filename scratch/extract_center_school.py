import os
import sys
sys.path.insert(0, r"D:\isl-translator")
import cv2
import numpy as np
from src.landmark_extractor import LandmarkExtractor

extractor = LandmarkExtractor()
video_path = r"D:\isl-translator\dataset\raw\school\hf_real\centerSchool.mp4"

cap = cv2.VideoCapture(video_path)
frame_idx = 0
hand_counts = []
selected_hands = []
hys = []
rys = []
sys_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = extractor.process_frame(frame, timestamp_ms=int(frame_idx * 33.3))
    num_hands = len(res.get("hands", []))
    hand_counts.append(num_hands)
    hc = res.get("hand_center", (0.5, 0.5))
    sc = res.get("shoulder_center", (0.5, 0.35))
    hys.append(hc[1])
    sys_list.append(sc[1])
    rys.append(hc[1] - sc[1])
    frame_idx += 1

cap.release()

print(f"centerSchool.mp4 processed {frame_idx} frames:")
print(f"Hand counts distribution: 0-hands: {hand_counts.count(0)}, 1-hand: {hand_counts.count(1)}, 2-hands: {hand_counts.count(2)}")
print(f"Mean Hy: {np.mean(hys):.4f}")
print(f"Mean Sy: {np.mean(sys_list):.4f}")
print(f"Mean Ry: {np.mean(rys):.4f}")
print(f"Frames where 2 hands detected: {[i for i, c in enumerate(hand_counts) if c == 2]}")
