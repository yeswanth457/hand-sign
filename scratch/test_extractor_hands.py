import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from pathlib import Path
from src.landmark_extractor import LandmarkExtractor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

extractor = LandmarkExtractor()
thalapathy_dir = Path("dataset/raw/thalapathy/hf_real")
video_files = sorted(list(thalapathy_dir.glob("*.mp4")))

print(f"Testing LandmarkExtractor on {len(video_files)} Thalapathy videos:")

global_ts = 0

for v_path in video_files:
    cap = cv2.VideoCapture(str(v_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count_0 = 0
    count_1 = 0
    count_2 = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        global_ts += 33
        res = extractor.process_frame(frame, timestamp_ms=global_ts)
        hands = res.get("hands", [])
        if len(hands) == 0:
            count_0 += 1
        elif len(hands) == 1:
            count_1 += 1
        else:
            count_2 += 1
            
    cap.release()
    print(f"Video {v_path.name} (total {total_frames} frames):")
    print(f"  0 hands: {count_0}, 1 hand: {count_1}, 2 hands: {count_2}")

extractor.close()
