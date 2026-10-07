import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from src.real_dataset_pipeline import RealISLDatasetPipeline

pipeline = RealISLDatasetPipeline()
raw_dir = os.path.join("dataset", "raw", "thalapathy", "hf_real")
for fname in sorted(os.listdir(raw_dir)):
    if not fname.endswith(".mp4"):
        continue
    fpath = os.path.join(raw_dir, fname)
    cap = cv2.VideoCapture(fpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(1, fps)
    cap.release()

    entry = {
        "video_path": fpath,
        "sign_class": "thalapathy",
        "signer_id": "hf_real",
        "video_id": os.path.splitext(fname)[0]
    }
    res = pipeline.process_video_file(entry)
    print(f"FILE: {fname}")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.2f}, Duration: {duration:.2f}s, TotalFrames: {total_frames}")
    print(f"  ValidLM: {res.get('valid_landmark_frames')}, Selected: {res.get('selected_frame_count')}, Tokens: {res.get('token_sequence_length')}")
    print(f"  Status: {res.get('status')}")
