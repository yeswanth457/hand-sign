import os
import cv2

video_dir = r"D:\isl-translator\dataset\raw\school\hf_real"
if os.path.exists(video_dir):
    vids = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"Total school mp4 videos: {len(vids)}")
    for v in sorted(vids)[:15]:
        vpath = os.path.join(video_dir, v)
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"  {v:<40} frames={count:<4} fps={fps:<4.1f} size={w}x{h}")
else:
    print("Video dir does not exist:", video_dir)
