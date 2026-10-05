import os
import cv2

video_path = r"D:\isl-translator\dataset\raw\school\hf_real\centerSchool.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("centerSchool.mp4 total frames:", total_frames)
    
    # Sample a frame and check brightness/dimensions
    ret, frame = cap.read()
    if ret:
        print("Frame size:", frame.shape)
        # Check average pixel value
        print("Mean pixel value:", frame.mean())
    cap.release()
