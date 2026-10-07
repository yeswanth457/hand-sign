import os
import glob
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from src.landmark_extractor import LandmarkExtractor

extractor = LandmarkExtractor(min_detection_confidence=0.25)

vids = glob.glob('dataset/raw/brother/**/*.mp4', recursive=True)
print(f"Total Brother MP4s found: {len(vids)}")

stats = {
    'total_videos': len(vids),
    'valid_videos': 0,
    'invalid_videos': 0,
    'total_frames': 0,
    'valid_hand_frames': 0,
    'left_only': 0,
    'right_only': 0,
    'both_hands': 0,
    'zero_hands': 0,
}

for v in sorted(vids):
    cap = cv2.VideoCapture(v)
    f_cnt, vld, l_cnt, r_cnt, b_cnt, z_cnt = 0, 0, 0, 0, 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        f_cnt += 1
        ld = extractor.process_frame(frame)
        lh = ld.get('left_hand') is not None
        rh = ld.get('right_hand') is not None
        if lh and rh:
            b_cnt += 1
            vld += 1
        elif lh:
            l_cnt += 1
            vld += 1
        elif rh:
            r_cnt += 1
            vld += 1
        else:
            z_cnt += 1
    cap.release()

    is_valid = vld >= 8
    if is_valid:
        stats['valid_videos'] += 1
    else:
        stats['invalid_videos'] += 1

    stats['total_frames'] += f_cnt
    stats['valid_hand_frames'] += vld
    stats['left_only'] += l_cnt
    stats['right_only'] += r_cnt
    stats['both_hands'] += b_cnt
    stats['zero_hands'] += z_cnt

    rel_name = os.path.relpath(v, 'dataset/raw/brother')
    print(f"{rel_name:<55} | Total:{f_cnt:<4} | Valid:{vld:<4} | 2H:{b_cnt:<3} | L:{l_cnt:<3} | R:{r_cnt:<3} | ValidVid:{is_valid}")

extractor.close()

print('=' * 75)
print('SUMMARY STATS (STEP 5):')
print(f"Total Brother videos:   {stats['total_videos']}")
print(f"Valid videos:           {stats['valid_videos']}")
print(f"Invalid videos:         {stats['invalid_videos']}")
print(f"Total frames:           {stats['total_frames']}")
print(f"Valid gesture frames:   {stats['valid_hand_frames']}")
print(f"Both hands frames:      {stats['both_hands']} ({stats['both_hands']/max(1, stats['valid_hand_frames'])*100:.1f}%)")
print(f"Left only frames:       {stats['left_only']} ({stats['left_only']/max(1, stats['valid_hand_frames'])*100:.1f}%)")
print(f"Right only frames:      {stats['right_only']} ({stats['right_only']/max(1, stats['valid_hand_frames'])*100:.1f}%)")
print(f"Zero hand frames:       {stats['zero_hands']}")
