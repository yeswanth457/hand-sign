import os
import sys
import csv
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath("."))

from config import CLASS_NAMES, CLASS_TO_INDEX, NUM_CLASSES, TOKEN_DIM, DATASET_DIR
from src.landmark_extractor import LandmarkExtractor
from src.gesture_tokenizer import GestureTokenizer
from src.frame_selector import MotionFrameSelector
from src.real_dataset_pipeline import RealISLDatasetPipeline
from src.build_real_split import build_real_splits

print("=" * 70)
print("REBUILDING 12D TOKEN DATASET ACROSS ALL 22 CLASSES")
print("=" * 70)

lm_dir = Path("dataset/landmarks")
tokens_dir = Path("dataset/tokens")
tokens_dir.mkdir(parents=True, exist_ok=True)

# 1. Reprocess raw Thalapathy videos
print("\n[Step 1] Processing raw Thalapathy videos with 2-hand extraction...")
raw_thalapathy_dir = Path("dataset/raw/thalapathy/hf_real")
thalapathy_videos = sorted(list(raw_thalapathy_dir.glob("*.mp4")))

pipeline = RealISLDatasetPipeline()
for v in thalapathy_videos:
    entry = {
        "video_path": str(v),
        "sign_class": "thalapathy",
        "signer_id": "hf_real",
        "video_id": v.stem
    }
    # Force remove old 6D files if any
    old_tk = tokens_dir / "thalapathy" / "hf_real" / f"{v.stem}.npz"
    if old_tk.exists():
        os.remove(old_tk)
    res = pipeline.process_video_file(entry)
    print(f"  Thalapathy video {v.name}: status={res['status']}, tokens={res['token_sequence_length']}")

# 2. Convert all existing landmarks to 12D tokens
print("\n[Step 2] Converting all existing landmark files to 12D tokens...")
tokenizer = GestureTokenizer()
total_converted = 0
active_classes = set(CLASS_NAMES)

csv_records = []

for c_dir in sorted(lm_dir.iterdir()):
    if not c_dir.is_dir():
        continue
    c_name = c_dir.name
    if c_name not in active_classes:
        continue
        
    for s_dir in c_dir.iterdir():
        if not s_dir.is_dir():
            continue
        s_name = s_dir.name
        
        for lm_file in s_dir.glob("*.npz"):
            v_id = lm_file.stem
            out_dir = tokens_dir / c_name / s_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_tk = out_dir / f"{v_id}.npz"
            
            # If thalapathy already reprocessed in step 1 and valid, keep it
            if c_name == "thalapathy" and out_tk.exists():
                try:
                    d = np.load(out_tk)
                    if d["tokens"].shape[1] == TOKEN_DIM:
                        csv_records.append({
                            "video_path": str(lm_file),
                            "sign_class": c_name,
                            "signer_id": s_name,
                            "video_id": v_id,
                            "fps": 30.0,
                            "total_frames": 90,
                            "valid_landmark_frames": len(d["tokens"]),
                            "selected_frame_count": len(d["tokens"]),
                            "selection_ratio": 1.0,
                            "token_sequence_length": len(d["tokens"]),
                            "landmark_file": str(lm_file),
                            "token_file": str(out_tk),
                            "status": "success",
                            "error": ""
                        })
                        continue
                except Exception:
                    pass

            try:
                lm_data = np.load(lm_file, allow_pickle=True)
                lms = lm_data["landmarks"]
                
                # Apply motion frame selector if needed or tokenize frames
                selector = MotionFrameSelector()
                for idx, frame_dict in enumerate(lms):
                    if isinstance(frame_dict, dict):
                        selector.process_frame(idx, frame_dict)
                        
                selected = selector.get_selected_sequence()
                if len(selected) < 3:
                    selected = [f for f in lms if isinstance(f, dict)]
                    
                tokens = tokenizer.tokenize_sequence(selected)
                
                if tokens.ndim == 2 and tokens.shape[1] == TOKEN_DIM and len(tokens) >= 3:
                    np.savez_compressed(
                        out_tk,
                        video_path=str(lm_file),
                        sign_class=c_name,
                        signer_id=s_name,
                        video_id=v_id,
                        total_frames=len(lms),
                        selected_frames=len(selected),
                        tokens=tokens
                    )
                    total_converted += 1
                    csv_records.append({
                        "video_path": str(lm_file),
                        "sign_class": c_name,
                        "signer_id": s_name,
                        "video_id": v_id,
                        "fps": 30.0,
                        "total_frames": len(lms),
                        "valid_landmark_frames": len(lms),
                        "selected_frame_count": len(selected),
                        "selection_ratio": round(len(selected)/max(1, len(lms)), 4),
                        "token_sequence_length": len(tokens),
                        "landmark_file": str(lm_file),
                        "token_file": str(out_tk),
                        "status": "success",
                        "error": ""
                    })
            except Exception as e:
                print(f"Error converting {lm_file}: {e}")

print(f"Total 12D token files generated: {len(csv_records)}")

# 3. Write metadata CSV
csv_path = Path("dataset/metadata/dataset.csv")
fieldnames = [
    "video_path", "sign_class", "signer_id", "video_id",
    "fps", "total_frames", "valid_landmark_frames",
    "selected_frame_count", "selection_ratio",
    "token_sequence_length", "landmark_file", "token_file",
    "status", "error"
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_records)
print(f"Updated metadata CSV at {csv_path} ({len(csv_records)} rows)")

# 4. Rebuild Splits
print("\n[Step 3] Building train/val/test splits...")
build_real_splits()
