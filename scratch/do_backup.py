import os
import shutil

backup_dir = "models/backup_two_hand_support"
os.makedirs(backup_dir, exist_ok=True)

files = [
    "src/landmark_extractor.py",
    "src/gesture_tokenizer.py",
    "src/frame_selector.py",
    "src/cnn_gru_model.py",
    "src/real_dataset_pipeline.py",
    "src/build_real_split.py",
    "src/train_21class_cnn_gru.py",
    "config.py",
    "app.py",
    "static/app.js",
    "static/index.html",
    "models/isl_cnn_gru_22class_best.pt",
    "models/isl_cnn_gru.pt"
]

for f in files:
    if os.path.exists(f):
        dst = os.path.join(backup_dir, os.path.basename(f))
        shutil.copy2(f, dst)
        print(f"Backed up: {f} -> {dst}")

print(f"Total backed up files: {len(os.listdir(backup_dir))}")
