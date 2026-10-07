import os
import shutil

backup_dir = os.path.join("models", "backup_water_final_temporal_fix")
os.makedirs(backup_dir, exist_ok=True)

files_to_copy = [
    "app.py", "config.py",
    "static/app.js", "static/index.html",
    "src/cnn_gru_model.py", "src/no_binary_model.py", "src/early_decision.py",
    "src/build_real_split.py", "src/real_dataset_pipeline.py", "src/landmark_extractor.py",
    "src/gesture_tokenizer.py", "src/sentence_processor.py"
]

for f in files_to_copy:
    if os.path.exists(f):
        dst = os.path.join(backup_dir, os.path.basename(f))
        shutil.copy2(f, dst)
        print(f"Backed up {f} -> {dst}")

for mf in os.listdir("models"):
    src_path = os.path.join("models", mf)
    if os.path.isfile(src_path) and (mf.endswith(".pth") or mf.endswith(".npy") or mf.endswith(".json")):
        dst = os.path.join(backup_dir, mf)
        shutil.copy2(src_path, dst)
        print(f"Backed up model artifact {mf} -> {dst}")

print("BACKUP COMPLETE")
