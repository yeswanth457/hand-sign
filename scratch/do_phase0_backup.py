import os
import shutil

backup_dir = "models/backup_friend_pre_modification"
os.makedirs(backup_dir, exist_ok=True)

files = [
    "config.py",
    "app.py",
    "src/cnn_gru_model.py",
    "src/early_decision.py",
    "src/gesture_tokenizer.py",
    "src/landmark_extractor.py",
    "src/frame_selector.py",
    "src/build_real_split.py",
    "src/train_cnn_gru.py",
    "src/train_real_model.py",
    "static/app.js",
    "static/index.html",
    "static/style.css",
    "models/isl_cnn_gru_father_fix.pt",
    "models/isl_cnn_gru_22class_brother_updated.pt",
    "models/isl_cnn_gru_22class_best.pt",
    "models/feature_mean.npy",
    "models/feature_std.npy"
]

print("=" * 60)
print("PHASE 0: EXECUTING COMPLETE SYSTEM BACKUP")
print("=" * 60)
backed = []
for f in files:
    if os.path.exists(f):
        dst = os.path.join(backup_dir, os.path.basename(f))
        shutil.copy2(f, dst)
        size = os.path.getsize(dst)
        backed.append((f, dst, size))
        print(f"  [OK] {f:35s} -> {dst} ({size:,} bytes)")
    else:
        print(f"  [SKIP] {f} not found")

print("-" * 60)
print(f"BACKUP STATUS: {len(backed)} files successfully frozen in {backup_dir}")
print("=" * 60)
