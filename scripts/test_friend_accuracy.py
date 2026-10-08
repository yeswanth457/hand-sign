"""
Measure actual Friend gesture accuracy on the 3 newest real Friend videos.
Strictly uses existing production pipeline and active model checkpoint.
Does NOT modify or retrain the model.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MAX_SEQ_LEN, MODEL_DIR, MODEL_PATH, DATASET_DIR
)
from src.real_dataset_pipeline import RealISLDatasetPipeline
from src.cnn_gru_model import ISL_CNN_GRU_Model, resample_tokens


def test_friend_accuracy(model_path=None):
    if model_path is None:
        model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH

    # 1. Print Active Checkpoint Exact Path
    print("=" * 60)
    print("REAL FRIEND GESTURE ACCURACY EVALUATION")
    print("=" * 60)
    print(f"Active Model Checkpoint: {model_path}")
    print(f"NUM_CLASSES:             {NUM_CLASSES}")
    
    # 2. Verify Expected Class and Index
    expected_class = "friend"
    class_idx = CLASS_TO_INDEX.get(expected_class)
    print(f"Expected Class:          {expected_class}")
    print(f"Class Index:             {class_idx}")
    assert class_idx == 18, f"Expected class index 18, got {class_idx}"
    assert NUM_CLASSES == 22, f"Expected 22 classes, got {NUM_CLASSES}"

    # 3. Locate the 3 Newest Friend Videos
    raw_friend_dir = os.path.join(DATASET_DIR, "raw", "friend")
    all_videos = glob.glob(os.path.join(raw_friend_dir, "**", "*.mp4"), recursive=True)
    all_videos += glob.glob(os.path.join(raw_friend_dir, "**", "*.avi"), recursive=True)
    all_videos += glob.glob(os.path.join(raw_friend_dir, "**", "*.mov"), recursive=True)

    if len(all_videos) == 0:
        print("[ERROR] No Friend videos found in dataset/raw/friend/.")
        return

    # Sort descending by modification time
    all_videos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    target_videos = all_videos[:3]

    print(f"\nLocating 3 Newest Friend Videos in {raw_friend_dir}:")
    for i, v in enumerate(target_videos, 1):
        mtime = os.path.getmtime(v)
        print(f"  {i}. {os.path.basename(v)} (Path: {v})")

    # 4. Load Pipeline, Normalization, and Active Model
    pipeline = RealISLDatasetPipeline()

    mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
    std_path = os.path.join(MODEL_DIR, "feature_std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print(f"[ERROR] Normalization stats missing at {mean_path} or {std_path}")
        return

    feat_mean = np.load(mean_path).astype(np.float32)
    feat_std = np.load(std_path).astype(np.float32)
    feat_std = np.where(feat_std < 1e-7, 1.0, feat_std)

    device = torch.device("cpu")
    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 5. Run Each Video Through Existing Pipeline
    results = []
    print("\n" + "=" * 60)
    print("INDIVIDUAL VIDEO INFERENCE RESULTS")
    print("=" * 60)

    for vid_path in target_videos:
        vid_name = os.path.basename(vid_path)
        signer_id = os.path.basename(os.path.dirname(vid_path))
        vid_stem = os.path.splitext(vid_name)[0]

        entry = {
            "video_path": vid_path,
            "sign_class": expected_class,
            "signer_id": signer_id,
            "video_id": vid_stem
        }

        # Process video with production pipeline
        proc_res = pipeline.process_video_file(entry)
        tk_file = proc_res.get("token_file")

        if not tk_file or not os.path.exists(tk_file):
            print(f"\nVideo:              {vid_name}")
            print(f"Expected:           {expected_class}")
            print(f"Predicted:          [FAILED TO EXTRACT TOKENS: {proc_res.get('error')}]")
            print(f"Friend probability: 0.0%")
            print(f"Top-1 confidence:   0.0%")
            print(f"Top-2:              --")
            print(f"Correct:            NO")
            results.append({
                "video": vid_name,
                "predicted": "failed",
                "friend_prob": 0.0,
                "top1_conf": 0.0,
                "top2": "--",
                "correct": False
            })
            continue

        raw_tokens = np.load(tk_file)["tokens"].astype(np.float32)
        tokens_25 = resample_tokens(raw_tokens, target_t=25)
        norm_tokens = (tokens_25 - feat_mean) / feat_std
        tensor_in = torch.tensor(norm_tokens, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_in)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top1_cls = INDEX_TO_CLASS[top1_idx]
        top1_conf = float(probs[top1_idx])

        friend_prob = float(probs[class_idx])

        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else -1
        top2_cls = INDEX_TO_CLASS[top2_idx] if top2_idx >= 0 else "--"
        top2_conf = float(probs[top2_idx]) if top2_idx >= 0 else 0.0

        is_correct = (top1_cls == expected_class)

        print(f"\nVideo:              {vid_name}")
        print(f"Expected:           {expected_class}")
        print(f"Predicted:          {top1_cls}")
        print(f"Friend probability: {friend_prob * 100:.2f}%")
        print(f"Top-1 confidence:   {top1_conf * 100:.2f}%")
        print(f"Top-2:              {top2_cls} ({top2_conf * 100:.2f}%)")
        print(f"Correct:            {'YES' if is_correct else 'NO'}")

        results.append({
            "video": vid_name,
            "predicted": top1_cls,
            "friend_prob": friend_prob * 100,
            "top1_conf": top1_conf * 100,
            "top2": f"{top2_cls} ({top2_conf * 100:.2f}%)",
            "correct": is_correct
        })

    # 6. Calculate Accuracy
    total_videos = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    incorrect_count = total_videos - correct_count
    accuracy = (correct_count / max(1, total_videos)) * 100

    print("\n" + "=" * 41)
    print("FRIEND ACCURACY")
    print("=" * 41)
    print(f"Total videos: {total_videos}")
    print(f"Correct:      {correct_count}")
    print(f"Incorrect:    {incorrect_count}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print("=" * 41)

    # 7. Print Confusion Breakdown
    print("\nConfusion Breakdown:")
    confusion_classes = ["friend", "house", "brother", "school"]
    counts = {c: 0 for c in confusion_classes}
    other_count = 0

    for r in results:
        pred = r["predicted"]
        if pred in counts:
            counts[pred] += 1
        else:
            other_count += 1

    print(f"Friend -> Friend:  {counts['friend']}")
    print(f"Friend -> House:   {counts['house']}")
    print(f"Friend -> Brother: {counts['brother']}")
    print(f"Friend -> School:  {counts['school']}")
    print(f"Friend -> Other:   {other_count}")

    # 8. Note on Confidence vs Accuracy
    print("\nNote: Model confidence is NOT accuracy.")
    print("Accuracy is strictly the percentage of correctly classified independent videos.")


if __name__ == "__main__":
    test_friend_accuracy()
