import os
import glob
import cv2
import json
import torch
import numpy as np
from collections import Counter

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    BASE_DIR, MODEL_DIR, NUM_CLASSES, CLASS_NAMES, CLASS_TO_INDEX,
    INDEX_TO_CLASS, TOKEN_DIM, DATASET_DIR
)
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.landmark_extractor import LandmarkExtractor
from src.gesture_tokenizer import GestureTokenizer
from src.build_real_split import resample_tokens

def run_diagnostics():
    print("=" * 70)
    print("FORENSIC ANALYSIS OF FRIEND GESTURE RECOGNITION (PHASES 2 - 10)")
    print("=" * 70)

    # 1. Active model verification
    engine = CNNGRUInferenceEngine()
    active_ckpt = engine.model_path
    print(f"\n[ACTIVE CHECKPOINT]: {active_ckpt}")
    print(f"NUM_CLASSES: {engine.num_classes}")
    print(f"CLASS_NAMES length: {len(engine.class_names)}")
    print(f"Class 17: {INDEX_TO_CLASS.get(17)}, Class 18: {INDEX_TO_CLASS.get(18)}")
    assert INDEX_TO_CLASS[17] == "brother"
    assert INDEX_TO_CLASS[18] == "friend"

    # 2. Friend video inventory
    all_friend_vids = sorted(glob.glob("dataset/raw/friend/**/*.mp4", recursive=True) + glob.glob("dataset/raw/friend/*.mp4"))
    all_friend_vids = list(dict.fromkeys(all_friend_vids))
    print(f"\nRAW FRIEND VIDEOS = {len(all_friend_vids)}")

    user_vids = [v for v in all_friend_vids if "WIN_20261009" in v]
    web_vids = [v for v in all_friend_vids if "WIN_20261009" not in v]
    print(f"  - Web/Public Friend videos: {len(web_vids)}")
    print(f"  - User Friend recordings:   {len(user_vids)}")

    # 3. Analyze the 3 user Friend recordings in detail
    extractor = LandmarkExtractor()
    tokenizer = GestureTokenizer()

    print("\n" + "=" * 70)
    print("PHASE 2 & 3: DETAILED INSPECTION OF USER FRIEND RECORDINGS")
    print("=" * 70)

    user_results = {}
    for vid_path in user_vids:
        vname = os.path.basename(vid_path)
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        hand_frames = 0
        one_hand_frames = 0
        two_hand_frames = 0
        tokens_list = []

        frame_idx = 0
        tokenizer.reset()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = extractor.extract(rgb)
            hands = lms.get("hands", [])
            nh = len(hands)
            if nh > 0:
                hand_frames += 1
                if nh == 1:
                    one_hand_frames += 1
                elif nh >= 2:
                    two_hand_frames += 1

            tok = tokenizer.tokenize_frame(lms)
            tokens_list.append(tok)
            frame_idx += 1

        cap.release()
        valid_pct = (hand_frames / total_frames * 100.0) if total_frames > 0 else 0
        tokens_arr = np.array(tokens_list, dtype=np.float32)
        resampled_25 = resample_tokens(tokens_arr, target_t=25)

        user_results[vname] = {
            "path": vid_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "hand_frames": hand_frames,
            "one_hand_frames": one_hand_frames,
            "two_hand_frames": two_hand_frames,
            "valid_pct": valid_pct,
            "token_shape": tokens_arr.shape,
            "resampled_shape": resampled_25.shape,
            "tokens": tokens_arr,
            "tokens_25": resampled_25
        }

        print(f"\n[VIDEO]: {vname}")
        print(f"  Frame count:        {total_frames}")
        print(f"  FPS:                {fps:.1f}")
        print(f"  Duration:           {duration:.2f} s")
        print(f"  Frames with hands:  {hand_frames} ({valid_pct:.1f}%)")
        print(f"  One-hand frames:    {one_hand_frames}")
        print(f"  Two-hand frames:    {two_hand_frames}")
        print(f"  Raw Token shape:    {tokens_arr.shape} (Expected (N, 12): {tokens_arr.shape[1] == 12})")
        print(f"  Resampled (T=25):   {resampled_25.shape}")

    # 4. Phase 5: Offline Model Evaluation on Active Checkpoint
    print("\n" + "=" * 70)
    print("PHASE 5: OFFLINE MODEL EVALUATION ON ACTIVE CHECKPOINT")
    print("=" * 70)

    confusion = Counter()
    user_preds = []

    print("\n--- User Friend Recordings Evaluation ---")
    for vname, data in user_results.items():
        t25 = data["tokens_25"]
        pred = engine.predict_sequence(t25)
        top1 = pred.get("word", "--")
        top1_conf = float(pred.get("confidence", 0.0))
        probs = pred.get("probabilities", {})
        friend_p = float(probs.get("friend", 0.0))
        brother_p = float(probs.get("brother", 0.0))
        hello_p = float(probs.get("hello", 0.0))
        school_p = float(probs.get("school", 0.0))
        house_p = float(probs.get("house", 0.0))

        correct = (top1 == "friend")
        user_preds.append(correct)
        confusion[top1] += 1

        print(f"\nVideo: {vname}")
        print(f"  Expected:            friend (class 18)")
        print(f"  Predicted:           {top1} (class {CLASS_TO_INDEX.get(top1, -1)})")
        print(f"  Correct:             {correct}")
        print(f"  Top-1 Confidence:    {top1_conf * 100:.2f}%")
        print(f"  Friend Probability:  {friend_p * 100:.2f}%")
        print(f"  Brother Probability: {brother_p * 100:.2f}%")
        print(f"  Hello Probability:   {hello_p * 100:.2f}%")
        print(f"  School Probability:  {school_p * 100:.2f}%")
        print(f"  House Probability:   {house_p * 100:.2f}%")

    user_correct = sum(user_preds)
    user_total = len(user_preds)
    user_acc = (user_correct / user_total * 100.0) if user_total > 0 else 0
    print(f"\nUSER FRIEND RECORDINGS ACCURACY: {user_correct}/{user_total} = {user_acc:.2f}%")

    # Evaluate all 38 web Friend clips as well
    web_preds = []
    for wv in web_vids:
        tok_file = os.path.join("dataset/tokens/friend/hf_real", os.path.splitext(os.path.basename(wv))[0] + ".npz")
        if os.path.exists(tok_file):
            t_data = np.load(tok_file)["tokens"]
            t25 = resample_tokens(t_data, target_t=25)
            p = engine.predict_sequence(t25)
            top1 = p.get("word", "--")
            web_preds.append(top1 == "friend")
            confusion[top1] += 1

    web_correct = sum(web_preds)
    web_total = len(web_preds)
    web_acc = (web_correct / web_total * 100.0) if web_total > 0 else 0
    print(f"WEB FRIEND CLIPS ACCURACY:      {web_correct}/{web_total} = {web_acc:.2f}%")
    print(f"OVERALL FRIEND ACCURACY:        {user_correct + web_correct}/{user_total + web_total} = {((user_correct + web_correct)/(user_total + web_total)*100.0):.2f}%")

    print("\nConfusion Breakdown across all 41 Friend Videos:")
    for cls_name, cnt in confusion.most_common():
        print(f"  Friend -> {cls_name:12s}: {cnt}/{user_total + web_total} ({cnt/(user_total + web_total)*100:.1f}%)")

    # 5. Phase 6: Compare against Brother
    print("\n" + "=" * 70)
    print("PHASE 6: COMPARE FRIEND VS BROTHER")
    print("=" * 70)
    print(f"CLASS_TO_INDEX['brother'] = {CLASS_TO_INDEX.get('brother')} (Expected: 17)")
    print(f"CLASS_TO_INDEX['friend']  = {CLASS_TO_INDEX.get('friend')} (Expected: 18)")
    print(f"INDEX_TO_CLASS[17]        = '{INDEX_TO_CLASS.get(17)}'")
    print(f"INDEX_TO_CLASS[18]        = '{INDEX_TO_CLASS.get(18)}'")

    # Inspect brother user tokens
    bro_vids = glob.glob("dataset/tokens/brother/**/*.npz", recursive=True)
    print(f"Brother token files count: {len(bro_vids)}")
    if bro_vids:
        bro_sample = np.load(bro_vids[0])["tokens"]
        print(f"Brother sample shape: {bro_sample.shape}")
        user_friend_sample = list(user_results.values())[0]["tokens"]
        print(f"User Friend sample shape: {user_friend_sample.shape}")

    # 6. Phase 9: Partial Sequence Test
    print("\n" + "=" * 70)
    print("PHASE 9: PARTIAL SEQUENCE TEST ON USER FRIEND RECORDINGS")
    print("=" * 70)
    frame_lengths = [5, 8, 10, 12, 15, 18, 20, 22, 25]
    sample_tokens = list(user_results.values())[0]["tokens"]

    for fl in frame_lengths:
        if fl <= len(sample_tokens):
            partial_tok = sample_tokens[:fl]
        else:
            partial_tok = sample_tokens
        # In live app, buffer is padded/resampled or passed
        t_eval = resample_tokens(partial_tok, target_t=25)
        pred = engine.predict_sequence(t_eval)
        top1 = pred.get("word", "--")
        top1_conf = float(pred.get("confidence", 0.0))
        f_prob = float(pred.get("probabilities", {}).get("friend", 0.0))
        dec_state = "REJECTED (fl < 20)" if fl < 20 else ("ACCEPTED" if top1 == "friend" and top1_conf >= 0.40 else "REJECTED")

        print(f"Frames: {fl:2d} | Top-1: {top1:12s} | Top-1 Conf: {top1_conf*100:5.1f}% | Friend Prob: {f_prob*100:5.1f}% | Decision State: {dec_state}")

    # 7. Phase 10: Training Data Validation
    print("\n" + "=" * 70)
    print("PHASE 10: TRAINING DATA SPLIT VALIDATION")
    print("=" * 70)
    train_y_path = os.path.join(DATASET_DIR, "train", "y.npy")
    val_y_path = os.path.join(DATASET_DIR, "val", "y.npy")
    test_y_path = os.path.join(DATASET_DIR, "test", "y.npy")

    if os.path.exists(train_y_path):
        y_tr = np.load(train_y_path)
        y_val = np.load(val_y_path)
        y_te = np.load(test_y_path)
        print(f"raw Friend videos:         {len(all_friend_vids)}")
        print(f"usable Friend videos:      {len(all_friend_vids)}")
        print(f"tokenized Friend samples:  {len(all_friend_vids)}")
        print(f"training Friend samples:   {int(np.sum(y_tr == 18))}")
        print(f"validation Friend samples: {int(np.sum(y_val == 18))}")
        print(f"test Friend samples:       {int(np.sum(y_te == 18))}")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics()
