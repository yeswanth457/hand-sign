"""
Validation Script for Binary NO Classifier across 7 Specified Classes:
1. NO
2. hello
3. thank_you
4. yes
5. goodbye
6. please
7. stop

Evaluates every available NPZ recording for these 7 classes using the trained
binary model models/no_binary_classifier.pt and reports predictions.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR
from src.gesture_tokenizer import GestureTokenizer
from src.no_binary_model import NOBinaryInferenceEngine, BINARY_MODEL_PATH
from src.build_real_split import resample_tokens


def load_token_matrix(filepath, tokenizer):
    data = np.load(filepath, allow_pickle=True)
    if 'tokens' in data.files:
        raw = data['tokens']
    elif 'landmarks' in data.files:
        raw = tokenizer.tokenize_sequence(data['landmarks'])
    elif 'landmarks_array' in data.files:
        raw = tokenizer.tokenize_sequence(data['landmarks_array'])
    else:
        raise ValueError(f"No landmarks found in {filepath}")

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    return resample_tokens(raw, target_t=25)


def validate_7_classes():
    print("=" * 75)
    print("      BINARY NO CLASSIFIER VALIDATION ON 7 SPECIFIED CLASSES")
    print("=" * 75)

    if not os.path.exists(BINARY_MODEL_PATH):
        print(f"Error: Binary model missing at {BINARY_MODEL_PATH}")
        return

    tokenizer = GestureTokenizer()
    engine = NOBinaryInferenceEngine(model_path=BINARY_MODEL_PATH, threshold=0.60)

    target_classes = ["no", "hello", "thank_you", "yes", "goodbye", "please", "stop"]
    landmarks_dir = os.path.join(BASE_DIR, "dataset", "landmarks")

    all_results = []
    class_summary = []

    for cls in target_classes:
        cls_dir = os.path.join(landmarks_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory missing for class {cls}")
            continue

        files = sorted(glob.glob(os.path.join(cls_dir, "**", "*.npz"), recursive=True))
        print(f"\n--- Class: {cls.upper()} ({len(files)} recordings) ---")

        no_classified_count = 0
        total_count = len(files)

        for f in files:
            fname = os.path.basename(f)
            try:
                tokens = load_token_matrix(f, tokenizer)
                res = engine.predict_sequence(tokens, max_seq_len=25)

                is_no = res["is_no"]
                no_prob = res["no_probability"]
                pred_label = res["prediction"]

                if is_no:
                    no_classified_count += 1

                all_results.append({
                    "class": cls,
                    "filename": fname,
                    "no_probability": no_prob,
                    "prediction": pred_label,
                    "is_correct": (pred_label == "NO") if cls == "no" else (pred_label == "NOT_NO")
                })
                print(f"  {fname:50s} | NO Prob: {no_prob:.4f} | Prediction: {pred_label}")
            except Exception as e:
                print(f"  {fname:50s} | ERROR: {e}")

        rate = (no_classified_count / total_count * 100) if total_count > 0 else 0.0
        class_summary.append({
            "class": cls,
            "recordings": total_count,
            "classified_as_no": no_classified_count,
            "no_percentage": round(rate, 2)
        })

    print("\n=" * 75)
    print("      7-CLASS VALIDATION SUMMARY")
    print("=" * 75)
    df_summary = pd.DataFrame(class_summary)
    print(df_summary.to_string(index=False))

    # Save detailed CSV
    df_details = pd.DataFrame(all_results)
    csv_path = os.path.join(MODEL_DIR, "no_binary_7_class_validation.csv")
    df_details.to_csv(csv_path, index=False)
    print(f"\nSaved detailed 7-class validation results to: {csv_path}")


if __name__ == "__main__":
    validate_7_classes()
