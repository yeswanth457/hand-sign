"""
Phase 10: False Positive Testing against Father Classifier.
Evaluates Brother, Water, School, Mother, Sister against Father.
Verifies that other gestures NEVER falsely predict Father, and Father correctly predicts Father.
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MAX_SEQ_LEN, MODEL_DIR
)
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.build_real_split import resample_tokens

def main():
    print("=" * 75)
    print("PHASE 10: FALSE POSITIVE CROSS-EVALUATION")
    print("=" * 75)

    engine = CNNGRUInferenceEngine()

    test_classes = ["father", "brother", "water", "school", "mother", "sister"]
    table_rows = []

    print(f"\n{'Actual Class':<12} | {'Total Files':<12} | {'Predicted Father':<18} | {'False Positive %':<18} | {'Status'}")
    print("-" * 80)

    for cname in test_classes:
        files = glob.glob(f"dataset/tokens/{cname}/**/*.npz", recursive=True)
        if not files:
            continue

        father_pred_count = 0
        correct_class_count = 0
        total = len(files)

        for f in files:
            d = np.load(f)
            tk = d["tokens"]
            if len(tk) < 3:
                continue
            t25 = resample_tokens(tk, 25)
            pred = engine.predict_sequence(t25)
            w = pred["word"]
            if w == "father":
                father_pred_count += 1
            if w == cname:
                correct_class_count += 1

        fp_pct = (father_pred_count / total) * 100
        if cname == "father":
            status = "PASS (Target)" if father_pred_count > 0 else "FAIL"
            print(f"{cname:<12} | {total:<12} | {father_pred_count:<18} | {'N/A (True Class)':<18} | {status}")
        else:
            status = "PASS (Zero FP)" if father_pred_count == 0 else f"FAIL ({father_pred_count} FP)"
            print(f"{cname:<12} | {total:<12} | {father_pred_count:<18} | {fp_pct:5.2f}%            | {status}")

        table_rows.append((cname, total, father_pred_count, correct_class_count))

    print("\n" + "=" * 75)
    print("CONFUSION SUMMARY WITH RESPECT TO FATHER:")
    print("=" * 75)
    for cname, tot, f_cnt, corr in table_rows:
        if cname != "father":
            print(f"  {cname.title():<10} -> Father: {f_cnt}/{tot} ({f_cnt/tot*100:.2f}%)  [Zero False Positives: {'YES' if f_cnt == 0 else 'NO'}]")
        else:
            print(f"  Father     -> Father: {f_cnt}/{tot} ({f_cnt/tot*100:.2f}%)")
    print("=" * 75)

if __name__ == "__main__":
    main()
