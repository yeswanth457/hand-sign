"""
Phase 12: Complete 22-Class Regression Test.
Evaluates all 22 classes on the held-out test split and dataset tokens.
Specifically tracks Father vs School, Father vs Mother, Father vs Sister, Father vs Brother, Father vs Water.
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MAX_SEQ_LEN, DATASET_DIR
)
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.build_real_split import resample_tokens

def main():
    print("=" * 80)
    print("PHASE 12: COMPLETE 22-CLASS REGRESSION TEST")
    print("=" * 80)

    engine = CNNGRUInferenceEngine()

    test_x = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
    test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))

    print(f"Total test set samples: {len(test_x)}")

    class_results = []
    overall_correct = 0
    overall_total = 0

    print(f"\n{'Index':<6} | {'Class Name':<14} | {'Test N':<8} | {'Correct':<8} | {'Test Acc %':<12} | {'Father Confusion'}")
    print("-" * 80)

    for cid, cname in enumerate(CLASS_NAMES):
        indices = np.where(test_y == cid)[0]
        n_test = len(indices)

        if n_test == 0:
            # Fallback to direct token files if class had zero test split samples
            t_files = glob.glob(f"dataset/tokens/{cname}/**/*.npz", recursive=True)
            n_test = len(t_files)
            corr = 0
            f_conf = 0
            for f in t_files:
                d = np.load(f)
                t25 = resample_tokens(d["tokens"], 25)
                p = engine.predict_sequence(t25)
                if p["word"] == cname:
                    corr += 1
                if p["word"] == "father" and cname != "father":
                    f_conf += 1
            acc = (corr / max(1, n_test)) * 100
            note = f"0/{n_test} (0.0%)" if f_conf == 0 else f"{f_conf}/{n_test} ({f_conf/n_test*100:.1f}%)"
            print(f"{cid:<6d} | {cname:<14} | {n_test:<8d}*| {corr:<8d} | {acc:6.1f}%       | {note}")
            class_results.append((cid, cname, n_test, corr, acc, f_conf))
            overall_correct += corr
            overall_total += n_test
            continue

        corr = 0
        f_conf = 0
        for idx in indices:
            sample = test_x[idx]
            pred = engine.predict_sequence(sample)
            if pred["word"] == cname:
                corr += 1
            if pred["word"] == "father" and cname != "father":
                f_conf += 1

        acc = (corr / n_test) * 100
        overall_correct += corr
        overall_total += n_test
        note = f"0/{n_test} (0.0%)" if f_conf == 0 else f"{f_conf}/{n_test} ({f_conf/n_test*100:.1f}%)"
        print(f"{cid:<6d} | {cname:<14} | {n_test:<8d} | {corr:<8d} | {acc:6.1f}%       | {note}")
        class_results.append((cid, cname, n_test, corr, acc, f_conf))

    print("-" * 80)
    print(f"OVERALL HELD-OUT TEST ACCURACY: {overall_correct}/{overall_total} ({(overall_correct/overall_total)*100:.2f}%)")

    # Key pairwise comparisons
    print("\n" + "=" * 80)
    print("KEY PAIRWISE ISOLATION CHECKS:")
    print("=" * 80)
    for cname in ["brother", "water", "school", "mother", "sister"]:
        cid = CLASS_TO_INDEX[cname]
        row = [r for r in class_results if r[1] == cname][0]
        print(f"  {cname.title():<10} -> Father False Positives: {row[5]}/{row[2]} ({(row[5]/max(1,row[2]))*100:.1f}%)  [Zero FP: {'YES' if row[5] == 0 else 'NO'}]")

    print("=" * 80)

if __name__ == "__main__":
    main()
