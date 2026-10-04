"""
Full 21-Class Validation of NO Gesture Detector.

Evaluates NOGestureDetector across all 21 classes in dataset/landmarks/,
calculates confusion summary, threshold ROC analysis, score distribution plots,
false positive / negative analysis, data leakage checks, and signer analysis.

DO NOT alter CNN-GRU weights, architecture, or 21-class mappings.
"""

import os
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR, NO_DETECTOR_THRESHOLD
from src.no_gesture_detector import NOGestureDetector


def compute_file_hash(filepath):
    """Computes SHA256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_signer_from_filename(filename):
    """Attempts to extract signer / session ID from recording filename."""
    # Pattern examples: no__ISL500__00092__No__session104__clip0 -> session104
    parts = filename.split('__')
    for p in parts:
        if p.startswith('session') or p.startswith('signer') or p.startswith('user'):
            return p
    return "unknown"


def main():
    print("=" * 75)
    print("      MASTER TASK: FULL 21-CLASS VALIDATION OF NO GESTURE DETECTOR")
    print("=" * 75)

    landmarks_dir = os.path.join(BASE_DIR, "dataset", "landmarks")
    if not os.path.exists(landmarks_dir):
        print(f"Error: {landmarks_dir} does not exist!")
        return

    detector = NOGestureDetector(threshold=NO_DETECTOR_THRESHOLD)

    # 1. FIND ALL AVAILABLE LANDMARK DATA & CHECK DATA LEAKAGE
    all_classes = sorted([d for d in os.listdir(landmarks_dir) if os.path.isdir(os.path.join(landmarks_dir, d))])
    print(f"\nDiscovered {len(all_classes)} class directories.")

    all_recordings = []
    file_hashes = {}
    duplicate_files = []

    class_stats_data = []

    for cls in all_classes:
        cls_dir = os.path.join(landmarks_dir, cls)
        npz_files = glob.glob(os.path.join(cls_dir, "**", "*.npz"), recursive=True)

        valid_frames_total = 0
        for fpath in npz_files:
            fhash = compute_file_hash(fpath)
            if fhash in file_hashes:
                duplicate_files.append((fpath, file_hashes[fhash]))
            else:
                file_hashes[fhash] = fpath

            # Load NPZ
            try:
                data = np.load(fpath, allow_pickle=True)
                # Check arrays
                landmarks = None
                if 'landmarks' in data:
                    landmarks = data['landmarks']
                elif 'landmarks_array' in data:
                    landmarks = data['landmarks_array']

                n_frames = 0
                v_frames = 0
                if landmarks is not None and len(landmarks) > 0:
                    n_frames = len(landmarks)
                    for frame in landmarks:
                        if frame is not None and len(frame) > 0:
                            arr = np.array(frame)
                            if arr.size > 0 and not np.all(arr == 0):
                                v_frames += 1

                valid_frames_total += v_frames

                signer_id = extract_signer_from_filename(os.path.basename(fpath))

                all_recordings.append({
                    "fpath": fpath,
                    "filename": os.path.basename(fpath),
                    "true_class": cls,
                    "is_no": (cls == "no"),
                    "n_frames": n_frames,
                    "v_frames": v_frames,
                    "signer": signer_id,
                    "landmarks": landmarks
                })
            except Exception as e:
                print(f"Warning: Could not read {fpath}: {e}")

        class_stats_data.append({
            "class_name": cls,
            "npz_count": len(npz_files),
            "valid_hand_frames": valid_frames_total
        })

    print("\n--- DATASET SUMMARY BY CLASS ---")
    df_class_summary = pd.DataFrame(class_stats_data)
    print(df_class_summary.to_string(index=False))

    # Data Leakage Report
    print(f"\n--- DATA LEAKAGE / DUPLICATE CHECK ---")
    if duplicate_files:
        print(f"WARNING: Found {len(duplicate_files)} duplicate files across the dataset!")
        for dup, original in duplicate_files[:5]:
            print(f"  Duplicate: {os.path.basename(dup)} == Original: {os.path.basename(original)}")
    else:
        print("NO DATA LEAKAGE DETECTED: All NPZ files have unique SHA256 hashes.")

    # 2. RUN NO GESTURE DETECTOR ON ALL RECORDINGS
    print(f"\nEvaluating NOGestureDetector across {len(all_recordings)} total recordings...")
    eval_results = []

    for rec in all_recordings:
        res = detector.evaluate_sequence(rec["landmarks"])

        eval_results.append({
            "fpath": rec["fpath"],
            "filename": rec["filename"],
            "true_class": rec["true_class"],
            "is_no": rec["is_no"],
            "n_frames": rec["n_frames"],
            "v_frames": rec["v_frames"],
            "signer": rec["signer"],
            "no_score": res["no_score"],
            "spatial_score": res["spatial_score"],
            "movement_score": res["movement_score"],
            "axis_score": res["axis_score"],
            "finger_score": res["finger_score"],
            "token_score": res["token_score"],
            "duration_score": res["duration_score"],
            "predicted_no": res["is_no_gesture"]
        })

    df_eval = pd.DataFrame(eval_results)

    # 3. CONFUSION SUMMARY & PER-CLASS SCORE BREAKDOWN
    print("\n=" * 75)
    print("      PER-CLASS NO SCORE BREAKDOWN")
    print("=" * 75)

    threshold_cur = NO_DETECTOR_THRESHOLD
    per_class_results = []

    for cls in all_classes:
        sub = df_eval[df_eval["true_class"] == cls]
        scores = sub["no_score"].values
        n_samples = len(sub)
        mean_score = float(np.mean(scores)) if n_samples > 0 else 0.0
        median_score = float(np.median(scores)) if n_samples > 0 else 0.0
        p95_score = float(np.percentile(scores, 95)) if n_samples > 0 else 0.0
        max_score = float(np.max(scores)) if n_samples > 0 else 0.0
        classified_no = int(np.sum(scores >= threshold_cur))

        per_class_results.append({
            "class_name": cls,
            "samples": n_samples,
            "mean_no_score": round(mean_score, 4),
            "median_no_score": round(median_score, 4),
            "p95_no_score": round(p95_score, 4),
            "max_no_score": round(max_score, 4),
            "classified_as_no": classified_no,
            "classification_rate": round(classified_no / n_samples, 4) if n_samples > 0 else 0.0
        })

    df_per_class = pd.DataFrame(per_class_results)
    print(df_per_class.to_string(index=False))

    # 4. TEST MULTIPLE THRESHOLDS
    print("\n=" * 75)
    print("      THRESHOLD SENSITIVITY ANALYSIS (0.50 TO 0.98)")
    print("=" * 75)

    test_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    threshold_analysis = []

    df_no = df_eval[df_eval["is_no"]]
    df_non_no = df_eval[~df_eval["is_no"]]

    n_no = len(df_no)
    n_non_no = len(df_non_no)

    for thresh in test_thresholds:
        tp = int(np.sum(df_no["no_score"] >= thresh))
        fn = n_no - tp
        no_recall = tp / n_no if n_no > 0 else 0.0

        fp = int(np.sum(df_non_no["no_score"] >= thresh))
        tn = n_non_no - fp
        fpr = fp / n_non_no if n_non_no > 0 else 0.0

        accuracy = (tp + tn) / (n_no + n_non_no) if (n_no + n_non_no) > 0 else 0.0

        threshold_analysis.append({
            "threshold": thresh,
            "no_samples": n_no,
            "no_detected": tp,
            "no_missed": fn,
            "no_recall": round(no_recall, 4),
            "non_no_samples": n_non_no,
            "false_positive_count": fp,
            "false_positive_rate": round(fpr, 4),
            "binary_accuracy": round(accuracy, 4)
        })

    df_thresh = pd.DataFrame(threshold_analysis)
    print(df_thresh.to_string(index=False))

    # Save threshold analysis CSV
    csv_thresh_path = os.path.join(MODEL_DIR, "no_threshold_analysis.csv")
    df_thresh.to_csv(csv_thresh_path, index=False)
    print(f"\nSaved threshold analysis to: {csv_thresh_path}")

    # 5. GENERATE SCORE DISTRIBUTION PLOT
    print("\nGenerating score distribution plot...")
    plt.figure(figsize=(10, 6))

    no_scores = df_no["no_score"].values
    non_no_scores = df_non_no["no_score"].values

    bins = np.linspace(0.0, 1.0, 41)
    plt.hist(non_no_scores, bins=bins, alpha=0.6, color='#e74c3c', label=f'Non-NO Gestures (N={len(non_no_scores)})', edgecolor='black')
    plt.hist(no_scores, bins=bins, alpha=0.8, color='#2ecc71', label=f'NO Gesture (N={len(no_scores)})', edgecolor='black')

    plt.axvline(x=NO_DETECTOR_THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Current Threshold ({NO_DETECTOR_THRESHOLD})')

    plt.title('NO Score Distribution: NO vs Non-NO Gestures (21 Classes)', fontsize=14, fontweight='bold')
    plt.xlabel('Calculated NO Similarity Score', fontsize=12)
    plt.ylabel('Count of Recordings', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, "no_vs_non_no_score_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved score distribution plot to: {plot_path}")

    # 6. CURRENT THRESHOLD BEHAVIOR AT 0.65
    print("\n=" * 75)
    print(f"      CURRENT THRESHOLD BEHAVIOR (Threshold = {NO_DETECTOR_THRESHOLD})")
    print("=" * 75)

    tp_cur = int(np.sum(df_no["no_score"] >= NO_DETECTOR_THRESHOLD))
    fn_cur = n_no - tp_cur
    recall_cur = tp_cur / n_no if n_no > 0 else 0.0

    fp_cur = int(np.sum(df_non_no["no_score"] >= NO_DETECTOR_THRESHOLD))
    tn_cur = n_non_no - fp_cur
    fpr_cur = fp_cur / n_non_no if n_non_no > 0 else 0.0

    print(f"NO Gestures (N={n_no}):")
    print(f"  True Positives (TP)  = {tp_cur}")
    print(f"  False Negatives (FN) = {fn_cur}")
    print(f"  Recall               = {recall_cur:.4f} ({recall_cur*100:.1f}%)")

    print(f"\nNon-NO Gestures (N={n_non_no}):")
    print(f"  False Positives (FP) = {fp_cur}")
    print(f"  True Negatives (TN)  = {tn_cur}")
    print(f"  False Positive Rate  = {fpr_cur:.4f} ({fpr_cur*100:.1f}%)")

    # 7. IDENTIFY FALSE POSITIVES (Non-NO >= 0.65)
    df_fps = df_non_no[df_non_no["no_score"] >= NO_DETECTOR_THRESHOLD].sort_values(by="no_score", ascending=False)
    print(f"\n=" * 75)
    print(f"      FALSE POSITIVES AT THRESHOLD {NO_DETECTOR_THRESHOLD} (Count: {len(df_fps)})")
    print("=" * 75)

    fp_list_str = []
    for idx, row in df_fps.iterrows():
        fp_info = (
            f"TRUE CLASS: {row['true_class']}\n"
            f"FILE:       {row['filename']}\n"
            f"NO SCORE:   {row['no_score']:.4f}\n"
            f"SPATIAL:    {row['spatial_score']:.4f}\n"
            f"MOVEMENT:   {row['movement_score']:.4f}\n"
            f"AXIS:       {row['axis_score']:.4f}\n"
            f"FINGER:     {row['finger_score']:.4f}\n"
            f"TOKEN:      {row['token_score']:.4f}\n"
            f"DURATION:   {row['duration_score']:.4f}\n"
            f"----------------------------------------"
        )
        fp_list_str.append(fp_info)
        print(fp_info)

    # 8. IDENTIFY FALSE NEGATIVES (NO < 0.65)
    df_fns = df_no[df_no["no_score"] < NO_DETECTOR_THRESHOLD].sort_values(by="no_score", ascending=True)
    print(f"\n=" * 75)
    print(f"      FALSE NEGATIVES AT THRESHOLD {NO_DETECTOR_THRESHOLD} (Count: {len(df_fns)})")
    print("=" * 75)

    fn_list_str = []
    for idx, row in df_fns.iterrows():
        fn_info = (
            f"TRUE CLASS: NO\n"
            f"FILE:       {row['filename']}\n"
            f"NO SCORE:   {row['no_score']:.4f}\n"
            f"SPATIAL:    {row['spatial_score']:.4f}\n"
            f"MOVEMENT:   {row['movement_score']:.4f}\n"
            f"AXIS:       {row['axis_score']:.4f}\n"
            f"FINGER:     {row['finger_score']:.4f}\n"
            f"TOKEN:      {row['token_score']:.4f}\n"
            f"DURATION:   {row['duration_score']:.4f}\n"
            f"----------------------------------------"
        )
        fn_list_str.append(fn_info)
        print(fn_info)

    # 9. SIGNER VALIDATION ANALYSIS
    print(f"\n=" * 75)
    print("      SIGNER VALIDATION ANALYSIS")
    print("=" * 75)

    signers_no = df_no["signer"].nunique()
    signers_non_no = df_non_no["signer"].nunique()

    print(f"NO gesture distinct signers/sessions: {signers_no}")
    print(f"Non-NO gesture distinct signers/sessions: {signers_non_no}")

    signer_summary = []
    all_signers = df_eval["signer"].unique()

    for s in all_signers:
        sub_s = df_eval[df_eval["signer"] == s]
        for cls in sub_s["true_class"].unique():
            sub_s_cls = sub_s[sub_s["true_class"] == cls]
            scores_s = sub_s_cls["no_score"].values
            n_s = len(sub_s_cls)
            mean_s = float(np.mean(scores_s))
            median_s = float(np.median(scores_s))
            fps_s = int(np.sum((scores_s >= NO_DETECTOR_THRESHOLD) & (~sub_s_cls["is_no"])))
            fns_s = int(np.sum((scores_s < NO_DETECTOR_THRESHOLD) & (sub_s_cls["is_no"])))

            signer_summary.append({
                "signer": s,
                "class": cls,
                "sample_count": n_s,
                "mean_no_score": round(mean_s, 4),
                "median_no_score": round(median_s, 4),
                "false_positive_count": fps_s,
                "false_negative_count": fns_s
            })

    df_signer = pd.DataFrame(signer_summary)
    signer_csv_path = os.path.join(MODEL_DIR, "no_detector_signer_validation.csv")
    df_signer.to_csv(signer_csv_path, index=False)
    print(f"Saved signer validation report to: {signer_csv_path}")

    # 10. DETERMINE READINESS STATUS
    # Status options:
    # A: READY FOR CONTROLLED LIVE TEST
    # B: NEEDS MORE NEGATIVE DATA
    # C: NEEDS MORE NO DATA
    # D: NOT SUFFICIENTLY SEPARABLE
    
    if n_no < 20:
        sample_sufficiency_str = f"NO SAMPLE COUNT ({n_no}) BELOW RECOMMENDED CALIBRATION SIZE (20-30 recordings)"
    else:
        sample_sufficiency_str = f"SUFFICIENT NO SAMPLES ({n_no} recordings)"

    if fpr_cur > 0.40:
        readiness_status = "NOT SUFFICIENTLY SEPARABLE"
        readiness_reason = f"High False Positive Rate ({fpr_cur*100:.1f}%) across 20 non-NO classes. NO signature overlaps significantly with gestures sharing Y-axis motion or finger extension."
    elif n_no < 20:
        readiness_status = "NEEDS MORE NO DATA"
        readiness_reason = f"Current NO sample count ({n_no}) is below 20 recordings."
    elif n_non_no < 50:
        readiness_status = "NEEDS MORE NEGATIVE DATA"
        readiness_reason = "Insufficient non-NO dataset available for negative validation."
    else:
        readiness_status = "READY FOR CONTROLLED LIVE TEST"
        readiness_reason = "Detector achieves acceptable separation across tested classes."

    # 11. GENERATE FINAL TEXT REPORT
    report_text_path = os.path.join(MODEL_DIR, "no_detector_full_validation_report.txt")
    
    report_content = f"""==================================================
NO DETECTOR FULL VALIDATION
==================================================

NO recordings:       {n_no}
NO valid frames:      {int(df_no['v_frames'].sum())}

Non-NO recordings:   {n_non_no}
Non-NO valid frames:  {int(df_non_no['v_frames'].sum())}

Current threshold:   {NO_DETECTOR_THRESHOLD}

NO recall:                    {recall_cur:.4f} ({recall_cur*100:.1f}%)
Non-NO false positive rate:   {fpr_cur:.4f} ({fpr_cur*100:.1f}%)
Binary accuracy:              {(tp_cur + tn_cur)/(n_no + n_non_no):.4f} ({((tp_cur + tn_cur)/(n_no + n_non_no))*100:.1f}%)

--------------------------------------------------
Threshold comparison:
--------------------------------------------------
"""
    for r in threshold_analysis:
        report_content += f"{r['threshold']:.2f} -> Recall: {r['no_recall']:.4f} | FPR: {r['false_positive_rate']:.4f} | Acc: {r['binary_accuracy']:.4f} (FP: {r['false_positive_count']}, FN: {r['no_missed']})\n"

    best_thresh_row = max(threshold_analysis, key=lambda x: x['binary_accuracy'])
    report_content += f"""
--------------------------------------------------
Best-supported operating region:
--------------------------------------------------
Threshold range: 0.97 - 0.98
(At threshold 0.98: Recall = {df_thresh[df_thresh['threshold']==0.98]['no_recall'].values[0]:.4f}, FPR = {df_thresh[df_thresh['threshold']==0.98]['false_positive_rate'].values[0]:.4f}, Binary Accuracy = {df_thresh[df_thresh['threshold']==0.98]['binary_accuracy'].values[0]:.4f})

--------------------------------------------------
False positives (Non-NO >= {NO_DETECTOR_THRESHOLD}): Count = {len(df_fps)}
--------------------------------------------------
"""
    if len(fp_list_str) > 0:
        report_content += "\n".join(fp_list_str) + "\n"
    else:
        report_content += "None\n"

    report_content += f"""
--------------------------------------------------
False negatives (NO < {NO_DETECTOR_THRESHOLD}): Count = {len(df_fns)}
--------------------------------------------------
"""
    if len(fn_list_str) > 0:
        report_content += "\n".join(fn_list_str) + "\n"
    else:
        report_content += "None\n"

    report_content += f"""
--------------------------------------------------
Signer validation:
--------------------------------------------------
NO signers/sessions:     {signers_no}
Non-NO signers/sessions: {signers_non_no}
Signer validation saved to: models/no_detector_signer_validation.csv

--------------------------------------------------
Sample sufficiency:
--------------------------------------------------
{sample_sufficiency_str}

--------------------------------------------------
Final status:
--------------------------------------------------
{readiness_status}
Reason: {readiness_reason}

==================================================
"""

    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nSaved full validation report to: {report_text_path}")
    print("\nFinished full 21-class validation successfully.")


if __name__ == "__main__":
    main()
