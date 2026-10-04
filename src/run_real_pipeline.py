"""
Master Pipeline: Real ISL Dataset → CNN-GRU Training.

Complete end-to-end pipeline that:
1. Calls the HF Dataset API to discover and download real MP4 videos
2. Runs MediaPipe landmark extraction (num_hands=2)
3. Generates 6D gesture tokens [Hx, Hy, Mx, My, Rx, Ry]
4. Builds 25-frame temporal sequences
5. Creates signer-level train/val/test splits
6. Trains the CNN-GRU model
7. Evaluates with confusion matrix and per-class metrics

All data is REAL — no synthetic generation, no fake samples.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from config import (
    RAW_DATA_DIR, DATASET_DIR, MODEL_DIR, MODEL_PATH,
    CLASS_NAMES, CLASS_TO_INDEX, NUM_CLASSES, TOKEN_DIM, MAX_SEQ_LEN
)


def step1_download_from_hf():
    """Step 1: Use HF Dataset API to download real MP4 videos."""
    print("\n" + "=" * 70)
    print("  STEP 1: HuggingFace Dataset API → Download Real Videos")
    print("=" * 70)
    
    from src.hf_dataset_api import run_full_api_pipeline
    result = run_full_api_pipeline(RAW_DATA_DIR)
    
    # Save status
    report_path = os.path.join(DATASET_DIR, "metadata", "hf_api_status.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    
    return result


def step2_extract_landmarks_and_tokens():
    """Step 2: Process all downloaded videos through MediaPipe → 6D tokens."""
    print("\n" + "=" * 70)
    print("  STEP 2: MediaPipe Landmark Extraction → 6D Token Generation")
    print("=" * 70)
    
    from src.real_dataset_pipeline import RealISLDatasetPipeline
    
    pipeline = RealISLDatasetPipeline()
    summary = pipeline.process_dataset()
    pipeline.print_summary_report(summary)
    
    return summary


def step3_build_splits():
    """Step 3: Build signer-level train/val/test splits from extracted tokens."""
    print("\n" + "=" * 70)
    print("  STEP 3: Build Train/Val/Test Splits (Signer-Level Separation)")
    print("=" * 70)
    
    from src.build_real_split import build_real_splits
    build_real_splits()
    
    # Verify splits
    split_info = {}
    for split_name in ["train", "val", "test"]:
        x_path = os.path.join(DATASET_DIR, split_name, "X.npy")
        y_path = os.path.join(DATASET_DIR, split_name, "y.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            X = np.load(x_path)
            y = np.load(y_path)
            split_info[split_name] = {"X_shape": X.shape, "y_shape": y.shape, "count": len(y)}
            
            # Per-class counts
            class_counts = {}
            for cid, cname in enumerate(CLASS_NAMES):
                count = int((y == cid).sum())
                if count > 0:
                    class_counts[cname] = count
            split_info[split_name]["class_counts"] = class_counts
        else:
            split_info[split_name] = {"count": 0}
    
    return split_info


def step4_train_model(epochs=50, lr=0.001, batch_size=16):
    """Step 4: Train the 21-class CNN-GRU model on real data."""
    print("\n" + "=" * 70)
    print("  STEP 4: Train 21-Class CNN-GRU Model on Real ISL Data")
    print("=" * 70)
    
    from src.train_21class_cnn_gru import execute_pipeline
    result = execute_pipeline()
    return result


def train_model(epochs=50, lr=0.001, batch_size=16):
    """Actual training function delegating to 21-class training pipeline."""
    from src.train_21class_cnn_gru import execute_pipeline
    return execute_pipeline()


def step5_evaluate_model():
    """Step 5: Verify evaluation reports and confusion matrix from 21-class training."""
    print("\n" + "=" * 70)
    print("  STEP 5: 21-Class Evaluation & Confusion Matrix")
    print("=" * 70)
    
    report_path = os.path.join(MODEL_DIR, "cnn_gru_21class_training_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "completed"}


def step6_verify_webcam_integration():
    """Step 6: Verify the trained model loads correctly for webcam inference."""
    print("\n" + "=" * 70)
    print("  STEP 6: Verify Webcam Pipeline Integration")
    print("=" * 70)
    
    import torch
    from src.cnn_gru_model import ISL_CNN_GRU_Model, CNNGRUInferenceEngine
    
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return {"status": "error", "message": "Model file not found"}
    
    # Load and verify
    engine = CNNGRUInferenceEngine(MODEL_PATH)
    
    # Test with dummy input
    dummy_seq = np.random.randn(25, 6).astype(np.float32)
    result = engine.predict_sequence(dummy_seq)
    
    print(f"  Model loaded: {engine.model_loaded}")
    print(f"  Output classes: {engine.num_classes}")
    print(f"  Class mapping: {engine.id_to_word}")
    print(f"  Dummy prediction: {result['word']} ({result['confidence']*100:.1f}%)")
    
    # Verify normalization stats
    mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
    std_path = os.path.join(MODEL_DIR, "feature_std.npy")
    norm_ok = os.path.exists(mean_path) and os.path.exists(std_path)
    print(f"  Feature normalization: {'OK' if norm_ok else 'MISSING'}")
    
    return {
        "status": "ok",
        "model_loaded": engine.model_loaded,
        "num_classes": engine.num_classes,
        "normalization": norm_ok,
    }


def generate_final_report(api_result, extract_summary, split_info, train_result, eval_report, webcam_result):
    """Generate comprehensive final report."""
    print("\n" + "=" * 70)
    print("  FINAL PIPELINE REPORT")
    print("=" * 70)
    
    # 1. Dataset API Status
    print(f"\n1. DATASET API STATUS")
    print(f"   Config: {api_result.get('config', 'N/A')}")
    print(f"   Split: {api_result.get('split', 'N/A')}")
    print(f"   Total API rows: {api_result.get('total_api_rows', 0)}")
    
    # 2. Downloaded Videos
    inventory = api_result.get('inventory', {})
    total_videos = api_result.get('total_videos', 0)
    print(f"\n2. DOWNLOADED VIDEOS: {total_videos} total")
    for cls in CLASS_NAMES:
        count = inventory.get(cls, 0)
        print(f"   {cls:15s}: {count:3d}")
    
    # 3. Missing Classes
    still_missing = api_result.get('still_missing', [])
    low_count = api_result.get('low_count', [])
    print(f"\n3. MISSING CLASSES: {still_missing if still_missing else 'None'}")
    if low_count:
        print(f"   LOW COUNT (<3): {low_count}")
    
    # 4. Extracted Sequences
    total_sequences = 0
    if extract_summary:
        total_sequences = extract_summary.get('successful', 0)
    print(f"\n4. EXTRACTED SEQUENCES: {total_sequences}")
    
    # 5. Train/Val/Test Counts
    print(f"\n5. TRAIN/VAL/TEST SPLIT:")
    for split_name in ["train", "val", "test"]:
        info = split_info.get(split_name, {})
        count = info.get('count', 0)
        shape = info.get('X_shape', 'N/A')
        print(f"   {split_name:5s}: {count:4d} samples  (shape: {shape})")
    
    # 6. Evaluation Metrics
    print(f"\n6. EVALUATION METRICS:")
    if eval_report and isinstance(eval_report, dict):
        acc = eval_report.get('accuracy', 0)
        print(f"   Overall Accuracy: {acc*100:.1f}%")
        
        # Per-class precision/recall/F1
        print(f"\n   {'Class':15s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Sup':>5s}")
        print(f"   {'-'*40}")
        for cls in CLASS_NAMES:
            metrics = eval_report.get(cls, {})
            if metrics:
                p = metrics.get('precision', 0)
                r = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                sup = metrics.get('support', 0)
                print(f"   {cls:15s} {p:6.2f} {r:6.2f} {f1:6.2f} {sup:5.0f}")
    else:
        print("   No evaluation results available.")
    
    # 7. Model Output Classes
    print(f"\n7. MODEL OUTPUT CLASSES: {NUM_CLASSES}")
    for i, cls in enumerate(CLASS_NAMES):
        has_data = inventory.get(cls, 0) > 0
        print(f"   [{i:2d}] {cls:15s} {'✓ real data' if has_data else '✗ NO DATA'}")
    
    # 8. Classes needing additional data
    needs_data = [cls for cls in CLASS_NAMES if inventory.get(cls, 0) < 3]
    print(f"\n8. CLASSES REQUIRING ADDITIONAL REAL VIDEO DATA: {needs_data if needs_data else 'None'}")
    
    supported = [cls for cls in CLASS_NAMES if inventory.get(cls, 0) >= 3]
    print(f"\n   Fully supported classes: {len(supported)}/{len(CLASS_NAMES)}")
    
    if len(supported) < len(CLASS_NAMES):
        print(f"\n   *** WARNING: Cannot claim all {len(CLASS_NAMES)} classes are supported ***")
        print(f"   *** {len(CLASS_NAMES) - len(supported)} classes still need real video data ***")
    
    print("\n" + "=" * 70)
    
    return {
        "api_status": "connected",
        "config": api_result.get("config"),
        "split": api_result.get("split"),
        "total_videos": total_videos,
        "inventory": inventory,
        "missing_classes": still_missing,
        "total_sequences": total_sequences,
        "train_count": split_info.get("train", {}).get("count", 0),
        "val_count": split_info.get("val", {}).get("count", 0),
        "test_count": split_info.get("test", {}).get("count", 0),
        "model_classes": NUM_CLASSES,
        "supported_classes": len(supported),
        "needs_data": needs_data,
    }


def run_full_pipeline(skip_download=False, skip_extract=False, epochs=50):
    """Run the complete pipeline end-to-end."""
    t_start = time.time()
    
    print("\n" + "#" * 70)
    print("#  REAL ISL DATASET → CNN-GRU TRAINING PIPELINE")
    print("#  Dataset: vidit031/isl-isolated-40words")
    print("#  Target Classes: 21")
    print("#" * 70)
    
    # Step 1: Download
    api_result = {}
    if not skip_download:
        api_result = step1_download_from_hf()
    else:
        print("\n[SKIP] Step 1: Download (using existing videos)")
        from src.hf_dataset_api import get_final_inventory
        inventory, total = get_final_inventory(RAW_DATA_DIR, CLASS_NAMES)
        api_result = {
            "config": "default", "split": "train",
            "inventory": inventory, "total_videos": total,
            "still_missing": [c for c in CLASS_NAMES if inventory.get(c, 0) == 0],
            "low_count": [c for c in CLASS_NAMES if 0 < inventory.get(c, 0) < 3],
            "total_api_rows": 0,
        }
    
    # Step 2: Extract landmarks & tokens
    extract_summary = {}
    if not skip_extract:
        extract_summary = step2_extract_landmarks_and_tokens()
    else:
        print("\n[SKIP] Step 2: Extraction (using existing tokens)")
    
    # Step 3: Build splits
    split_info = step3_build_splits()
    
    # Check if we have enough data to train
    train_count = split_info.get("train", {}).get("count", 0)
    if train_count == 0:
        print("\n[ERROR] No training data available. Cannot proceed with training.")
        return generate_final_report(api_result, extract_summary, split_info, {}, {}, {})
    
    # Step 4: Train
    train_result = train_model(epochs=epochs)
    
    # Step 5: Evaluate
    eval_report = step5_evaluate_model()
    
    # Step 6: Verify webcam integration
    webcam_result = step6_verify_webcam_integration()
    
    elapsed = time.time() - t_start
    print(f"\nTotal pipeline time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    # Final report
    report = generate_final_report(api_result, extract_summary, split_info, train_result, eval_report, webcam_result)
    
    # Save report
    report_path = os.path.join(DATASET_DIR, "metadata", "pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to: {report_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real ISL Dataset Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip HF download step")
    parser.add_argument("--skip-extract", action="store_true", help="Skip landmark extraction step")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()
    
    run_full_pipeline(
        skip_download=args.skip_download,
        skip_extract=args.skip_extract,
        epochs=args.epochs
    )
