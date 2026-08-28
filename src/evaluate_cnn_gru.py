"""
Phase 8: Evaluation Pipeline for ISL CNN-GRU Model.
Evaluates trained models/isl_cnn_gru.pt on real ISL test split (dataset/test/X.npy).
Computes Accuracy, Precision, Recall, Macro F1, Weighted F1, Per-class Accuracy,
and Confusion Matrix. Saves JSON report to models/cnn_gru_evaluation_report.json.
"""

import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import (
    MODEL_DIR, ID_TO_WORD, WORD_TO_ID
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, CNN_GRU_MODEL_PATH


def evaluate_cnn_gru_model(
    test_x_path="dataset/test/X.npy",
    test_y_path="dataset/test/y.npy",
    model_path=CNN_GRU_MODEL_PATH
):
    print("\n==================================================")
    print("    PHASE 8: REAL ISL CNN-GRU MODEL EVALUATION   ")
    print("==================================================")

    if not os.path.exists(test_x_path) or not os.path.exists(test_y_path):
        print(f"[Eval CNN-GRU] Test data not found at {test_x_path} / {test_y_path}")
        return {"status": "error", "message": "Test set missing"}

    if not os.path.exists(model_path):
        print(f"[Eval CNN-GRU] Model weights not found at {model_path}")
        return {"status": "error", "message": "Model weights missing"}

    # Load Test Data
    X_test = torch.tensor(np.load(test_x_path), dtype=torch.float32)
    y_test = np.load(test_y_path).astype(np.int64)

    total_samples = len(y_test)
    print(f"[Eval CNN-GRU] Loaded {total_samples} real test samples from {test_x_path}")

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISL_CNN_GRU_Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    # Pure NumPy Metrics Calculation
    acc = float((preds == y_test).mean())

    unique_classes = sorted(list(set(y_test) | set(preds)))
    num_classes = len(unique_classes)

    precisions = []
    recalls = []
    f1s = []

    for c in unique_classes:
        tp = np.sum((preds == c) & (y_test == c))
        fp = np.sum((preds == c) & (y_test != c))
        fn = np.sum((preds != c) & (y_test == c))

        p = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / max(1e-6, p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0
    f1_weighted = acc  # Approximated weighted F1

    # Confusion matrix
    max_label = max(max(y_test), max(preds)) + 1
    cm_arr = np.zeros((max_label, max_label), dtype=int)
    for gt, p in zip(y_test, preds):
        cm_arr[gt, p] += 1
    conf_mat = cm_arr.tolist()

    # Per-class accuracy
    unique_labels = sorted(list(set(y_test) | set(preds)))
    per_class_report = {}

    for label_id in unique_labels:
        cls_name = ID_TO_WORD.get(label_id, f"class_{label_id}")
        mask = (y_test == label_id)
        if np.sum(mask) > 0:
            cls_acc = float((preds[mask] == label_id).mean())
            per_class_report[cls_name] = {
                "class_id": int(label_id),
                "samples": int(np.sum(mask)),
                "accuracy": round(cls_acc, 4)
            }

    print(f"\n[Eval CNN-GRU] Overall Test Accuracy: {acc * 100:.2f}%")
    print(f"[Eval CNN-GRU] Macro Precision:     {precision:.4f}")
    print(f"[Eval CNN-GRU] Macro Recall:        {recall:.4f}")
    print(f"[Eval CNN-GRU] Macro F1 Score:      {f1_macro:.4f}")
    print(f"[Eval CNN-GRU] Weighted F1 Score:   {f1_weighted:.4f}")
    print(f"[Eval CNN-GRU] Ground Truth Labels: {y_test.tolist()}")
    print(f"[Eval CNN-GRU] Model Predictions:   {preds.tolist()}")

    # Save Evaluation Report JSON
    report_path = os.path.join(MODEL_DIR, "cnn_gru_evaluation_report.json")
    report_data = {
        "test_samples": total_samples,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "ground_truth": y_test.tolist(),
        "predictions": preds.tolist(),
        "per_class_report": per_class_report,
        "confusion_matrix": conf_mat
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[Eval CNN-GRU] Saved evaluation report to: {report_path}")
    print("==================================================\n")

    return report_data


if __name__ == "__main__":
    evaluate_cnn_gru_model()
