"""
Phase 8: Model Evaluation Pipeline for Real ISL Dataset.
Evaluates trained PyTorch TemporalMemoryTransformer model on real 6D gesture token sequences,
calculating Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

import sys
import os
import json
import csv
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    METADATA_CSV_PATH, TOKENS_DIR, MODEL_PATH, MODEL_DIR,
    TOKEN_DIM, MAX_SEQ_LEN, ISL_VOCABULARY, VOCAB_SIZE
)
from src.temporal_transformer import TemporalMemoryTransformer
from src.train_real_model import load_real_token_dataset, RealISLTokenDataset


def evaluate_real_model(model_path=MODEL_PATH, csv_path=METADATA_CSV_PATH):
    """
    Loads trained PyTorch model and evaluates performance metrics on real ISL gesture token sequences.
    """
    print("\n==================================================")
    print("    PHASE 8: REAL ISL MODEL EVALUATION PIPELINE")
    print("==================================================")

    if not os.path.exists(model_path):
        print(f"[Eval] Error: Model checkpoint not found at {model_path}")
        return None

    token_seqs, labels, class_to_id = load_real_token_dataset(csv_path)
    total_samples = len(token_seqs)

    if total_samples == 0:
        print("[Eval] Error: No real token sequences found for evaluation.")
        return None

    id_to_class = {v: k for k, v in class_to_id.items()}

    # Load Model Weights
    model = TemporalMemoryTransformer(
        token_dim=TOKEN_DIM,
        d_model=64,
        n_heads=4,
        num_layers=2,
        num_classes=VOCAB_SIZE
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Form Dataset Tensors
    dataset = RealISLTokenDataset(token_seqs, labels, max_seq_len=MAX_SEQ_LEN)
    x_tensors = torch.tensor(dataset.samples) # (N, MAX_SEQ_LEN, 6)
    y_true = np.array(labels)

    with torch.no_grad():
        logits = model(x_tensors)
        y_pred = torch.argmax(logits, dim=-1).numpy()

    # Compute Metrics
    correct = np.sum(y_pred == y_true)
    accuracy = float(correct / total_samples)

    # Per-class metrics
    num_classes = len(class_to_id)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    per_class_metrics = {}
    f1_scores = []

    for c_id, c_name in id_to_class.items():
        tp = int(confusion_matrix[c_id, c_id])
        fp = int(np.sum(confusion_matrix[:, c_id]) - tp)
        fn = int(np.sum(confusion_matrix[c_id, :]) - tp)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[c_name] = {
            "class_id": c_id,
            "samples": int(np.sum(y_true == c_id)),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    eval_report = {
        "total_samples": total_samples,
        "num_classes": num_classes,
        "overall_accuracy": round(accuracy, 4),
        "macro_f1_score": round(macro_f1, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist()
    }

    # Save Evaluation JSON Report
    eval_json_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    # Print Summary Report
    print(f"[Eval] Total Evaluation Samples: {total_samples}")
    print(f"[Eval] Overall Accuracy:         {accuracy * 100:.1f}%")
    print(f"[Eval] Macro F1-Score:            {macro_f1 * 100:.1f}%")

    print("\n--- PER-CLASS PERFORMANCE METRICS ---")
    print(f"{'Class Name':<15} | {'Samples':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for c_name, m in per_class_metrics.items():
        print(f"{c_name:<15} | {m['samples']:<8} | {m['precision']*100:<9.1f}% | {m['recall']*100:<9.1f}% | {m['f1_score']*100:<9.1f}%")

    print("\n--- CONFUSION MATRIX ---")
    print("Pred ->", [id_to_class[i] for i in range(num_classes)])
    for i in range(num_classes):
        print(f"True {id_to_class[i]:<10}:", confusion_matrix[i].tolist())

    print(f"\n[Eval] Evaluation report saved to: {eval_json_path}")
    print("==================================================\n")

    return eval_report


if __name__ == "__main__":
    evaluate_real_model()
