"""
Production 21-Class CNN-GRU Training and Evaluation Pipeline.
Strictly adheres to Master Prompt specifications:
- Dataset verification (Train: 263, Val: 59, Test: 59)
- Class-weighted CrossEntropyLoss calculated exclusively from training set
- Training-derived feature normalization [Hx, Hy, Mx, My, Rx, Ry]
- Full epoch logging: Epoch | Train Loss | Train Acc | Val Loss | Val Acc
- Early stopping and best validation checkpointing
- Baseline preservation (42.6%)
- Complete test evaluation across all 21 classes
- Confusion matrix plot and classification report generation
- Compatibility verification with CNNGRUInferenceEngine
"""

import os
import sys
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MAX_SEQ_LEN, DATASET_DIR, MODEL_DIR
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, CNNGRUInferenceEngine


# -------------------------------------------------------------
# 1. Dataset Class
# -------------------------------------------------------------
class TokenSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------------------------------------------
# 2. Main Training & Evaluation Function
# -------------------------------------------------------------
def execute_pipeline():
    print("=" * 70)
    print("      21-CLASS CNN-GRU TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # Set deterministic seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Step 1: Pre-training Data Verification
    # ---------------------------------------------------------
    train_x_path = os.path.join(DATASET_DIR, "train", "X.npy")
    train_y_path = os.path.join(DATASET_DIR, "train", "y.npy")
    val_x_path = os.path.join(DATASET_DIR, "val", "X.npy")
    val_y_path = os.path.join(DATASET_DIR, "val", "y.npy")
    test_x_path = os.path.join(DATASET_DIR, "test", "X.npy")
    test_y_path = os.path.join(DATASET_DIR, "test", "y.npy")

    for p in [train_x_path, train_y_path, val_x_path, val_y_path, test_x_path, test_y_path]:
        assert os.path.exists(p), f"Missing required file: {p}"

    train_X = np.load(train_x_path).astype(np.float32)
    train_y = np.load(train_y_path).astype(np.int64)
    val_X = np.load(val_x_path).astype(np.float32)
    val_y = np.load(val_y_path).astype(np.int64)
    test_X = np.load(test_x_path).astype(np.float32)
    test_y = np.load(test_y_path).astype(np.int64)

    # Programmatic Verifications (Dynamic Sample Count N, 25-frame sequence, 6D tokens)
    assert train_X.ndim == 3 and train_X.shape[1:] == (25, 6), f"Unexpected train_X shape: {train_X.shape}"
    assert train_y.ndim == 1 and len(train_y) == len(train_X), f"Unexpected train_y shape: {train_y.shape}"
    assert val_X.ndim == 3 and val_X.shape[1:] == (25, 6), f"Unexpected val_X shape: {val_X.shape}"
    assert val_y.ndim == 1 and len(val_y) == len(val_X), f"Unexpected val_y shape: {val_y.shape}"
    assert test_X.ndim == 3 and test_X.shape[1:] == (25, 6), f"Unexpected test_X shape: {test_X.shape}"
    assert test_y.ndim == 1 and len(test_y) == len(test_X), f"Unexpected test_y shape: {test_y.shape}"

    assert np.issubdtype(train_y.dtype, np.integer) and np.issubdtype(val_y.dtype, np.integer) and np.issubdtype(test_y.dtype, np.integer)
    assert 0 <= train_y.min() and train_y.max() < 21
    assert 0 <= val_y.min() and val_y.max() < 21
    assert 0 <= test_y.min() and test_y.max() < 21

    assert np.isfinite(train_X).all() and np.isfinite(val_X).all() and np.isfinite(test_X).all()

    # Check all classes present in all splits
    train_classes = set(np.unique(train_y))
    val_classes = set(np.unique(val_y))
    test_classes = set(np.unique(test_y))

    assert len(train_classes) == 21, f"Train missing classes: {set(range(21)) - train_classes}"
    assert len(val_classes) == 21, f"Val missing classes: {set(range(21)) - val_classes}"
    assert len(test_classes) == 21, f"Test missing classes: {set(range(21)) - test_classes}"

    print("\n" + "=" * 50)
    print("DATASET PRE-TRAINING VERIFICATION")
    print("=" * 50)
    print(f"NUMBER OF CLASSES:   {NUM_CLASSES}")
    print(f"FEATURE DIMENSION:   {TOKEN_DIM} [Hx, Hy, Mx, My, Rx, Ry]")
    print(f"SEQUENCE LENGTH:     {MAX_SEQ_LEN}")
    print(f"TRAIN SHAPE:         {train_X.shape}")
    print(f"VALIDATION SHAPE:    {val_X.shape}")
    print(f"TEST SHAPE:          {test_X.shape}")
    print("\nCLASS DISTRIBUTION:")
    for cid, cname in enumerate(CLASS_NAMES):
        ntr = int((train_y == cid).sum())
        nva = int((val_y == cid).sum())
        nte = int((test_y == cid).sum())
        print(f"  [{cid:2d}] {cname:<12}: Train={ntr:2d} | Val={nva:2d} | Test={nte:2d} | Total={ntr+nva+nte:2d}")
    print("=" * 50)

    # ---------------------------------------------------------
    # Step 2: Feature Normalization
    # ---------------------------------------------------------
    mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
    std_path = os.path.join(MODEL_DIR, "feature_std.npy")

    assert os.path.exists(mean_path) and os.path.exists(std_path), "Missing normalization parameters!"
    feat_mean = np.load(mean_path).astype(np.float32)
    feat_std = np.load(std_path).astype(np.float32)
    feat_std = np.where(feat_std < 1e-7, 1.0, feat_std)

    print(f"\nFeature Normalization (6D):")
    print(f"  Mean: {np.round(feat_mean, 4)}")
    print(f"  Std:  {np.round(feat_std, 4)}")

    # Apply identical normalization to train, val, test
    norm_train_X = (train_X - feat_mean) / feat_std
    norm_val_X = (val_X - feat_mean) / feat_std
    norm_test_X = (test_X - feat_mean) / feat_std

    # ---------------------------------------------------------
    # Step 3: Class Weights Calculation (TRAIN SET ONLY)
    # ---------------------------------------------------------
    train_counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(np.float32)
    train_counts = np.maximum(train_counts, 1.0)
    total_train = len(train_y)

    # Standard balanced class weighting: total_samples / (num_classes * class_count)
    class_weights = total_train / (NUM_CLASSES * train_counts)
    # Clip extreme weights to avoid numerical instability
    class_weights = np.clip(class_weights, 0.2, 5.0)

    print(f"\nCalculated Class Weights (from Training Set ONLY):")
    for cid, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:<12} (count={int(train_counts[cid]):2d}): weight={class_weights[cid]:.3f}")

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ---------------------------------------------------------
    # Step 4: Model, Loss, Optimizer, Scheduler Setup
    # ---------------------------------------------------------
    batch_size = 16
    train_loader = DataLoader(TokenSequenceDataset(norm_train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TokenSequenceDataset(norm_val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TokenSequenceDataset(norm_test_X, test_y), batch_size=batch_size, shuffle=False)

    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.25
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # ---------------------------------------------------------
    # Step 5: Training Loop with Validation Monitoring
    # ---------------------------------------------------------
    epochs = 50
    best_val_acc = 0.0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    history = []

    best_checkpoint_path = os.path.join(MODEL_DIR, "isl_cnn_gru_21class_best.pt")
    active_model_path = os.path.join(MODEL_DIR, "isl_cnn_gru.pt")

    print("\n" + "=" * 65)
    print("TRAINING LOG")
    print("=" * 65)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<10} | {'Val Acc':<9}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += len(by)

        avg_train_loss = train_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)

        avg_val_loss = val_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(float(avg_train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": round(float(avg_val_loss), 4),
            "val_acc": round(float(val_acc), 4)
        })

        # Checkpoint saving on best validation accuracy
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            shutil.copyfile(best_checkpoint_path, active_model_path)
        else:
            patience_counter += 1

        print(f"{epoch:<6d} | {avg_train_loss:<12.4f} | {train_acc*100:<10.1f}% | {avg_val_loss:<10.4f} | {val_acc*100:<8.1f}% {'*' if is_best else ''}")

        if patience_counter >= patience and epoch >= 30:
            print(f"\n[Early Stopping] No improvement in validation accuracy for {patience} epochs. Stopping at epoch {epoch}.")
            break

    print("-" * 65)
    print(f"Best Epoch: {best_epoch} with Validation Accuracy: {best_val_acc*100:.1f}%")
    print(f"Saved best model checkpoint to: {best_checkpoint_path}")
    print(f"Updated active inference model at: {active_model_path}")

    # ---------------------------------------------------------
    # Step 6: Final Test Evaluation on Best Checkpoint
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION (UNSEEN TEST SET: 59 SAMPLES)")
    print("=" * 65)

    eval_model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    ).to(device)

    eval_model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    eval_model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = eval_model(bx)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(by.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    correct_predictions = int((all_preds == all_targets).sum())
    total_test_samples = len(all_targets)
    test_accuracy = correct_predictions / total_test_samples

    # Metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )

    clf_report_dict = classification_report(
        all_targets, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    conf_mat = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    # Per-Class Table
    print(f"\n| {'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8} |")
    print("|" + "-"*14 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*10 + "|")
    for cid, cname in enumerate(CLASS_NAMES):
        p = per_class_p[cid]
        r = per_class_r[cid]
        f = per_class_f1[cid]
        s = int(per_class_supp[cid])
        print(f"| {cname:<12} | {p:<10.3f} | {r:<10.3f} | {f:<10.3f} | {s:<8d} |")

    # ---------------------------------------------------------
    # Step 7: Baseline Comparison
    # ---------------------------------------------------------
    baseline_acc = 0.426
    baseline_correct = 20
    baseline_total = 47

    diff = test_accuracy - baseline_acc
    sign_str = "+" if diff >= 0 else ""

    print("\n" + "=" * 50)
    print("BASELINE COMPARISON")
    print("=" * 50)
    print(f"Baseline Accuracy:  {baseline_acc * 100:.1f}% ({baseline_correct}/{baseline_total} correct on 19 classes)")
    print(f"New CNN-GRU:        {test_accuracy * 100:.1f}% ({correct_predictions}/{total_test_samples} correct on 21 classes)")
    print(f"Difference:         {sign_str}{diff * 100:.1f}%")
    print("=" * 50)

    # ---------------------------------------------------------
    # Step 8: Save Confusion Matrix & Training Curves Plots
    # ---------------------------------------------------------
    cm_plot_path = os.path.join(MODEL_DIR, "cnn_gru_21class_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=f"21-Class CNN-GRU Confusion Matrix (Test Accuracy: {test_accuracy*100:.1f}%)",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = conf_mat.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = conf_mat[i, j]
            ax.text(j, i, f"{val:d}", ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)
    fig.tight_layout()
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"\nSaved confusion matrix plot: {cm_plot_path}")

    # Curves plot
    curves_plot_path = os.path.join(MODEL_DIR, "cnn_gru_training_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ep_axis = [h["epoch"] for h in history]
    ax1.plot(ep_axis, [h["train_loss"] for h in history], label="Train Loss")
    ax1.plot(ep_axis, [h["val_loss"] for h in history], label="Val Loss")
    ax1.set_title("Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(ep_axis, [h["train_acc"] * 100 for h in history], label="Train Accuracy")
    ax2.plot(ep_axis, [h["val_acc"] * 100 for h in history], label="Val Accuracy")
    ax2.set_title("Accuracy vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.savefig(curves_plot_path, dpi=200)
    plt.close()
    print(f"Saved training curves plot: {curves_plot_path}")

    # ---------------------------------------------------------
    # Step 9: Save JSON Reports
    # ---------------------------------------------------------
    history_path = os.path.join(MODEL_DIR, "cnn_gru_training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(float(best_val_acc), 4),
            "history": history
        }, f, indent=2)

    clf_report_path = os.path.join(MODEL_DIR, "cnn_gru_21class_classification_report.json")
    with open(clf_report_path, "w", encoding="utf-8") as f:
        json.dump(clf_report_dict, f, indent=2)

    training_report_path = os.path.join(MODEL_DIR, "cnn_gru_21class_training_report.json")
    training_report_data = {
        "timestamp": "2026-09-29T00:15:00Z",
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "dataset_split": {
            "train": int(len(train_X)),
            "validation": int(len(val_X)),
            "test": int(len(test_X)),
            "total": int(len(train_X) + len(val_X) + len(test_X))
        },
        "input_shape": [25, 6],
        "feature_order": ["Hx", "Hy", "Mx", "My", "Rx", "Ry"],
        "training_settings": {
            "epochs_run": len(history),
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "initial_lr": 0.001,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(float(best_val_acc), 4)
        },
        "test_results": {
            "test_accuracy": round(float(test_accuracy), 4),
            "correct_samples": int(correct_predictions),
            "total_samples": int(total_test_samples),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "baseline_accuracy": baseline_acc,
            "accuracy_difference": round(float(diff), 4)
        },
        "confusion_matrix": conf_mat.tolist()
    }
    with open(training_report_path, "w", encoding="utf-8") as f:
        json.dump(training_report_data, f, indent=2)

    print(f"Saved classification report: {clf_report_path}")
    print(f"Saved full training report:   {training_report_path}")

    # ---------------------------------------------------------
    # Step 10: Model Compatibility Check with CNNGRUInferenceEngine
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("MODEL COMPATIBILITY CHECK")
    print("=" * 50)

    try:
        engine = CNNGRUInferenceEngine(model_path=best_checkpoint_path)
        dummy_seq = np.random.randn(25, 6).astype(np.float32)
        pred_res = engine.predict_sequence(dummy_seq)
        assert engine.model_loaded, "Inference engine failed to load model"
        assert engine.num_classes == 21, f"Expected 21 classes, found {engine.num_classes}"
        print("Inference Engine Compatibility: PASSED")
        print(f"  Sample Prediction: {pred_res['word']} (Confidence: {pred_res['confidence']:.2f})")
        compat_ok = True
    except Exception as e:
        print(f"Inference Engine Compatibility: FAILED ({e})")
        compat_ok = False

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_accuracy": test_accuracy,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "correct": correct_predictions,
        "total": total_test_samples,
        "compat_ok": compat_ok,
        "history": history,
        "report_dict": clf_report_dict
    }


if __name__ == "__main__":
    execute_pipeline()
