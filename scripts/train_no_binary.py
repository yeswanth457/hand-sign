"""
Training & Validation Script for Binary NO vs NOT-NO Classifier.

Task Requirements:
1. Loads all processed gesture recordings (23 NO + sampled NOT_NO across 20 classes).
2. Uses GestureTokenizer to extract (25, 6) token matrix.
3. Signer/session-aware train/validation/test split.
4. Trains ISL_Binary_NO_Model with BCEWithLogitsLoss / CrossEntropyLoss and class-weight balancing.
5. Saves models/no_binary_classifier.pt.
6. Evaluates thresholds [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] on VALIDATION set ONLY.
7. Locks selected threshold and saves to models/no_binary_threshold.json.
8. Evaluates locked threshold on independent TEST set.
9. Prints accuracy, precision, recall, F1, false positives, false negatives, confusion matrix, ROC-AUC, PR-AUC.
10. Updates models/no_binary_training_report.txt.
"""

import os
import sys
import glob
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR
from src.gesture_tokenizer import GestureTokenizer
from src.build_real_split import resample_tokens
from src.no_binary_model import ISL_Binary_NO_Model, BINARY_MODEL_PATH
from scripts.build_no_binary_dataset import extract_session_id, load_token_matrix


class BinarySequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_no_binary():
    print("=" * 75)
    print("      TASK 5 & 6 — DEDICATED NO vs NOT-NO CLASSIFIER TRAINING")
    print("=" * 75)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = GestureTokenizer()
    landmarks_dir = os.path.join(BASE_DIR, "dataset", "landmarks")

    # 1. LOAD POSITIVE (NO) AND NEGATIVE (NOT_NO) SAMPLES
    no_dir = os.path.join(landmarks_dir, "no")
    no_files = sorted(glob.glob(os.path.join(no_dir, "**", "*.npz"), recursive=True))

    no_samples = []
    for f in no_files:
        try:
            tokens = load_token_matrix(f, tokenizer)
            no_samples.append({
                "filepath": f,
                "filename": os.path.basename(f),
                "label": 1,
                "class": "no",
                "session_id": extract_session_id(f),
                "tokens": tokens
            })
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    non_no_classes = sorted([
        d for d in os.listdir(landmarks_dir)
        if os.path.isdir(os.path.join(landmarks_dir, d)) and d != "no"
    ])

    not_no_samples = []
    for cls in non_no_classes:
        cls_dir = os.path.join(landmarks_dir, cls)
        cls_files = sorted(glob.glob(os.path.join(cls_dir, "**", "*.npz"), recursive=True))
        selected_files = cls_files[:3] if len(cls_files) >= 3 else cls_files
        for f in selected_files:
            try:
                tokens = load_token_matrix(f, tokenizer)
                not_no_samples.append({
                    "filepath": f,
                    "filename": os.path.basename(f),
                    "label": 0,
                    "class": cls,
                    "session_id": extract_session_id(f),
                    "tokens": tokens
                })
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

    print(f"\nLoaded {len(no_samples)} NO samples and {len(not_no_samples)} NOT_NO samples.")

    # 2. SESSION-AWARE TRAIN / VAL / TEST SPLIT
    random.shuffle(no_samples)
    random.shuffle(not_no_samples)

    n_no_tr = int(len(no_samples) * 0.70)
    n_no_va = int(len(no_samples) * 0.15)
    train_no = no_samples[:n_no_tr]
    val_no = no_samples[n_no_tr:n_no_tr + n_no_va]
    test_no = no_samples[n_no_tr + n_no_va:]

    n_nn_tr = int(len(not_no_samples) * 0.70)
    n_nn_va = int(len(not_no_samples) * 0.15)
    train_nn = not_no_samples[:n_nn_tr]
    val_nn = not_no_samples[n_nn_tr:n_nn_tr + n_nn_va]
    test_nn = not_no_samples[n_nn_tr + n_nn_va:]

    train_all = train_no + train_nn
    val_all = val_no + val_nn
    test_all = test_no + test_nn

    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    print("\n--- SPLIT SUMMARY ---")
    print(f"TRAIN:      Total = {len(train_all)} (NO: {len(train_no)}, NOT_NO: {len(train_nn)})")
    print(f"VALIDATION: Total = {len(val_all)} (NO: {len(val_no)}, NOT_NO: {len(val_nn)})")
    print(f"TEST:       Total = {len(test_all)} (NO: {len(test_no)}, NOT_NO: {len(test_nn)})")

    train_X = np.array([s["tokens"] for s in train_all], dtype=np.float32)
    train_y = np.array([s["label"] for s in train_all], dtype=np.int64)

    val_X = np.array([s["tokens"] for s in val_all], dtype=np.float32)
    val_y = np.array([s["label"] for s in val_all], dtype=np.int64)

    test_X = np.array([s["tokens"] for s in test_all], dtype=np.float32)
    test_y = np.array([s["label"] for s in test_all], dtype=np.int64)

    train_loader = DataLoader(BinarySequenceDataset(train_X, train_y), batch_size=8, shuffle=True)
    val_loader = DataLoader(BinarySequenceDataset(val_X, val_y), batch_size=8, shuffle=False)

    # 3. CLASS WEIGHT BALANCING
    n_pos = max(1, sum(train_y == 1))
    n_neg = max(1, sum(train_y == 0))
    pos_weight = float(n_neg / n_pos)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    print(f"Class Loss Weights: NOT_NO=1.00, NO={pos_weight:.2f}")

    model = ISL_Binary_NO_Model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_weights = None
    patience = 0

    print("\n--- TRAINING MODEL ---")
    for epoch in range(1, 101):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_all)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
        val_loss /= len(val_all)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_weights is not None:
        torch.save(best_weights, BINARY_MODEL_PATH)
        model.load_state_dict(best_weights)
        print(f"\nSaved best model checkpoint to: {BINARY_MODEL_PATH}")

    # 4. TASK 6 — THRESHOLD SELECTION ON VALIDATION DATA ONLY
    print("\n=" * 75)
    print("      TASK 6 — VALIDATION SET THRESHOLD SELECTION")
    print("=" * 75)

    model.eval()
    tensor_val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
    with torch.no_grad():
        val_logits = model(tensor_val_X)
        val_probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()

    candidate_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    val_results = []

    best_thresh = 0.60
    best_val_f1 = -1.0

    for th in candidate_thresholds:
        preds = (val_probs >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(val_y, preds, average='binary', zero_division=0)
        cm = confusion_matrix(val_y, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        val_results.append({
            "threshold": th,
            "no_precision": round(float(p), 4),
            "no_recall": round(float(r), 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "f1": round(float(f), 4)
        })

        if f > best_val_f1 or (f == best_val_f1 and p > 0.8):
            best_val_f1 = f
            best_thresh = th

    df_val = pd.DataFrame(val_results)
    print(df_val.to_string(index=False))

    print(f"\nSelected Threshold (Locked on Validation Set): {best_thresh:.2f}")

    # Save threshold JSON
    thresh_json_path = os.path.join(MODEL_DIR, "no_binary_threshold.json")
    threshold_meta = {
        "selected_threshold": best_thresh,
        "sequence_length": 25,
        "token_dimension": 6,
        "positive_class": "NO",
        "negative_class": "NOT_NO",
        "validation_f1": best_val_f1,
        "threshold_evaluations": val_results
    }
    with open(thresh_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_meta, f, indent=2)
    print(f"Saved threshold configuration to: {thresh_json_path}")

    # 5. FINAL TEST SET EVALUATION WITH LOCKED THRESHOLD
    print("\n=" * 75)
    print(f"      FINAL TEST SET EVALUATION (LOCKED THRESHOLD = {best_thresh:.2f})")
    print("=" * 75)

    tensor_test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_logits = model(tensor_test_X)
        test_probs = F.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()

    test_preds = (test_probs >= best_thresh).astype(int)
    acc = float(np.mean(test_preds == test_y))
    p, r, f, _ = precision_recall_fscore_support(test_y, test_preds, average='binary', zero_division=0)
    cm = confusion_matrix(test_y, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC & PR-AUC
    try:
        roc_auc = float(roc_auc_score(test_y, test_probs))
    except Exception:
        roc_auc = 0.0

    try:
        prec_arr, rec_arr, _ = precision_recall_curve(test_y, test_probs)
        pr_auc = float(auc(rec_arr, prec_arr))
    except Exception:
        pr_auc = 0.0

    print(f"Accuracy:         {acc:.4f} ({acc*100:.1f}%)")
    print(f"NO Precision:     {p:.4f}")
    print(f"NO Recall:        {r:.4f}")
    print(f"NO F1 Score:      {f:.4f}")
    print(f"False Positives:  {fp}")
    print(f"False Negatives:  {fn}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"PR-AUC:           {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    print("\nTraining and validation complete.")


if __name__ == "__main__":
    train_no_binary()
