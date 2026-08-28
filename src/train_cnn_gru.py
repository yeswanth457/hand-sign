"""
Phase 6 & 7: Real Training Pipeline for ISL CNN-GRU Model.
Loads extracted real 6D gesture token sequences [T x 6] from dataset.csv,
formats fixed-length padded tensors (N, 25, 6), trains the PyTorch ISL_CNN_GRU_Model,
and saves trained checkpoint weights separately to models/isl_cnn_gru.pt.
"""

import sys
import os
import json
import csv
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    METADATA_CSV_PATH, TOKENS_DIR, MODEL_DIR,
    TOKEN_DIM, MAX_SEQ_LEN, ISL_VOCABULARY, VOCAB_SIZE, WORD_TO_ID
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, CNN_GRU_MODEL_PATH
from src.train_real_model import RealISLTokenDataset, load_real_token_dataset


def train_cnn_gru_model(epochs=50, lr=0.002, model_path=CNN_GRU_MODEL_PATH):
    """
    Trains PyTorch ISL_CNN_GRU_Model on real token dataset (dataset/train/ and dataset/val/).
    Saves trained checkpoint weights to model_path (models/isl_cnn_gru.pt).
    """
    print("\n==================================================")
    print("    PHASE 16.8: REAL ISL CNN-GRU MODEL TRAINING  ")
    print("==================================================")

    train_x_path = os.path.join("dataset", "train", "X.npy")
    train_y_path = os.path.join("dataset", "train", "y.npy")
    val_x_path = os.path.join("dataset", "val", "X.npy")
    val_y_path = os.path.join("dataset", "val", "y.npy")

    if not (os.path.exists(train_x_path) and os.path.exists(train_y_path)):
        print("[Train CNN-GRU] ERROR: Training tensors missing.")
        return {"status": "error", "message": "Missing training tensors"}

    X_train = np.load(train_x_path).astype(np.float32)
    y_train = np.load(train_y_path).astype(np.int64)

    X_val = np.load(val_x_path).astype(np.float32) if os.path.exists(val_x_path) else np.empty((0, 25, 6), dtype=np.float32)
    y_val = np.load(val_y_path).astype(np.int64) if os.path.exists(val_y_path) else np.empty((0,), dtype=np.int64)

    total_train = len(X_train)
    total_val = len(X_val)

    print(f"[Train CNN-GRU] Loaded Real Training Sequences   : {total_train} (shape: {X_train.shape})")
    print(f"[Train CNN-GRU] Loaded Real Validation Sequences : {total_val} (shape: {X_val.shape})")

    # PyTorch Datasets
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=min(4, max(1, total_train)), shuffle=True)

    val_loader = None
    if total_val > 0:
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=min(4, max(1, total_val)), shuffle=False)

    # Initialize ISL_CNN_GRU_Model
    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=VOCAB_SIZE,
        dropout=0.1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    best_train_acc = 0.0

    print(f"\n[Train CNN-GRU] Starting PyTorch training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += len(by)

        scheduler.step()

        train_acc = train_correct / max(1, train_total)
        avg_train_loss = train_loss / max(1, train_total)

        # Validation Step
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for bx, by in val_loader:
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item() * len(by)
                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == by).sum().item()
                    val_total += len(by)

        val_acc = (val_correct / max(1, val_total)) if val_total > 0 else 0.0
        avg_val_loss = (val_loss / max(1, val_total)) if val_total > 0 else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": round(val_acc, 4)
        })

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.1f}%")

        if epoch == epochs or (avg_val_loss < best_val_loss and val_total > 0):
            best_val_loss = avg_val_loss
            best_train_acc = max(best_train_acc, train_acc)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    # Save Training History JSON
    history_path = os.path.join(MODEL_DIR, "cnn_gru_training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_samples": total_train,
            "val_samples": total_val,
            "epochs": epochs,
            "best_train_accuracy": best_train_acc,
            "history": history
        }, f, indent=2)

    print(f"\n[Train CNN-GRU] SUCCESS! CNN-GRU Model trained on real ISL data.")
    print(f"[Train CNN-GRU] Saved trained weights to: {model_path}")
    print(f"[Train CNN-GRU] Saved training log to:    {history_path}")
    print("==================================================\n")

    return {
        "status": "success",
        "train_samples": total_train,
        "val_samples": total_val,
        "best_train_accuracy": best_train_acc,
        "model_path": model_path
    }


if __name__ == "__main__":
    train_cnn_gru_model(epochs=50)
