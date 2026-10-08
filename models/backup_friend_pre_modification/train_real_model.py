"""
Phase 7: Real Data Training Pipeline for Temporal Memory Transformer.
Loads real extracted 6D gesture token sequences [T x 6], formats fixed-length
padded tensors (N, MAX_SEQ_LEN, 6), trains the PyTorch TemporalMemoryTransformer model,
and saves trained weights to models/isl_temporal_transformer.pt.
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
    METADATA_CSV_PATH, TOKENS_DIR, MODEL_PATH, MODEL_DIR,
    TOKEN_DIM, MAX_SEQ_LEN, ISL_VOCABULARY, VOCAB_SIZE, WORD_TO_ID
)
from src.temporal_transformer import TemporalMemoryTransformer


class RealISLTokenDataset(Dataset):
    def __init__(self, token_sequences, labels, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.samples = []
        self.labels = []

        for seq, label in zip(token_sequences, labels):
            seq_len = len(seq)
            # Pad or truncate to max_seq_len
            if seq_len < max_seq_len:
                pad = np.zeros((max_seq_len - seq_len, TOKEN_DIM), dtype=np.float32)
                padded_seq = np.vstack([seq, pad])
            else:
                padded_seq = seq[:max_seq_len]

            self.samples.append(padded_seq.astype(np.float32))
            self.labels.append(label)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])


def load_real_token_dataset(csv_path=METADATA_CSV_PATH):
    """Loads all successful token sequences and class labels from dataset.csv."""
    if not os.path.exists(csv_path):
        print(f"[Train] Metadata CSV not found at {csv_path}")
        return [], [], {}

    token_sequences = []
    labels = []
    class_to_id = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"] == "success" and os.path.exists(r["token_file"]):
                sign_cls = r["sign_class"].lower().strip()
                if sign_cls in WORD_TO_ID:
                    cls_id = WORD_TO_ID[sign_cls]
                    class_to_id[sign_cls] = cls_id

                    try:
                        with np.load(r["token_file"]) as data:
                            tk_arr = data["tokens"]
                            if tk_arr.ndim == 2 and tk_arr.shape[1] == TOKEN_DIM:
                                token_sequences.append(tk_arr)
                                labels.append(cls_id)
                    except Exception as e:
                        print(f"[Train] Error loading {r['token_file']}: {e}")

    return token_sequences, labels, class_to_id


def train_real_isl_model(epochs=30, lr=0.001, model_path=MODEL_PATH):
    """
    Trains PyTorch TemporalMemoryTransformer on real token dataset.
    Saves trained checkpoint weights to model_path.
    """
    print("\n==================================================")
    print("    PHASE 7: REAL ISL MODEL TRAINING PIPELINE")
    print("==================================================")

    token_seqs, labels, class_to_id = load_real_token_dataset()
    total_samples = len(token_seqs)

    if total_samples == 0:
        print("[Train] ERROR: No valid real token sequences found in dataset.")
        return {"status": "error", "message": "No real token sequences"}

    print(f"[Train] Loaded {total_samples} real ISL token sequences across {len(class_to_id)} sign classes.")
    for cls_name, cls_id in class_to_id.items():
        count = sum(1 for l in labels if l == cls_id)
        print(f"  - Class '{cls_name}' (ID {cls_id}): {count} sequence(s)")

    # Prepare Dataset & DataLoader
    num_classes = len(class_to_id)
    dataset = RealISLTokenDataset(token_seqs, labels, max_seq_len=MAX_SEQ_LEN)
    
    batch_size = min(8, total_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Temporal Memory Transformer
    model = TemporalMemoryTransformer(
        token_dim=TOKEN_DIM,
        d_model=64,
        n_heads=4,
        num_layers=2,
        num_classes=VOCAB_SIZE  # Supports full vocabulary head
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_acc = 0.0

    print(f"\n[Train] Starting PyTorch model training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by in dataloader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)

        history.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "accuracy": round(acc, 4)
        })

        if epoch % 5 == 0 or epoch == epochs or acc >= best_acc:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc * 100:.1f}%")

        if acc >= best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    # Save Training History JSON
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples": total_samples,
            "classes": class_to_id,
            "epochs": epochs,
            "final_accuracy": best_acc,
            "history": history
        }, f, indent=2)

    print(f"\n[Train] SUCCESS! Model trained on real ISL data.")
    print(f"[Train] Saved trained weights to: {model_path}")
    print(f"[Train] Saved training log to:    {history_path}")
    print("==================================================\n")

    return {
        "status": "success",
        "total_samples": total_samples,
        "classes": class_to_id,
        "best_accuracy": best_acc,
        "model_path": model_path
    }


if __name__ == "__main__":
    train_real_isl_model(epochs=25)
