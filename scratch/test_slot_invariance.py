"""
Test script for general single-hand slot invariance.
Trains CNN-GRU where single-handed samples are trained symmetrically in both slots.
Evaluates both left-slot and right-slot versions of Father, Brother, Water, School, etc.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MAX_SEQ_LEN, DATASET_DIR, MODEL_DIR
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, CNNGRUInferenceEngine
from src.build_real_split import resample_tokens

def swap_slots(tokens_25):
    """Swaps LH (0..5) and RH (6..11) in a 12D token sequence (25, 12)."""
    swapped = np.zeros_like(tokens_25)
    swapped[:, :6] = tokens_25[:, 6:]
    swapped[:, 6:] = tokens_25[:, :6]
    return swapped

def is_single_handed(tokens_25):
    """Returns True if only one hand is active in the sequence."""
    lh_active = np.any(tokens_25[:, :6] != 0)
    rh_active = np.any(tokens_25[:, 6:] != 0)
    return (lh_active and not rh_active) or (rh_active and not lh_active)

def main():
    print("=" * 70)
    print("TESTING GENERAL SINGLE-HAND SLOT INVARIANCE")
    print("=" * 70)

    train_X = np.load(os.path.join(DATASET_DIR, "train", "X.npy"))
    train_y = np.load(os.path.join(DATASET_DIR, "train", "y.npy"))
    val_X = np.load(os.path.join(DATASET_DIR, "val", "X.npy"))
    val_y = np.load(os.path.join(DATASET_DIR, "val", "y.npy"))
    test_X = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
    test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))

    print(f"Original Train: X={train_X.shape}, y={train_y.shape}")

    # Build symmetrically augmented training set for single-handed classes
    aug_X = list(train_X)
    aug_y = list(train_y)

    for i in range(len(train_X)):
        sample = train_X[i]
        label = train_y[i]
        if is_single_handed(sample):
            # Add swapped slot version
            swapped = swap_slots(sample)
            aug_X.append(swapped)
            aug_y.append(label)

    arr_train_X = np.array(aug_X, dtype=np.float32)
    arr_train_y = np.array(aug_y, dtype=np.int64)
    print(f"Symmetrically Augmented Train: X={arr_train_X.shape}, y={arr_train_y.shape}")

    # Normalization parameters (using train set)
    all_feat = arr_train_X.reshape(-1, TOKEN_DIM)
    feat_mean = np.mean(all_feat, axis=0).astype(np.float32)
    feat_std = np.std(all_feat, axis=0).astype(np.float32)
    feat_std = np.where(feat_std < 1e-7, 1.0, feat_std)

    norm_train_X = (arr_train_X - feat_mean) / feat_std
    norm_val_X = (val_X - feat_mean) / feat_std
    norm_test_X = (test_X - feat_mean) / feat_std

    # Class weights from train set
    train_counts = np.bincount(arr_train_y, minlength=NUM_CLASSES).astype(np.float32)
    train_counts = np.maximum(train_counts, 1.0)
    class_weights = np.sqrt(len(arr_train_y) / (NUM_CLASSES * train_counts))
    class_weights = np.clip(class_weights, 0.6, 2.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    class SimpleDataset(Dataset):
        def __init__(self, X, y, augment=False):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            self.augment = augment
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx].clone()
            y = self.y[idx]
            if self.augment and torch.rand(1).item() > 0.3:
                x = x + torch.randn_like(x) * 0.015
            return x, y

    sample_weights = [1.0 / train_counts[label] for label in arr_train_y]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(arr_train_y), replacement=True)

    train_loader = DataLoader(SimpleDataset(norm_train_X, arr_train_y, augment=True), batch_size=16, sampler=sampler)
    val_loader = DataLoader(SimpleDataset(norm_val_X, val_y), batch_size=16, shuffle=False)
    test_loader = DataLoader(SimpleDataset(norm_test_X, test_y), batch_size=16, shuffle=False)

    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.25
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-5)

    epochs = 45
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            total_correct += (preds == by).sum().item()
            total_n += len(by)
        scheduler.step()

        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                val_n += len(by)
        val_acc = val_correct / max(1, val_n)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {total_loss/total_n:.4f} | Train Acc: {total_correct/total_n*100:.1f}% | Val Acc: {val_acc*100:.1f}% (Best: {best_val_acc*100:.1f}%)")

    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    # Save to candidate checkpoint
    cand_path = os.path.join(MODEL_DIR, "isl_cnn_gru_22class_father_accuracy_fix.pt")
    torch.save(model.state_dict(), cand_path)
    print(f"\nSaved slot-invariant candidate checkpoint to: {cand_path}")

    # Evaluate BOTH Left slot and Right slot for Father
    d_father = np.load("dataset/tokens/father/hf_real/WIN_20261008_00_45_29_Pro.npz")
    f_lh = resample_tokens(d_father["tokens"], 25)
    f_rh = swap_slots(f_lh)

    def eval_seq(tokens_25):
        norm_t = (tokens_25 - feat_mean) / feat_std
        with torch.no_grad():
            logits = model(torch.tensor(norm_t, dtype=torch.float32).unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pid = np.argmax(probs)
        return INDEX_TO_CLASS[pid], probs[pid]

    w_lh, c_lh = eval_seq(f_lh)
    w_rh, c_rh = eval_seq(f_rh)
    print("\nSLOT INVARIANCE EVALUATION:")
    print(f"  User Father in LEFT slot:  Predicted = {w_lh:<10} Confidence = {c_lh*100:.2f}%")
    print(f"  User Father in RIGHT slot: Predicted = {w_rh:<10} Confidence = {c_rh*100:.2f}%")

    # Evaluate Brother in both slots
    d_brother = np.load("dataset/tokens/brother/hf_real/WIN_20261007_22_50_15_Pro.npz")
    b_seq = resample_tokens(d_brother["tokens"], 25)
    b_lh = b_seq if is_single_handed(b_seq) and np.any(b_seq[:, :6] != 0) else swap_slots(b_seq)
    b_rh = swap_slots(b_lh)
    b_w_lh, b_c_lh = eval_seq(b_lh)
    b_w_rh, b_c_rh = eval_seq(b_rh)
    print(f"  User Brother in LEFT slot:  Predicted = {b_w_lh:<10} Confidence = {b_c_lh*100:.2f}%")
    print(f"  User Brother in RIGHT slot: Predicted = {b_w_rh:<10} Confidence = {b_c_rh*100:.2f}%")

    # Overall test accuracy
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(by.cpu().numpy())
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"\nOverall Unseen Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
