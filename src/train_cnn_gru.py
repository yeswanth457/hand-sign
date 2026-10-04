"""
Train 10-Class CNN-GRU Model on Real ISL Data.
Loads train/val splits, trains the ISL_CNN_GRU_Model with proper evaluation,
and saves the best checkpoint.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_DIR, MODEL_DIR, MODEL_PATH, TOKEN_DIM, MAX_SEQ_LEN,
    NUM_CLASSES, CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS
)
from src.cnn_gru_model import ISL_CNN_GRU_Model


class ISLTokenDataset(Dataset):
    """Dataset for 25x6 token sequences."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_split(split_name):
    """Load a dataset split (train/val/test)."""
    x_path = os.path.join(DATASET_DIR, split_name, "X.npy")
    y_path = os.path.join(DATASET_DIR, split_name, "y.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[ERROR] Missing {split_name} split files: {x_path}, {y_path}")
        return None, None
    
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    
    print(f"  {split_name}: X={X.shape}, y={y.shape}")
    return X, y


def apply_normalization(X, mean, std):
    """Apply feature normalization."""
    return (X - mean) / std


def train_10class_model(epochs=50, lr=0.001, batch_size=16):
    """Train the 10-class CNN-GRU model on real ISL data."""
    print("\n" + "=" * 60)
    print("  10-CLASS CNN-GRU TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load splits
    print("\nLoading data splits...")
    train_X, train_y = load_split("train")
    val_X, val_y = load_split("val")
    
    if train_X is None or len(train_X) == 0:
        print("[ERROR] No training data available. Run dataset processing first.")
        return {"status": "error", "message": "No training data"}
    
    # 2. Load feature normalization
    mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
    std_path = os.path.join(MODEL_DIR, "feature_std.npy")
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        feature_mean = np.load(mean_path).astype(np.float32)
        feature_std = np.load(std_path).astype(np.float32)
        feature_std = np.where(feature_std < 1e-7, 1.0, feature_std)
        
        train_X = apply_normalization(train_X, feature_mean, feature_std)
        if val_X is not None and len(val_X) > 0:
            val_X = apply_normalization(val_X, feature_mean, feature_std)
        print(f"Applied feature normalization: mean={feature_mean}, std={feature_std}")
    
    # 3. Print class distribution
    print(f"\nTraining set class distribution:")
    for cid, cname in enumerate(CLASS_NAMES):
        count = int((train_y == cid).sum())
        print(f"  [{cid}] {cname:12s}: {count} samples")
    
    # 4. Compute class weights for imbalanced data
    class_counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)  # Prevent division by zero
    total_samples = len(train_y)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    print(f"\nClass weights: {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in class_weights]))}")
    
    # 5. Create DataLoaders
    train_dataset = ISLTokenDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_X)), shuffle=True, drop_last=False)
    
    val_loader = None
    if val_X is not None and len(val_X) > 0:
        val_dataset = ISLTokenDataset(val_X, val_y)
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_X)), shuffle=False)
    
    # 6. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. Loss and optimizer
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 8. Training loop
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {epochs} epochs...\n")
    
    for epoch in range(1, epochs + 1):
        # Train phase
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
        
        train_acc = train_correct / max(1, train_total)
        avg_train_loss = train_loss / max(1, train_total)
        
        # Validation phase
        val_acc = 0.0
        avg_val_loss = 0.0
        if val_loader:
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
            
            val_acc = val_correct / max(1, val_total)
            avg_val_loss = val_loss / max(1, val_total)
        
        scheduler.step()
        
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": round(val_acc, 4)
        })
        
        # Save best model
        save_metric = val_acc if val_loader else train_acc
        if save_metric >= best_val_acc:
            best_val_acc = save_metric
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
    
    print(f"\nBest epoch: {best_epoch} (acc: {best_val_acc*100:.1f}%)")
    print(f"Model saved to: {MODEL_PATH}")
    
    # 9. Save training history
    history_path = os.path.join(MODEL_DIR, "cnn_gru_training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "train_samples": int(len(train_X)),
            "val_samples": int(len(val_X)) if val_X is not None else 0,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(best_val_acc, 4),
            "history": history
        }, f, indent=2)
    
    # 10. Verify saved model
    print("\n" + "=" * 60)
    print("MODEL VERIFICATION")
    print("=" * 60)
    
    verify_model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    )
    
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    verify_model.load_state_dict(state_dict)
    verify_model.eval()
    
    # Check output shape
    dummy_input = torch.randn(1, 25, 6)
    with torch.no_grad():
        output = verify_model(dummy_input)
    
    print(f"Input shape:    (25, 6)")
    print(f"Output classes: {output.shape[1]}")
    print(f"Expected:       {NUM_CLASSES}")
    assert output.shape[1] == NUM_CLASSES, f"Output mismatch: {output.shape[1]} != {NUM_CLASSES}"
    
    print(f"\nClasses:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name}")
    
    print(f"\n[OK] Model verification PASSED")
    print("=" * 60)
    
    return {
        "status": "success",
        "model_path": MODEL_PATH,
        "num_classes": NUM_CLASSES,
        "best_epoch": best_epoch,
        "best_accuracy": best_val_acc,
        "train_samples": len(train_X),
    }


def evaluate_on_test():
    """Evaluate the saved model on the held-out test set."""
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)
    
    test_X, test_y = load_split("test")
    if test_X is None or len(test_X) == 0:
        # Fall back to validation set if no test set
        print("No test set found, evaluating on validation set...")
        test_X, test_y = load_split("val")
        if test_X is None or len(test_X) == 0:
            print("[ERROR] No test or validation data available.")
            return
    
    # Apply normalization
    mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
    std_path = os.path.join(MODEL_DIR, "feature_std.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        feature_mean = np.load(mean_path).astype(np.float32)
        feature_std = np.load(std_path).astype(np.float32)
        feature_std = np.where(feature_std < 1e-7, 1.0, feature_std)
        test_X = apply_normalization(test_X, feature_mean, feature_std)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    ).to(device)
    
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run inference
    X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    
    # Classification report
    present_labels = sorted(set(test_y.tolist()) | set(preds.tolist()))
    present_names = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]
    
    report = classification_report(
        test_y, preds,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0
    )
    
    print("\nClassification Report:")
    print(classification_report(
        test_y, preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(test_y, preds, labels=list(range(NUM_CLASSES)))
    print("Confusion Matrix:")
    print(f"{'':>10}", end="")
    for name in CLASS_NAMES:
        print(f"{name[:6]:>7}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>10}", end="")
        for val in row:
            print(f"{val:>7}", end="")
        print()
    
    # Save report
    report_path = os.path.join(MODEL_DIR, "cnn_gru_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "test_samples": len(test_X),
            "accuracy": report.get("accuracy", 0),
            "per_class": {name: report.get(name, {}) for name in present_names},
            "confusion_matrix": cm.tolist()
        }, f, indent=2)
    
    print(f"\nSaved evaluation report to: {report_path}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    result = train_10class_model(epochs=50)
    if result.get("status") == "success":
        evaluate_on_test()
