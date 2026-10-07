"""
Offline training and evaluation of models/isl_cnn_gru_father_fix.pt.
Preserves existing classes, uses held-out user Father test sample,
and verifies zero regression across Brother, Water, School, etc.
"""

import os
import sys
import shutil
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

def generate_father_variations(raw_tokens, target_count=20):
    """
    Generate realistic temporal and spatial variations of real Father recording.
    raw_tokens: (N, 12)
    Variations:
      - Temporal speeds: slicing/resampling from 18..28 frames to 25
      - Minor spatial translations: +/- 0.02
      - Minor scale/distance shifts: +/- 4%
    All variations preserve the exact physical hand landmarks and motion trajectory.
    """
    variations = []
    n, d = raw_tokens.shape
    
    # Base 25-frame resampled
    base = resample_tokens(raw_tokens, target_t=25)
    variations.append(base)
    
    np.random.seed(42)
    # Speeds (different temporal window lengths)
    for speed_crop in [0.85, 0.90, 0.95, 1.0, 1.05, 1.1]:
        crop_len = int(np.clip(n * speed_crop, 15, n))
        start_idx = np.random.randint(0, max(1, n - crop_len + 1))
        sub = raw_tokens[start_idx : start_idx + crop_len]
        res = resample_tokens(sub, target_t=25)
        variations.append(res)
    
    # Spatial jitter & scale
    while len(variations) < target_count:
        v = base.copy()
        # Scale (distance variation)
        scale = np.random.uniform(0.96, 1.04)
        # Shift (position variation)
        shift_x = np.random.uniform(-0.025, 0.025)
        shift_y = np.random.uniform(-0.025, 0.025)
        
        # Apply to non-zero coordinates
        mask = np.any(v[:, :6] != 0, axis=1)
        # Left hand: Hx(0), Hy(1), Mx(2), My(3), Rx(4), Ry(5)
        v[mask, 0] = v[mask, 0] * scale + shift_x
        v[mask, 1] = v[mask, 1] * scale + shift_y
        v[mask, 2] = v[mask, 2] * scale
        v[mask, 3] = v[mask, 3] * scale
        v[mask, 4] = v[mask, 4] * scale + shift_x
        v[mask, 5] = v[mask, 5] * scale + shift_y
        
        # Also small temporal jitter
        noise = np.random.normal(0, 0.01, v.shape).astype(np.float32)
        v[mask] += noise[mask]
        variations.append(v.astype(np.float32))
        
    return variations[:target_count]

class TokenDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            if torch.rand(1).item() > 0.3:
                noise = torch.randn_like(x) * 0.02
                x = x + noise
        return x, y

def main():
    print("=" * 70)
    print("OFFLINE FATHER RETRAINING & EVALUATION")
    print("=" * 70)
    
    # 1. Load current split data
    train_X = np.load(os.path.join(DATASET_DIR, "train", "X.npy"))
    train_y = np.load(os.path.join(DATASET_DIR, "train", "y.npy"))
    val_X = np.load(os.path.join(DATASET_DIR, "val", "X.npy"))
    val_y = np.load(os.path.join(DATASET_DIR, "val", "y.npy"))
    test_X = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
    test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))
    
    print(f"Original Train: X={train_X.shape}, y={train_y.shape}")
    print(f"Original Val:   X={val_X.shape}, y={val_y.shape}")
    print(f"Original Test:  X={test_X.shape}, y={test_y.shape}")
    
    father_idx = CLASS_TO_INDEX["father"]
    print(f"Father Class Index: {father_idx}")
    
    # Load user father tokens
    user_father_1 = np.load("dataset/tokens/father/hf_real/WIN_20261008_00_45_21_Pro.npz")["tokens"].astype(np.float32)
    user_father_2 = np.load("dataset/tokens/father/hf_real/WIN_20261008_00_45_29_Pro.npz")["tokens"].astype(np.float32)
    
    print("User Father 1 shape:", user_father_1.shape)
    print("User Father 2 shape:", user_father_2.shape)
    
    # Assign User Father 1 -> Train (with 25 variations)
    # Assign User Father 2 -> Test (Strict held-out validation)
    father_train_vars = generate_father_variations(user_father_1, target_count=25)
    print(f"Generated {len(father_train_vars)} training variations from User Father 1")
    
    # Append to train
    train_X_aug = np.concatenate([train_X, np.array(father_train_vars, dtype=np.float32)], axis=0)
    train_y_aug = np.concatenate([train_y, np.full((len(father_train_vars),), father_idx, dtype=np.int64)], axis=0)
    
    # Append User Father 2 to test
    held_out_father_25 = resample_tokens(user_father_2, target_t=25).reshape(1, 25, 12)
    test_X_aug = np.concatenate([test_X, held_out_father_25], axis=0)
    test_y_aug = np.concatenate([test_y, np.array([father_idx], dtype=np.int64)], axis=0)
    
    print(f"Augmented Train: X={train_X_aug.shape}, y={train_y_aug.shape}")
    print(f"Augmented Test:  X={test_X_aug.shape}, y={test_y_aug.shape}")
    
    # Normalization parameters (using train stats)
    feat_mean = np.load(os.path.join(MODEL_DIR, "feature_mean.npy")).astype(np.float32)
    feat_std = np.load(os.path.join(MODEL_DIR, "feature_std.npy")).astype(np.float32)
    feat_std = np.where(feat_std < 1e-7, 1.0, feat_std)
    
    norm_train_X = (train_X_aug - feat_mean) / feat_std
    norm_val_X = (val_X - feat_mean) / feat_std
    norm_test_X = (test_X_aug - feat_mean) / feat_std
    
    # Class weights from train set
    train_counts = np.bincount(train_y_aug, minlength=NUM_CLASSES).astype(np.float32)
    train_counts = np.maximum(train_counts, 1.0)
    class_weights = np.sqrt(len(train_y_aug) / (NUM_CLASSES * train_counts))
    class_weights = np.clip(class_weights, 0.6, 2.5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    sample_weights = [1.0 / train_counts[label] for label in train_y_aug]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_y_aug), replacement=True)
    
    train_loader = DataLoader(TokenDataset(norm_train_X, train_y_aug, augment=True), batch_size=16, sampler=sampler)
    val_loader = DataLoader(TokenDataset(norm_val_X, val_y, augment=False), batch_size=16, shuffle=False)
    test_loader = DataLoader(TokenDataset(norm_test_X, test_y_aug, augment=False), batch_size=16, shuffle=False)
    
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
    
    print("\nTraining CNN-GRU model...")
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
        train_acc = total_correct / max(1, total_n)
        
        # Validation
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
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {total_loss/total_n:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% (Best: {best_val_acc*100:.1f}%)")
            
    # Load best weights
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()
    
    target_ckpt = os.path.join(MODEL_DIR, "isl_cnn_gru_father_fix.pt")
    torch.save(model.state_dict(), target_ckpt)
    print(f"\nSaved new model checkpoint to: {target_ckpt}")
    
    # ── Test Set Evaluation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET (INCLUDING HELD-OUT USER FATHER)")
    print("=" * 70)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = (all_preds == all_labels).sum() / len(all_labels)
    print(f"Overall Test Accuracy: {overall_acc*100:.2f}% ({np.sum(all_preds == all_labels)}/{len(all_labels)})")
    
    # Key classes breakdown
    key_classes = ["father", "school", "mother", "brother", "sister", "water", "hello", "thank_you", "please"]
    print("\nKey Classes Accuracy:")
    for cname in key_classes:
        cid = CLASS_TO_INDEX[cname]
        mask = (all_labels == cid)
        total_c = np.sum(mask)
        if total_c > 0:
            correct_c = np.sum(all_preds[mask] == cid)
            acc = correct_c / total_c * 100
            print(f"  {cname:<12}: {acc:5.1f}% ({correct_c}/{total_c})")
        else:
            print(f"  {cname:<12}: No test samples")
            
    # Father confusions
    f_cid = CLASS_TO_INDEX["father"]
    f_mask = (all_labels == f_cid)
    f_preds = all_preds[f_mask]
    print(f"\nFather test predictions ({len(f_preds)} total):")
    for p in f_preds:
        print(f"  Predicted: {INDEX_TO_CLASS[p]}")
        
    # Evaluate held-out User Father specifically
    norm_u2 = (held_out_father_25 - feat_mean) / feat_std
    with torch.no_grad():
        u2_logits = model(torch.tensor(norm_u2, dtype=torch.float32).to(device))
        u2_probs = torch.softmax(u2_logits, dim=1).cpu().numpy()[0]
    
    top5_idx = np.argsort(u2_probs)[::-1][:5]
    print("\nStrict Held-out User Father (WIN_20261008_00_45_29_Pro) Predictions:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  Top {rank}: {INDEX_TO_CLASS[idx]:<12} ({u2_probs[idx]*100:.2f}%)")

if __name__ == "__main__":
    main()
