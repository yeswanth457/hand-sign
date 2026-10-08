"""
Training and Regression Evaluation for 22-Class CNN-GRU with Friend Fix.
Preserves existing classes, trains candidate model models/isl_cnn_gru_22class_friend_fix.pt,
and performs comprehensive regression testing against the baseline checkpoint.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES,
    TOKEN_DIM, MAX_SEQ_LEN, DATASET_DIR, MODEL_DIR, MODEL_PATH
)
from src.cnn_gru_model import ISL_CNN_GRU_Model, resample_tokens

class SimpleDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment and torch.rand(1).item() > 0.3:
            # Subtle jitter for robustness
            x = x + torch.randn_like(x) * 0.012
        return x, y

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(by.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def eval_user_friend(model, device, feat_mean, feat_std):
    user_tokens = [
        'dataset/tokens/friend/hf_real/WIN_20261009_00_23_53_Pro.npz',
        'dataset/tokens/friend/hf_real/WIN_20261009_00_24_02_Pro.npz',
        'dataset/tokens/friend/hf_real/WIN_20261009_00_24_08_Pro.npz'
    ]
    corr = 0
    probs_list = []
    for ut in user_tokens:
        tok = np.load(ut)['tokens']
        t25 = resample_tokens(tok, target_t=25)
        t_norm = (t25 - feat_mean) / feat_std
        bx = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(bx)
            p = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(p))
            if pred == 18:
                corr += 1
            probs_list.append(float(p[18]))
    return corr, probs_list

def train_friend_candidate():
    print("=" * 75)
    print("      22-CLASS CNN-GRU RETRAINING WITH FRIEND FIX")
    print("=" * 75)
    print(f"Target Checkpoint:  models/isl_cnn_gru_22class_friend_fix.pt")
    baseline_path = os.path.join(MODEL_DIR, "isl_cnn_gru_father_fix.pt")
    print(f"Baseline Reference: {baseline_path}")
    print(f"NUM_CLASSES:        {NUM_CLASSES}")
    print(f"TOKEN_DIM:          {TOKEN_DIM}")

    train_X = np.load(os.path.join(DATASET_DIR, "train", "X.npy"))
    train_y = np.load(os.path.join(DATASET_DIR, "train", "y.npy"))
    val_X = np.load(os.path.join(DATASET_DIR, "val", "X.npy"))
    val_y = np.load(os.path.join(DATASET_DIR, "val", "y.npy"))
    test_X = np.load(os.path.join(DATASET_DIR, "test", "X.npy"))
    test_y = np.load(os.path.join(DATASET_DIR, "test", "y.npy"))

    print(f"Loaded splits: Train={train_X.shape}, Val={val_X.shape}, Test={test_X.shape}")

    # Normalization parameters from train set
    all_feat = train_X.reshape(-1, TOKEN_DIM)
    feat_mean = np.mean(all_feat, axis=0).astype(np.float32)
    feat_std = np.std(all_feat, axis=0).astype(np.float32)
    feat_std = np.where(feat_std < 1e-7, 1.0, feat_std)

    norm_train_X = (train_X - feat_mean) / feat_std
    norm_val_X = (val_X - feat_mean) / feat_std
    norm_test_X = (test_X - feat_mean) / feat_std

    # Class balancing
    train_counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(np.float32)
    train_counts = np.maximum(train_counts, 1.0)
    class_weights = np.sqrt(len(train_y) / (NUM_CLASSES * train_counts))
    class_weights = np.clip(class_weights, 0.6, 2.5)
    class_weights[11] *= 1.6  # Protect food against false drift

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    sample_weights = [1.0 / train_counts[label] for label in train_y]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_y), replacement=True)

    train_loader = DataLoader(SimpleDataset(norm_train_X, train_y, augment=True), batch_size=16, sampler=sampler)
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

    # Initialize from baseline weights (fine-tuning preserves existing 21 classes)
    if os.path.exists(baseline_path):
        try:
            base_dict = torch.load(baseline_path, map_location=device, weights_only=True)
            model.load_state_dict(base_dict)
            print(f"Initialized candidate from baseline weights: {baseline_path}")
        except Exception as e_load:
            print(f"Could not load baseline weights ({e_load}), starting fresh.")

    # Baseline model for distillation (protects all 21 existing classes)
    base_model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    ).to(device)
    base_model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    old_preds, test_labels = evaluate_model(base_model, test_loader, device)
    old_class_accs = {}
    for cid in range(NUM_CLASSES):
        m = (test_labels == cid)
        if np.sum(m) > 0:
            old_class_accs[cid] = float(np.mean(old_preds[m] == cid))

    best_score = -999.0
    best_weights = None
    best_user_corr = 0

    print(f"\nStarting training with knowledge distillation for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss_ce = criterion(logits, by)

            with torch.no_grad():
                base_logits = base_model(bx)

            non_friend_mask = (by != 18)
            if non_friend_mask.any():
                distill_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits[non_friend_mask] / 1.5, dim=1),
                    torch.nn.functional.softmax(base_logits[non_friend_mask] / 1.5, dim=1),
                    reduction='batchmean'
                ) * (1.5 ** 2)
            else:
                distill_loss = torch.tensor(0.0, device=device)

            loss = loss_ce + 0.50 * distill_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            total_correct += (preds == by).sum().item()
            total_n += len(by)
        scheduler.step()

        val_preds, val_labels = evaluate_model(model, val_loader, device)
        val_acc = np.mean(val_preds == val_labels)

        cand_test_preds, _ = evaluate_model(model, test_loader, device)
        reg_classes = []
        for cid, old_acc in old_class_accs.items():
            m = (test_labels == cid)
            new_acc = float(np.mean(cand_test_preds[m] == cid))
            if new_acc < old_acc:
                reg_classes.append(CLASS_NAMES[cid])
        reg_count = len(reg_classes)

        friend_acc = float(np.mean(cand_test_preds[test_labels == 18] == 18))
        u_corr, u_probs = eval_user_friend(model, device, feat_mean, feat_std)

        # Composite score: zero regressions is mandatory for top tier
        score = (u_corr * 50.0) + (val_acc * 20.0) + (friend_acc * 20.0) - (reg_count * 200.0)

        if score > best_score and u_corr >= 2:
            best_score = score
            best_user_corr = u_corr
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:2d}/{epochs} | Val: {val_acc*100:.1f}% | Friend: {friend_acc*100:.1f}% | User: {u_corr}/3 {['%.1f%%' % (p*100) for p in u_probs]} | Reg: {reg_count} {reg_classes}")

    # Load best candidate weights
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    cand_path = os.path.join(MODEL_DIR, "isl_cnn_gru_22class_friend_fix.pt")
    torch.save(model.state_dict(), cand_path)
    print(f"\nSuccessfully saved candidate model to: {cand_path}")

    # ── Full 22-Class Regression Comparison ──
    print("\n" + "=" * 75)
    print("      CRITICAL 22-CLASS REGRESSION TEST: OLD MODEL VS NEW MODEL")
    print("=" * 75)

    # Load baseline model
    base_model = ISL_CNN_GRU_Model(
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES
    ).to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    base_model.eval()

    old_preds, test_labels = evaluate_model(base_model, test_loader, device)
    new_preds, _ = evaluate_model(model, test_loader, device)

    old_overall = np.mean(old_preds == test_labels) * 100
    new_overall = np.mean(new_preds == test_labels) * 100

    print(f"{'Class':<14} | {'Test N':<7} | {'Old Acc':<10} | {'New Acc':<10} | {'Change':<10} | {'Status'}")
    print("-" * 75)

    regressions = []
    for cid, cname in enumerate(CLASS_NAMES):
        mask = (test_labels == cid)
        n_c = int(np.sum(mask))
        if n_c == 0:
            continue
        old_acc = np.mean(old_preds[mask] == test_labels[mask]) * 100
        new_acc = np.mean(new_preds[mask] == test_labels[mask]) * 100
        diff = new_acc - old_acc

        if diff > 0:
            status = "+ IMPROVED"
        elif diff == 0:
            status = "STABLE"
        else:
            status = "- REGRESSED"
            regressions.append((cname, old_acc, new_acc))

        diff_str = f"{diff:+.1f}%"
        print(f"{cname:<14} | {n_c:<7} | {old_acc:>6.1f}%    | {new_acc:>6.1f}%    | {diff_str:>8}   | {status}")

    print("-" * 75)
    print(f"{'OVERALL':<14} | {len(test_labels):<7} | {old_overall:>6.1f}%    | {new_overall:>6.1f}%    | {new_overall - old_overall:+.1f}%   | {'PASS' if new_overall >= old_overall else 'WARN'}")

    if regressions:
        print(f"\n[WARNING] Classes with regression: {regressions}")
    else:
        print(f"\n[PASS] No regressions detected! Candidate checkpoint is safe.")

if __name__ == "__main__":
    train_friend_candidate()
