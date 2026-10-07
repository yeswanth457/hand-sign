"""
CNN-GRU Deep Neural Network for Sign Language Recognition.
Combines 1D Convolutional Neural Network (Spatial Feature Extraction) with
Bidirectional Gated Recurrent Unit (Temporal Feature Extraction) and Linear Classifier Head.

Data Contract:
- Input shape: (Batch_Size, 25, 6)
- Output shape: (Batch_Size, 21) logits over 21 ISL sign classes
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TOKEN_DIM, MAX_SEQ_LEN, NUM_CLASSES, MODEL_DIR,
    CLASS_NAMES, INDEX_TO_CLASS, CLASS_TO_INDEX
)

_brother_updated_path = os.path.join(MODEL_DIR, "isl_cnn_gru_22class_brother_updated.pt")
_best_22_path = os.path.join(MODEL_DIR, "isl_cnn_gru_22class_best.pt")
_best_21_path = os.path.join(MODEL_DIR, "isl_cnn_gru_21class_best.pt")
_default_path = os.path.join(MODEL_DIR, "isl_cnn_gru.pt")
if os.path.exists(_brother_updated_path):
    CNN_GRU_MODEL_PATH = _brother_updated_path
elif os.path.exists(_best_22_path):
    CNN_GRU_MODEL_PATH = _best_22_path
elif os.path.exists(_best_21_path):
    CNN_GRU_MODEL_PATH = _best_21_path
else:
    CNN_GRU_MODEL_PATH = _default_path


class ISL_CNN_GRU_Model(nn.Module):
    def __init__(
        self,
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.2
    ):
        super(ISL_CNN_GRU_Model, self).__init__()

        self.token_dim = token_dim
        self.num_classes = num_classes

        # 1D CNN Spatial Feature Extractor
        self.conv1 = nn.Conv1d(
            in_channels=token_dim,
            out_channels=cnn_channels[0],
            kernel_size=3,
            padding=1
        )
        self.ln1 = nn.LayerNorm(cnn_channels[0])

        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels[0],
            out_channels=cnn_channels[1],
            kernel_size=3,
            padding=1
        )
        self.ln2 = nn.LayerNorm(cnn_channels[1])

        # GRU Temporal Feature Extractor
        self.gru = nn.GRU(
            input_size=cnn_channels[1],
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0
        )

        gru_out_dim = gru_hidden_dim * 2  # Bidirectional

        # Classifier Head
        self.fc1 = nn.Linear(gru_out_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward Pass:
        x: (B, T, C) tensor where T=25, C=6
        Returns: logits of shape (B, num_classes)
        """
        B, T, C = x.shape

        # Transpose for 1D CNN: (B, T, C) -> (B, C, T)
        x_cnn = x.transpose(1, 2)

        # 1D Conv Layer 1
        x_cnn = F.relu(self.conv1(x_cnn)).transpose(1, 2)  # (B, T, 32)
        x_cnn = self.ln1(x_cnn).transpose(1, 2)             # (B, 32, T)

        # 1D Conv Layer 2
        x_cnn = F.relu(self.conv2(x_cnn)).transpose(1, 2)  # (B, T, 64)
        x_cnn = self.ln2(x_cnn)                             # (B, T, 64)

        # Apply Bidirectional GRU
        gru_out, _ = self.gru(x_cnn)  # (B, T, 256)

        # Mean pooling across sequence length T
        temporal_embedding = torch.mean(gru_out, dim=1)  # (B, 256)

        # Linear Classifier Head
        feat = F.relu(self.fc1(temporal_embedding))
        feat = self.dropout(feat)
        logits = self.classifier(feat)  # (B, num_classes)

        return logits


def resample_tokens(tokens, target_t=25):
    """Resamples token sequence (N, C) to exact target length target_t using smooth temporal interpolation."""
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    n, c = tokens.shape
    if n == target_t:
        return tokens.astype(np.float32)
    if n == 1:
        return np.repeat(tokens.astype(np.float32), target_t, axis=0)
    old_t = np.linspace(0.0, 1.0, max(1, n))
    new_t = np.linspace(0.0, 1.0, target_t)
    resampled = np.zeros((target_t, c), dtype=np.float32)
    for j in range(c):
        resampled[:, j] = np.interp(new_t, old_t, tokens[:, j])
    return resampled


class CNNGRUInferenceEngine:
    """Inference engine for the 21-class ISL CNN-GRU model."""
    
    def __init__(self, model_path=CNN_GRU_MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = NUM_CLASSES  # Exactly 21 classes
        self.class_names = CLASS_NAMES
        self.id_to_word = INDEX_TO_CLASS
        self.model_loaded = False

        # Feature normalization stats
        mean_path = os.path.join(MODEL_DIR, "feature_mean.npy")
        std_path = os.path.join(MODEL_DIR, "feature_std.npy")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.feature_mean = np.load(mean_path).astype(np.float32)
            self.feature_std = np.load(std_path).astype(np.float32)
            # Prevent division by zero
            self.feature_std = np.where(self.feature_std < 1e-7, 1.0, self.feature_std)
            print(f"[CNNGRUInferenceEngine] Feature normalization active.")
        else:
            self.feature_mean = None
            self.feature_std = None

        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                
                # Verify checkpoint has matching token_dim and output classes
                if isinstance(state_dict, dict):
                    if "conv1.weight" in state_dict:
                        ckpt_dim = state_dict["conv1.weight"].shape[1]
                        if ckpt_dim != TOKEN_DIM:
                            print(f"[CNNGRUInferenceEngine] WARNING: Checkpoint has token_dim={ckpt_dim}, expected {TOKEN_DIM}.")
                            self._init_fresh_model()
                            return
                    if "classifier.weight" in state_dict:
                        ckpt_classes = state_dict["classifier.weight"].shape[0]
                        if ckpt_classes != NUM_CLASSES:
                            print(f"[CNNGRUInferenceEngine] WARNING: Checkpoint has {ckpt_classes} classes, expected {NUM_CLASSES}.")
                            self._init_fresh_model()
                            return

                self.model = ISL_CNN_GRU_Model(
                    token_dim=TOKEN_DIM,
                    cnn_channels=(32, 64),
                    gru_hidden_dim=128,
                    gru_num_layers=2,
                    num_classes=NUM_CLASSES
                ).to(self.device)

                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_loaded = True
                print(f"[CNNGRUInferenceEngine] Loaded CNN-GRU weights from {self.model_path} ({NUM_CLASSES} classes)")
            except Exception as e:
                print(f"[CNNGRUInferenceEngine] Warning: Failed to load weights: {e}")
                self._init_fresh_model()
        else:
            print(f"[CNNGRUInferenceEngine] Checkpoint not found at {self.model_path}. Using initial weights.")
            self._init_fresh_model()
    
    def _init_fresh_model(self):
        """Initialize model with random weights (untrained)."""
        self.model = ISL_CNN_GRU_Model(
            token_dim=TOKEN_DIM,
            cnn_channels=(32, 64),
            gru_hidden_dim=128,
            gru_num_layers=2,
            num_classes=NUM_CLASSES
        ).to(self.device)
        self.model.eval()
        self.model_loaded = False

    def predict_sequence(self, token_sequence, max_seq_len=25):
        """
        Predict ISL Sign from gesture token sequence [T x 6].
        
        Input sequence is padded/truncated to exactly 25 tokens.
        Returns dict with predicted class, confidence, and top probabilities.
        """
        if token_sequence is None or len(token_sequence) == 0:
            return {
                "class_id": -1,
                "word": "--",
                "confidence": 0.0,
                "probabilities": {}
            }

        tokens = np.array(token_sequence, dtype=np.float32)
        if tokens.ndim == 1:
            tokens = tokens.reshape(1, -1)
        
        seq_len = len(tokens)

        # Smoothly resample sequence to exactly max_seq_len (25) matching training pipeline
        token_sub = tokens[:, :TOKEN_DIM]
        if seq_len == max_seq_len:
            padded = token_sub.astype(np.float32)
        else:
            padded = resample_tokens(token_sub, target_t=max_seq_len)

        # Apply feature normalization (same as training)
        if self.feature_mean is not None and self.feature_std is not None:
            padded = (padded - self.feature_mean) / self.feature_std

        tensor_in = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        class_name = self.id_to_word.get(pred_id, f"class_{pred_id}")

        # Section 25: Uncertainty & Ambiguity Gating
        # If model is uncertain (low confidence or top two classes are tied),
        # DO NOT force output into SORRY or STOP! Return WAITING FOR CLEAR GESTURE.
        sorted_probs = np.sort(probs)[::-1]
        top1 = sorted_probs[0]
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2

        if confidence < 0.40 or (confidence < 0.50 and margin < 0.10):
            class_name = "WAITING FOR CLEAR GESTURE"
            pred_id = -1

        top_probs = {
            self.id_to_word.get(idx, f"class_{idx}"): round(float(probs[idx]), 4)
            for idx in np.argsort(probs)[::-1][:min(5, self.num_classes)]
        }

        return {
            "class_id": pred_id,
            "word": class_name,
            "confidence": confidence,
            "probabilities": top_probs
        }

