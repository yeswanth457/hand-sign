"""
Temporal Binary NO vs NOT-NO Classifier Model.

Architecture: 1D CNN + Bidirectional GRU + Linear Head.
Input: (Batch_Size, 25, 6) tensor of 6D tokens [Hx, Hy, Mx, My, Rx, Ry].
Output: Logits of shape (Batch_Size, 2) where class 0 = NOT_NO, class 1 = NO.

Does NOT modify the 21-class CNN-GRU model or weights.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import TOKEN_DIM, MAX_SEQ_LEN, MODEL_DIR

BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "no_binary_classifier.pt")


class ISL_Binary_NO_Model(nn.Module):
    def __init__(
        self,
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        dropout=0.2
    ):
        super(ISL_Binary_NO_Model, self).__init__()

        self.token_dim = token_dim

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

        # Classifier Head (2 classes: 0 = NOT_NO, 1 = NO)
        self.fc1 = nn.Linear(gru_out_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        """
        Forward Pass:
        x: (B, T, C) tensor where T=25, C=6
        Returns: logits of shape (B, 2)
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

        # Classifier Head
        feat = F.relu(self.fc1(temporal_embedding))
        feat = self.dropout(feat)
        logits = self.classifier(feat)  # (B, 2)

        return logits


class NOBinaryInferenceEngine:
    """Inference Engine for Binary NO vs NOT-NO Classifier."""

    def __init__(self, model_path=BINARY_MODEL_PATH, threshold=0.60):
        self.model_path = model_path
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.model = None

        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model = ISL_Binary_NO_Model().to(self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_loaded = True
                print(f"[NOBinaryInferenceEngine] Loaded binary model from {self.model_path}")
            except Exception as e:
                print(f"[NOBinaryInferenceEngine] Warning: Could not load binary model: {e}")
                self.model = None
        else:
            print(f"[NOBinaryInferenceEngine] Binary model not found at {self.model_path}")

    def predict_sequence(self, token_sequence, max_seq_len=25):
        """
        Predicts binary NO probability and classification from gesture token sequence [T x 6].
        """
        if self.model is None or not self.model_loaded:
            return {
                "is_no": False,
                "no_probability": 0.0,
                "prediction": "NOT_NO",
                "status": "MODEL_NOT_LOADED"
            }

        if token_sequence is None or len(token_sequence) == 0:
            return {
                "is_no": False,
                "no_probability": 0.0,
                "prediction": "NOT_NO",
                "status": "EMPTY_SEQUENCE"
            }

        tokens = np.array(token_sequence, dtype=np.float32)
        if tokens.ndim == 1:
            tokens = tokens.reshape(1, -1)

        n, c = tokens.shape
        if n != max_seq_len:
            # Smooth resampling to 25 frames
            old_t = np.linspace(0.0, 1.0, max(1, n))
            new_t = np.linspace(0.0, 1.0, max_seq_len)
            padded = np.zeros((max_seq_len, c), dtype=np.float32)
            for j in range(c):
                padded[:, j] = np.interp(new_t, old_t, tokens[:, j])
        else:
            padded = tokens.astype(np.float32)

        # Canonical spatial alignment for binary NO model
        # The training data for NO was recorded in canonical right-hand coordinate space (Hx ~ 0.33, Rx ~ -0.20).
        # If live video hand center Hx > 0.50 (e.g. un-flipped webcam or right side of frame), mirror X coordinates.
        aligned_padded = padded.copy()
        if aligned_padded[:, 0].mean() > 0.50:
            aligned_padded[:, 0] = 1.0 - aligned_padded[:, 0]  # Hx
            aligned_padded[:, 2] = -aligned_padded[:, 2]      # Mx
            aligned_padded[:, 4] = -aligned_padded[:, 4]      # Rx

        tensor_in = torch.tensor(aligned_padded, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Print actual tensor statistics immediately before model inference
        t_mean = tensor_in.mean().item()
        t_std = tensor_in.std().item()
        t_min = tensor_in.min().item()
        t_max = tensor_in.max().item()

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        no_prob = float(probs[1])
        is_no = no_prob >= self.threshold
        prediction = "NO" if is_no else "NOT_NO"

        return {
            "is_no": is_no,
            "no_probability": round(no_prob, 4),
            "not_no_probability": round(float(probs[0]), 4),
            "prediction": prediction,
            "status": "SUCCESS"
        }
