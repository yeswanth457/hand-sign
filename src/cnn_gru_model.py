"""
Phase 3, 4, 5: CNN-GRU Deep Neural Network for Sign Language Recognition.
Combines 1D Convolutional Neural Network (Spatial Feature Extraction) with
Bidirectional Gated Recurrent Unit (Temporal Feature Extraction) and Linear Classifier Head.

Data Contract:
- Input shape: (Batch_Size, Sequence_Length, Feature_Dim) = (B, 25, 6)
- CNN Output shape: (B, 25, 64) spatial feature vectors
- GRU Output shape: (B, 256) aggregated temporal feature representation
- Classifier Output shape: (B, num_classes) logits over vocabulary
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
    TOKEN_DIM, MAX_SEQ_LEN, VOCAB_SIZE, MODEL_DIR,
    ISL_VOCABULARY, ID_TO_WORD, WORD_TO_ID
)

CNN_GRU_MODEL_PATH = os.path.join(MODEL_DIR, "isl_cnn_gru.pt")


class ISL_CNN_GRU_Model(nn.Module):
    def __init__(
        self,
        token_dim=TOKEN_DIM,
        cnn_channels=(32, 64),
        gru_hidden_dim=128,
        gru_num_layers=2,
        num_classes=VOCAB_SIZE,
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


class CNNGRUInferenceEngine:
    def __init__(self, model_path=CNN_GRU_MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ISL_CNN_GRU_Model(
            token_dim=TOKEN_DIM,
            cnn_channels=(32, 64),
            gru_hidden_dim=128,
            gru_num_layers=2,
            num_classes=VOCAB_SIZE
        ).to(self.device)

        self.model_loaded = False
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_loaded = True
                print(f"[CNNGRUInferenceEngine] Loaded CNN-GRU weights from {self.model_path}")
            except Exception as e:
                print(f"[CNNGRUInferenceEngine] Warning: Failed to load weights: {e}")
                self.model.eval()
        else:
            print(f"[CNNGRUInferenceEngine] Checkpoint not found at {self.model_path}. Using initial weights.")
            self.model.eval()

    def predict_sequence(self, token_sequence, max_seq_len=MAX_SEQ_LEN):
        """
        Predict ISL Sign from gesture token sequence [T x 6].
        Returns: (predicted_class_id, class_name, confidence, probabilities_dict)
        """
        if token_sequence is None or len(token_sequence) == 0:
            return {
                "class_id": 0,
                "word": ID_TO_WORD.get(0, "unknown"),
                "confidence": 0.0,
                "probabilities": {}
            }

        tokens = np.array(token_sequence, dtype=np.float32)
        seq_len = len(tokens)

        # Pad or truncate to MAX_SEQ_LEN
        if seq_len < max_seq_len:
            padded = np.zeros((max_seq_len, TOKEN_DIM), dtype=np.float32)
            padded[:seq_len] = tokens
        else:
            padded = tokens[:max_seq_len]

        tensor_in = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        class_name = ID_TO_WORD.get(pred_id, f"class_{pred_id}")

        top_probs = {
            ID_TO_WORD.get(idx, f"class_{idx}"): round(float(probs[idx]), 4)
            for idx in np.argsort(probs)[::-1][:5]
        }

        return {
            "class_id": pred_id,
            "word": class_name,
            "confidence": confidence,
            "probabilities": top_probs
        }
