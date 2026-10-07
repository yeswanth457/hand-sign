"""
Phase 2: Dataset creation and management for ISL dynamic signs.
Supports raw recording, synthetic ISL landmark generation, and dataset split management.
"""

import os
import json
import numpy as np
from config import ISL_VOCABULARY, WORD_TO_ID, DATASET_DIR, TOKEN_DIM


class ISLDatasetManager:
    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.raw_dir = os.path.join(dataset_dir, "raw")
        self.processed_dir = os.path.join(dataset_dir, "processed")
        self.train_dir = os.path.join(dataset_dir, "train")
        self.val_dir = os.path.join(dataset_dir, "val")
        self.test_dir = os.path.join(dataset_dir, "test")
        
        self.ensure_directories()

    def ensure_directories(self):
        """Creates dataset directories for all 50 vocabulary classes."""
        for word in ISL_VOCABULARY:
            os.makedirs(os.path.join(self.raw_dir, word), exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir, word), exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def generate_synthetic_dataset(self, samples_per_class=30, seq_length=25):
        """
        Generates synthetic 6D gesture token sequences for 50 ISL sign classes.
        Each gesture class has a distinct spatial trajectory pattern (circles, waves, vertical drops, 
        sweeps, pulses, etc.) with realistic human variance and noise.
        """
        np.random.seed(42)
        print(f"[DatasetManager] Generating synthetic dataset ({samples_per_class} samples/class)...")

        dataset_x = []
        dataset_y = []

        t = np.linspace(0, np.pi, seq_length)

        for class_id, word in enumerate(ISL_VOCABULARY):
            for sample_idx in range(samples_per_class):
                # Base dynamic frequency and phase shift per sample
                freq = 1.0 + (class_id % 5) * 0.4 + np.random.uniform(-0.1, 0.1)
                phase = (class_id * 0.2) + np.random.uniform(-0.05, 0.05)
                scale = 0.3 + (class_id % 3) * 0.2 + np.random.uniform(-0.02, 0.02)
                
                # Hand center trajectories Hx, Hy (0 to 1 range)
                base_hx = 0.5 + scale * np.sin(freq * t + phase)
                base_hy = 0.5 + scale * np.cos(freq * t + phase)
                
                # Add class-specific trajectory variations
                if class_id % 4 == 1:
                    base_hy += 0.2 * np.sin(2 * freq * t) # W-shape motion
                elif class_id % 4 == 2:
                    base_hx *= np.exp(-0.2 * t)          # Dampened sweep
                elif class_id % 4 == 3:
                    base_hy += 0.15 * t                  # Downward sweep

                # Motion vectors Mx, My (first order derivative)
                mx = np.gradient(base_hx)
                my = np.gradient(base_hy)

                # Relative body offset Rx, Ry (hand position relative to shoulder center ~ 0.5, 0.3)
                rx = base_hx - (0.5 + np.random.uniform(-0.02, 0.02))
                ry = base_hy - (0.3 + np.random.uniform(-0.02, 0.02))

                # Combine into (seq_length, 6) gesture token sequence
                token_seq = np.stack([base_hx, base_hy, mx, my, rx, ry], axis=-1)

                # Add sensor/signer jitter noise
                noise = np.random.normal(0, 0.015, token_seq.shape)
                token_seq += noise

                dataset_x.append(token_seq.astype(np.float32))
                dataset_y.append(class_id)

        dataset_x = np.array(dataset_x, dtype=np.float32)  # Shape: (N, seq_len, 6)
        dataset_y = np.array(dataset_y, dtype=np.int64)    # Shape: (N,)

        # Train / Val / Test Split (70% train, 15% val, 15% test)
        indices = np.arange(len(dataset_x))
        np.random.shuffle(indices)

        num_samples = len(indices)
        train_end = int(0.70 * num_samples)
        val_end = int(0.85 * num_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        np.save(os.path.join(self.train_dir, "X.npy"), dataset_x[train_idx])
        np.save(os.path.join(self.train_dir, "y.npy"), dataset_y[train_idx])

        np.save(os.path.join(self.val_dir, "X.npy"), dataset_x[val_idx])
        np.save(os.path.join(self.val_dir, "y.npy"), dataset_y[val_idx])

        np.save(os.path.join(self.test_dir, "X.npy"), dataset_x[test_idx])
        np.save(os.path.join(self.test_dir, "y.npy"), dataset_y[test_idx])

        print(f"[DatasetManager] Synthetic dataset saved: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        return {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "total": num_samples
        }

    def load_dataset(self, split="train"):
        """Loads dataset split arrays (X, y)."""
        split_dir = getattr(self, f"{split}_dir", self.train_dir)
        x_path = os.path.join(split_dir, "X.npy")
        y_path = os.path.join(split_dir, "y.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"[DatasetManager] Dataset split '{split}' not found. Generating synthetic dataset...")
            self.generate_synthetic_dataset()

        x = np.load(x_path)
        y = np.load(y_path)
        return x, y


if __name__ == "__main__":
    manager = ISLDatasetManager()
    stats = manager.generate_synthetic_dataset()
    print("Dataset generation complete:", stats)
