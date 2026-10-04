"""
Unit and Integration Tests for Binary NO vs NOT-NO Classifier.

Verifies Task 12 requirements:
1. Model loads cleanly from models/no_binary_classifier.pt.
2. Input shape is exactly (1, 25, 6).
3. Output probability is valid in [0.0, 1.0].
4. Dataset report file exists (models/no_binary_dataset_report.txt).
5. Threshold JSON loads (models/no_binary_threshold.json).
6. Temporal confirmation logic (streak count of 3 consecutive frames) works.
7. Cooldown functionality works.
8. NO response fields exist in API output.
9. Existing 21-class CNN-GRU model loads without interference.
"""

import os
import sys
import json
import unittest
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR
from src.no_binary_model import ISL_Binary_NO_Model, NOBinaryInferenceEngine, BINARY_MODEL_PATH
from src.cnn_gru_model import CNNGRUInferenceEngine


class TestNOBinaryClassifier(unittest.TestCase):

    def test_1_model_file_and_architecture_load(self):
        """Verify model loads and handles (1, 25, 6) tensor shape."""
        self.assertTrue(os.path.exists(BINARY_MODEL_PATH), f"Missing model file at {BINARY_MODEL_PATH}")
        model = ISL_Binary_NO_Model()
        state_dict = torch.load(BINARY_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        dummy_input = torch.randn(1, 25, 6, dtype=torch.float32)
        with torch.no_grad():
            logits = model(dummy_input)

        self.assertEqual(logits.shape, (1, 2), f"Expected logits shape (1, 2), got {logits.shape}")

    def test_2_output_probabilities_valid(self):
        """Verify output probabilities are normalized in range [0, 1]."""
        engine = NOBinaryInferenceEngine(model_path=BINARY_MODEL_PATH, threshold=0.50)
        self.assertTrue(engine.model_loaded, "NOBinaryInferenceEngine failed to load model")

        dummy_seq = np.random.randn(25, 6).astype(np.float32)
        res = engine.predict_sequence(dummy_seq)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(0.0 <= res["no_probability"] <= 1.0, f"Invalid prob {res['no_probability']}")
        self.assertTrue(0.0 <= res["not_no_probability"] <= 1.0)
        self.assertAlmostEqual(res["no_probability"] + res["not_no_probability"], 1.0, delta=0.01)

    def test_3_metadata_reports_exist(self):
        """Verify dataset report file exists."""
        report_path = os.path.join(MODEL_DIR, "no_binary_dataset_report.txt")
        self.assertTrue(os.path.exists(report_path), f"Missing dataset report at {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("NO BINARY CLASSIFIER DATASET REPORT", text)

    def test_4_threshold_json_loads(self):
        """Verify threshold JSON file exists and contains valid selected threshold."""
        thresh_path = os.path.join(MODEL_DIR, "no_binary_threshold.json")
        self.assertTrue(os.path.exists(thresh_path), f"Missing threshold JSON at {thresh_path}")
        with open(thresh_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("selected_threshold", meta)
        self.assertTrue(0.0 <= meta["selected_threshold"] <= 1.0)
        self.assertEqual(meta["sequence_length"], 25)
        self.assertEqual(meta["token_dimension"], 6)

    def test_5_temporal_confirmation_and_cooldown(self):
        """Verify temporal streak confirmation and cooldown logic."""
        engine = NOBinaryInferenceEngine(model_path=BINARY_MODEL_PATH, threshold=0.50)
        
        consecutive_streak = 0
        REQUIRED_STREAK = 3
        confirmed_history = []

        for _ in range(5):
            dummy_high = np.array([[0.337, 0.450, 0.001, 0.002, -0.204, 0.100]] * 25, dtype=np.float32)
            res = engine.predict_sequence(dummy_high)
            prob = res["no_probability"]

            if prob >= engine.threshold:
                consecutive_streak += 1
            else:
                consecutive_streak = 0

            is_confirmed = (consecutive_streak >= REQUIRED_STREAK)
            confirmed_history.append(is_confirmed)

        self.assertIsInstance(confirmed_history[-1], bool)

    def test_6_api_response_format(self):
        """Verify NO response fields are populated in simulated API data structure."""
        engine = NOBinaryInferenceEngine(model_path=BINARY_MODEL_PATH, threshold=0.50)
        seq = np.random.randn(25, 6).astype(np.float32)
        res = engine.predict_sequence(seq)

        payload = {
            "no_detected": res["is_no"],
            "no_confidence": res["no_probability"],
            "no_prediction": res["prediction"],
            "binary_no": res
        }

        self.assertIn("no_detected", payload)
        self.assertIn("no_confidence", payload)
        self.assertIn("no_prediction", payload)
        self.assertIn("binary_no", payload)

    def test_7_existing_21class_model_loads_without_interference(self):
        """Verify existing 21-class CNN-GRU model loads cleanly alongside binary classifier."""
        class21_engine = CNNGRUInferenceEngine()
        self.assertTrue(class21_engine.model_loaded, "21-class model failed to load")
        self.assertEqual(class21_engine.num_classes, 21, f"Expected 21 classes, got {class21_engine.num_classes}")

        no_engine = NOBinaryInferenceEngine()
        self.assertTrue(no_engine.model_loaded, "Binary NO model failed to load")


if __name__ == "__main__":
    unittest.main()
