import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_vocabulary(self):
        response = self.client.get("/api/vocabulary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("vocabulary", data)
        self.assertEqual(data["total"], 21)

    def test_pipeline_status(self):
        response = self.client.get("/api/pipeline_status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("backend_mediapipe_available", data)

    def test_process_token_no_hand(self):
        payload = {
            "token": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "frame_id": 1,
            "has_hand": False
        }
        response = self.client.post("/api/process_token", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["early_decision"]["state"], "NO_HAND")

    def test_process_token_with_hand(self):
        payload = {
            "token": [0.55, 0.45, 0.01, -0.02, 0.05, 0.10],
            "frame_id": 2,
            "has_hand": True,
            "hand_center": [0.55, 0.45],
            "shoulder_center": [0.50, 0.35]
        }
        response = self.client.post("/api/process_token", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("motion_energy", data)

    def test_process_frame_image(self):
        # Create a blank BGR image frame (480x640x3)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white circle to simulate a visual pattern
        cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)
        _, buffer = cv2.imencode('.jpg', img)
        base64_str = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "image_base64": f"data:image/jpeg;base64,{base64_str}",
            "frame_id": 3,
            "timestamp_ms": 1000.0
        }
        response = self.client.post("/api/process_frame_image", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("frame_id", data)
        self.assertIn("prediction", data)
        self.assertEqual(data["pipeline_source"], "Backend MediaPipe (Python)")

    def test_dataset_browser(self):
        response = self.client.get("/api/dataset_browser?offset=0&length=2")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rows", data)
        self.assertIn("total", data)
        self.assertGreater(data["total"], 0)
        self.assertEqual(len(data["rows"]), 2)
        self.assertIn("word", data["rows"][0])

if __name__ == "__main__":
    unittest.main()

