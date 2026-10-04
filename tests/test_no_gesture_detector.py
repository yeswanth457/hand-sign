"""
Unit and Integration Tests for NOGestureDetector (Standard unittest).

Tests:
1. Valid NO sequence
2. Empty sequence
3. No hand
4. One-frame sequence
5. Short sequence
6. Sequence with NaN
7. Sequence with Inf
8. Coordinates outside range
9. Movement in wrong dominant axis
10. Correct NO-like movement
11. Two-hand input
12. Left/right handedness
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.no_gesture_detector import NOGestureDetector


def make_dummy_hand(x_offset=0.35, y_offset=0.50, z_offset=0.0):
    """Generates a valid (21, 3) hand landmark array."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [x_offset, y_offset, z_offset]
    for i in range(1, 5):
        pts[i] = [x_offset + 0.02 * i, y_offset - 0.02 * i, z_offset - 0.01 * i]
    for i in range(5, 9):
        pts[i] = [x_offset + 0.01 * (i - 4), y_offset - 0.04 * (i - 4), z_offset - 0.02 * (i - 4)]
    for i in range(9, 13):
        pts[i] = [x_offset + 0.005 * (i - 8), y_offset - 0.045 * (i - 8), z_offset - 0.02 * (i - 8)]
    for i in range(13, 17):
        pts[i] = [x_offset - 0.005 * (i - 12), y_offset - 0.015 * (i - 12), z_offset - 0.01 * (i - 12)]
    for i in range(17, 21):
        pts[i] = [x_offset - 0.01 * (i - 16), y_offset - 0.01 * (i - 16), z_offset - 0.01 * (i - 16)]
    return pts


class TestNOGestureDetector(unittest.TestCase):
    def setUp(self):
        self.detector = NOGestureDetector()

    # 1. Valid NO sequence
    def test_valid_no_sequence(self):
        sequence = []
        for t in range(40):
            y_shift = 0.02 * np.sin(t * 0.3)
            pts = make_dummy_hand(x_offset=0.36, y_offset=0.55 + y_shift)
            sequence.append({
                "hands": [pts],
                "pose": {"LS": (0.4, 0.35, 0.0), "RS": (0.6, 0.35, 0.0)},
                "hand_center": (0.36, 0.55 + y_shift)
            })
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 40)
        self.assertGreater(res["no_score"], 0.5)
        self.assertEqual(res["dominant_axis"], "Y")

    # 2. Empty sequence
    def test_empty_sequence(self):
        res = self.detector.evaluate_sequence([])
        self.assertEqual(res["frames_analyzed"], 0)
        self.assertEqual(res["no_score"], 0.0)
        self.assertFalse(res["is_no_gesture"])

    # 3. No hand
    def test_no_hand(self):
        sequence = [{"hands": [], "pose": {}}, {"hands": None, "pose": {}}]
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 0)
        self.assertEqual(res["no_score"], 0.0)
        self.assertFalse(res["is_no_gesture"])

    # 4. One-frame sequence
    def test_one_frame_sequence(self):
        pts = make_dummy_hand(x_offset=0.36, y_offset=0.55)
        sequence = [{"hands": [pts], "pose": {}, "hand_center": (0.36, 0.55)}]
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 1)
        self.assertIn("no_score", res)

    # 5. Short sequence
    def test_short_sequence(self):
        sequence = []
        for t in range(5):
            pts = make_dummy_hand(x_offset=0.36, y_offset=0.55 + 0.01 * t)
            sequence.append({"hands": [pts], "hand_center": (0.36, 0.55 + 0.01 * t)})
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 5)
        self.assertLess(res["duration_score"], 1.0)

    # 6. Sequence with NaN
    def test_sequence_with_nan(self):
        sequence = []
        for t in range(25):
            pts = make_dummy_hand()
            if t == 10:
                pts[8, 0] = np.nan
            sequence.append({"hands": [pts], "hand_center": (0.36, 0.55)})
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 24)
        self.assertGreater(res["no_score"], 0.0)

    # 7. Sequence with Inf
    def test_sequence_with_inf(self):
        sequence = []
        for t in range(25):
            pts = make_dummy_hand()
            if t == 5:
                pts[0, 1] = np.inf
            sequence.append({"hands": [pts], "hand_center": (0.36, 0.55)})
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 24)

    # 8. Coordinates outside range
    def test_coordinates_outside_range(self):
        sequence = []
        for t in range(30):
            pts = make_dummy_hand(x_offset=0.99, y_offset=0.99)
            sequence.append({"hands": [pts], "hand_center": (0.99, 0.99)})
        res = self.detector.evaluate_sequence(sequence)
        self.assertLess(res["spatial_score"], 0.5)
        self.assertFalse(res["is_no_gesture"])

    # 9. Movement in wrong dominant axis
    def test_movement_in_wrong_dominant_axis(self):
        sequence = []
        for t in range(30):
            x_shift = 0.05 * t
            pts = make_dummy_hand(x_offset=0.36 + x_shift, y_offset=0.55)
            sequence.append({"hands": [pts], "hand_center": (0.36 + x_shift, 0.55)})
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["dominant_axis"], "X")
        self.assertLess(res["axis_score"], 0.5)

    # 10. Correct NO-like movement
    def test_correct_no_like_movement(self):
        sequence = []
        for t in range(40):
            y_shift = 0.03 * np.sin(t * 0.2)
            pts = make_dummy_hand(x_offset=0.38, y_offset=0.58 + y_shift)
            sequence.append({"hands": [pts], "hand_center": (0.38, 0.58 + y_shift)})
        res = self.detector.evaluate_sequence(sequence)
        self.assertGreater(res["no_score"], 0.70)
        self.assertTrue(res["is_no_gesture"])

    # 11. Two-hand input
    def test_two_hand_input(self):
        sequence = []
        for t in range(30):
            h1 = make_dummy_hand(x_offset=0.36, y_offset=0.55 + 0.01 * np.sin(t))
            h2 = make_dummy_hand(x_offset=0.70, y_offset=0.70)
            sequence.append({
                "hands": [h1, h2],
                "hand_center": (0.36, 0.55 + 0.01 * np.sin(t))
            })
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 30)
        self.assertGreater(res["no_score"], 0.50)

    # 12. Left/right handedness
    def test_left_right_handedness(self):
        sequence = []
        for t in range(30):
            pts = make_dummy_hand(x_offset=0.38, y_offset=0.55)
            sequence.append({
                "hands": [pts],
                "handedness": "Left",
                "hand_center": (0.38, 0.55)
            })
        res = self.detector.evaluate_sequence(sequence)
        self.assertEqual(res["frames_analyzed"], 30)
        self.assertIn("no_score", res)


if __name__ == "__main__":
    unittest.main()
