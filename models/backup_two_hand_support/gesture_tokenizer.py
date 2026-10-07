"""
Phase 6: Gesture Token Generation.
Converts extracted landmarks into compact 6D vectors:
[Hx, Hy, Mx, My, Rx, Ry]
- Hx, Hy: Normalized hand center position
- Mx, My: Hand motion velocity vector (ΔHx, ΔHy)
- Rx, Ry: Relative hand-to-body shoulder center position (Hx - Sx, Hy - Sy)
"""

import numpy as np
from config import TOKEN_DIM


class GestureTokenizer:
    def __init__(self):
        self.prev_hand_center = None

    def tokenize_frame(self, landmark_data):
        """
        Converts landmark_data dict into a 6D float token vector:
        [Hx, Hy, Mx, My, Rx, Ry]
        """
        hx, hy = landmark_data.get("hand_center", (0.5, 0.5))
        sx, sy = landmark_data.get("shoulder_center", (0.5, 0.35))

        # Calculate Motion Vector Mx, My
        if self.prev_hand_center is not None:
            mx = hx - self.prev_hand_center[0]
            my = hy - self.prev_hand_center[1]
        else:
            mx, my = 0.0, 0.0

        self.prev_hand_center = (hx, hy)

        # Calculate Relative Body Vector Rx, Ry
        rx = hx - sx
        ry = hy - sy

        token = np.array([hx, hy, mx, my, rx, ry], dtype=np.float32)
        return token

    def tokenize_sequence(self, landmark_sequence):
        """
        Converts a list of landmark_data frame dictionaries into an (N, 6) token matrix.
        """
        self.reset()
        tokens = []
        for frame_data in landmark_sequence:
            token = self.tokenize_frame(frame_data)
            tokens.append(token)
        
        if len(tokens) == 0:
            return np.zeros((0, TOKEN_DIM), dtype=np.float32)
        return np.array(tokens, dtype=np.float32)

    def reset(self):
        self.prev_hand_center = None
