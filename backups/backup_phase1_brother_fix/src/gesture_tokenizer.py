"""
Phase 6: Gesture Token Generation (Two-Hand Supported).
Converts extracted landmarks into compact 12D vectors:
[LHx, LHy, LMx, LMy, LRx, LRy, RHx, RHy, RMx, RMy, RRx, RRy]
- Left Hand (0..5): Position, velocity, body-relative offset (or 0 if missing)
- Right Hand (6..11): Position, velocity, body-relative offset (or 0 if missing)
"""

import numpy as np
from config import TOKEN_DIM


class GestureTokenizer:
    def __init__(self, max_occlusion_grace=4):
        self.prev_left_center = None
        self.prev_right_center = None
        self.last_valid_left_center = None
        self.last_valid_right_center = None
        self.left_missing_frames = 0
        self.right_missing_frames = 0
        self.max_occlusion_grace = max_occlusion_grace

    def tokenize_frame(self, landmark_data):
        """
        Converts landmark_data dict into a 12D float token vector:
        [LHx, LHy, LMx, LMy, LRx, LRy, RHx, RHy, RMx, RMy, RRx, RRy]
        Supports two-hand occlusion grace period (3-5 frames carry-forward)
        to prevent temporary overlap from destroying dual-hand representation.
        """
        pose = landmark_data.get("pose", {})
        ls = pose.get("LS", (0.4, 0.35, 0.0))
        rs = pose.get("RS", (0.6, 0.35, 0.0))
        lsx, lsy = float(ls[0]), float(ls[1])
        rsx, rsy = float(rs[0]), float(rs[1])
        mid_x = (lsx + rsx) / 2.0

        lh_center = landmark_data.get("left_hand_center")
        rh_center = landmark_data.get("right_hand_center")
        hands = landmark_data.get("hands", [])

        # Spatial fallback if left/right centers are not explicitly separated
        if lh_center is None and rh_center is None and len(hands) > 0:
            if len(hands) == 1:
                pts = np.array(hands[0])
                c = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
                if c[0] < mid_x:
                    lh_center = c
                else:
                    rh_center = c
            elif len(hands) >= 2:
                c0 = (float(np.mean(np.array(hands[0])[:, 0])), float(np.mean(np.array(hands[0])[:, 1])))
                c1 = (float(np.mean(np.array(hands[1])[:, 0])), float(np.mean(np.array(hands[1])[:, 1])))
                if c0[0] < c1[0]:
                    lh_center, rh_center = c0, c1
                else:
                    lh_center, rh_center = c1, c0
        elif lh_center is None and rh_center is None and "hand_center" in landmark_data:
            c = landmark_data["hand_center"]
            if c[0] < mid_x:
                lh_center = c
            else:
                rh_center = c

        left_carried_forward = False
        right_carried_forward = False

        # ── Left Hand Features (0..5) ──
        if lh_center is not None:
            lhx, lhy = float(lh_center[0]), float(lh_center[1])
            if self.prev_left_center is not None:
                lmx = lhx - self.prev_left_center[0]
                lmy = lhy - self.prev_left_center[1]
            else:
                lmx, lmy = 0.0, 0.0
            lrx = lhx - lsx
            lry = lhy - lsy
            self.prev_left_center = (lhx, lhy)
            self.last_valid_left_center = (lhx, lhy)
            self.left_missing_frames = 0
            left_token = [lhx, lhy, lmx, lmy, lrx, lry]
        elif (self.last_valid_left_center is not None and 
              self.left_missing_frames < self.max_occlusion_grace and 
              (rh_center is not None or self.last_valid_right_center is not None)):
            # Step 7: Occlusion Grace Period -- carry forward last valid left hand position with 0 velocity
            lhx, lhy = self.last_valid_left_center
            lmx, lmy = 0.0, 0.0
            lrx = lhx - lsx
            lry = lhy - lsy
            self.left_missing_frames += 1
            left_carried_forward = True
            left_token = [lhx, lhy, lmx, lmy, lrx, lry]
        else:
            self.prev_left_center = None
            self.last_valid_left_center = None
            self.left_missing_frames = 0
            left_token = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ── Right Hand Features (6..11) ──
        if rh_center is not None:
            rhx, rhy = float(rh_center[0]), float(rh_center[1])
            if self.prev_right_center is not None:
                rmx = rhx - self.prev_right_center[0]
                rmy = rhy - self.prev_right_center[1]
            else:
                rmx, rmy = 0.0, 0.0
            rrx = rhx - rsx
            rry = rhy - rsy
            self.prev_right_center = (rhx, rhy)
            self.last_valid_right_center = (rhx, rhy)
            self.right_missing_frames = 0
            right_token = [rhx, rhy, rmx, rmy, rrx, rry]
        elif (self.last_valid_right_center is not None and 
              self.right_missing_frames < self.max_occlusion_grace and 
              (lh_center is not None or self.last_valid_left_center is not None)):
            # Step 7: Occlusion Grace Period -- carry forward last valid right hand position with 0 velocity
            rhx, rhy = self.last_valid_right_center
            rmx, rmy = 0.0, 0.0
            rrx = rhx - rsx
            rry = rhy - rsy
            self.right_missing_frames += 1
            right_carried_forward = True
            right_token = [rhx, rhy, rmx, rmy, rrx, rry]
        else:
            self.prev_right_center = None
            self.last_valid_right_center = None
            self.right_missing_frames = 0
            right_token = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        token = np.array(left_token + right_token, dtype=np.float32)
        return token

    def tokenize_sequence(self, landmark_sequence):
        """
        Converts a list of landmark_data frame dictionaries into an (N, 12) token matrix.
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
        self.prev_left_center = None
        self.prev_right_center = None
        self.last_valid_left_center = None
        self.last_valid_right_center = None
        self.left_missing_frames = 0
        self.right_missing_frames = 0

