"""
Phase 8: Early Decision and Real-Time Cooldown State Machine.
Triggers positive sign decision when prediction confidence > threshold for N consecutive frames.
Maintains state machine (IDLE, SIGNING, CONFIRMED, COOLDOWN) to prevent duplicate output loops.
"""

from collections import deque
from config import (
    CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES, COOLDOWN_FRAMES,
    IDLE_ENERGY_THRESHOLD, SIGNING_MOTION_THRESHOLD, CLASS_NAMES
)


class State:
    NO_HAND = "NO_HAND"
    READY = "READY"
    COLLECTING = "COLLECTING"
    PREDICTING = "PREDICTING"
    ACCEPTED = "ACCEPTED"
    LOCKED = "LOCKED"
    IDLE = "IDLE"
    SIGNING = "SIGNING"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


class EarlyDecisionEngine:
    def __init__(
        self,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sustained_frames=SUSTAINED_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES,
        idle_threshold=SIGNING_MOTION_THRESHOLD
    ):
        self.confidence_threshold = confidence_threshold
        self.sustained_frames = sustained_frames
        self.cooldown_frames = cooldown_frames
        self.idle_threshold = idle_threshold
        self.valid_classes = set(CLASS_NAMES)

        self.current_state = State.NO_HAND
        self.prediction_window = deque(maxlen=sustained_frames)
        self.cooldown_counter = 0
        self.last_accepted_sign = None

    def process_prediction(self, prediction_res, motion_energy):
        """
        Processes real-time frame model prediction dict:
        {"word": str, "confidence": float, "class_id": int}
        and motion_energy float.
        Enforces strict motion gating and cooldown to prevent rapid duplicate loops.
        """
        word = prediction_res.get("word")
        confidence = prediction_res.get("confidence", 0.0)

        accepted_word = None
        is_accepted = False

        # 1. Cooldown enforcement: tick down cooldown on every frame
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter > 0:
                self.current_state = State.COOLDOWN
                self.prediction_window.clear()
                return {
                    "accepted": False,
                    "word": None,
                    "state": State.COOLDOWN,
                    "confidence": confidence,
                    "cooldown_remaining": self.cooldown_counter,
                    "last_accepted_sign": self.last_accepted_sign,
                    "sustained_count": 0,
                    "sustained_target": self.sustained_frames
                }

        # 2. Motion Gating: Resting / motionless hand is IDLE, UNLESS high-confidence sign is held
        if motion_energy < self.idle_threshold and confidence < 0.65:
            self.current_state = State.IDLE
            self.prediction_window.clear()
            return {
                "accepted": False,
                "word": None,
                "state": State.IDLE,
                "confidence": confidence,
                "cooldown_remaining": 0,
                "last_accepted_sign": self.last_accepted_sign,
                "sustained_count": 0,
                "sustained_target": self.sustained_frames
            }

        # 3. Active signing with sufficient motion
        self.current_state = State.SIGNING

        # Accumulate ONLY valid trained class names with high confidence
        if word and word in self.valid_classes and confidence >= self.confidence_threshold:
            self.prediction_window.append((word, confidence))
        else:
            self.prediction_window.clear()

        # Require at least sustained_frames consistent predictions with high confidence
        if len(self.prediction_window) >= self.sustained_frames:
            words = [w for w, c in self.prediction_window]
            confidences = [c for w, c in self.prediction_window]

            # All frames in window must agree on the same sign with confidence >= threshold
            if len(set(words)) == 1 and all(c >= self.confidence_threshold for c in confidences):
                candidate_word = words[-1]
                accepted_word = candidate_word
                is_accepted = True
                self.last_accepted_sign = candidate_word
                self.current_state = State.ACCEPTED
                self.cooldown_counter = self.cooldown_frames
                self.prediction_window.clear()

        return {
            "accepted": is_accepted,
            "word": accepted_word,
            "state": self.current_state,
            "confidence": confidence,
            "cooldown_remaining": self.cooldown_counter,
            "last_accepted_sign": self.last_accepted_sign,
            "sustained_count": len(self.prediction_window),
            "sustained_target": self.sustained_frames
        }


    def reset(self):
        self.current_state = State.NO_HAND
        self.prediction_window.clear()
        self.cooldown_counter = 0
        self.last_accepted_sign = None
