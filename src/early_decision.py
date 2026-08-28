"""
Phase 8: Early Decision and Real-Time Cooldown State Machine.
Triggers positive sign decision when prediction confidence > threshold for N consecutive frames.
Maintains state machine (IDLE, SIGNING, CONFIRMED, COOLDOWN) to prevent duplicate output loops.
"""

from collections import deque
from config import CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES, COOLDOWN_FRAMES, IDLE_ENERGY_THRESHOLD


class State:
    IDLE = "IDLE"
    SIGNING = "SIGNING"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


class EarlyDecisionEngine:
    def __init__(
        self,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sustained_frames=SUSTAINED_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES
    ):
        self.confidence_threshold = confidence_threshold
        self.sustained_frames = sustained_frames
        self.cooldown_frames = cooldown_frames

        self.current_state = State.IDLE
        self.prediction_window = deque(maxlen=sustained_frames)
        self.cooldown_counter = 0
        self.last_accepted_sign = None

    def process_prediction(self, prediction_res, motion_energy):
        """
        Processes real-time frame model prediction dict:
        {"word": str, "confidence": float, "class_id": int}
        and motion_energy float.

        Returns dict with status:
        - accepted: bool
        - word: str or None
        - state: current State
        - confidence: float
        - cooldown_remaining: int
        """
        word = prediction_res.get("word")
        confidence = prediction_res.get("confidence", 0.0)

        # Update prediction rolling window
        self.prediction_window.append((word, confidence))

        accepted_word = None
        is_accepted = False

        # State Machine Transitions
        if self.current_state == State.COOLDOWN:
            self.cooldown_counter -= 1
            # Reset to IDLE if cooldown completed AND motion drops to idle level
            if self.cooldown_counter <= 0 or motion_energy < IDLE_ENERGY_THRESHOLD:
                self.current_state = State.IDLE
                self.cooldown_counter = 0
                self.prediction_window.clear()

        elif self.current_state in [State.IDLE, State.SIGNING]:
            # Evaluate if prediction is sustained with high confidence
            if len(self.prediction_window) >= self.sustained_frames:
                words = [w for w, c in self.prediction_window]
                confidences = [c for w, c in self.prediction_window]

                # All N frames predict the exact same word with high confidence
                if len(set(words)) == 1 and all(c >= self.confidence_threshold for c in confidences):
                    candidate_word = words[0]
                    
                    # Accept sign if it's new or motion has reset
                    accepted_word = candidate_word
                    is_accepted = True
                    self.last_accepted_sign = candidate_word
                    self.current_state = State.CONFIRMED
                    
                    # Transition immediately into COOLDOWN to prevent duplicates
                    self.current_state = State.COOLDOWN
                    self.cooldown_counter = self.cooldown_frames
                    self.prediction_window.clear()
                elif motion_energy > IDLE_ENERGY_THRESHOLD:
                    self.current_state = State.SIGNING
                else:
                    self.current_state = State.IDLE

        return {
            "accepted": is_accepted,
            "word": accepted_word,
            "state": self.current_state,
            "confidence": confidence,
            "cooldown_remaining": self.cooldown_counter,
            "last_accepted_sign": self.last_accepted_sign
        }

    def reset(self):
        self.current_state = State.IDLE
        self.prediction_window.clear()
        self.cooldown_counter = 0
        self.last_accepted_sign = None
