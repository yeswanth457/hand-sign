"""
Phase 8: Early Decision and Real-Time Cooldown State Machine.
Triggers positive sign decision when prediction confidence > threshold for N consecutive frames.
Maintains state machine (IDLE, SIGNING, CONFIRMED, COOLDOWN) to prevent duplicate output loops.
"""

from collections import deque
from config import CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES, COOLDOWN_FRAMES, IDLE_ENERGY_THRESHOLD


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
        cooldown_frames=COOLDOWN_FRAMES
    ):
        self.confidence_threshold = confidence_threshold
        self.sustained_frames = sustained_frames
        self.cooldown_frames = cooldown_frames

        self.current_state = State.NO_HAND
        self.prediction_window = deque(maxlen=sustained_frames)
        self.cooldown_counter = 0
        self.last_accepted_sign = None

    def process_prediction(self, prediction_res, motion_energy):
        """
        Processes real-time frame model prediction dict:
        {"word": str, "confidence": float, "class_id": int}
        and motion_energy float.
        """
        word = prediction_res.get("word")
        confidence = prediction_res.get("confidence", 0.0)

        # Only accumulate valid predictions into window (ignore BUFFERING and empty predictions)
        if word and word != "BUFFERING" and word != "--" and confidence > 0.0:
            self.prediction_window.append((word, confidence))

        accepted_word = None
        is_accepted = False

        # 1. Automatic state transition from initial/idle states when valid predictions arrive
        if self.current_state in [State.NO_HAND, State.READY, State.COLLECTING, State.PREDICTING]:
            if word and word != "BUFFERING" and word != "--" and confidence > 0.0:
                self.current_state = State.SIGNING if motion_energy > IDLE_ENERGY_THRESHOLD else State.IDLE

        # 2. State Machine Transitions & Lock/Cooldown Handling
        if self.current_state == State.LOCKED:
            # Release LOCKED state when hand rests (motion < 0.035) or hand leaves frame (confidence == 0)
            if motion_energy < 0.035 or confidence == 0.0:
                self.current_state = State.IDLE
                self.prediction_window.clear()

        elif self.current_state == State.COOLDOWN:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0 or motion_energy < 0.035:
                self.current_state = State.IDLE
                self.cooldown_counter = 0
                self.prediction_window.clear()

        # 3. Evaluate prediction acceptance (instant on high confidence >= threshold)
        if self.current_state in [State.IDLE, State.SIGNING, State.READY, State.COLLECTING, State.PREDICTING]:
            if len(self.prediction_window) >= 1:
                words = [w for w, c in self.prediction_window]
                confidences = [c for w, c in self.prediction_window]

                # High confidence prediction (>= threshold) accepts immediately
                if all(c >= self.confidence_threshold for c in confidences):
                    candidate_word = words[-1]
                    
                    accepted_word = candidate_word
                    is_accepted = True
                    self.last_accepted_sign = candidate_word
                    self.current_state = State.LOCKED
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
            "last_accepted_sign": self.last_accepted_sign,
            "sustained_count": len(self.prediction_window),
            "sustained_target": self.sustained_frames
        }

    def reset(self):
        self.current_state = State.NO_HAND
        self.prediction_window.clear()
        self.cooldown_counter = 0
