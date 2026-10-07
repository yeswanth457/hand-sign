"""
Phase 4: Adaptive Motion Energy Engine.
Calculates frame-to-frame joint movement energy:
E_t = sum(distance(landmark_t, landmark_{t-1}))
Computes adaptive threshold: τ = μ_E + λ * σ_E
"""

import numpy as np
from collections import deque
from config import MOTION_WINDOW_SIZE, LAMBDA_SIGMA, IDLE_ENERGY_THRESHOLD


class MotionEnergyCalculator:
    def __init__(self, window_size=MOTION_WINDOW_SIZE, lambda_sigma=LAMBDA_SIGMA):
        self.window_size = window_size
        self.lambda_sigma = lambda_sigma
        self.energy_history = deque(maxlen=window_size)
        self.prev_landmarks = None

    def calculate_energy(self, current_landmarks):
        """
        Calculates joint displacement energy E_t between current and previous frame.
        current_landmarks can be a dictionary of pose landmarks {"LW": (x,y,z), ...} or hand centers.
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return 0.0

        total_energy = 0.0
        joint_count = 0

        if isinstance(current_landmarks, dict) and isinstance(self.prev_landmarks, dict):
            for joint_name, curr_pos in current_landmarks.items():
                if joint_name in self.prev_landmarks:
                    prev_pos = self.prev_landmarks[joint_name]
                    dist = np.sqrt(
                        (curr_pos[0] - prev_pos[0]) ** 2 +
                        (curr_pos[1] - prev_pos[1]) ** 2 +
                        (curr_pos[2] - prev_pos[2]) ** 2
                    )
                    total_energy += dist
                    joint_count += 1
        elif isinstance(current_landmarks, (list, tuple, np.ndarray)):
            curr_arr = np.array(current_landmarks)
            prev_arr = np.array(self.prev_landmarks)
            if curr_arr.shape == prev_arr.shape:
                total_energy = np.sum(np.linalg.norm(curr_arr - prev_arr, axis=-1))
                joint_count = len(curr_arr)

        self.prev_landmarks = current_landmarks

        # Normalize energy by joint count if > 0
        norm_energy = float(total_energy / max(1, joint_count))
        self.energy_history.append(norm_energy)
        return norm_energy

    def get_adaptive_threshold(self):
        """
        Calculates adaptive motion threshold: τ = μ_E + λ * σ_E
        """
        if len(self.energy_history) == 0:
            return IDLE_ENERGY_THRESHOLD

        mu_E = np.mean(self.energy_history)
        sigma_E = np.std(self.energy_history)

        tau = float(mu_E + self.lambda_sigma * sigma_E)
        # Ensure threshold does not drop below baseline idle noise floor
        return max(IDLE_ENERGY_THRESHOLD, tau)

    def is_active_motion(self, current_energy):
        """
        Evaluates whether current energy exceeds the dynamic adaptive threshold τ.
        """
        threshold = self.get_adaptive_threshold()
        return current_energy > threshold, threshold

    def reset(self):
        self.energy_history.clear()
        self.prev_landmarks = None
