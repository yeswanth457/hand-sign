"""
Phase 5: Motion-triggered frame selection.
Filters idle/static frames and retains key signing frames in a temporal buffer.
"""

from collections import deque
from config import MAX_SEQ_LEN
from src.motion_energy import MotionEnergyCalculator


class MotionFrameSelector:
    def __init__(self, max_buffer_len=MAX_SEQ_LEN):
        self.max_buffer_len = max_buffer_len
        self.motion_calculator = MotionEnergyCalculator()
        self.selected_frames_buffer = deque(maxlen=max_buffer_len)
        self.selected_landmarks_buffer = deque(maxlen=max_buffer_len)

    def process_frame(self, frame_id, landmark_data):
        """
        Evaluates current frame landmark data.
        Returns tuple: (is_selected, motion_energy, threshold, buffer_length)
        """
        # Calculate motion energy using pose joints or hand center
        pose_joints = landmark_data.get("pose", {})
        energy = self.motion_calculator.calculate_energy(pose_joints)
        is_selected, threshold = self.motion_calculator.is_active_motion(energy)

        if is_selected:
            self.selected_frames_buffer.append(frame_id)
            self.selected_landmarks_buffer.append(landmark_data)

        return {
            "is_selected": is_selected,
            "motion_energy": energy,
            "threshold": threshold,
            "selected_frames_count": len(self.selected_frames_buffer)
        }

    def get_selected_sequence(self):
        """Returns the list of buffered selected landmark frames."""
        return list(self.selected_landmarks_buffer)

    def clear_buffer(self):
        """Clears selection buffers after recognition early decision."""
        self.selected_frames_buffer.clear()
        self.selected_landmarks_buffer.clear()
        self.motion_calculator.reset()
