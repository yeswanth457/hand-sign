"""
Phase 3: Video Preprocessing and Pose + Hand Landmark Extraction.
Uses MediaPipe Tasks API (MediaPipe 1.0.0+) to extract real body and hand landmarks.
Extracts: Left/Right Wrist, Left/Right Elbow, Left/Right Shoulder, Hand Landmark arrays.
"""

import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


class LandmarkExtractor:
    def __init__(self, pose_model_path="models/pose_landmarker.task", hand_model_path="models/hand_landmarker.task", min_detection_confidence=0.5):
        self.mp_available = MP_AVAILABLE
        self.pose_landmarker = None
        self.hand_landmarker = None

        if self.mp_available:
            try:
                # Check if task models exist relative to cwd or project root
                if not os.path.exists(pose_model_path):
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    pose_model_path = os.path.join(base_dir, "models", "pose_landmarker.task")
                    hand_model_path = os.path.join(base_dir, "models", "hand_landmarker.task")

                if os.path.exists(pose_model_path) and os.path.exists(hand_model_path):
                    pose_options = vision.PoseLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(model_asset_path=pose_model_path),
                        running_mode=vision.RunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=min_detection_confidence,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    hand_options = vision.HandLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=2,
                        min_hand_detection_confidence=min_detection_confidence,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
                    self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                    self.frame_timestamp_ms = 0
                    self.is_fallback = False
                else:
                    print(f"[LandmarkExtractor] Model files missing ({pose_model_path}, {hand_model_path}). Fallback active.")
                    self.mp_available = False
                    self.is_fallback = True
            except Exception as e:
                print(f"[LandmarkExtractor] MediaPipe Tasks init warning: {e}. Fallback active.")
                self.mp_available = False
                self.is_fallback = True
        else:
            self.is_fallback = True

    def close(self):
        """Safely closes MediaPipe landmarker instances to release C++ memory and threads."""
        if hasattr(self, "pose_landmarker") and self.pose_landmarker:
            try:
                self.pose_landmarker.close()
            except Exception:
                pass
            self.pose_landmarker = None
        if hasattr(self, "hand_landmarker") and self.hand_landmarker:
            try:
                self.hand_landmarker.close()
            except Exception:
                pass
            self.hand_landmarker = None

    def extract(self, frame_bgr, timestamp_ms=None):
        """
        Backward-compatible public extraction method.
        Delegates to process_frame().
        """
        return self.process_frame(frame_bgr, timestamp_ms=timestamp_ms)

    def process_frame(self, frame_bgr, timestamp_ms=None):
        """
        Processes a single BGR video frame in VIDEO tracking mode using monotonic timestamps.
        """
        if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
            return {
                "pose": {"LS": (0.4, 0.35, 0.0), "RS": (0.6, 0.35, 0.0), "LW": (0.4, 0.65, 0.0), "RW": (0.6, 0.65, 0.0)},
                "hands": [],
                "hand_center": (0.5, 0.5),
                "shoulder_center": (0.5, 0.35),
                "is_fallback": True
            }

        h, w, c = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pose_res = None
        hands_res = None

        if timestamp_ms is not None and int(timestamp_ms) > self.frame_timestamp_ms:
            current_ts = int(timestamp_ms)
        else:
            self.frame_timestamp_ms += 33
            current_ts = self.frame_timestamp_ms
        self.frame_timestamp_ms = current_ts

        if self.mp_available:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                # Process hand landmarker first (critical for sign gesture tokens)
                if self.hand_landmarker:
                    try:
                        hands_res = self.hand_landmarker.detect_for_video(mp_image, current_ts)
                    except Exception:
                        hands_res = None

                # Process pose landmarker independently to prevent pose delays from blocking hands
                if self.pose_landmarker:
                    try:
                        pose_res = self.pose_landmarker.detect_for_video(mp_image, current_ts)
                    except Exception:
                        pose_res = None
            except Exception:
                pass

        # Extract Pose Joints (Left/Right Wrist, Elbow, Shoulder)
        # MediaPipe Pose Landmark Indices:
        # 11: LEFT_SHOULDER, 12: RIGHT_SHOULDER, 13: LEFT_ELBOW, 14: RIGHT_ELBOW, 15: LEFT_WRIST, 16: RIGHT_WRIST
        pose_dict = {}
        if pose_res and pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            lms = pose_res.pose_landmarks[0]
            pose_dict = {
                "LW": (lms[15].x, lms[15].y, lms[15].z),
                "RW": (lms[16].x, lms[16].y, lms[16].z),
                "LE": (lms[13].x, lms[13].y, lms[13].z),
                "RE": (lms[14].x, lms[14].y, lms[14].z),
                "LS": (lms[11].x, lms[11].y, lms[11].z),
                "RS": (lms[12].x, lms[12].y, lms[12].z),
            }
        else:
            # Fallback dynamic estimation based on visual frame motion centroid
            cx, cy = 0.60, 0.65
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            if hasattr(self, "prev_frame_gray") and self.prev_frame_gray is not None and self.prev_frame_gray.shape == gray.shape:
                diff = cv2.absdiff(gray, self.prev_frame_gray)
                _, diff_thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
                M_diff = cv2.moments(diff_thresh)
                if M_diff["m00"] > 0:
                    cx = float((M_diff["m10"] / M_diff["m00"]) / max(1, w))
                    cy = float((M_diff["m01"] / M_diff["m00"]) / max(1, h))
                else:
                    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                    M = cv2.moments(thresh)
                    if M["m00"] > 0:
                        cx = float((M["m10"] / M["m00"]) / max(1, w))
                        cy = float((M["m01"] / M["m00"]) / max(1, h))
            else:
                _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                M = cv2.moments(thresh)
                if M["m00"] > 0:
                    cx = float((M["m10"] / M["m00"]) / max(1, w))
                    cy = float((M["m01"] / M["m00"]) / max(1, h))

            self.prev_frame_gray = gray

            pose_dict = {
                "LW": (1.0 - cx, cy, 0.0),
                "RW": (cx, cy, 0.0),
                "LE": (0.5 * ((1.0 - cx) + 0.35), 0.5 * (cy + 0.50), 0.0),
                "RE": (0.5 * (cx + 0.65), 0.5 * (cy + 0.50), 0.0),
                "LS": (0.40, 0.35, 0.0),
                "RS": (0.60, 0.35, 0.0),
            }

        # Extract Hand Landmarks (21 points per hand)
        hands_list = []
        if hands_res and hands_res.hand_landmarks:
            for hand_lms in hands_res.hand_landmarks:
                pts = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                hands_list.append(pts)

        # Dominant Hand Center Hx, Hy with stable temporal tracking
        if hands_list and len(hands_list) > 0:
            centers = []
            for h_pts in hands_list:
                pts = np.array(h_pts)
                centers.append((float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))

            if len(centers) == 1:
                hx, hy = centers[0]
            elif hasattr(self, "prev_hand_center") and self.prev_hand_center is not None:
                px, py = self.prev_hand_center
                dists = [(cx - px)**2 + (cy - py)**2 for cx, cy in centers]
                best_idx = int(np.argmin(dists))
                hx, hy = centers[best_idx]
            else:
                best_idx = int(np.argmin([c[1] for c in centers]))
                hx, hy = centers[best_idx]

            self.prev_hand_center = (hx, hy)
        else:
            hx, hy = pose_dict["RW"][0], pose_dict["RW"][1]
            self.prev_hand_center = (hx, hy)

        # Shoulder Center Sx, Sy
        ls = pose_dict["LS"]
        rs = pose_dict["RS"]
        sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0

        return {
            "pose": pose_dict,
            "hands": hands_list,
            "hand_center": (float(hx), float(hy)),
            "shoulder_center": (float(sx), float(sy)),
            "is_fallback": getattr(self, "is_fallback", False),
        }

    def process_raw_landmarks(self, pose_dict, hand_pts=None):
        """Processes pre-extracted numerical landmarks directly."""
        if hand_pts and len(hand_pts) > 0:
            pts = np.array(hand_pts)
            hx, hy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        elif "RW" in pose_dict:
            hx, hy = pose_dict["RW"][0], pose_dict["RW"][1]
        else:
            hx, hy = 0.5, 0.5

        ls = pose_dict.get("LS", (0.4, 0.35, 0))
        rs = pose_dict.get("RS", (0.6, 0.35, 0))
        sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0

        return {
            "pose": pose_dict,
            "hands": [hand_pts] if hand_pts else [],
            "hand_center": (float(hx), float(hy)),
            "shoulder_center": (float(sx), float(sy)),
        }
