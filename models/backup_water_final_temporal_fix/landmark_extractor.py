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

        # Extract Hand Landmarks (21 points per hand) with explicit Left/Right separation
        hands_list = []
        left_hand_pts = None
        right_hand_pts = None
        left_hand_center = None
        right_hand_center = None

        if hands_res and hands_res.hand_landmarks:
            for i, hand_lms in enumerate(hands_res.hand_landmarks):
                pts = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                hands_list.append(pts)
                
                label = "Right"
                if hasattr(hands_res, "handedness") and hands_res.handedness and i < len(hands_res.handedness):
                    h_cat = hands_res.handedness[i][0]
                    label = getattr(h_cat, "category_name", None) or getattr(h_cat, "display_name", None) or "Right"
                
                pts_arr = np.array(pts, dtype=np.float32)
                c_x, c_y = float(np.mean(pts_arr[:, 0])), float(np.mean(pts_arr[:, 1]))
                
                if label == "Left" and left_hand_pts is None:
                    left_hand_pts = pts
                    left_hand_center = (c_x, c_y)
                elif label == "Right" and right_hand_pts is None:
                    right_hand_pts = pts
                    right_hand_center = (c_x, c_y)
                elif left_hand_pts is None:
                    left_hand_pts = pts
                    left_hand_center = (c_x, c_y)
                elif right_hand_pts is None:
                    right_hand_pts = pts
                    right_hand_center = (c_x, c_y)

        # Shoulder Center Sx, Sy
        ls = pose_dict["LS"]
        rs = pose_dict["RS"]
        sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0

        # Fallback spatial assignment if both hands are detected but assigned to the same side
        if len(hands_list) >= 2 and (left_hand_pts is None or right_hand_pts is None):
            c0 = (float(np.mean(np.array(hands_list[0])[:, 0])), float(np.mean(np.array(hands_list[0])[:, 1])))
            c1 = (float(np.mean(np.array(hands_list[1])[:, 0])), float(np.mean(np.array(hands_list[1])[:, 1])))
            if c0[0] < c1[0]:
                left_hand_pts, left_hand_center = hands_list[0], c0
                right_hand_pts, right_hand_center = hands_list[1], c1
            else:
                left_hand_pts, left_hand_center = hands_list[1], c1
                right_hand_pts, right_hand_center = hands_list[0], c0

        # Dominant Hand Center Hx, Hy for backward compatibility
        if right_hand_center is not None:
            hx, hy = right_hand_center
        elif left_hand_center is not None:
            hx, hy = left_hand_center
        else:
            hx, hy = pose_dict["RW"][0], pose_dict["RW"][1]

        return {
            "pose": pose_dict,
            "hands": hands_list,
            "left_hand": left_hand_pts,
            "right_hand": right_hand_pts,
            "left_hand_center": left_hand_center,
            "right_hand_center": right_hand_center,
            "hand_center": (float(hx), float(hy)),
            "shoulder_center": (float(sx), float(sy)),
            "raw_hand_count": len(hands_list),
            "is_fallback": getattr(self, "is_fallback", False),
        }

    def reset(self):
        """Resets temporal tracking state."""
        self.frame_timestamp_ms = 0
        if hasattr(self, "prev_hand_center"):
            self.prev_hand_center = None
        if hasattr(self, "prev_left_center"):
            self.prev_left_center = None
        if hasattr(self, "prev_right_center"):
            self.prev_right_center = None
        if hasattr(self, "prev_frame_gray"):
            self.prev_frame_gray = None

    def process_raw_landmarks(self, pose_dict, hand_pts=None, left_hand_pts=None, right_hand_pts=None):
        """Processes pre-extracted numerical landmarks directly."""
        ls = pose_dict.get("LS", (0.4, 0.35, 0))
        rs = pose_dict.get("RS", (0.6, 0.35, 0))
        sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0

        lh_center = None
        rh_center = None
        hands_list = []

        if left_hand_pts:
            pts = np.array(left_hand_pts)
            lh_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            hands_list.append(left_hand_pts)

        if right_hand_pts:
            pts = np.array(right_hand_pts)
            rh_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            hands_list.append(right_hand_pts)

        if hand_pts and not left_hand_pts and not right_hand_pts:
            pts = np.array(hand_pts)
            c = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            hands_list.append(hand_pts)
            if c[0] < sx:
                lh_center = c
                left_hand_pts = hand_pts
            else:
                rh_center = c
                right_hand_pts = hand_pts

        hx = rh_center[0] if rh_center else (lh_center[0] if lh_center else 0.5)
        hy = rh_center[1] if rh_center else (lh_center[1] if lh_center else 0.5)

        return {
            "pose": pose_dict,
            "hands": hands_list,
            "left_hand": left_hand_pts,
            "right_hand": right_hand_pts,
            "left_hand_center": lh_center,
            "right_hand_center": rh_center,
            "hand_center": (float(hx), float(hy)),
            "shoulder_center": (float(sx), float(sy)),
            "raw_hand_count": len(hands_list)
        }
