"""
NO Gesture Profile Range Detector.

Evaluates a sequence of MediaPipe hand landmarks against the calibrated NO gesture profile.
Calculates continuous multi-factor similarity score (0.0 to 1.0) based on:
1. Spatial Landmark Positions (with weighted landmark importance: middle tip #12, index tip #8, etc.)
2. Frame-to-Frame Movement Velocity
3. Axis Movement Dominance (Y-axis dominance)
4. Finger Relationship Distances (Thumb-Index, Thumb-Middle, Index-Middle)
5. Project 6D Tokens [Hx, Hy, Mx, My, Rx, Ry]
6. Temporal Sequence & Phase Duration

Does NOT alter the CNN-GRU model, 21-class mapping, or 6D token representation.
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR, NO_RANGE_TOLERANCE, NO_DETECTOR_THRESHOLD

LANDMARK_NAMES = [
    "wrist_0",
    "thumb_cmc_1", "thumb_mcp_2", "thumb_ip_3", "thumb_tip_4",
    "index_mcp_5", "index_pip_6", "index_dip_7", "index_tip_8",
    "middle_mcp_9", "middle_pip_10", "middle_dip_11", "middle_tip_12",
    "ring_mcp_13", "ring_pip_14", "ring_dip_15", "ring_tip_16",
    "pinky_mcp_17", "pinky_pip_18", "pinky_dip_19", "pinky_tip_20"
]

# Weights for landmarks: high weight for active moving fingers, low weight for wrist anchor
LANDMARK_WEIGHTS = {
    0: 0.3,   # Wrist (anchor)
    1: 0.5, 2: 0.5, 3: 0.7, 4: 1.2,   # Thumb
    5: 0.8, 6: 0.9, 7: 1.2, 8: 1.5,   # Index
    9: 0.8, 10: 0.9, 11: 1.3, 12: 1.5, # Middle (top moving)
    13: 0.7, 14: 0.8, 15: 0.9, 16: 1.0, # Ring
    17: 0.6, 18: 0.7, 19: 0.8, 20: 0.9  # Pinky
}


class NOGestureDetector:
    def __init__(self, profile_path=None, tolerance=NO_RANGE_TOLERANCE, threshold=NO_DETECTOR_THRESHOLD):
        self.tolerance = tolerance
        self.threshold = threshold

        if profile_path is None:
            profile_path = os.path.join(MODEL_DIR, "no_gesture_range_profile.json")

        self.profile = self._load_profile(profile_path)

    def _load_profile(self, profile_path):
        """Loads reference NO gesture profile or uses robust defaults if file missing."""
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[NOGestureDetector] Warning: Could not load profile {profile_path}: {e}")

        # Fallback profile based on empirical NO measurements
        return {
            "gesture": "no",
            "axis_distribution": {"X_pct": 18.1, "Y_pct": 64.9, "Z_pct": 17.0, "dominant_axis": "Y"},
            "spatial_ranges": {
                "x": {"p05": 0.2666, "p95": 0.5276},
                "y": {"p05": 0.3745, "p95": 0.8571},
                "z": {"p05": -0.0794, "p95": 0.0}
            },
            "movement_ranges": {
                "dx": {"p05": -0.01327, "p95": 0.01533},
                "dy": {"p05": -0.05084, "p95": 0.06717},
                "dz": {"p05": -0.01391, "p95": 0.01536},
                "magnitude": {"p05": 0.002, "p95": 0.08}
            },
            "finger_relationships": {
                "thumb_index": {"p05": 0.04, "p95": 0.25},
                "thumb_middle": {"p05": 0.05, "p95": 0.28},
                "index_middle": {"p05": 0.02, "p95": 0.15}
            },
            "token_ranges": {
                "Hx": {"p05": 0.2953, "p95": 0.4920},
                "Hy": {"p05": 0.4249, "p95": 0.8239},
                "Mx": {"p05": -0.0102, "p95": 0.0116},
                "My": {"p05": -0.0446, "p95": 0.0587},
                "Rx": {"p05": -0.2295, "p95": -0.0629},
                "Ry": {"p05": -0.1987, "p95": 0.1649}
            },
            "duration": {
                "mean_total_gesture_frames": 43.0,
                "p05_duration": 25.0,
                "p95_duration": 60.0
            }
        }

    def _range_score(self, val, p05, p95):
        """
        Calculates a soft score (0.0 to 1.0) indicating how well `val` falls within [p05, p95]
        with a soft penalty scaling into the tolerance zone.
        """
        if p05 is None or p95 is None or np.isnan(val):
            return 0.5

        span = max(1e-4, p95 - p05)
        tol_margin = span * self.tolerance
        low_bound = p05 - tol_margin
        high_bound = p95 + tol_margin

        if p05 <= val <= p95:
            return 1.0
        elif low_bound <= val < p05:
            return float(1.0 - (p05 - val) / max(1e-4, tol_margin))
        elif p95 < val <= high_bound:
            return float(1.0 - (val - p95) / max(1e-4, tol_margin))
        else:
            dist = (p05 - val) if val < low_bound else (val - p95)
            return float(max(0.0, 1.0 - dist / max(1e-4, span * 2.0)))

    def extract_features(self, sequence_landmarks):
        """
        Extracts structured kinematic features from a list of frame landmark dicts or numpy arrays.
        Handles missing hands, 1-frame input, NaNs, and two-hand selection safely.
        """
        valid_frames = []

        for frame_item in sequence_landmarks:
            if isinstance(frame_item, dict):
                hands = frame_item.get("hands", [])
                if not hands or len(hands) == 0:
                    continue
                # Active hand selection (prefer hand with highest total motion or first hand)
                hand_pts = np.array(hands[0], dtype=np.float32)
                pose_data = frame_item.get("pose", {})
                ls = pose_data.get("LS", (0.5, 0.35, 0.0))
                rs = pose_data.get("RS", (0.5, 0.35, 0.0))
                sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
                hx, hy = frame_item.get("hand_center", (float(np.mean(hand_pts[:, 0])), float(np.mean(hand_pts[:, 1]))))
            elif isinstance(frame_item, (np.ndarray, list)):
                pts_arr = np.array(frame_item, dtype=np.float32)
                if pts_arr.ndim == 3:  # (num_hands, 21, 3)
                    hand_pts = pts_arr[0]
                elif pts_arr.ndim == 2:  # (21, 3)
                    hand_pts = pts_arr
                else:
                    continue
                sx, sy = 0.5, 0.35
                hx, hy = float(np.mean(hand_pts[:, 0])), float(np.mean(hand_pts[:, 1]))
            else:
                continue

            pts = np.array(hand_pts, dtype=np.float32)
            if pts.shape[0] != 21:
                continue

            # Ensure 3D shape (21, 3)
            if pts.shape[1] == 2:
                pts = np.hstack([pts, np.full((21, 1), np.nan, dtype=np.float32)])

            if not np.isfinite(pts[:, :2]).all():
                continue

            valid_frames.append({
                "landmarks": pts,
                "hand_center": (float(hx), float(hy)),
                "shoulder_center": (float(sx), float(sy))
            })

        if len(valid_frames) == 0:
            return None

        # Tensor of shape (T, 21, 3)
        coords = np.array([f["landmarks"] for f in valid_frames], dtype=np.float32)
        n_frames = len(coords)

        # Spatial landmarks
        spatial = {
            "wrist_0": coords[:, 0],
            "thumb_tip_4": coords[:, 4],
            "index_tip_8": coords[:, 8],
            "middle_tip_12": coords[:, 12],
            "ring_tip_16": coords[:, 16],
            "pinky_tip_20": coords[:, 20]
        }

        # Movement features (deltas)
        if n_frames > 1:
            deltas = coords[1:] - coords[:-1]
            mags = np.sqrt(deltas[:, :, 0]**2 + deltas[:, :, 1]**2)
        else:
            deltas = np.zeros((0, 21, 3), dtype=np.float32)
            mags = np.zeros((0, 21), dtype=np.float32)

        # Axis contribution
        abs_dx = float(np.sum(np.abs(deltas[:, :, 0]))) if len(deltas) > 0 else 0.0
        abs_dy = float(np.sum(np.abs(deltas[:, :, 1]))) if len(deltas) > 0 else 0.0
        abs_dz = float(np.sum(np.abs(deltas[:, :, 2]))) if (len(deltas) > 0 and np.isfinite(deltas[:, :, 2]).all()) else 0.0
        tot_motion = abs_dx + abs_dy + abs_dz

        if tot_motion > 0:
            x_ratio = (abs_dx / tot_motion) * 100.0
            y_ratio = (abs_dy / tot_motion) * 100.0
            z_ratio = (abs_dz / tot_motion) * 100.0
        else:
            x_ratio, y_ratio, z_ratio = 0.0, 0.0, 0.0

        dominant_axis = "Y" if y_ratio >= max(x_ratio, z_ratio) else ("X" if x_ratio >= z_ratio else "Z")

        # Finger relationship distances & curl ratio
        w0 = coords[:, 0, :2]
        t4 = coords[:, 4, :2]
        i8 = coords[:, 8, :2]
        m12 = coords[:, 12, :2]
        r16 = coords[:, 16, :2]
        p20 = coords[:, 20, :2]

        ti_dist = np.linalg.norm(t4 - i8, axis=1)
        tm_dist = np.linalg.norm(t4 - m12, axis=1)
        im_dist = np.linalg.norm(i8 - m12, axis=1)

        i_ext = np.linalg.norm(i8 - w0, axis=1)
        m_ext = np.linalg.norm(m12 - w0, axis=1)
        r_ext = np.linalg.norm(r16 - w0, axis=1)
        p_ext = np.linalg.norm(p20 - w0, axis=1)
        curl_ratios = (r_ext + p_ext) / np.maximum(1e-4, i_ext + m_ext)

        # 6D Tokens [Hx, Hy, Mx, My, Rx, Ry]
        tokens = []
        prev_hc = None
        for f in valid_frames:
            hx, hy = f["hand_center"]
            sx, sy = f["shoulder_center"]
            if prev_hc is not None:
                mx, my = hx - prev_hc[0], hy - prev_hc[1]
            else:
                mx, my = 0.0, 0.0
            prev_hc = (hx, hy)
            rx, ry = hx - sx, hy - sy
            tokens.append([hx, hy, mx, my, rx, ry])
        tokens = np.array(tokens, dtype=np.float32)

        return {
            "n_frames": n_frames,
            "coords": coords,
            "deltas": deltas,
            "mags": mags,
            "spatial": spatial,
            "x_ratio": x_ratio,
            "y_ratio": y_ratio,
            "z_ratio": z_ratio,
            "dominant_axis": dominant_axis,
            "ti_dist": ti_dist,
            "tm_dist": tm_dist,
            "im_dist": im_dist,
            "curl_ratios": curl_ratios,
            "tokens": tokens
        }

    def evaluate_sequence(self, sequence_landmarks):
        """
        Evaluates sequence against reference NO profile.
        Returns detailed scoring dictionary.
        """
        features = self.extract_features(sequence_landmarks)

        if features is None or features["n_frames"] == 0:
            return {
                "gesture": "no",
                "is_no_gesture": False,
                "no_score": 0.0,
                "spatial_score": 0.0,
                "movement_score": 0.0,
                "axis_score": 0.0,
                "finger_score": 0.0,
                "token_score": 0.0,
                "duration_score": 0.0,
                "dominant_axis": "NONE",
                "frames_analyzed": 0
            }

        prof = self.profile

        # 1. Spatial Score (weighted across 21 landmarks)
        spatial_scores = []
        total_weight = 0.0

        for i, name in enumerate(LANDMARK_NAMES):
            w = LANDMARK_WEIGHTS.get(i, 1.0)
            pts = features["coords"][:, i]
            lm_ref = prof.get("landmark_ranges", {}).get(name, {})

            x_ref = lm_ref.get("x", prof.get("spatial_ranges", {}).get("x", {}))
            y_ref = lm_ref.get("y", prof.get("spatial_ranges", {}).get("y", {}))

            sx = self._range_score(np.mean(pts[:, 0]), x_ref.get("p05"), x_ref.get("p95"))
            sy = self._range_score(np.mean(pts[:, 1]), y_ref.get("p05"), y_ref.get("p95"))

            lm_score = 0.5 * (sx + sy)
            spatial_scores.append(lm_score * w)
            total_weight += w

        spatial_score = float(np.sum(spatial_scores) / max(1e-4, total_weight))

        # 2. Movement Score
        deltas = features["deltas"]
        if len(deltas) > 0:
            m_ref = prof.get("movement_ranges", {})
            dx_mean = np.mean(deltas[:, :, 0])
            dy_mean = np.mean(deltas[:, :, 1])
            mag_mean = np.mean(features["mags"])

            s_dx = self._range_score(dx_mean, m_ref.get("dx", {}).get("p05"), m_ref.get("dx", {}).get("p95"))
            s_dy = self._range_score(dy_mean, m_ref.get("dy", {}).get("p05"), m_ref.get("dy", {}).get("p95"))
            s_mag = self._range_score(mag_mean, m_ref.get("magnitude", {}).get("p05"), m_ref.get("magnitude", {}).get("p95"))
            movement_score = float(0.35 * s_dx + 0.45 * s_dy + 0.20 * s_mag)
        else:
            movement_score = 0.5

        # 3. Axis Dominance Score
        ax_ref = prof.get("axis_distribution", {})
        y_ref_pct = ax_ref.get("Y_pct", 64.9)
        # Soft penalty if Y dominance departs significantly from ~64.9%
        y_diff = abs(features["y_ratio"] - y_ref_pct)
        axis_score = float(max(0.0, 1.0 - y_diff / 40.0))

        # 4. Finger Relationship Score
        f_ref = prof.get("finger_relationships", {})
        s_ti = self._range_score(np.mean(features["ti_dist"]), f_ref.get("thumb_index", {}).get("p05"), f_ref.get("thumb_index", {}).get("p95"))
        s_tm = self._range_score(np.mean(features["tm_dist"]), f_ref.get("thumb_middle", {}).get("p05"), f_ref.get("thumb_middle", {}).get("p95"))
        s_im = self._range_score(np.mean(features["im_dist"]), f_ref.get("index_middle", {}).get("p05"), f_ref.get("index_middle", {}).get("p95"))
        cr_ref = f_ref.get("finger_curl_ratio", {})
        s_cr = self._range_score(np.mean(features["curl_ratios"]), cr_ref.get("p05", 0.30), cr_ref.get("p95", 0.75))
        finger_score = float((s_ti + s_tm + s_im + 2.0 * s_cr) / 5.0)

        # 5. 6D Token Score
        t_ref = prof.get("token_ranges", {})
        token_scores = []
        token_feats = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]
        for idx, feat in enumerate(token_feats):
            vals = features["tokens"][:, idx]
            ref_i = t_ref.get(feat, {})
            score_i = self._range_score(np.mean(vals), ref_i.get("p05"), ref_i.get("p95"))
            token_scores.append(score_i)
        token_score = float(np.mean(token_scores))

        # 6. Duration Score
        dur_ref = prof.get("duration", {})
        mean_dur = dur_ref.get("mean_total_gesture_frames", 43.0)
        p05_dur = dur_ref.get("p05_duration", 25.0)
        p95_dur = dur_ref.get("p95_duration", 60.0)

        n_fr = features["n_frames"]
        if p05_dur <= n_fr <= p95_dur:
            duration_score = 1.0
        else:
            diff = abs(n_fr - mean_dur)
            duration_score = float(max(0.0, 1.0 - diff / 40.0))

        # Overall NO Score
        no_score = float(
            0.25 * spatial_score +
            0.20 * movement_score +
            0.15 * axis_score +
            0.15 * finger_score +
            0.15 * token_score +
            0.10 * duration_score
        )

        is_no = bool(no_score >= self.threshold)

        return {
            "gesture": "no",
            "is_no_gesture": is_no,
            "no_score": round(no_score, 4),
            "spatial_score": round(spatial_score, 4),
            "movement_score": round(movement_score, 4),
            "axis_score": round(axis_score, 4),
            "finger_score": round(finger_score, 4),
            "token_score": round(token_score, 4),
            "duration_score": round(duration_score, 4),
            "dominant_axis": features["dominant_axis"],
            "frames_analyzed": features["n_frames"]
        }
