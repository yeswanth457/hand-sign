"""
NO Gesture Range Calibration Script.

1. Loads all NO gesture recordings from dataset/landmarks/no.
2. Extracts spatial, movement, axis contribution, finger relationship, 6D token, and duration statistics.
3. Calculates P05, P25, median, P75, P95, min, max, mean, std.
4. Generates updated models/no_gesture_range_profile.json and models/no_gesture_range_report.txt.
"""

import os
import sys
import json
import glob
import numpy as np

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR, NO_DETECTOR_THRESHOLD

LANDMARK_NAMES = [
    "wrist_0",
    "thumb_cmc_1", "thumb_mcp_2", "thumb_ip_3", "thumb_tip_4",
    "index_mcp_5", "index_pip_6", "index_dip_7", "index_tip_8",
    "middle_mcp_9", "middle_pip_10", "middle_dip_11", "middle_tip_12",
    "ring_mcp_13", "ring_pip_14", "ring_dip_15", "ring_tip_16",
    "pinky_mcp_17", "pinky_pip_18", "pinky_dip_19", "pinky_tip_20"
]


def compute_array_stats(arr):
    """Computes comprehensive distribution statistics for a 1D numpy array."""
    if len(arr) == 0:
        return {
            "p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0,
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0
        }
    arr_clean = arr[np.isfinite(arr)]
    if len(arr_clean) == 0:
        return {
            "p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0,
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0
        }
    return {
        "p05": float(np.percentile(arr_clean, 5)),
        "p25": float(np.percentile(arr_clean, 25)),
        "median": float(np.median(arr_clean)),
        "p75": float(np.percentile(arr_clean, 75)),
        "p95": float(np.percentile(arr_clean, 95)),
        "min": float(np.min(arr_clean)),
        "max": float(np.max(arr_clean)),
        "mean": float(np.mean(arr_clean)),
        "std": float(np.std(arr_clean))
    }


def calibrate_no_gesture_profile(landmark_dir=None):
    if landmark_dir is None:
        landmark_dir = os.path.join(BASE_DIR, "dataset", "landmarks", "no")

    files = sorted(glob.glob(os.path.join(landmark_dir, "**", "*.npz"), recursive=True))
    if not files:
        print(f"[Error] No landmark NPZ files found in {landmark_dir}")
        return None

    all_coords = []      # List of (T_i, 21, 3)
    all_deltas = []      # List of (T_i-1, 21, 3)
    all_tokens = []      # List of (T_i, 6)
    all_thumb_index_dist = []
    all_thumb_middle_dist = []
    all_index_middle_dist = []
    all_curl_ratios = []
    sample_durations = []
    start_frames = []
    peak_frames = []
    end_frames = []

    total_valid_frames = 0
    total_invalid_frames = 0
    processed_samples = 0

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        raw_frames = data["landmarks"]

        sample_valid_coords = []
        sample_tokens = []
        sample_ti_dist = []
        sample_tm_dist = []
        sample_im_dist = []
        sample_curl_ratio = []

        prev_hc = None

        for fr in raw_frames:
            if not isinstance(fr, dict):
                total_invalid_frames += 1
                continue
            hands = fr.get("hands", [])
            if not hands or len(hands) == 0:
                total_invalid_frames += 1
                continue
            pts = np.array(hands[0], dtype=np.float32)
            if pts.shape[0] != 21 or not np.isfinite(pts[:, :2]).all():
                total_invalid_frames += 1
                continue

            if pts.shape[1] == 2:
                pts = np.hstack([pts, np.full((21, 1), np.nan, dtype=np.float32)])

            sample_valid_coords.append(pts)
            total_valid_frames += 1

            # Finger relationships
            w0, t4, i8, m12, r16, p20 = pts[0], pts[4], pts[8], pts[12], pts[16], pts[20]
            ti_d = float(np.linalg.norm(t4[:2] - i8[:2]))
            tm_d = float(np.linalg.norm(t4[:2] - m12[:2]))
            im_d = float(np.linalg.norm(i8[:2] - m12[:2]))

            i_ext = np.linalg.norm(i8[:2] - w0[:2])
            m_ext = np.linalg.norm(m12[:2] - w0[:2])
            r_ext = np.linalg.norm(r16[:2] - w0[:2])
            p_ext = np.linalg.norm(p20[:2] - w0[:2])
            curl_r = float((r_ext + p_ext) / max(1e-4, i_ext + m_ext))

            sample_ti_dist.append(ti_d)
            sample_tm_dist.append(tm_d)
            sample_im_dist.append(im_d)
            sample_curl_ratio.append(curl_r)

            # 6D token calculation
            pose_data = fr.get("pose", {})
            ls = pose_data.get("LS", (0.5, 0.35, 0.0))
            rs = pose_data.get("RS", (0.5, 0.35, 0.0))
            sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
            hx, hy = fr.get("hand_center", (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))

            if prev_hc is not None:
                mx = hx - prev_hc[0]
                my = hy - prev_hc[1]
            else:
                mx, my = 0.0, 0.0
            prev_hc = (hx, hy)
            rx = hx - sx
            ry = hy - sy

            sample_tokens.append([hx, hy, mx, my, rx, ry])

        if len(sample_valid_coords) > 0:
            processed_samples += 1
            sample_coords_arr = np.array(sample_valid_coords, dtype=np.float32)
            all_coords.append(sample_coords_arr)

            sample_tokens_arr = np.array(sample_tokens, dtype=np.float32)
            all_tokens.append(sample_tokens_arr)

            all_thumb_index_dist.extend(sample_ti_dist)
            all_thumb_middle_dist.extend(sample_tm_dist)
            all_index_middle_dist.extend(sample_im_dist)
            all_curl_ratios.extend(sample_curl_ratio)

            n_frames = len(sample_valid_coords)
            sample_durations.append(n_frames)

            if n_frames > 1:
                deltas = sample_coords_arr[1:] - sample_coords_arr[:-1]
                all_deltas.append(deltas)
                move_2d = np.mean(np.sqrt(deltas[:, :, 0]**2 + deltas[:, :, 1]**2), axis=1)
                e_thresh = np.percentile(move_2d, 30) + 1e-4
                act = np.where(move_2d >= e_thresh)[0]
                if len(act) > 0:
                    start_frames.append(int(act[0]))
                    end_frames.append(int(act[-1]) + 1)
                    peak_frames.append(int(np.argmax(move_2d)))
                else:
                    start_frames.append(0)
                    end_frames.append(n_frames - 1)
                    peak_frames.append(n_frames // 2)
            else:
                start_frames.append(0)
                end_frames.append(0)
                peak_frames.append(0)

    # Aggregate global tensors
    concat_coords = np.concatenate(all_coords, axis=0)
    concat_deltas = np.concatenate(all_deltas, axis=0) if all_deltas else np.zeros((0, 21, 3), dtype=np.float32)
    concat_tokens = np.concatenate(all_tokens, axis=0)

    z_available = bool(np.isfinite(concat_coords[:, :, 2]).all())

    # Spatial Ranges
    spatial_ranges = {
        "x": compute_array_stats(concat_coords[:, :, 0].flatten()),
        "y": compute_array_stats(concat_coords[:, :, 1].flatten()),
        "z": compute_array_stats(concat_coords[:, :, 2].flatten()) if z_available else {
            "p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0
        }
    }

    # Movement Ranges
    movement_mags = np.sqrt(concat_deltas[:, :, 0]**2 + concat_deltas[:, :, 1]**2) if len(concat_deltas) > 0 else np.array([0.0])
    movement_ranges = {
        "dx": compute_array_stats(concat_deltas[:, :, 0].flatten()) if len(concat_deltas) > 0 else compute_array_stats(np.array([0.0])),
        "dy": compute_array_stats(concat_deltas[:, :, 1].flatten()) if len(concat_deltas) > 0 else compute_array_stats(np.array([0.0])),
        "dz": compute_array_stats(concat_deltas[:, :, 2].flatten()) if (z_available and len(concat_deltas) > 0) else compute_array_stats(np.array([0.0])),
        "magnitude": compute_array_stats(movement_mags.flatten())
    }

    # Axis Distribution
    abs_dx = float(np.sum(np.abs(concat_deltas[:, :, 0]))) if len(concat_deltas) > 0 else 0.0
    abs_dy = float(np.sum(np.abs(concat_deltas[:, :, 1]))) if len(concat_deltas) > 0 else 0.0
    abs_dz = float(np.sum(np.abs(concat_deltas[:, :, 2]))) if (z_available and len(concat_deltas) > 0) else 0.0
    tot_motion = abs_dx + abs_dy + abs_dz
    if tot_motion > 0:
        x_pct = (abs_dx / tot_motion) * 100.0
        y_pct = (abs_dy / tot_motion) * 100.0
        z_pct = (abs_dz / tot_motion) * 100.0 if z_available else 0.0
    else:
        x_pct = y_pct = z_pct = 0.0

    dom_axis = "Y" if y_pct >= max(x_pct, z_pct) else ("X" if x_pct >= z_pct else "Z")

    axis_distribution = {
        "X_pct": x_pct,
        "Y_pct": y_pct,
        "Z_pct": z_pct,
        "dominant_axis": dom_axis
    }

    # Landmark Ranges
    landmark_ranges = {}
    for i, name in enumerate(LANDMARK_NAMES):
        x_st = compute_array_stats(concat_coords[:, i, 0])
        y_st = compute_array_stats(concat_coords[:, i, 1])
        z_st = compute_array_stats(concat_coords[:, i, 2]) if z_available else {
            "p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0
        }
        landmark_ranges[name] = {
            "x": x_st,
            "y": y_st,
            "z": z_st
        }

    # Finger Relationships
    finger_relationships = {
        "thumb_index": compute_array_stats(np.array(all_thumb_index_dist, dtype=np.float32)),
        "thumb_middle": compute_array_stats(np.array(all_thumb_middle_dist, dtype=np.float32)),
        "index_middle": compute_array_stats(np.array(all_index_middle_dist, dtype=np.float32)),
        "finger_curl_ratio": compute_array_stats(np.array(all_curl_ratios, dtype=np.float32))
    }

    # 6D Token Ranges
    token_features = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]
    token_ranges = {}
    for idx, feat in enumerate(token_features):
        token_ranges[feat] = compute_array_stats(concat_tokens[:, idx])

    # Duration Statistics
    durations_arr = np.array(sample_durations, dtype=np.float32)
    duration_stats = {
        "mean_start_frame": int(np.round(np.mean(start_frames))),
        "mean_peak_frame": int(np.round(np.mean(peak_frames))),
        "mean_end_frame": int(np.round(np.mean(end_frames))),
        "mean_total_gesture_frames": float(np.mean(durations_arr)),
        "p05_duration": float(np.percentile(durations_arr, 5)),
        "p95_duration": float(np.percentile(durations_arr, 95)),
        "min_duration": float(np.min(durations_arr)),
        "max_duration": float(np.max(durations_arr)),
        "std_duration": float(np.std(durations_arr))
    }

    # Construct Profile
    is_robust = (processed_samples >= 20)
    profile = {
        "gesture": "no",
        "scope": "Observed range for collected NO samples",
        "sample_count": processed_samples,
        "valid_frame_count": total_valid_frames,
        "is_robust": is_robust,
        "robustness_notice": (
            "SUFFICIENT SAMPLES (>=20)" if is_robust
            else f"INSUFFICIENT SAMPLES FOR A ROBUST NO RANGE ({processed_samples} samples available; recommended: 20-30 samples across multiple signers)"
        ),
        "spatial_ranges": spatial_ranges,
        "movement_ranges": movement_ranges,
        "axis_distribution": axis_distribution,
        "landmark_ranges": landmark_ranges,
        "finger_relationships": finger_relationships,
        "token_ranges": token_ranges,
        "duration": duration_stats,
        "threshold": {
            "recommended": NO_DETECTOR_THRESHOLD,
            "status": "THRESHOLD NOT FULLY VALIDATED"
        },
        "sample_quality": {
            "total_samples": processed_samples,
            "valid_frames": total_valid_frames,
            "invalid_frames": total_invalid_frames
        }
    }

    # Save Profile JSON
    json_path = os.path.join(MODEL_DIR, "no_gesture_range_profile.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    # Save Text Report
    report_path = os.path.join(MODEL_DIR, "no_gesture_range_report.txt")
    lines = [
        "=" * 75,
        "   NO GESTURE X/Y/Z AXIS RANGE CALIBRATION REPORT",
        "=" * 75,
        f"Scope: {profile['scope']}",
        f"Analyzed Samples: {processed_samples}",
        f"Total Valid Frames: {total_valid_frames}",
        f"Total Rejected Frames: {total_invalid_frames}",
        f"Z-Axis Available: {'YES' if z_available else 'NO'}",
        f"Robustness: {profile['robustness_notice']}\n",
        "=" * 75,
        "AXIS MOVEMENT DISTRIBUTION",
        "=" * 75,
        f"X Movement: {axis_distribution['X_pct']:.1f}%",
        f"Y Movement: {axis_distribution['Y_pct']:.1f}%",
        f"Z Movement: {axis_distribution['Z_pct']:.1f}%",
        f"Dominant Axis: {axis_distribution['dominant_axis']} AXIS\n",
        "=" * 75,
        "SPATIAL RANGES (X / Y / Z)",
        "=" * 75,
        f"X: P05={spatial_ranges['x']['p05']:.4f} .. P95={spatial_ranges['x']['p95']:.4f} | Min={spatial_ranges['x']['min']:.4f} .. Max={spatial_ranges['x']['max']:.4f}",
        f"Y: P05={spatial_ranges['y']['p05']:.4f} .. P95={spatial_ranges['y']['p95']:.4f} | Min={spatial_ranges['y']['min']:.4f} .. Max={spatial_ranges['y']['max']:.4f}",
        f"Z: P05={spatial_ranges['z']['p05']:.4f} .. P95={spatial_ranges['z']['p95']:.4f} | Min={spatial_ranges['z']['min']:.4f} .. Max={spatial_ranges['z']['max']:.4f}\n",
        "=" * 75,
        "MOVEMENT VELOCITY RANGES",
        "=" * 75,
        f"DX: P05={movement_ranges['dx']['p05']:.5f} .. P95={movement_ranges['dx']['p95']:.5f}",
        f"DY: P05={movement_ranges['dy']['p05']:.5f} .. P95={movement_ranges['dy']['p95']:.5f}",
        f"DZ: P05={movement_ranges['dz']['p05']:.5f} .. P95={movement_ranges['dz']['p95']:.5f}",
        f"Magnitude: P05={movement_ranges['magnitude']['p05']:.5f} .. P95={movement_ranges['magnitude']['p95']:.5f}\n",
        "=" * 75,
        "FINGER RELATIONSHIP DISTANCES (NORMALIZED 2D EUCLIDEAN)",
        "=" * 75,
        f"Thumb-Index  (4-8) : P05={finger_relationships['thumb_index']['p05']:.4f} .. P95={finger_relationships['thumb_index']['p95']:.4f} | Mean={finger_relationships['thumb_index']['mean']:.4f}",
        f"Thumb-Middle (4-12): P05={finger_relationships['thumb_middle']['p05']:.4f} .. P95={finger_relationships['thumb_middle']['p95']:.4f} | Mean={finger_relationships['thumb_middle']['mean']:.4f}",
        f"Index-Middle (8-12): P05={finger_relationships['index_middle']['p05']:.4f} .. P95={finger_relationships['index_middle']['p95']:.4f} | Mean={finger_relationships['index_middle']['mean']:.4f}",
        f"Curl Ratio (R+P/I+M): P05={finger_relationships['finger_curl_ratio']['p05']:.4f} .. P95={finger_relationships['finger_curl_ratio']['p95']:.4f} | Mean={finger_relationships['finger_curl_ratio']['mean']:.4f}\n",
        "=" * 75,
        "6D TOKEN RANGES [Hx, Hy, Mx, My, Rx, Ry]",
        "=" * 75
    ]

    for feat in ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]:
        tr = token_ranges[feat]
        lines.append(f"{feat:<3}: P05={tr['p05']:.4f} .. P95={tr['p95']:.4f} | Mean={tr['mean']:.4f} | Median={tr['median']:.4f}")

    lines.extend([
        "\n" + "=" * 75,
        "GESTURE PHASE DURATION",
        "=" * 75,
        f"Mean Start Frame: {duration_stats['mean_start_frame']}",
        f"Mean Peak Frame:  {duration_stats['mean_peak_frame']}",
        f"Mean End Frame:   {duration_stats['mean_end_frame']}",
        f"Mean Total Active Frames: {duration_stats['mean_total_gesture_frames']:.1f}",
        f"P05 Duration: {duration_stats['p05_duration']:.1f} frames | P95 Duration: {duration_stats['p95_duration']:.1f} frames\n"
    ])

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Successfully calibrated NO gesture profile!")
    print(f"  Profile JSON saved to: {json_path}")
    print(f"  Report TXT saved to:   {report_path}")

    return profile


if __name__ == "__main__":
    calibrate_no_gesture_profile()
