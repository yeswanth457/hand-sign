"""
NO Gesture X/Y/Z Axis Range Analyzer & Gesture Range Generator.

Data-driven movement range analyzer specifically for the ISL gesture: 'NO'.
Analyzes real MediaPipe hand landmark coordinates across all 21 hand landmarks (X, Y, Z),
computes robust 5th-95th percentile positional and movement signatures,
identifies dominant movement axes, segments gesture phases, maps to the 6D token representation,
and exports machine-readable profile (JSON), textual report, raw CSV samples, and visualization plots.
"""

import os
import sys
import re
import csv
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import MODEL_DIR, TOKEN_DIM, CLASS_NAMES

LANDMARK_NAMES = [
    "wrist_0",
    "thumb_cmc_1", "thumb_mcp_2", "thumb_ip_3", "thumb_tip_4",
    "index_mcp_5", "index_pip_6", "index_dip_7", "index_tip_8",
    "middle_mcp_9", "middle_pip_10", "middle_dip_11", "middle_tip_12",
    "ring_mcp_13", "ring_pip_14", "ring_dip_15", "ring_tip_16",
    "pinky_mcp_17", "pinky_pip_18", "pinky_dip_19", "pinky_tip_20"
]

KEY_LANDMARK_INDICES = {
    0: "Wrist #0",
    4: "Thumb tip #4",
    8: "Index tip #8",
    12: "Middle tip #12",
    16: "Ring tip #16",
    20: "Pinky tip #20"
}


def parse_console_log_text(log_content):
    """
    Parses browser console debug output containing:
    [LANDMARK COORD VERIFICATION]
    WRIST #0: MP (x, y[, z])
    INDEX TIP #8: MP (x, y[, z])
    """
    wrist_coords = []
    index_coords = []
    z_logged = False

    lines = log_content.splitlines()
    for line in lines:
        wrist_match = re.search(r"WRIST #0:\s*MP\s*\(\s*([\d\.-]+)\s*,\s*([\d\.-]+)(?:\s*,\s*([\d\.-]+))?\s*\)", line)
        if wrist_match:
            x = float(wrist_match.group(1))
            y = float(wrist_match.group(2))
            z = float(wrist_match.group(3)) if wrist_match.group(3) is not None else None
            if z is not None:
                z_logged = True
            wrist_coords.append((x, y, z))

        idx_match = re.search(r"INDEX TIP #8:\s*MP\s*\(\s*([\d\.-]+)\s*,\s*([\d\.-]+)(?:\s*,\s*([\d\.-]+))?\s*\)", line)
        if idx_match:
            x = float(idx_match.group(1))
            y = float(idx_match.group(2))
            z = float(idx_match.group(3)) if idx_match.group(3) is not None else None
            if z is not None:
                z_logged = True
            index_coords.append((x, y, z))

    return {
        "wrist": wrist_coords,
        "index_tip": index_coords,
        "z_logged": z_logged
    }


def load_real_no_landmark_samples(landmark_dir=None):
    """
    Loads all real NO gesture recordings from dataset/landmarks/no.
    Returns a list of structured sample dicts.
    """
    if landmark_dir is None:
        landmark_dir = os.path.join(BASE_DIR, "dataset", "landmarks", "no")

    files = sorted(glob.glob(os.path.join(landmark_dir, "**", "*.npz"), recursive=True))
    samples = []

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        raw_frames = data["landmarks"]
        video_id = str(data.get("video_id", os.path.splitext(os.path.basename(fpath))[0]))
        signer_id = str(data.get("signer_id", "hf_real"))

        valid_frames = []
        invalid_frames_count = 0

        for frame_idx, frame_data in enumerate(raw_frames):
            if not isinstance(frame_data, dict):
                invalid_frames_count += 1
                continue

            hands = frame_data.get("hands", [])
            if not hands or len(hands) == 0:
                invalid_frames_count += 1
                continue

            pts = hands[0]
            if len(pts) != 21:
                invalid_frames_count += 1
                continue

            # Verify finite numeric coordinates
            pts_arr = np.array(pts, dtype=np.float32)
            if not np.isfinite(pts_arr).all():
                invalid_frames_count += 1
                continue

            # Ensure 3D shape (21, 3)
            if pts_arr.shape[1] == 2:
                # If only 2D, pad with NaNs for z
                pts_arr = np.hstack([pts_arr, np.full((21, 1), np.nan, dtype=np.float32)])

            pose_data = frame_data.get("pose", {})
            ls = pose_data.get("LS", (0.5, 0.35, 0.0))
            rs = pose_data.get("RS", (0.5, 0.35, 0.0))
            shoulder_center = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            hand_center = frame_data.get("hand_center", (float(np.mean(pts_arr[:, 0])), float(np.mean(pts_arr[:, 1]))))

            valid_frames.append({
                "frame_idx": frame_idx,
                "landmarks": pts_arr,  # (21, 3)
                "hand_center": hand_center,
                "shoulder_center": shoulder_center,
                "handedness": "Right"  # Dominant signing hand in ISL
            })

        if len(valid_frames) > 0:
            samples.append({
                "sample_id": video_id,
                "file_path": fpath,
                "signer_id": signer_id,
                "total_frames": len(raw_frames),
                "valid_frames_count": len(valid_frames),
                "invalid_frames_count": invalid_frames_count,
                "frames": valid_frames
            })

    return samples


def compute_sample_statistics(sample):
    """
    Computes positional, movement, phase, and 6D token metrics for a single sample.
    """
    frames = sample["frames"]
    n_frames = len(frames)

    # 3D tensor: (T, 21, 3)
    coords = np.array([f["landmarks"] for f in frames], dtype=np.float32)

    # Deltas: (T-1, 21, 3)
    if n_frames > 1:
        deltas = coords[1:] - coords[:-1]
        movements_2d = np.sqrt(deltas[:, :, 0]**2 + deltas[:, :, 1]**2)
        if np.isfinite(coords[:, :, 2]).all():
            movements_3d = np.sqrt(deltas[:, :, 0]**2 + deltas[:, :, 1]**2 + deltas[:, :, 2]**2)
        else:
            movements_3d = None
    else:
        deltas = np.zeros((0, 21, 3), dtype=np.float32)
        movements_2d = np.zeros((0, 21), dtype=np.float32)
        movements_3d = None

    # Motion energy per frame based on mean 2D movement across hand
    if n_frames > 1:
        frame_energies = np.mean(movements_2d, axis=1)  # (T-1,)
    else:
        frame_energies = np.zeros(1, dtype=np.float32)

    # Detect gesture phases: neutral start -> active movement -> stabilization end
    if len(frame_energies) >= 3:
        energy_threshold = np.percentile(frame_energies, 30) + 1e-4
        active_indices = np.where(frame_energies >= energy_threshold)[0]
        if len(active_indices) > 0:
            start_frame = int(active_indices[0])
            end_frame = int(active_indices[-1]) + 1
            peak_frame = int(np.argmax(frame_energies))
        else:
            start_frame = 0
            end_frame = n_frames - 1
            peak_frame = n_frames // 2
    else:
        start_frame = 0
        end_frame = n_frames - 1
        peak_frame = n_frames // 2

    # 6D Tokens: [Hx, Hy, Mx, My, Rx, Ry]
    tokens_6d = []
    prev_hc = None
    for f in frames:
        hx, hy = f["hand_center"]
        sx, sy = f["shoulder_center"]
        if prev_hc is not None:
            mx = hx - prev_hc[0]
            my = hy - prev_hc[1]
        else:
            mx = 0.0
            my = 0.0
        prev_hc = (hx, hy)
        rx = hx - sx
        ry = hy - sy
        tokens_6d.append([hx, hy, mx, my, rx, ry])
    tokens_6d = np.array(tokens_6d, dtype=np.float32)

    return {
        "coords": coords,
        "deltas": deltas,
        "movements_2d": movements_2d,
        "movements_3d": movements_3d,
        "frame_energies": frame_energies,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "peak_frame": peak_frame,
        "total_gesture_frames": max(1, end_frame - start_frame + 1),
        "tokens_6d": tokens_6d
    }


def analyze_no_gesture_dataset(samples):
    """
    Aggregates metrics across all NO gesture samples and produces comprehensive statistics.
    """
    total_valid_frames = sum(s["valid_frames_count"] for s in samples)
    total_invalid_frames = sum(s["invalid_frames_count"] for s in samples)

    # Collect all valid coordinate arrays
    all_coords = []
    all_deltas = []
    all_tokens_6d = []
    sample_analyses = []

    for s in samples:
        analysis = compute_sample_statistics(s)
        sample_analyses.append(analysis)
        all_coords.append(analysis["coords"])
        if len(analysis["deltas"]) > 0:
            all_deltas.append(analysis["deltas"])
        all_tokens_6d.append(analysis["tokens_6d"])

    # Concatenate across all frames: (Total_Frames, 21, 3)
    concat_coords = np.concatenate(all_coords, axis=0)
    concat_deltas = np.concatenate(all_deltas, axis=0) if all_deltas else np.zeros((0, 21, 3), dtype=np.float32)
    concat_tokens = np.concatenate(all_tokens_6d, axis=0)

    # Check Z-axis availability
    z_available = np.isfinite(concat_coords[:, :, 2]).all()

    # Per-landmark statistics
    landmark_stats = {}
    total_displacements = []

    for i in range(21):
        x = concat_coords[:, i, 0]
        y = concat_coords[:, i, 1]
        z = concat_coords[:, i, 2] if z_available else None

        dx = concat_deltas[:, i, 0] if len(concat_deltas) > 0 else np.array([0.0])
        dy = concat_deltas[:, i, 1] if len(concat_deltas) > 0 else np.array([0.0])
        dz = concat_deltas[:, i, 2] if (z_available and len(concat_deltas) > 0) else None

        move_2d = np.sqrt(dx**2 + dy**2)
        total_disp = float(np.sum(move_2d))
        mean_disp = float(np.mean(move_2d))
        p95_disp = float(np.percentile(move_2d, 95))

        total_displacements.append({
            "index": i,
            "name": LANDMARK_NAMES[i],
            "total_disp": total_disp,
            "mean_disp": mean_disp,
            "p95_disp": p95_disp
        })

        stat_entry = {
            "x": {
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "p05": float(np.percentile(x, 5)),
                "p95": float(np.percentile(x, 95))
            },
            "y": {
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "p05": float(np.percentile(y, 5)),
                "p95": float(np.percentile(y, 95))
            },
            "z": {
                "min": float(np.min(z)) if z_available else None,
                "max": float(np.max(z)) if z_available else None,
                "mean": float(np.mean(z)) if z_available else None,
                "median": float(np.median(z)) if z_available else None,
                "p05": float(np.percentile(z, 5)) if z_available else None,
                "p95": float(np.percentile(z, 95)) if z_available else None
            },
            "movement": {
                "dx_min": float(np.min(dx)),
                "dx_max": float(np.max(dx)),
                "dx_p05": float(np.percentile(dx, 5)),
                "dx_p95": float(np.percentile(dx, 95)),
                "dy_min": float(np.min(dy)),
                "dy_max": float(np.max(dy)),
                "dy_p05": float(np.percentile(dy, 5)),
                "dy_p95": float(np.percentile(dy, 95)),
                "dz_min": float(np.min(dz)) if z_available else None,
                "dz_max": float(np.max(dz)) if z_available else None,
                "dz_p05": float(np.percentile(dz, 5)) if z_available else None,
                "dz_p95": float(np.percentile(dz, 95)) if z_available else None,
                "mean_movement_2d": mean_disp,
                "p95_movement_2d": p95_disp,
                "max_movement_2d": float(np.max(move_2d))
            }
        }
        landmark_stats[LANDMARK_NAMES[i]] = stat_entry

    # Rank landmarks by displacement
    total_displacements.sort(key=lambda d: d["total_disp"], reverse=True)

    # Global Axis Movement Analysis
    total_abs_dx = float(np.sum(np.abs(concat_deltas[:, :, 0]))) if len(concat_deltas) > 0 else 0.0
    total_abs_dy = float(np.sum(np.abs(concat_deltas[:, :, 1]))) if len(concat_deltas) > 0 else 0.0
    total_abs_dz = float(np.sum(np.abs(concat_deltas[:, :, 2]))) if (z_available and len(concat_deltas) > 0) else 0.0

    total_motion = total_abs_dx + total_abs_dy + total_abs_dz
    if total_motion > 0:
        x_dominance = (total_abs_dx / total_motion) * 100.0
        y_dominance = (total_abs_dy / total_motion) * 100.0
        z_dominance = (total_abs_dz / total_motion) * 100.0 if z_available else 0.0
    else:
        x_dominance, y_dominance, z_dominance = 0.0, 0.0, 0.0

    # Determine dominant axis mathematically
    axis_totals = [("X", total_abs_dx), ("Y", total_abs_dy)]
    if z_available:
        axis_totals.append(("Z", total_abs_dz))
    dominant_axis = max(axis_totals, key=lambda a: a[1])[0]

    # Global X, Y, Z ranges
    global_x_min = float(np.min(concat_coords[:, :, 0]))
    global_x_max = float(np.max(concat_coords[:, :, 0]))
    global_x_p05 = float(np.percentile(concat_coords[:, :, 0], 5))
    global_x_p95 = float(np.percentile(concat_coords[:, :, 0], 95))

    global_y_min = float(np.min(concat_coords[:, :, 1]))
    global_y_max = float(np.max(concat_coords[:, :, 1]))
    global_y_p05 = float(np.percentile(concat_coords[:, :, 1], 5))
    global_y_p95 = float(np.percentile(concat_coords[:, :, 1], 95))

    if z_available:
        global_z_min = float(np.min(concat_coords[:, :, 2]))
        global_z_max = float(np.max(concat_coords[:, :, 2]))
        global_z_p05 = float(np.percentile(concat_coords[:, :, 2], 5))
        global_z_p95 = float(np.percentile(concat_coords[:, :, 2], 95))
    else:
        global_z_min, global_z_max, global_z_p05, global_z_p95 = None, None, None, None

    # Global deltas P05 & P95
    if len(concat_deltas) > 0:
        dx_p05 = float(np.percentile(concat_deltas[:, :, 0], 5))
        dx_p95 = float(np.percentile(concat_deltas[:, :, 0], 95))
        dy_p05 = float(np.percentile(concat_deltas[:, :, 1], 5))
        dy_p95 = float(np.percentile(concat_deltas[:, :, 1], 95))
        dz_p05 = float(np.percentile(concat_deltas[:, :, 2], 5)) if z_available else None
        dz_p95 = float(np.percentile(concat_deltas[:, :, 2], 95)) if z_available else None
    else:
        dx_p05 = dx_p95 = dy_p05 = dy_p95 = dz_p05 = dz_p95 = 0.0

    # 6D Token Statistics
    token_features = ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]
    token_ranges = {}
    for idx, feat in enumerate(token_features):
        vals = concat_tokens[:, idx]
        token_ranges[feat] = {
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        }

    # Mean phase frames
    mean_start = int(np.round(np.mean([a["start_frame"] for a in sample_analyses])))
    mean_peak = int(np.round(np.mean([a["peak_frame"] for a in sample_analyses])))
    mean_end = int(np.round(np.mean([a["end_frame"] for a in sample_analyses])))
    mean_total_gesture = int(np.round(np.mean([a["total_gesture_frames"] for a in sample_analyses])))

    return {
        "num_samples": len(samples),
        "total_valid_frames": total_valid_frames,
        "total_invalid_frames": total_invalid_frames,
        "z_available": z_available,
        "dominant_axis": dominant_axis,
        "axis_analysis": {
            "X": {
                "min": global_x_min,
                "max": global_x_max,
                "range": global_x_max - global_x_min,
                "p05": global_x_p05,
                "p95": global_x_p95,
                "total_movement": total_abs_dx,
                "dominance_pct": x_dominance
            },
            "Y": {
                "min": global_y_min,
                "max": global_y_max,
                "range": global_y_max - global_y_min,
                "p05": global_y_p05,
                "p95": global_y_p95,
                "total_movement": total_abs_dy,
                "dominance_pct": y_dominance
            },
            "Z": {
                "min": global_z_min,
                "max": global_z_max,
                "range": (global_z_max - global_z_min) if z_available else None,
                "p05": global_z_p05,
                "p95": global_z_p95,
                "total_movement": total_abs_dz if z_available else None,
                "dominance_pct": z_dominance if z_available else 0.0
            }
        },
        "movement_ranges": {
            "dx_p05": dx_p05,
            "dx_p95": dx_p95,
            "dy_p05": dy_p05,
            "dy_p95": dy_p95,
            "dz_p05": dz_p05,
            "dz_p95": dz_p95
        },
        "top_moving_landmarks": total_displacements,
        "landmark_stats": landmark_stats,
        "token_ranges_6d": token_ranges,
        "phases": {
            "mean_start_frame": mean_start,
            "mean_peak_frame": mean_peak,
            "mean_end_frame": mean_end,
            "mean_total_gesture_frames": mean_total_gesture
        },
        "sample_analyses": sample_analyses
    }


def save_no_gesture_profile(results, output_path=None):
    """
    Saves the structured NO_GESTURE_PROFILE to JSON.
    """
    if output_path is None:
        output_path = os.path.join(MODEL_DIR, "no_gesture_range_profile.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile = {
        "label": "no",
        "scope": "Observed range for collected NO samples",
        "signers_count": results["num_samples"],
        "handedness": "Right",
        "sample_count": results["num_samples"],
        "is_robust": results["num_samples"] >= 20,
        "robustness_notice": (
            "SUFFICIENT SAMPLES (>=20)" if results["num_samples"] >= 20
            else f"INSUFFICIENT SAMPLES FOR A ROBUST NO RANGE ({results['num_samples']} samples available; recommended: 20-30 samples across multiple signers)"
        ),
        "dominant_axis": results["dominant_axis"],
        "axis_dominance": {
            "X_pct": results["axis_analysis"]["X"]["dominance_pct"],
            "Y_pct": results["axis_analysis"]["Y"]["dominance_pct"],
            "Z_pct": results["axis_analysis"]["Z"]["dominance_pct"]
        },
        "global_ranges": {
            "x": {"min": results["axis_analysis"]["X"]["min"], "max": results["axis_analysis"]["X"]["max"], "p05": results["axis_analysis"]["X"]["p05"], "p95": results["axis_analysis"]["X"]["p95"]},
            "y": {"min": results["axis_analysis"]["Y"]["min"], "max": results["axis_analysis"]["Y"]["max"], "p05": results["axis_analysis"]["Y"]["p05"], "p95": results["axis_analysis"]["Y"]["p95"]},
            "z": {"min": results["axis_analysis"]["Z"]["min"], "max": results["axis_analysis"]["Z"]["max"], "p05": results["axis_analysis"]["Z"]["p05"], "p95": results["axis_analysis"]["Z"]["p95"]}
        },
        "landmarks": {},
        "movement": results["movement_ranges"],
        "token_ranges_6d": results["token_ranges_6d"],
        "phases": results["phases"],
        "top_moving_landmarks": [
            {"rank": idx + 1, "landmark": d["name"], "total_displacement": d["total_disp"], "mean_displacement": d["mean_disp"]}
            for idx, d in enumerate(results["top_moving_landmarks"])
        ]
    }

    for lm_name, stats in results["landmark_stats"].items():
        profile["landmarks"][lm_name] = {
            "x_p05": stats["x"]["p05"],
            "x_p95": stats["x"]["p95"],
            "y_p05": stats["y"]["p05"],
            "y_p95": stats["y"]["p95"],
            "z_p05": stats["z"]["p05"],
            "z_p95": stats["z"]["p95"]
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    return output_path, profile


def save_coordinate_samples_csv(samples, output_path=None):
    """
    Saves raw frame-level landmark coordinate data to CSV.
    Columns: sample_id, frame, landmark_id, x, y, z, dx, dy, dz, handedness
    """
    if output_path is None:
        output_path = os.path.join(MODEL_DIR, "no_gesture_coordinate_samples.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "frame", "landmark_id", "landmark_name", "x", "y", "z", "dx", "dy", "dz", "handedness"])

        for s in samples:
            sample_id = s["sample_id"]
            frames = s["frames"]
            for fr_idx, fr in enumerate(frames):
                pts = fr["landmarks"]  # (21, 3)
                handedness = fr.get("handedness", "Right")

                # Compute frame-to-frame delta if previous frame exists
                prev_pts = frames[fr_idx - 1]["landmarks"] if fr_idx > 0 else None

                for lm_idx in range(21):
                    x = float(pts[lm_idx, 0])
                    y = float(pts[lm_idx, 1])
                    z = float(pts[lm_idx, 2]) if np.isfinite(pts[lm_idx, 2]) else ""

                    if prev_pts is not None:
                        dx = float(x - prev_pts[lm_idx, 0])
                        dy = float(y - prev_pts[lm_idx, 1])
                        dz = float(z - prev_pts[lm_idx, 2]) if (z != "" and np.isfinite(prev_pts[lm_idx, 2])) else ""
                    else:
                        dx, dy, dz = 0.0, 0.0, 0.0 if z != "" else ""

                    writer.writerow([
                        sample_id, fr["frame_idx"], lm_idx, LANDMARK_NAMES[lm_idx],
                        f"{x:.6f}", f"{y:.6f}", f"{z:.6f}" if z != "" else "null",
                        f"{dx:.6f}", f"{dy:.6f}", f"{dz:.6f}" if dz != "" else "null",
                        handedness
                    ])

    return output_path


def generate_visualizations(samples, sample_analyses, output_path=None):
    """
    Generates high-resolution multi-panel plots for the NO gesture:
    1. Wrist X over time
    2. Wrist Y over time
    3. Wrist Z over time
    4. Index-tip X over time
    5. Index-tip Y over time
    6. Index-tip Z over time
    7. X/Y movement trajectory
    8. Movement magnitude over time
    """
    if output_path is None:
        output_path = os.path.join(MODEL_DIR, "no_gesture_movement_analysis.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use first sample as primary exemplar for trajectory, and overlay all samples
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle("Observed Range for Collected NO Samples — MediaPipe Kinematic Analysis", fontsize=16, fontweight="bold", y=0.99)

    # Plot 1: Wrist X over time
    ax = axes[0, 0]
    for idx, s in enumerate(samples):
        coords = s["frames"]
        wx = [f["landmarks"][0, 0] for f in coords]
        alpha = 0.8 if idx == 0 else 0.3
        lw = 2.0 if idx == 0 else 1.0
        ax.plot(wx, alpha=alpha, linewidth=lw, label=f"Sample {idx+1:02d}" if idx < 3 else None)
    ax.set_title("1. Wrist #0: X Position Over Time (Horizontal)", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized X")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # Plot 2: Wrist Y over time
    ax = axes[0, 1]
    for idx, s in enumerate(samples):
        coords = s["frames"]
        wy = [f["landmarks"][0, 1] for f in coords]
        alpha = 0.8 if idx == 0 else 0.3
        lw = 2.0 if idx == 0 else 1.0
        ax.plot(wy, alpha=alpha, linewidth=lw, label=f"Sample {idx+1:02d}" if idx < 3 else None)
    ax.set_title("2. Wrist #0: Y Position Over Time (Vertical)", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Y")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Wrist Z over time (Depth)
    ax = axes[1, 0]
    has_z = False
    for idx, s in enumerate(samples):
        coords = s["frames"]
        wz = [f["landmarks"][0, 2] for f in coords]
        if np.isfinite(wz).all():
            has_z = True
            alpha = 0.8 if idx == 0 else 0.3
            lw = 2.0 if idx == 0 else 1.0
            ax.plot(wz, alpha=alpha, linewidth=lw, label=f"Sample {idx+1:02d}" if idx < 3 else None)
    if has_z:
        ax.set_title("3. Wrist #0: Z Position Over Time (Depth)", fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("MediaPipe Depth Z")
    else:
        ax.text(0.5, 0.5, "Z AXIS: NOT AVAILABLE IN RECORDING", ha="center", va="center", transform=ax.transAxes, color="red")
        ax.set_title("3. Wrist #0: Z Position Over Time (Depth - Unavailable)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 4: Index-tip X over time
    ax = axes[1, 1]
    for idx, s in enumerate(samples):
        coords = s["frames"]
        ix = [f["landmarks"][8, 0] for f in coords]
        alpha = 0.8 if idx == 0 else 0.3
        lw = 2.0 if idx == 0 else 1.0
        ax.plot(ix, alpha=alpha, linewidth=lw, label=f"Sample {idx+1:02d}" if idx < 3 else None)
    ax.set_title("4. Index Tip #8: X Position Over Time (Oscillation Sweep)", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized X")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 5: Index-tip Y over time
    ax = axes[2, 0]
    for idx, s in enumerate(samples):
        coords = s["frames"]
        iy = [f["landmarks"][8, 1] for f in coords]
        alpha = 0.8 if idx == 0 else 0.3
        lw = 2.0 if idx == 0 else 1.0
        ax.plot(iy, alpha=alpha, linewidth=lw, label=f"Sample {idx+1:02d}" if idx < 3 else None)
    ax.set_title("5. Index Tip #8: Y Position Over Time", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Y")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 6: Index-tip Z over time
    ax = axes[2, 1]
    if has_z:
        for idx, s in enumerate(samples):
            coords = s["frames"]
            iz = [f["landmarks"][8, 2] for f in coords]
            if np.isfinite(iz).all():
                alpha = 0.8 if idx == 0 else 0.3
                lw = 2.0 if idx == 0 else 1.0
                ax.plot(iz, alpha=alpha, linewidth=lw, label=f"Sample {idx+1:02d}" if idx < 3 else None)
        ax.set_title("6. Index Tip #8: Z Position Over Time (Depth)", fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("MediaPipe Depth Z")
    else:
        ax.text(0.5, 0.5, "Z AXIS: NOT AVAILABLE IN RECORDING", ha="center", va="center", transform=ax.transAxes, color="red")
        ax.set_title("6. Index Tip #8: Z Position Over Time (Depth - Unavailable)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 7: X/Y 2D Movement Trajectory (Wrist vs Index Tip)
    ax = axes[3, 0]
    if len(samples) > 0:
        ex = samples[0]["frames"]
        wx = [f["landmarks"][0, 0] for f in ex]
        wy = [f["landmarks"][0, 1] for f in ex]
        ix = [f["landmarks"][8, 0] for f in ex]
        iy = [f["landmarks"][8, 1] for f in ex]
        ax.plot(wx, wy, "b-o", markersize=3, alpha=0.7, label="Wrist #0 Trajectory")
        ax.plot(ix, iy, "r-s", markersize=3, alpha=0.7, label="Index Tip #8 Trajectory")
        # Invert Y to match screen coordinate system
        ax.invert_yaxis()
    ax.set_title("7. X/Y Movement Trajectory (Exemplar Sample)", fontweight="bold")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y (Screen Inverted)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 8: Movement Magnitude Over Time
    ax = axes[3, 1]
    for idx, sa in enumerate(sample_analyses):
        energies = sa["frame_energies"]
        alpha = 0.8 if idx == 0 else 0.3
        lw = 2.0 if idx == 0 else 1.0
        ax.plot(energies, alpha=alpha, linewidth=lw, label=f"Sample {idx+1:02d}" if idx < 3 else None)
    ax.set_title("8. Movement Magnitude (Motion Velocity) Over Time", fontweight="bold")
    ax.set_xlabel("Frame Index (t)")
    ax.set_ylabel("Mean 2D Velocity (pixels/norm)")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def generate_text_report(results, samples, output_path=None):
    """
    Generates human-readable report covering all phases.
    """
    if output_path is None:
        output_path = os.path.join(MODEL_DIR, "no_gesture_range_report.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("   NO GESTURE X/Y/Z AXIS RANGE ANALYSIS & GESTURE PROFILE REPORT")
    lines.append("=" * 70)
    lines.append("Scope: Observed range for collected NO samples")
    lines.append(f"Analyzed Samples: {results['num_samples']}")
    lines.append(f"Total Valid Frames: {results['total_valid_frames']}")
    lines.append(f"Total Rejected Frames: {results['total_invalid_frames']}")
    lines.append(f"Handedness: Right (Standard dominant hand in ISL)")
    lines.append(f"Z-Axis Available: {'YES' if results['z_available'] else 'NO (Z NOT LOGGED IN FRONTEND CONSOLE)'}")
    lines.append("")

    if results["num_samples"] < 20:
        lines.append(">>> NOTICE: INSUFFICIENT SAMPLES FOR A ROBUST NO RANGE")
        lines.append(f"    Currently analyzed: {results['num_samples']} samples. Recommended: 20-30 samples across multiple signers.\n")
    else:
        lines.append(">>> STATUS: SUFFICIENT SAMPLES AVAILABLE (>= 20)\n")

    lines.append("=" * 70)
    lines.append("PER-SAMPLE INVENTORY & BREAKDOWN")
    lines.append("=" * 70)
    for idx, s in enumerate(samples, 1):
        lines.append(f"NO SAMPLE {idx:02d}: {s['sample_id']} | Signer: {s['signer_id']} | Valid Frames: {s['valid_frames_count']} / {s['total_frames']}")

    lines.append("\n" + "=" * 70)
    lines.append("COMBINED NO GESTURE RANGE — AXIS ANALYSIS")
    lines.append("=" * 70)

    for ax_name in ["X", "Y", "Z"]:
        info = results["axis_analysis"][ax_name]
        lines.append(f"{ax_name}:")
        if info["min"] is not None:
            lines.append(f"  min:       {info['min']:.4f}")
            lines.append(f"  max:       {info['max']:.4f}")
            lines.append(f"  range:     {info['range']:.4f}")
            lines.append(f"  P05:       {info['p05']:.4f}")
            lines.append(f"  P95:       {info['p95']:.4f}")
            lines.append(f"  movement:  {info['total_movement']:.4f} ({info['dominance_pct']:.1f}% dominance)")
        else:
            lines.append("  NOT AVAILABLE — Z NOT LOGGED")

    lines.append(f"\nDOMINANT AXIS: {results['dominant_axis']} AXIS")

    lines.append("\n" + "=" * 70)
    lines.append("MOVEMENT ANALYSIS (FRAME-TO-FRAME VELOCITY)")
    lines.append("=" * 70)
    m = results["movement_ranges"]
    lines.append(f"DX:  P05 = {m['dx_p05']:.5f}  |  P95 = {m['dx_p95']:.5f}")
    lines.append(f"DY:  P05 = {m['dy_p05']:.5f}  |  P95 = {m['dy_p95']:.5f}")
    if m["dz_p05"] is not None:
        lines.append(f"DZ:  P05 = {m['dz_p05']:.5f}  |  P95 = {m['dz_p95']:.5f}")
    else:
        lines.append("DZ:  NOT AVAILABLE — Z NOT LOGGED")
    lines.append(f"Dominant movement axis: {results['dominant_axis']}")

    lines.append("\n" + "=" * 70)
    lines.append("TOP MOVING LANDMARKS (RANKED BY TOTAL DISPLACEMENT)")
    lines.append("=" * 70)
    for idx, d in enumerate(results["top_moving_landmarks"], 1):
        lines.append(f"{idx:2d}. {d['name']:<16} : Total Disp={d['total_disp']:.4f}, Mean Velocity={d['mean_disp']:.5f}, P95 Velocity={d['p95_disp']:.5f}")

    lines.append("\n" + "=" * 70)
    lines.append("GESTURE PHASES (CALCULATED FROM MOTION ENERGY)")
    lines.append("=" * 70)
    ph = results["phases"]
    lines.append(f"Gesture start frame:  {ph['mean_start_frame']}")
    lines.append(f"Gesture peak frame:   {ph['mean_peak_frame']}")
    lines.append(f"Gesture end frame:    {ph['mean_end_frame']}")
    lines.append(f"Total gesture frames: {ph['mean_total_gesture_frames']}")

    lines.append("\n" + "=" * 70)
    lines.append("NO 6D TOKEN RANGE (PROJECT INFERENCE PIPELINE COMPATIBILITY)")
    lines.append("=" * 70)
    for feat, tr in results["token_ranges_6d"].items():
        lines.append(f"{feat:<3}: P05 = {tr['p05']:.4f}  |  P95 = {tr['p95']:.4f}  |  Mean = {tr['mean']:.4f}  |  Median = {tr['median']:.4f}  |  Min = {tr['min']:.4f}  |  Max = {tr['max']:.4f}")

    lines.append("\n" + "=" * 70)
    lines.append("KEY LANDMARKS 5th-95th PERCENTILE ROBUST RANGES")
    lines.append("=" * 70)
    for idx, name in KEY_LANDMARK_INDICES.items():
        lm_key = LANDMARK_NAMES[idx]
        st = results["landmark_stats"][lm_key]
        z_str = f"Z: [{st['z']['p05']:.4f} .. {st['z']['p95']:.4f}]" if st['z']['p05'] is not None else "Z: N/A"
        lines.append(f"{name:<16}: X: [{st['x']['p05']:.4f} .. {st['x']['p95']:.4f}]  |  Y: [{st['y']['p05']:.4f} .. {st['y']['p95']:.4f}]  |  {z_str}")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return output_path, report_text


def execute_analysis(landmark_dir=None, console_log_path=None):
    """
    Main entry point for NO gesture range analysis.
    """
    print("=" * 70)
    print("        NO GESTURE RANGE ANALYSIS (DATA-DRIVEN GENERATOR)")
    print("=" * 70)

    # 1. Load real recorded samples
    samples = load_real_no_landmark_samples(landmark_dir)
    print(f"Loaded {len(samples)} real NO gesture recording files.")

    # 2. Check console log file if provided
    if console_log_path and os.path.exists(console_log_path):
        with open(console_log_path, "r", encoding="utf-8") as f:
            console_data = parse_console_log_text(f.read())
        print(f"Parsed console log: {len(console_data['wrist'])} wrist entries, {len(console_data['index_tip'])} index tip entries.")
        if not console_data["z_logged"]:
            print("Z AXIS: NOT CURRENTLY LOGGED IN CONSOLE LOGS")

    # 3. Perform statistical analysis
    results = analyze_no_gesture_dataset(samples)

    # 4. Save structured outputs
    profile_path, profile = save_no_gesture_profile(results)
    print(f"Saved machine-readable profile to: {profile_path}")

    csv_path = save_coordinate_samples_csv(samples)
    print(f"Saved raw coordinate samples CSV to: {csv_path}")

    plot_path = generate_visualizations(samples, results["sample_analyses"])
    print(f"Saved kinematic visualizations to: {plot_path}")

    report_path, report_text = generate_text_report(results, samples)
    print(f"Saved textual report to: {report_path}")

    # 5. Print validation output block
    print("\n" + "=" * 70)
    print("NO GESTURE AXIS ANALYSIS")
    print("=" * 70)
    print(f"Valid frames:   {results['total_valid_frames']}")
    print(f"Invalid frames: {results['total_invalid_frames']}")

    print("\nX RANGE:")
    print(f"min: {results['axis_analysis']['X']['min']:.4f}")
    print(f"max: {results['axis_analysis']['X']['max']:.4f}")
    print(f"P05: {results['axis_analysis']['X']['p05']:.4f}")
    print(f"P95: {results['axis_analysis']['X']['p95']:.4f}")

    print("\nY RANGE:")
    print(f"min: {results['axis_analysis']['Y']['min']:.4f}")
    print(f"max: {results['axis_analysis']['Y']['max']:.4f}")
    print(f"P05: {results['axis_analysis']['Y']['p05']:.4f}")
    print(f"P95: {results['axis_analysis']['Y']['p95']:.4f}")

    if results["z_available"]:
        print("\nZ RANGE:")
        print(f"min: {results['axis_analysis']['Z']['min']:.4f}")
        print(f"max: {results['axis_analysis']['Z']['max']:.4f}")
        print(f"P05: {results['axis_analysis']['Z']['p05']:.4f}")
        print(f"P95: {results['axis_analysis']['Z']['p95']:.4f}")
    else:
        print("\nZ RANGE:")
        print("NOT AVAILABLE — Z NOT LOGGED")

    print("\n==================================================")
    print("MOVEMENT ANALYSIS")
    print("==================================================")
    print(f"DX:\nP05: {results['movement_ranges']['dx_p05']:.5f}\nP95: {results['movement_ranges']['dx_p95']:.5f}")
    print(f"DY:\nP05: {results['movement_ranges']['dy_p05']:.5f}\nP95: {results['movement_ranges']['dy_p95']:.5f}")
    if results["movement_ranges"]["dz_p05"] is not None:
        print(f"DZ:\nP05: {results['movement_ranges']['dz_p05']:.5f}\nP95: {results['movement_ranges']['dz_p95']:.5f}")
    else:
        print("DZ:\nNOT AVAILABLE — Z NOT LOGGED")
    print(f"\nDominant movement axis: {results['dominant_axis']} AXIS")

    print("\n==================================================")
    print("6D TOKEN ANALYSIS")
    print("==================================================")
    for feat in ["Hx", "Hy", "Mx", "My", "Rx", "Ry"]:
        tr = results["token_ranges_6d"][feat]
        print(f"{feat}: P05={tr['p05']:.4f}, P95={tr['p95']:.4f}, Mean={tr['mean']:.4f}")

    print("\n==================================================")
    print("FINAL NO GESTURE PROFILE")
    print("==================================================")
    print(f"Profile saved to: {profile_path}")
    print(f"Observed scope: {profile['scope']}")
    print(f"Status: {profile['robustness_notice']}")

    return results


if __name__ == "__main__":
    execute_analysis()
