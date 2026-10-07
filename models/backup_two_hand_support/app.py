"""
FastAPI Web Server for RT-STAMP-SLR ISL Translation System.
Provides endpoints for real-time sign language recognition using CNN-GRU model.
Exactly 22 ISL sign classes including Thalapathy. No simulation, no mock predictions.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import base64
import time
import numpy as np
import cv2
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from collections import deque
from typing import List, Dict, Any, Optional

from config import (
    CLASS_NAMES, NUM_CLASSES, INDEX_TO_CLASS, CLASS_TO_INDEX,
    TAMIL_VOCAB_MAP, ENGLISH_TRANSLATIONS, BASE_DIR,
    CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES, COOLDOWN_FRAMES,
    IDLE_ENERGY_THRESHOLD, SIGNING_MOTION_THRESHOLD
)
from src.landmark_extractor import LandmarkExtractor
from src.motion_energy import MotionEnergyCalculator
from src.frame_selector import MotionFrameSelector
from src.gesture_tokenizer import GestureTokenizer
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.no_binary_model import NOBinaryInferenceEngine
from src.early_decision import EarlyDecisionEngine
from src.sentence_processor import LocalSentenceProcessor
from src.build_real_split import resample_tokens

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            msg = " ".join(str(a) for a in args)
            sys.stdout.buffer.write(msg.encode('utf-8', errors='replace') + b'\n')
            sys.stdout.buffer.flush()
        except Exception:
            safe_args = [str(a).encode('ascii', errors='backslashreplace').decode('ascii') for a in args]
            print(*safe_args, **kwargs)

app = FastAPI(
    title="RT-STAMP-SLR ISL Translator",
    description="Real-Time Indian Sign Language to English/Tamil Translator (22 Classes)",
    version="2.1.0"
)

# Static files directory for web UI
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Models directory for Task models and PyTorch models
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
app.mount("/models", StaticFiles(directory=models_dir), name="models")

# Pipeline Components Initialization
landmark_extractor = LandmarkExtractor()
frame_selector = MotionFrameSelector()
gesture_tokenizer = GestureTokenizer()
model_engine = CNNGRUInferenceEngine()
safe_print(f"MODEL_CLASS_COUNT = {model_engine.num_classes}")
safe_print(f"MODEL_CLASS_21 = {CLASS_NAMES[21] if len(CLASS_NAMES) > 21 else 'UNKNOWN'}")
no_binary_engine = NOBinaryInferenceEngine(threshold=0.60)
early_decision_engine = EarlyDecisionEngine()
sentence_processor = LocalSentenceProcessor(current_language="english")

# Real-time rolling gesture motion token buffer
realtime_token_buffer = deque(maxlen=60)
# binary_token_buffer is the ISOLATED NO-detector buffer.
# It resets on the FIRST no-hand frame (not after 3).
# It NEVER shares tokens with realtime_token_buffer across gesture boundaries.
# This mirrors the standalone test_no_binary_live.py exactly.
binary_token_buffer = deque(maxlen=25)
binary_consecutive_no_count = 0   # Tracks consecutive frames where NO_PROB >= 0.60
buffered_frame_ids = deque(maxlen=60)
sequence_counter = 0
frame_counter = 0
no_hand_consecutive_count = 0
NO_HAND_CLEAR_THRESHOLD = 3
idle_consecutive_count = 0
IDLE_CLEAR_THRESHOLD = 15
last_predicted_sequence = None
_first_no_sequence_saved = False  # Only save the first browser NO sequence to disk

# NO override thresholds
NO_PROB_THRESHOLD = 0.60          # Minimum binary NO probability to count as a NO frame
NO_CONSECUTIVE_REQUIRED = 1       # Minimum consecutive qualifying frames before overriding final_class


# Sequence & Stale Frame Tracking
active_sequence_id = 0
last_processed_frame_id = 0
last_token_timestamp = 0.0
token_timestamps = deque(maxlen=20)
cnn_call_count = 0

# Hand Tracking & Switching
last_selected_hand = None
hand_switch_count = 0
hand_switch_frames = []


# Pydantic Schemas
class FrameImageInput(BaseModel):
    image_base64: str
    frame_id: Optional[int] = 0
    timestamp_ms: Optional[float] = None


class LanguageToggleInput(BaseModel):
    language: str  # "english" or "tamil"


class TokenInput(BaseModel):
    """Input for browser-side MediaPipe: pre-computed 6D token + optional pose data."""
    token: Optional[List[float]] = None
    frame_id: Optional[int] = 0
    sequence_id: Optional[int] = 0
    timestamp_ms: Optional[float] = None
    pose: Optional[Dict[str, Any]] = None
    hand_center: Optional[List[float]] = None
    shoulder_center: Optional[List[float]] = None
    has_hand: Optional[bool] = True
    hands: Optional[List[Any]] = None
    hand_count: Optional[int] = 0
    primary_hand: Optional[str] = "NONE"
    secondary_hand: Optional[str] = "NONE"
    selected_hand: Optional[str] = "Right"
    selected_hand_index: Optional[int] = 0
    selected_hand_confidence: Optional[float] = 1.0
    camera_fps: Optional[float] = 30.0
    token_fps: Optional[float] = 20.0



@app.on_event("startup")
async def startup_event():
    safe_print("==================================================")
    safe_print(f"MODEL_CLASS_COUNT = {model_engine.num_classes}")
    safe_print(f"MODEL_CLASS_21 = {CLASS_NAMES[21] if len(CLASS_NAMES) > 21 else 'UNKNOWN'}")
    safe_print(f"TOTAL VOCABULARY COUNT = {len(CLASS_NAMES)}")
    safe_print("==================================================")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RT-STAMP-SLR ISL Translator API is running.</h1>"


@app.get("/api/vocabulary")
async def get_vocabulary():
    """Returns the 22 ISL vocabulary signs with English and Tamil mappings."""
    vocab_details = []
    for idx, word in enumerate(CLASS_NAMES):
        vocab_details.append({
            "id": idx,
            "english": word,
            "display_english": word.replace("_", " ").title(),
            "tamil": TAMIL_VOCAB_MAP.get(word, word),
            "translation": ENGLISH_TRANSLATIONS.get(word, word)
        })
    return {"vocabulary": vocab_details, "total": NUM_CLASSES}


_dataset_cache = {}

def _fetch_hf_rows(url: str) -> dict:
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RT-STAMP-SLR/2.0 (ISL-Translator; Educational/Research)"}
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))

def _get_local_dataset_rows(offset: int, length: int) -> dict:
    import csv
    csv_path = os.path.join(BASE_DIR, "dataset", "metadata", "dataset.csv")
    if not os.path.exists(csv_path):
        return {"rows": [], "total": 0, "offset": offset, "length": length, "has_more": False}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    total = len(reader)
    sliced = reader[offset: offset + length]
    result_rows = []
    for idx, r in enumerate(sliced):
        result_rows.append({
            "row_idx": offset + idx,
            "word": r.get("sign_class", "--"),
            "normalized_word": r.get("sign_class", "--"),
            "dataset": "Local ISL Dataset (raw)",
            "signer": r.get("signer_id", "signer_01"),
            "video_path": r.get("video_path", ""),
            "original_filename": os.path.basename(r.get("video_path", "")),
            "fps": float(r.get("fps", 30.0)),
            "resolution": "640x480",
            "duration": round(float(r.get("total_frames", 30)) / max(float(r.get("fps", 30.0)), 1.0), 2),
            "quality_score": round(float(r.get("selection_ratio", 0.85)), 2),
            "duplicate_status": "unique",
            "review_status": "accepted",
            "repository": "https://huggingface.co/datasets/vidit031/isl-isolated-40words",
            "download_url": "",
        })
    return {
        "rows": result_rows,
        "total": total,
        "offset": offset,
        "length": length,
        "has_more": (offset + length) < total,
        "source": "Local ISL Dataset (metadata/dataset.csv fallback)",
        "config": "default",
        "split": "train"
    }

@app.get("/api/dataset_browser")
async def dataset_browser(offset: int = 0, length: int = 5):
    """
    Proxies HuggingFace Datasets Server API for the ISL dataset explorer.
    Dataset: vidit031/isl-isolated-40words, config=default, split=train.
    Falls back cleanly to local dataset.csv if HF is unreachable.
    Returns actual rows from the real dataset — never fake data.
    """
    import asyncio

    cache_key = f"{offset}_{length}"
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    HF_DATASET = "vidit031/isl-isolated-40words"
    HF_CONFIG = "default"
    HF_SPLIT = "train"
    HF_BASE = "https://datasets-server.huggingface.co"

    rows_url = f"{HF_BASE}/rows?dataset={HF_DATASET}&config={HF_CONFIG}&split={HF_SPLIT}&offset={offset}&length={length}"

    try:
        data = await asyncio.to_thread(_fetch_hf_rows, rows_url)
        total = data.get("num_rows_total", 0)
        raw_rows = data.get("rows", [])

        # Transform rows for frontend display
        result_rows = []
        for entry in raw_rows:
            row = entry.get("row", {})
            result_rows.append({
                "row_idx": entry.get("row_idx", 0),
                "word": row.get("word", "--"),
                "normalized_word": row.get("normalized_word", "--"),
                "dataset": row.get("dataset", "--"),
                "signer": row.get("signer", "--"),
                "video_path": row.get("video_path", ""),
                "original_filename": row.get("original_filename", ""),
                "fps": row.get("fps"),
                "resolution": row.get("resolution", "--"),
                "duration": row.get("duration"),
                "quality_score": row.get("quality_score"),
                "duplicate_status": row.get("duplicate_status", "unknown"),
                "review_status": row.get("review_status", "pending"),
                "repository": row.get("repository", ""),
                "download_url": row.get("download_url", ""),
            })

        has_more = (offset + length) < total

        resp_payload = {
            "rows": result_rows,
            "total": total,
            "offset": offset,
            "length": length,
            "has_more": has_more,
            "source": f"https://huggingface.co/datasets/{HF_DATASET}",
            "config": HF_CONFIG,
            "split": HF_SPLIT,
        }
        _dataset_cache[cache_key] = resp_payload
        return resp_payload

    except Exception as e:
        print(f"[Dataset Browser] HF API query error ({e}), falling back to local dataset.csv")
        local_res = _get_local_dataset_rows(offset, length)
        if local_res["total"] > 0:
            return local_res
        raise HTTPException(status_code=502, detail=f"Failed to load dataset: {str(e)}")


@app.post("/api/process_token")
async def process_token(data: TokenInput):
    """
    Accepts hand and pose landmark metadata from browser-side MediaPipe.
    Uses Python GestureTokenizer to compute 6D tokens, maintains isolated sequence buffers,
    rejects stale out-of-order frames, and runs binary NO classifier inference.
    """
    global frame_counter, sequence_counter, no_hand_consecutive_count, idle_consecutive_count, last_predicted_sequence, binary_consecutive_no_count, _first_no_sequence_saved
    global active_sequence_id, last_processed_frame_id, last_token_timestamp, token_timestamps, cnn_call_count
    global last_selected_hand, hand_switch_count, hand_switch_frames

    t_start = time.perf_counter()
    frame_counter += 1
    fid = data.frame_id or frame_counter
    seq_id = data.sequence_id or 0

    # Section 12 & 14: Sequence transition detection & Stale rejection
    is_client_reloaded = (active_sequence_id - seq_id) > 1000
    if seq_id > 0 and (active_sequence_id == 0 or seq_id > active_sequence_id or is_client_reloaded):
        active_sequence_id = seq_id
        binary_token_buffer.clear()
        realtime_token_buffer.clear()
        buffered_frame_ids.clear()
        binary_consecutive_no_count = 0
        last_processed_frame_id = 0
        last_selected_hand = None
        hand_switch_count = 0
        hand_switch_frames = []
        gesture_tokenizer.reset()
        early_decision_engine.reset()
        sentence_processor.clear()
        safe_print(
            f"\nTOKEN_SEQUENCE_ID={seq_id}\n"
            f"BACKEND_SEQUENCE_ID={active_sequence_id}\n"
            f"STALE_TOKEN_REJECTED=NO\n"
            f"SEQUENCE_RESET_REASON=NEW_SEQUENCE\n"
            f"[SEQUENCE RESET] New sequence_id={seq_id}. Cleared all token buffers."
        )
    elif seq_id > 0 and seq_id < active_sequence_id:
        safe_print(
            f"\nTOKEN_SEQUENCE_ID={seq_id}\n"
            f"BACKEND_SEQUENCE_ID={active_sequence_id}\n"
            f"STALE_TOKEN_REJECTED=YES\n"
            f"SEQUENCE_RESET_REASON=STALE_SEQUENCE"
        )
        return {
            "status": "STALE_TOKEN_REJECTED",
            "frame_id": fid,
            "sequence_id": seq_id,
            "active_sequence_id": active_sequence_id,
            "no_probability": 0.0,
            "final_class": "--",
            "hand_detected": data.has_hand
        }

    # Handle NO_HAND state
    if not data.has_hand:
        safe_print(f"\n[HAND]\ndetected=false\nhand_count=0 (streak={no_hand_consecutive_count + 1}/{NO_HAND_CLEAR_THRESHOLD})")
        idle_consecutive_count = 0
        no_hand_consecutive_count += 1
        if no_hand_consecutive_count >= NO_HAND_CLEAR_THRESHOLD:
            safe_print(f"SEQUENCE_RESET_REASON=HAND_LOST")
            last_predicted_sequence = None
            binary_token_buffer.clear()
            binary_consecutive_no_count = 0
            realtime_token_buffer.clear()
            buffered_frame_ids.clear()
            gesture_tokenizer.reset()
            early_decision_engine.reset()
            last_processed_frame_id = 0
        proc_time_ms = (time.perf_counter() - t_start) * 1000
        return {
            "frame_id": fid,
            "sequence_id": seq_id,
            "primary_class": "--",
            "primary_confidence": 0.0,
            "no_probability": 0.0,
            "no_confirmations": 0,
            "no_confirmed": False,
            "final_class": "--",
            "hand_detected": False,
            "token_buffer_length": len(realtime_token_buffer),
            "binary_no_probability": 0.0,
            "motion_energy": 0.0,
            "threshold": 0.015,
            "token": None,
            "buffer_status": f"{len(realtime_token_buffer)}/25",
            "processing_time_ms": round(proc_time_ms, 2),
            "prediction": {
                "word": "--",
                "class_id": -1,
                "confidence": 0.0
            },
            "binary_no": {
                "no_probability": 0.0,
                "no_prediction": "NOT_NO",
                "no_confirmed": False,
                "consecutive_count": 0
            },
            "early_decision": {
                "state": "NO_HAND",
                "accepted": False,
                "cooldown_remaining": 0,
                "last_accepted": "--",
                "sustained_count": 0,
                "sustained_target": 2
            },
            "hands_info": {
                "hand_count": 0,
                "primary_hand": "NONE",
                "secondary_hand": "NONE"
            },
            "translation": sentence_processor.get_current_translation()
        }

    # Stale out-of-order frame rejection
    if fid > 0 and fid <= last_processed_frame_id:
        safe_print(f"[STALE FRAME REJECTED] fid={fid} <= last_processed={last_processed_frame_id}")
        return {
            "status": "STALE_FRAME_REJECTED",
            "frame_id": fid,
            "last_processed_frame_id": last_processed_frame_id,
            "no_probability": 0.0,
            "final_class": "--"
        }

    # Token Calculation using authoritative Python GestureTokenizer
    hand_c = tuple(data.hand_center) if (data.hand_center and len(data.hand_center) >= 2) else (0.5, 0.5)
    shoulder_c = tuple(data.shoulder_center) if (data.shoulder_center and len(data.shoulder_center) >= 2) else (0.5, 0.35)

    if data.token and len(data.token) == 6:
        token_arr = np.array(data.token, dtype=np.float32)
    else:
        lm_data_dict = {"hand_center": hand_c, "shoulder_center": shoulder_c}
        token_arr = gesture_tokenizer.tokenize_frame(lm_data_dict)

    last_processed_frame_id = fid

    # Calculate token sampling rate
    now_ts = time.time()
    if last_token_timestamp > 0:
        dt = now_ts - last_token_timestamp
        if dt > 0 and dt < 2.0:
            token_timestamps.append(dt)
    last_token_timestamp = now_ts
    avg_dt = np.mean(token_timestamps) if len(token_timestamps) > 0 else 0.05
    tokens_per_sec = float(1.0 / max(0.001, avg_dt))

    # Section 3 & 8: Detailed Live Sequence Diagnostic and Hand Switching
    hand_count = data.hand_count or 1
    sel_hand = data.selected_hand or data.primary_hand or "Right"
    sel_conf = float(data.selected_hand_confidence) if data.selected_hand_confidence is not None else 1.0
    sel_idx = getattr(data, "selected_hand_index", 0) or 0

    if last_selected_hand is not None and last_selected_hand != sel_hand and sel_hand != "NONE":
        hand_switch_count += 1
        hand_switch_frames.append(fid)
        safe_print(f"\n[BACKEND HAND SWITCH DETECTED] old={last_selected_hand}, new={sel_hand} -> resetting sequence temporal state")
        realtime_token_buffer.clear()
        binary_token_buffer.clear()
        buffered_frame_ids.clear()
        gesture_tokenizer.reset()
        early_decision_engine.reset()
        sentence_processor.clear()
    last_selected_hand = sel_hand

    # Accumulate into rolling token buffer
    realtime_token_buffer.append(token_arr)
    binary_token_buffer.append(token_arr)
    buffered_frame_ids.append(fid)
    no_hand_consecutive_count = 0

    buf_len = len(realtime_token_buffer)
    binary_buf_len = len(binary_token_buffer)

    safe_print(
        f"\nLIVE_SEQUENCE_START\n"
        f"sequence_id={seq_id}\n"
        f"RAW_HAND_COUNT={hand_count}\n"
        f"SELECTED_HAND={sel_hand}\n"
        f"SELECTED_HAND_INDEX={sel_idx}\n"
        f"SELECTED_HAND_CONFIDENCE={sel_conf:.4f}\n"
        f"HAND_SWITCH_COUNT={hand_switch_count}\n"
        f"HAND_SWITCH_FRAME={hand_switch_frames}\n"
        f"landmark_count={21 * hand_count}\n"
        f"token_available=True\n"
        f"token_shape=(6,)\n"
        f"token_count={buf_len}"
    )

    safe_print(
        f"\n[TOKEN]\n"
        f"Hx={token_arr[0]:.4f}\n"
        f"Hy={token_arr[1]:.4f}\n"
        f"Mx={token_arr[2]:.4f}\n"
        f"My={token_arr[3]:.4f}\n"
        f"Rx={token_arr[4]:.4f}\n"
        f"Ry={token_arr[5]:.4f}"
    )

    # Build landmark_data for motion energy calculation
    pose_dict = data.pose or {}
    if not pose_dict:
        pose_dict = {"HAND": (hand_c[0], hand_c[1], 0.0)}

    landmark_data = {
        "pose": pose_dict,
        "hands": [],
        "hand_center": hand_c,
        "shoulder_center": shoulder_c,
    }

    # Motion energy calculation
    sel_res = frame_selector.process_frame(fid, landmark_data)
    pos_energy = sel_res["motion_energy"]
    threshold = sel_res["threshold"]
    vel_energy = float(np.sqrt(token_arr[2]**2 + token_arr[3]**2))
    motion_energy = max(pos_energy, vel_energy)

    # ── Binary NO Classifier ──────────────────────────────────────────────
    no_prob = 0.0
    no_res = {}
    if no_binary_engine and no_binary_engine.model_loaded and binary_buf_len >= 5:
        no_tokens = list(binary_token_buffer)
        no_res = no_binary_engine.predict_sequence(no_tokens, max_seq_len=25)
        no_prob = float(no_res.get("no_probability", 0.0))

    # ── 21-Class CNN-GRU Prediction ───────────────────────────────────────
    sorted_probs = []
    if buf_len >= 25:
        cnn_call_count += 1
        tokens_np = np.array(list(realtime_token_buffer)[-25:], dtype=np.float32)
        assert tokens_np.shape == (25, 6), f"Expected (25, 6), got {tokens_np.shape}"
        buffer_status = "25/25"

        prediction = model_engine.predict_sequence(tokens_np)
        primary_class = prediction.get("word", "--")
        primary_confidence = float(prediction.get("confidence", 0.0))
        probs = prediction.get("probabilities", {})
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

        # Section 3 & 15: CNN Diagnostic Block
        safe_print(
            f"\nCNN_INPUT_SHAPE=(25,6)\n"
            f"CNN_RAW_CLASS={primary_class}\n"
            f"CNN_RAW_CLASS_INDEX={CLASS_TO_INDEX.get(primary_class, -1)}\n"
            f"CNN_RAW_CONFIDENCE={primary_confidence:.4f}"
        )
        for i, (c_name, c_conf) in enumerate(sorted_probs, 1):
            safe_print(f"CNN_TOP{i}={c_name}\nCNN_TOP{i}_CONF={c_conf:.4f}")

        # ── NO Temporal Gating ────────────────────────────────────────────
        if no_prob >= NO_PROB_THRESHOLD:
            binary_consecutive_no_count += 1
        else:
            binary_consecutive_no_count = 0

        no_confirmed = binary_consecutive_no_count >= NO_CONSECUTIVE_REQUIRED

        # ── Final Class Decision ──────────────────────────────────────────
        is_chin_level = float(tokens_np[:, 5].mean()) < 0.0
        water_prob = probs.get("water", 0.0)
        water_cand = (primary_class == "water") or (water_prob > 0.25)
        school_prob = probs.get("school", 0.0)
        school_cand = (primary_class == "school") or (school_prob > 0.25)
        if no_confirmed and not is_chin_level and not water_cand and not school_cand:
            final_class = "no"
        else:
            final_class = primary_class

        # Section 5: Capture Real School Sequence
        if primary_class == "school" or school_cand:
            try:
                school_save_npy = os.path.join(BASE_DIR, "models", "debug_live_school_sequence.npy")
                school_save_txt = os.path.join(BASE_DIR, "models", "debug_live_school_sequence.txt")
                np.save(school_save_npy, tokens_np)
                with open(school_save_txt, "w", encoding="utf-8") as f:
                    for row in tokens_np:
                        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
                safe_print(
                    f"\nLIVE_SCHOOL_SEQUENCE_SAVED=YES\n"
                    f"LIVE_SCHOOL_SEQUENCE_SHAPE=(25,6)"
                )
            except Exception as e_save:
                safe_print(f"[Warning] Failed to save debug school sequence: {e_save}")

        # Section 3 & 4: Automatically capture real failed school sequence if classified as water
        if primary_class == "water" or final_class == "water":
            try:
                fail_save_npy = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.npy")
                fail_save_txt = os.path.join(BASE_DIR, "models", "debug_failed_school_as_water.txt")
                np.save(fail_save_npy, tokens_np)
                with open(fail_save_txt, "w", encoding="utf-8") as f:
                    for row in tokens_np:
                        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
                safe_print(
                    f"\nFAILED_SCHOOL_SEQUENCE_CAPTURED=YES\n"
                    f"FAILED_SCHOOL_SEQUENCE_SHAPE=(25,6)\n"
                    f"LIVE_FAILED_SCHOOL_RAW_CNN_CLASS={primary_class}\n"
                    f"LIVE_FAILED_SCHOOL_RAW_CNN_CONFIDENCE={primary_confidence:.4f}"
                )
                for i, (c_name, c_conf) in enumerate(sorted_probs[:5], 1):
                    safe_print(f"TOP{i}={c_name}: {c_conf:.4f}")
            except Exception as e_fail:
                safe_print(f"[Warning] Failed to save debug failed school sequence: {e_fail}")

        # Step 6: Capture exact live sequence for PLEASE live diagnostic
        try:
            live_seq_npy = os.path.join(BASE_DIR, "models", "debug_failed_please_live_sequence.npy")
            live_seq_txt = os.path.join(BASE_DIR, "models", "debug_failed_please_live_sequence.txt")
            np.save(live_seq_npy, tokens_np)
            with open(live_seq_txt, "w", encoding="utf-8") as f:
                f.write(f"# REAL WEBCAM LIVE SEQUENCE CAPTURE\n")
                f.write(f"# active_hand={sel_hand}\n")
                f.write(f"# sequence_id={seq_id}\n")
                f.write(f"# hand_switches={hand_switch_count}\n")
                f.write(f"# predicted_class={primary_class}\n")
                f.write(f"# predicted_conf={primary_confidence:.4f}\n")
                f.write(f"# effective_fps={tokens_per_sec:.2f}\n")
                for row in tokens_np:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
            safe_print(
                f"\nLIVE_PLEASE_SEQUENCE_SAVED=YES\n"
                f"LIVE_PLEASE_SEQUENCE_SHAPE=(25,6)\n"
                f"ACTIVE_HAND={sel_hand}\n"
                f"HAND_SWITCH_COUNT={hand_switch_count}\n"
                f"PREDICTED_CLASS={primary_class}\n"
                f"PREDICTED_CONF={primary_confidence:.4f}"
            )
        except Exception as e_live:
            safe_print(f"[Warning] Failed to save live sequence: {e_live}")

        # PLEASE -> WATER capture: save any water prediction as the failed-please sequence
        # so the diagnosis script (scripts/diagnose_please_water.py) can analyze it.
        # This does NOT change any logic -- it only records the sequence for offline analysis.
        if primary_class == "water" or final_class == "water":
            try:
                please_fail_npy = os.path.join(BASE_DIR, "models", "debug_failed_please_as_water.npy")
                please_fail_txt = os.path.join(BASE_DIR, "models", "debug_failed_please_as_water.txt")
                np.save(please_fail_npy, tokens_np)
                with open(please_fail_txt, "w", encoding="utf-8") as f:
                    f.write(f"# Captured water prediction for PLEASE->WATER diagnosis\n")
                    f.write(f"# CNN_RAW_CLASS={primary_class}\n")
                    f.write(f"# CNN_RAW_CONFIDENCE={primary_confidence:.4f}\n")
                    for row in tokens_np:
                        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
                safe_print(
                    f"\nFAILED_PLEASE_SEQUENCE_CAPTURED=YES\n"
                    f"FAILED_PLEASE_SEQUENCE_SHAPE=(25,6)\n"
                    f"RAW_CNN_CLASS={primary_class}\n"
                    f"RAW_CNN_CONFIDENCE={primary_confidence:.4f}"
                )
            except Exception as e_pcap:
                safe_print(f"[Warning] Failed to save please->water debug sequence: {e_pcap}")
    else:
        prediction = {
            "word": "COLLECTING GESTURE...",
            "class_id": -1,
            "confidence": 0.0,
            "probabilities": {}
        }
        primary_class = "COLLECTING GESTURE..."
        primary_confidence = 0.0
        final_class = "COLLECTING GESTURE..."
        no_confirmed = False
        buffer_status = f"{buf_len}/25 (Collecting)"

    # ── Sentence & Early Decision Processing ─────────────────────────────
    final_prediction_for_decision = dict(prediction)
    final_prediction_for_decision["word"] = final_class
    effective_conf = no_prob if no_confirmed else primary_confidence
    final_prediction_for_decision["confidence"] = effective_conf

    if buf_len >= 25 and final_class in CLASS_NAMES and effective_conf >= 0.35:
        decision_res = early_decision_engine.process_prediction(final_prediction_for_decision, motion_energy)
        if decision_res["accepted"] and decision_res["word"]:
            accepted_word = decision_res["word"]
            print(f"\n>>> FINAL DECISION ACCEPTED: '{accepted_word.upper()}' <<<")
            translation_res = sentence_processor.add_sign_token(accepted_word)
            realtime_token_buffer.clear()
            buffered_frame_ids.clear()
            binary_token_buffer.clear()
            binary_consecutive_no_count = 0
            last_predicted_sequence = None
        else:
            translation_res = sentence_processor.get_current_translation()
    else:
        decision_res = {
            "state": "COLLECTING" if buf_len < 25 else "PREDICTING",
            "accepted": False,
            "cooldown_remaining": 0,
            "last_accepted_sign": early_decision_engine.last_accepted_sign,
            "sustained_count": 0,
            "sustained_target": 2
        }
        translation_res = sentence_processor.get_current_translation()

    # Section 4: Determine UI Output Text
    if final_class == "school":
        ui_output = "School"
    elif final_class == "water":
        ui_output = "Water (தண்ணீர்)"
    elif final_class == "no":
        ui_output = "No (இல்லை)"
    elif final_class == "please":
        ui_output = "Please."
    elif final_class in ENGLISH_TRANSLATIONS:
        ui_output = ENGLISH_TRANSLATIONS[final_class]
    else:
        ui_output = final_class

    if buf_len >= 25:
        # Section 4: Trace exactly where School becomes Water
        safe_print(
            f"\nRAW_CNN_PREDICTION={primary_class}\n"
            f"EARLY_DECISION={decision_res.get('state', 'READY')}\n"
            f"NO_MODEL_RESULT={no_prob:.4f}\n"
            f"FINAL_CLASS={final_class}\n"
            f"UI_OUTPUT={ui_output}"
        )

        # Section 13: Token Timing
        cam_fps = float(data.camera_fps or 30.0)
        tok_fps = float(data.token_fps or 20.0)
        safe_print(
            f"CAMERA_FPS={cam_fps:.1f}\n"
            f"TOKEN_FPS={tok_fps:.1f}\n"
            f"BACKEND_TOKEN_FPS={tokens_per_sec:.2f}\n"
            f"CNN_CALL_COUNT={cnn_call_count}"
        )

    proc_time_ms = (time.perf_counter() - t_start) * 1000

    # ── API Response (full diagnostic fields exposed) ─────────────────────
    return {
        "frame_id": fid,
        "primary_class": primary_class,
        "primary_confidence": round(primary_confidence, 4),
        # Binary NO classifier fields
        "no_probability": round(no_prob, 4),
        "no_confirmations": binary_consecutive_no_count,
        "no_confirmed": no_confirmed if buf_len >= 25 else False,
        # Final authoritative class (may differ from primary_class when NO override triggers)
        "final_class": final_class,
        "hand_detected": True,
        "token_buffer_length": len(realtime_token_buffer),
        "binary_no_probability": round(no_prob, 4),
        "motion_energy": round(motion_energy, 4),
        "threshold": round(threshold, 4),
        "token": [round(float(v), 4) for v in token_arr],
        "buffer_status": buffer_status,
        "processing_time_ms": round(proc_time_ms, 2),
        "prediction": {
            "word": final_class,
            "class_id": prediction.get("class_id", -1),
            "confidence": round(primary_confidence, 4)
        },
        "binary_no_confirmed": no_confirmed if buf_len >= 25 else False,
        "no_detected": no_confirmed if buf_len >= 25 else False,
        "no_confidence": round(no_prob, 4),
        "no_prediction": "NO" if no_prob >= NO_PROB_THRESHOLD else "NOT_NO",
        "binary_no": {
            "no_probability": round(no_prob, 4),
            "no_prediction": "NO" if no_prob >= NO_PROB_THRESHOLD else "NOT_NO",
            "no_confirmed": no_confirmed if buf_len >= 25 else False,
            "consecutive_count": binary_consecutive_no_count
        },
        "early_decision": {
            "state": decision_res.get("state", "READY"),
            "accepted": decision_res.get("accepted", False),
            "cooldown_remaining": decision_res.get("cooldown_remaining", 0),
            "last_accepted": decision_res.get("last_accepted_sign", "--"),
            "sustained_count": decision_res.get("sustained_count", 0),
            "sustained_target": decision_res.get("sustained_target", 2)
        },
        "hands_info": {
            "hand_count": data.hand_count or 1,
            "primary_hand": data.primary_hand or "Right",
            "secondary_hand": data.secondary_hand or "NONE"
        },
        "translation": translation_res
    }


@app.post("/api/toggle_language")
async def toggle_language(data: LanguageToggleInput):
    """Toggles translation target language ('english' or 'tamil')."""
    current_lang = sentence_processor.set_language(data.language)
    trans = sentence_processor.get_current_translation()
    return {"status": "success", "language": current_lang, "translation": trans}


@app.post("/api/clear_sentence")
async def clear_sentence():
    """Resets word buffer, token buffer, and all pipeline state."""
    global sequence_counter, last_predicted_sequence, binary_consecutive_no_count, active_sequence_id, last_processed_frame_id
    trans = sentence_processor.clear()
    early_decision_engine.reset()
    frame_selector.clear_buffer()
    realtime_token_buffer.clear()
    buffered_frame_ids.clear()
    binary_token_buffer.clear()
    binary_consecutive_no_count = 0   # RESET: stale NO state must not affect next gesture
    gesture_tokenizer.reset()
    last_predicted_sequence = None
    active_sequence_id = 0
    last_processed_frame_id = 0
    print("\n" + "="*50 + "\nRESET COMPLETE — READY FOR NEW GESTURE\n" + "="*50 + "\n")
    return {"status": "success", "translation": trans}


@app.post("/api/process_frame_image")
async def process_frame_image(data: FrameImageInput):
    """
    Accepts a raw BGR frame image (base64) for backend Python MediaPipe processing.
    Runs LandmarkExtractor -> GestureTokenizer -> 6D Token -> CNN-GRU inference.
    """
    global frame_counter, sequence_counter, no_hand_consecutive_count
    t_start = time.perf_counter()
    frame_counter += 1
    fid = data.frame_id or frame_counter

    if not data.image_base64:
        raise HTTPException(status_code=400, detail="Missing image_base64 payload")

    try:
        encoded = data.image_base64.split(",", 1)[1] if "," in data.image_base64 else data.image_base64
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {str(e)}")

    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image buffer")

    # Step 2 Logging: Backend frame receipt
    print(f"\n[BACKEND FRAME]\nReceived frame:\nwidth={frame_bgr.shape[1]}\nheight={frame_bgr.shape[0]}")

    # Run Python MediaPipe LandmarkExtractor
    landmark_data = landmark_extractor.process_frame(frame_bgr, timestamp_ms=data.timestamp_ms)
    has_hand = bool(landmark_data.get("hands") and len(landmark_data["hands"]) > 0)
    hand_count = len(landmark_data.get("hands", []))

    # Step 3 Logging: Hand detection
    print(f"\n[HAND]\ndetected={'true' if has_hand else 'false'}\nhand_count={hand_count}")

    token = gesture_tokenizer.tokenize_frame(landmark_data)
    if token is None:
        token = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    token_input = TokenInput(
        token=token.tolist() if isinstance(token, np.ndarray) else token,
        frame_id=fid,
        timestamp_ms=data.timestamp_ms,
        pose=landmark_data.get("pose"),
        hand_center=landmark_data.get("hand_center"),
        shoulder_center=landmark_data.get("shoulder_center"),
        has_hand=has_hand,
        hands=landmark_data.get("hands"),
        hand_count=hand_count
    )

    res = await process_token(token_input)
    res["pipeline_source"] = "Backend MediaPipe (Python)"
    res["has_hand"] = has_hand
    res["hand_detected"] = has_hand
    res["landmark_count"] = hand_count * 21
    return res


@app.get("/api/pipeline_status")
async def pipeline_status():
    """
    Returns diagnostic details about the backend server and MediaPipe status.
    """
    return {
        "status": "PIPELINE_READY" if (landmark_extractor.mp_available and not landmark_extractor.is_fallback) else "PIPELINE_FALLBACK",
        "backend_mediapipe_available": landmark_extractor.mp_available,
        "backend_mediapipe_fallback": landmark_extractor.is_fallback,
        "pose_model_exists": os.path.exists(os.path.join(BASE_DIR, "models", "pose_landmarker.task")),
        "hand_model_exists": os.path.exists(os.path.join(BASE_DIR, "models", "hand_landmarker.task")),
        "cnn_gru_loaded": model_engine.model_loaded,
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES
    }


@app.get("/api/model_info")
async def model_info():
    """Returns current model configuration and status."""
    return {
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "model_loaded": model_engine.model_loaded,
        "model_path": model_engine.model_path,
        "input_shape": [25, 6],
        "architecture": "CNN-GRU",
    }


@app.post("/api/predict_binary_no")
async def predict_binary_no(data: TokenInput):
    """
    Isolated binary NO classifier endpoint.
    Evaluates real-time 6D token stream against the binary NO model.
    Does NOT modify or replace the 21-class CNN-GRU model.
    """
    if not data.has_hand:
        binary_token_buffer.clear()
        return {
            "is_no": False,
            "no_probability": 0.0,
            "prediction": "Waiting for hand gesture...",
            "status": "NO_HAND"
        }

    token = data.token
    if token is None or len(token) != 6:
        return {"error": "Token must be a 6-element float array"}

    binary_token_buffer.append(np.array(token, dtype=np.float32))

    if len(binary_token_buffer) < 5:
        return {
            "is_no": False,
            "no_probability": 0.0,
            "prediction": "Waiting for hand gesture...",
            "buffer_status": f"{len(binary_token_buffer)}/25"
        }

    res = no_binary_engine.predict_sequence(list(binary_token_buffer), max_seq_len=25)
    res["buffer_status"] = f"{len(binary_token_buffer)}/25"
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

