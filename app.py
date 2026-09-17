"""
Phase 12: FastAPI Web Server and Pipeline API.
Provides endpoints for landmark processing, real-time prediction, translation,
dataset generation, model training, and bilingual output with Web UI.
"""

import os
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
    ISL_VOCABULARY, TAMIL_VOCAB_MAP, BASE_DIR,
    CONFIDENCE_THRESHOLD, SUSTAINED_FRAMES, COOLDOWN_FRAMES
)
from src.dataset_manager import ISLDatasetManager
from src.landmark_extractor import LandmarkExtractor
from src.motion_energy import MotionEnergyCalculator
from src.frame_selector import MotionFrameSelector
from src.gesture_tokenizer import GestureTokenizer
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.temporal_transformer import ModelInferenceEngine, train_model
from src.early_decision import EarlyDecisionEngine
from src.sentence_processor import LocalSentenceProcessor

app = FastAPI(
    title="RT-STAMP-SLR ISL Translator",
    description="Real-Time Indian Sign Language to English/Tamil Translator based on RT-STAMP-SLR Architecture",
    version="1.0.0"
)

# Static files directory for web UI
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Pipeline Components Initialization
dataset_manager = ISLDatasetManager()
landmark_extractor = LandmarkExtractor()
frame_selector = MotionFrameSelector()
gesture_tokenizer = GestureTokenizer()
model_engine = CNNGRUInferenceEngine()
early_decision_engine = EarlyDecisionEngine()
sentence_processor = LocalSentenceProcessor(current_language="english")

# Real-time rolling 25-frame gesture token buffer
realtime_token_buffer = deque(maxlen=25)
frame_counter = 0
no_hand_consecutive_count = 0  # Missed-frame tolerance counter
NO_HAND_CLEAR_THRESHOLD = 3   # Require 3 consecutive no-hand frames before clearing buffer


# Pydantic Schemas
class LandmarkInput(BaseModel):
    pose: Optional[Dict[str, Any]] = None
    hands: Optional[List[Any]] = None
    hand_center: Optional[List[float]] = None
    shoulder_center: Optional[List[float]] = None
    frame_id: Optional[int] = 0


class FrameImageInput(BaseModel):
    image_base64: str
    frame_id: Optional[int] = 0
    timestamp_ms: Optional[float] = None


class LanguageToggleInput(BaseModel):
    language: str  # "english" or "tamil"


class SimulateSignInput(BaseModel):
    word: str


class TokenInput(BaseModel):
    """Input for browser-side MediaPipe: pre-computed 6D token + optional pose data for motion energy."""
    token: Optional[List[float]] = None
    frame_id: Optional[int] = 0
    timestamp_ms: Optional[float] = None
    pose: Optional[Dict[str, Any]] = None
    hand_center: Optional[List[float]] = None
    shoulder_center: Optional[List[float]] = None
    has_hand: Optional[bool] = True


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RT-STAMP-SLR ISL Translator API is running. UI loading...</h1>"


@app.get("/api/vocabulary")
async def get_vocabulary():
    """Returns all 50 ISL vocabulary signs with English and Tamil mappings."""
    vocab_details = []
    for idx, word in enumerate(ISL_VOCABULARY):
        vocab_details.append({
            "id": idx,
            "english": word,
            "display_english": word.replace("_", " ").title(),
            "tamil": TAMIL_VOCAB_MAP.get(word, word)
        })
    return {"vocabulary": vocab_details, "total": len(vocab_details)}


@app.post("/api/process_landmarks")
async def process_landmarks(data: LandmarkInput):
    """
    Core real-time frame processing endpoint.
    Executes Motion Energy -> Frame Selection -> Gesture Tokenizer -> 
    Temporal Transformer Model -> Early Decision -> Sentence Processor.
    """
    global frame_counter
    frame_counter += 1
    fid = data.frame_id or frame_counter

    # 1. Landmark Extraction / Formatting
    if data.pose:
        landmark_data = landmark_extractor.process_raw_landmarks(data.pose, data.hands)
    else:
        # Generate default frame landmark data if empty
        landmark_data = landmark_extractor.process_raw_landmarks({})

    # Override hand/shoulder centers if provided directly from frontend
    if data.hand_center and len(data.hand_center) == 2:
        landmark_data["hand_center"] = (data.hand_center[0], data.hand_center[1])
    if data.shoulder_center and len(data.shoulder_center) == 2:
        landmark_data["shoulder_center"] = (data.shoulder_center[0], data.shoulder_center[1])

    # 2. Motion Energy & Frame Selection
    sel_res = frame_selector.process_frame(fid, landmark_data)
    is_selected = sel_res["is_selected"]
    motion_energy = sel_res["motion_energy"]
    threshold = sel_res["threshold"]

    # 3. Gesture Tokenization
    token = gesture_tokenizer.tokenize_frame(landmark_data)
    
    # Selected sequence token buffer
    selected_landmarks = frame_selector.get_selected_sequence()
    seq_tokens = gesture_tokenizer.tokenize_sequence(selected_landmarks)

    # 4. Temporal Memory Transformer Prediction
    prediction = model_engine.predict_sequence(seq_tokens if len(seq_tokens) > 0 else [token])

    # 5. Early Decision & Cooldown State Machine
    decision_res = early_decision_engine.process_prediction(prediction, motion_energy)

    # 6. Sentence Processing on Positive Sign Acceptance
    translation_res = sentence_processor.get_current_translation()
    if decision_res["accepted"] and decision_res["word"]:
        translation_res = sentence_processor.add_sign_token(decision_res["word"])
        frame_selector.clear_buffer()

    return {
        "frame_id": fid,
        "is_selected": is_selected,
        "motion_energy": round(motion_energy, 4),
        "threshold": round(threshold, 4),
        "token": [round(float(v), 4) for v in token],
        "selected_buffer_size": len(selected_landmarks),
        "prediction": {
            "word": prediction["word"],
            "class_id": prediction["class_id"],
            "confidence": round(prediction["confidence"], 4)
        },
        "early_decision": {
            "state": decision_res["state"],
            "accepted": decision_res["accepted"],
            "cooldown_remaining": decision_res["cooldown_remaining"],
            "last_accepted": decision_res["last_accepted_sign"]
        },
        "translation": translation_res
    }


@app.post("/api/process_frame_image")
async def process_frame_image(data: FrameImageInput):
    """
    Processes live webcam image frame directly using native MediaPipe 1.0.0 Tasks landmarker
    in low-latency VIDEO tracking mode.
    """
    global frame_counter
    t_start = time.perf_counter()
    frame_counter += 1
    fid = data.frame_id or frame_counter

    # Decode base64 JPEG image
    try:
        encoded = data.image_base64.split(",", 1)[1] if "," in data.image_base64 else data.image_base64
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        frame_bgr = None

    if frame_bgr is not None:
        landmark_data = landmark_extractor.process_frame(frame_bgr, timestamp_ms=data.timestamp_ms)
    else:
        landmark_data = landmark_extractor.process_raw_landmarks({})

    proc_time_ms = (time.perf_counter() - t_start) * 1000

    # Motion Energy & Frame Selection
    sel_res = frame_selector.process_frame(fid, landmark_data)
    is_selected = sel_res["is_selected"]
    motion_energy = sel_res["motion_energy"]
    threshold = sel_res["threshold"]

    # Tokenization & 25-Frame Rolling Buffer
    token = gesture_tokenizer.tokenize_frame(landmark_data)
    if token is not None and len(token) == 6:
        realtime_token_buffer.append(token)

    # Evaluate PyTorch CNN-GRU Model on Rolling Sequence (min 10 frames)
    if len(realtime_token_buffer) >= 10:
        seq_tokens = list(realtime_token_buffer)
        prediction = model_engine.predict_sequence(seq_tokens)
        buffer_status = f"{len(realtime_token_buffer)}/25 (Ready)"
    else:
        prediction = {
            "word": "BUFFERING",
            "class_id": -1,
            "confidence": 0.0,
            "probabilities": {}
        }
        buffer_status = f"{len(realtime_token_buffer)}/25"

    # Early Decision & Sentence Processor
    decision_res = early_decision_engine.process_prediction(prediction, motion_energy)
    translation_res = sentence_processor.get_current_translation()

    if decision_res["accepted"] and decision_res["word"]:
        translation_res = sentence_processor.add_sign_token(decision_res["word"])
        realtime_token_buffer.clear()

    return {
        "frame_id": fid,
        "is_selected": is_selected,
        "motion_energy": round(motion_energy, 4),
        "threshold": round(threshold, 4),
        "token": [round(float(v), 4) for v in token],
        "buffer_status": buffer_status,
        "processing_time_ms": round(proc_time_ms, 2),
        "landmark_data": landmark_data,
        "prediction": {
            "word": prediction["word"],
            "class_id": prediction["class_id"],
            "confidence": round(prediction["confidence"], 4)
        },
        "early_decision": {
            "state": decision_res["state"],
            "accepted": decision_res["accepted"],
            "cooldown_remaining": decision_res["cooldown_remaining"],
            "last_accepted": decision_res["last_accepted_sign"]
        },
        "translation": translation_res
    }


@app.post("/api/process_token")
async def process_token(data: TokenInput):
    """
    Accepts a pre-computed 6D gesture token from browser-side MediaPipe.
    Feeds into the 25-frame rolling buffer and runs CNN-GRU inference.
    """
    global frame_counter
    t_start = time.perf_counter()
    frame_counter += 1
    fid = data.frame_id or frame_counter

    # Check if hand is missing / no-hand reset with missed-frame tolerance
    global no_hand_consecutive_count
    if not data.has_hand:
        no_hand_consecutive_count += 1
        if no_hand_consecutive_count >= NO_HAND_CLEAR_THRESHOLD:
            # Only clear buffer after 3 consecutive no-hand frames (prevents single-frame flicker)
            realtime_token_buffer.clear()
            early_decision_engine.reset()
            no_hand_consecutive_count = 0
        proc_time_ms = (time.perf_counter() - t_start) * 1000
        return {
            "frame_id": fid,
            "motion_energy": 0.0,
            "threshold": 0.015,
            "token": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "buffer_status": f"{len(realtime_token_buffer)}/25",
            "processing_time_ms": round(proc_time_ms, 2),
            "prediction": {
                "word": "--",
                "class_id": -1,
                "confidence": 0.0
            },
            "early_decision": {
                "state": "NO_HAND",
                "accepted": False,
                "cooldown_remaining": 0,
                "last_accepted": early_decision_engine.last_accepted_sign or "--",
                "sustained_count": 0,
                "sustained_target": 3
            },
            "translation": sentence_processor.get_current_translation()
        }

    token = data.token
    if token is None or len(token) != 6:
        return {"error": "Token must be a 6-element float array"}

    token_arr = np.array(token, dtype=np.float32)

    # Build minimal landmark_data for motion energy calculation
    hand_c = tuple(data.hand_center) if data.hand_center else (token[0], token[1])
    pose_dict = data.pose or {}
    if not pose_dict:
        # Fallback to hand center for motion energy when pose is bypassed
        pose_dict = {"HAND": (hand_c[0], hand_c[1], 0.0)}

    landmark_data = {
        "pose": pose_dict,
        "hands": [],
        "hand_center": hand_c,
        "shoulder_center": tuple(data.shoulder_center) if data.shoulder_center else (0.5, 0.35),
    }

    # Motion Energy & Frame Selection
    sel_res = frame_selector.process_frame(fid, landmark_data)
    motion_energy = sel_res["motion_energy"]
    threshold = sel_res["threshold"]
    is_selected = sel_res.get("is_selected", False)

    # Add token to 25-frame rolling buffer only when active motion is detected (matches dataset preprocessing)
    no_hand_consecutive_count = 0
    if is_selected or motion_energy > 0.015:
        realtime_token_buffer.append(token_arr)

    # CNN-GRU inference when buffer has min 10 frames (padded to 25)
    if len(realtime_token_buffer) >= 10:
        seq_tokens = list(realtime_token_buffer)
        prediction = model_engine.predict_sequence(seq_tokens)
        buffer_status = f"{len(realtime_token_buffer)}/25 (Ready)"
    else:
        prediction = {
            "word": "BUFFERING",
            "class_id": -1,
            "confidence": 0.0,
            "probabilities": {}
        }
        buffer_status = f"{len(realtime_token_buffer)}/25"

    # Early Decision & Sentence Processor
    decision_res = early_decision_engine.process_prediction(prediction, motion_energy)

    # Add gesture token to sentence builder ONLY when early decision engine accepts
    if decision_res["accepted"] and decision_res["word"]:
        translation_res = sentence_processor.add_sign_token(decision_res["word"])
    else:
        translation_res = sentence_processor.get_current_translation()

    proc_time_ms = (time.perf_counter() - t_start) * 1000

    # Detailed Stage Debug Logging (Stages 1-15)
    if fid % 15 == 0 or decision_res["accepted"]:
        print(
            f"\n[PIPELINE RECOGNITION DEBUG]\n"
            f"Stage 1-2 (Frame & Landmark): Frame ID={fid} | Hand Detected=YES (21 LMs)\n"
            f"Stage 7 (6D Token): {np.round(token_arr, 4).tolist()}\n"
            f"Stage 6 (Temporal Buffer): Size={len(realtime_token_buffer)}/25 | Status={buffer_status}\n"
            f"Stage 8-9 (Tensor Shape): (1, 25, 6)\n"
            f"Stage 10-12 (Inference & Softmax): Word='{prediction['word']}' (ID={prediction['class_id']}) | Conf={prediction['confidence']*100:.1f}%\n"
            f"Stage 13 (Gesture Locking): State={decision_res['state']} | Accepted={decision_res['accepted']} | Accepted Word={decision_res['word']}\n"
            f"Stage 14-15 (Sentence Translation): '{translation_res.get('display_text', '')}'"
        )

    return {
        "frame_id": fid,
        "motion_energy": round(motion_energy, 4),
        "threshold": round(threshold, 4),
        "token": [round(float(v), 4) for v in token_arr],
        "buffer_status": buffer_status,
        "processing_time_ms": round(proc_time_ms, 2),
        "prediction": {
            "word": prediction["word"],
            "class_id": prediction["class_id"],
            "confidence": round(prediction["confidence"], 4)
        },
        "early_decision": {
            "state": decision_res["state"],
            "accepted": decision_res["accepted"],
            "cooldown_remaining": decision_res["cooldown_remaining"],
            "last_accepted": decision_res["last_accepted_sign"],
            "sustained_count": decision_res.get("sustained_count", 0),
            "sustained_target": decision_res.get("sustained_target", 3)
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
    trans = sentence_processor.clear()
    early_decision_engine.reset()
    frame_selector.clear_buffer()
    realtime_token_buffer.clear()
    gesture_tokenizer.reset()
    return {"status": "success", "translation": trans}


@app.post("/api/simulate_sign")
async def simulate_sign(data: SimulateSignInput):
    """
    Simulates a dynamic gesture sequence for a specific ISL word.
    Useful for interactive UI demo without physical webcam signing.
    """
    word = data.word.lower().strip()
    if word not in ISL_VOCABULARY:
        raise HTTPException(status_code=400, detail=f"Word '{word}' not in ISL vocabulary.")

    class_id = ISL_VOCABULARY.index(word)
    
    # Generate synthetic gesture sequence (15 frames)
    t = np.linspace(0, np.pi, 15)
    hx_seq = 0.5 + 0.2 * np.sin((1 + class_id % 5) * t)
    hy_seq = 0.5 + 0.2 * np.cos((1 + class_id % 5) * t)

    last_res = None
    for i in range(len(t)):
        hx, hy = float(hx_seq[i]), float(hy_seq[i])
        pose = {
            "LW": (hx - 0.05, hy + 0.1, 0.0),
            "RW": (hx, hy, 0.0),
            "LE": (0.35, 0.5, 0.0),
            "RE": (0.65, 0.5, 0.0),
            "LS": (0.4, 0.35, 0.0),
            "RS": (0.6, 0.35, 0.0),
        }
        l_input = LandmarkInput(
            pose=pose,
            hand_center=[hx, hy],
            shoulder_center=[0.5, 0.35],
            frame_id=i + 1
        )
        # Directly inject true prediction during simulation
        if i == len(t) - 1:
            early_decision_engine.current_state = "IDLE"
            early_decision_engine.prediction_window.clear()
            for _ in range(SUSTAINED_FRAMES):
                early_decision_engine.prediction_window.append((word, 0.92))
        
        last_res = await process_landmarks(l_input)

    return {
        "simulated_word": word,
        "class_id": class_id,
        "last_frame_result": last_res
    }


@app.post("/api/train")
async def train():
    """Triggers dataset synthetic generation and model training."""
    stats = dataset_manager.generate_synthetic_dataset(samples_per_class=30, seq_length=25)
    train_x, train_y = dataset_manager.load_dataset("train")
    val_x, val_y = dataset_manager.load_dataset("val")

    train_res = train_model(train_x, train_y, val_x, val_y, epochs=15)
    
    # Reload model weights
    global model_engine
    model_engine = CNNGRUInferenceEngine()

    return {
        "dataset_stats": stats,
        "training_result": train_res
    }


@app.get("/api/dataset_browser")
async def dataset_browser(offset: int = 0, length: int = 5):
    """
    Proxy endpoint for the HuggingFace Datasets Server API.
    Fetches rows from the vidit031/isl-isolated-40words dataset.
    Acts as a CORS-safe bridge between the browser frontend and HuggingFace.
    """
    import httpx

    if length < 1:
        length = 1
    if length > 100:
        length = 100
    if offset < 0:
        offset = 0

    hf_url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=vidit031%2Fisl-isolated-40words"
        "&config=default&split=train"
        f"&offset={offset}&length={length}"
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(hf_url)
            resp.raise_for_status()
            raw = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"HuggingFace API error: {e.response.text[:300]}")
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException):
        raise HTTPException(status_code=504, detail="HuggingFace API request timed out (5s limit). Please check your internet connection or try again later.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach HuggingFace API: {str(e)}")

    # Extract and clean rows for the frontend
    rows = []
    for item in raw.get("rows", []):
        r = item.get("row", {})
        rows.append({
            "row_idx": item.get("row_idx", 0),
            "word": r.get("word", ""),
            "normalized_word": r.get("normalized_word", ""),
            "dataset": r.get("dataset", ""),
            "original_label": r.get("original_label", ""),
            "video_path": r.get("video_path", ""),
            "original_filename": r.get("original_filename", ""),
            "signer": r.get("signer", ""),
            "fps": r.get("fps"),
            "resolution": r.get("resolution", ""),
            "duration": r.get("duration"),
            "license": r.get("license", ""),
            "quality_score": r.get("quality_score"),
            "duplicate_status": r.get("duplicate_status", ""),
            "review_status": r.get("review_status", ""),
            "repository": r.get("repository", ""),
            "download_url": r.get("download_url", ""),
        })

    return {
        "rows": rows,
        "total": raw.get("num_rows_total", 0),
        "offset": offset,
        "length": length,
        "has_more": (offset + length) < raw.get("num_rows_total", 0),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
