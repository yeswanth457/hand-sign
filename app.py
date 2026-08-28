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

    # Evaluate PyTorch CNN-GRU Model on 25-Frame Rolling Sequence
    if len(realtime_token_buffer) == 25:
        seq_tokens = list(realtime_token_buffer)
        prediction = model_engine.predict_sequence(seq_tokens)
        buffer_status = "25/25 (Ready)"
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


@app.post("/api/toggle_language")
async def toggle_language(data: LanguageToggleInput):
    """Toggles translation target language ('english' or 'tamil')."""
    current_lang = sentence_processor.set_language(data.language)
    trans = sentence_processor.get_current_translation()
    return {"status": "success", "language": current_lang, "translation": trans}


@app.post("/api/clear_sentence")
async def clear_sentence():
    """Resets word buffer and clear current sentence."""
    trans = sentence_processor.clear()
    early_decision_engine.reset()
    frame_selector.clear_buffer()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
