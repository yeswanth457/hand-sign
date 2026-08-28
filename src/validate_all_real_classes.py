"""
Pipeline Validation Script for All Real ISL Classes.
Runs extracted 6D token sequences for all real sign classes (food, hello, school, thank_you, water, welcome)
through the exact same UNIFIED model (models/isl_cnn_gru.pt) + EarlyDecisionEngine + LocalSentenceProcessor.
"""

import os
import sys
import glob
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ISL_VOCABULARY
from src.cnn_gru_model import CNNGRUInferenceEngine, CNN_GRU_MODEL_PATH
from src.early_decision import EarlyDecisionEngine
from src.sentence_processor import LocalSentenceProcessor


def validate_all_classes():
    print("\n==================================================")
    print("   END-TO-END PIPELINE VALIDATION: ALL REAL CLASSES")
    print("==================================================")

    # 1. Initialize Single Unified Model Engine & Processing Pipeline
    if not os.path.exists(CNN_GRU_MODEL_PATH):
        print(f"[Error] Model checkpoint not found at {CNN_GRU_MODEL_PATH}")
        return

    model_engine = CNNGRUInferenceEngine(model_path=CNN_GRU_MODEL_PATH)
    early_decision = EarlyDecisionEngine(confidence_threshold=0.65, sustained_frames=2)
    sentence_processor = LocalSentenceProcessor()

    token_files = glob.glob("dataset/tokens/**/*.npz", recursive=True)
    token_files.sort()

    print(f"Loaded Unified Model: {CNN_GRU_MODEL_PATH}")
    print(f"Found {len(token_files)} real token sequence files across dataset/tokens/\n")

    results = []

    for tk_file in token_files:
        data = np.load(tk_file)
        sign_class = str(data["sign_class"])
        signer_id = str(data["signer_id"])
        video_id = str(data["video_id"])
        raw_tokens = data["tokens"].astype(np.float32)

        # Pad or resample sequence to (25, 6)
        n, c = raw_tokens.shape
        if n >= 25:
            indices = np.linspace(0, n - 1, 25, dtype=int)
            seq_tokens = raw_tokens[indices]
        else:
            seq_tokens = np.pad(raw_tokens, ((0, 25 - n), (0, 0)), mode="edge")

        # Pass 25-frame sequence through single unified CNN-GRU model
        pred = model_engine.predict_sequence(seq_tokens)
        predicted_word = pred["word"]
        confidence_pct = pred["confidence"] * 100.0

        # Simulate continuous frame stream for EarlyDecisionEngine state machine
        motion_energy = 0.05
        early_decision.reset()

        is_accepted = False
        dec_res = None
        # Step through frames until decision accepts or reaches max sequence frames
        for f_idx in range(5):
            dec_res = early_decision.process_prediction(pred, motion_energy)
            if dec_res["accepted"]:
                is_accepted = True
                break

        if is_accepted and dec_res["word"]:
            translation = sentence_processor.add_sign_token(dec_res["word"])
        else:
            translation = sentence_processor.get_current_translation()

        res_entry = {
            "video_id": video_id,
            "target_class": sign_class,
            "signer_id": signer_id,
            "pred_word": predicted_word,
            "confidence_pct": round(confidence_pct, 2),
            "decision_accepted": is_accepted,
            "english_text": translation["english"],
            "tamil_text": translation["tamil"]
        }
        results.append(res_entry)

        print(f"Video: {video_id:<22} | Target: {sign_class:<10} | Signer: {signer_id:<9} | Model Pred: {predicted_word:<10} ({confidence_pct:.1f}%) | Accepted: {str(is_accepted):<5}")

    # Summary Table
    print("\n==================================================")
    print("           VALIDATION SUMMARY TABLE")
    print("==================================================")
    print(f"{'Target Sign':<12} | {'Signer':<9} | {'Model Prediction':<16} | {'Confidence':<10} | {'Decision'}")
    print("-" * 65)
    for r in results:
        dec_str = "ACCEPTED" if r["decision_accepted"] else "REJECTED (<65%)"
        print(f"{r['target_class']:<12} | {r['signer_id']:<9} | {r['pred_word']:<16} | {r['confidence_pct']:>6.1f}%    | {dec_str}")

    print("\nFinal Sentence Processing Buffer:")
    current_trans = sentence_processor.get_current_translation()
    print(f"  - English Text : \"{current_trans['english']}\"")
    print(f"  - Tamil Text   : \"{current_trans['tamil']}\"")
    print("==================================================\n")


if __name__ == "__main__":
    validate_all_classes()
