"""
Unit and integration tests for Phase 13 Real ISL Dataset Pipeline.
Tests video discovery, signer extraction, landmark/token creation, invalid file handling, and CSV metadata generation.
"""

import sys
import os
import tempfile
import shutil
import csv
from pathlib import Path
import numpy as np
import cv2

# Add project root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOKEN_DIM
from src.real_dataset_pipeline import RealISLDatasetPipeline


def create_dummy_video(file_path, width=640, height=480, fps=30, num_frames=30):
    """Generates a dummy synthetic .mp4 video file for pipeline testing."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw moving circle to simulate motion
        cx = int(width / 2 + 150 * np.sin(i * 0.4))
        cy = int(height / 2 + 100 * np.cos(i * 0.4))
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        writer.write(frame)

    writer.release()


def test_real_dataset_pipeline_discovery():
    print("[Test Phase 13] 1. Testing Video File Discovery & Signer Extraction...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir) / "raw"
        landmarks_dir = Path(tmp_dir) / "landmarks"
        tokens_dir = Path(tmp_dir) / "tokens"
        metadata_dir = Path(tmp_dir) / "metadata"
        csv_path = metadata_dir / "dataset.csv"

        # Create test video structure
        v1 = raw_dir / "hello" / "signer_01" / "test_vid_01.mp4"
        v2 = raw_dir / "water" / "signer_02" / "test_vid_02.avi"
        create_dummy_video(v1, num_frames=20)
        create_dummy_video(v2, num_frames=20)

        pipeline = RealISLDatasetPipeline(
            raw_dir=raw_dir,
            landmarks_dir=landmarks_dir,
            tokens_dir=tokens_dir,
            metadata_dir=metadata_dir,
            csv_path=csv_path
        )

        entries = pipeline.discover_video_files()
        assert len(entries) == 2, f"Expected 2 discovered videos, got {len(entries)}"
        
        classes = sorted([e["sign_class"] for e in entries])
        signers = sorted([e["signer_id"] for e in entries])
        assert classes == ["hello", "water"]
        assert signers == ["signer_01", "signer_02"]
        print("[PASSED] Video Discovery & Signer Extraction")


def test_real_dataset_pipeline_processing():
    print("[Test Phase 13] 2. Testing Video Processing, Landmarks & Token Output Shape...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir) / "raw"
        landmarks_dir = Path(tmp_dir) / "landmarks"
        tokens_dir = Path(tmp_dir) / "tokens"
        metadata_dir = Path(tmp_dir) / "metadata"
        csv_path = metadata_dir / "dataset.csv"

        v1 = raw_dir / "hello" / "signer_01" / "vid_01.mp4"
        create_dummy_video(v1, num_frames=25)

        pipeline = RealISLDatasetPipeline(
            raw_dir=raw_dir,
            landmarks_dir=landmarks_dir,
            tokens_dir=tokens_dir,
            metadata_dir=metadata_dir,
            csv_path=csv_path,
            min_tokens=2
        )

        summary = pipeline.process_dataset()
        assert summary["videos_discovered"] == 1
        assert summary["successful"] == 1
        assert summary["token_dim"] == TOKEN_DIM

        # Verify Metadata CSV
        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            row = reader[0]
            assert row["sign_class"] == "hello"
            assert row["signer_id"] == "signer_01"
            assert row["status"] == "success"
            assert int(row["token_sequence_length"]) >= 2

        # Verify Saved .npz Tokens
        tk_file = Path(row["token_file"])
        assert tk_file.exists()
        with np.load(tk_file) as tk_data:
            tokens = tk_data["tokens"]
            assert tokens.ndim == 2
            assert tokens.shape[1] == 6
            assert np.isfinite(tokens).all()
        print("[PASSED] Processing, Token Dimensions & CSV Metadata")


def test_real_dataset_pipeline_corrupt_file_handling():
    print("[Test Phase 13] 3. Testing Corrupt/Invalid Video Handling...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir) / "raw"
        landmarks_dir = Path(tmp_dir) / "landmarks"
        tokens_dir = Path(tmp_dir) / "tokens"
        metadata_dir = Path(tmp_dir) / "metadata"
        csv_path = metadata_dir / "dataset.csv"

        # Create fake corrupted video file (invalid text file with .mp4 extension)
        v_bad = raw_dir / "hello" / "signer_01" / "bad_video.mp4"
        v_bad.parent.mkdir(parents=True, exist_ok=True)
        with open(v_bad, "w") as f:
            f.write("Not a real video file content")

        pipeline = RealISLDatasetPipeline(
            raw_dir=raw_dir,
            landmarks_dir=landmarks_dir,
            tokens_dir=tokens_dir,
            metadata_dir=metadata_dir,
            csv_path=csv_path
        )

        summary = pipeline.process_dataset()
        assert summary["videos_discovered"] == 1
        assert summary["failed"] == 1
        assert summary["successful"] == 0

        # Verify CSV recorded failure status cleanly
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            assert reader[0]["status"] == "failed"
            assert len(reader[0]["error"]) > 0

        print("[PASSED] Corrupt Video Handling")


def run_all_phase_13_tests():
    print("==================================================")
    print(" Running Phase 13 Real ISL Dataset Test Suite")
    print("==================================================")
    test_real_dataset_pipeline_discovery()
    test_real_dataset_pipeline_processing()
    test_real_dataset_pipeline_corrupt_file_handling()
    print("==================================================")
    print(" ALL PHASE 13 TESTS READY & PASSED FIXTURE VERIFICATION!")
    print("==================================================")


if __name__ == "__main__":
    run_all_phase_13_tests()
