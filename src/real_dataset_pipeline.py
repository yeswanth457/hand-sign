"""
Phase 13: Real ISL Dataset Processing Pipeline.
Ingests real ISL video files, extracts MediaPipe pose/hand landmarks, calculates
motion energy with adaptive thresholding, selects motion frames, generates 6D gesture tokens,
and records metadata with signer identity preservation.
"""

import os
import csv
import logging
from pathlib import Path
import cv2
import numpy as np

from config import (
    RAW_DATA_DIR, LANDMARKS_DIR, TOKENS_DIR, METADATA_DIR, METADATA_CSV_PATH,
    ISL_VOCABULARY, TOKEN_DIM, FRAME_WIDTH, FRAME_HEIGHT
)
from src.landmark_extractor import LandmarkExtractor
from src.frame_selector import MotionFrameSelector
from src.gesture_tokenizer import GestureTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealISLPipeline")

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class RealISLDatasetPipeline:
    def __init__(
        self,
        raw_dir=RAW_DATA_DIR,
        landmarks_dir=LANDMARKS_DIR,
        tokens_dir=TOKENS_DIR,
        metadata_dir=METADATA_DIR,
        csv_path=METADATA_CSV_PATH,
        min_tokens=3
    ):
        self.raw_dir = Path(raw_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.tokens_dir = Path(tokens_dir)
        self.metadata_dir = Path(metadata_dir)
        self.csv_path = Path(csv_path)
        self.min_tokens = min_tokens

        self.ensure_directories()
        self.landmark_extractor = LandmarkExtractor()

    def ensure_directories(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def discover_video_files(self):
        """
        Discovers all video files in raw_dir preserving structure:
        raw_dir / <sign_class> / <signer_id> / <video_filename>
        or raw_dir / <sign_class> / <video_filename> (defaulting signer_id to 'signer_01').
        """
        video_entries = []
        if not self.raw_dir.exists():
            return video_entries

        for ext in SUPPORTED_EXTENSIONS:
            for video_path in self.raw_dir.glob(f"**/*{ext}"):
                rel_path = video_path.relative_to(self.raw_dir)
                parts = rel_path.parts

                if len(parts) >= 3:
                    sign_class = parts[0]
                    signer_id = parts[1]
                    video_id = video_path.stem
                elif len(parts) == 2:
                    sign_class = parts[0]
                    signer_id = "signer_01"
                    video_id = video_path.stem
                else:
                    sign_class = "unknown"
                    signer_id = "signer_01"
                    video_id = video_path.stem

                video_entries.append({
                    "video_path": str(video_path),
                    "sign_class": sign_class,
                    "signer_id": signer_id,
                    "video_id": video_id
                })

        video_entries.sort(key=lambda x: (x["sign_class"], x["signer_id"], x["video_id"]))
        return video_entries

    def process_video_file(self, video_entry):
        """
        Processes a single real ISL video:
        1. Reads video frames sequentially with OpenCV.
        2. Extracts MediaPipe pose & hand landmarks.
        3. Computes Motion Energy & applies adaptive frame selection.
        4. Generates 6D gesture tokens [Hx, Hy, Mx, My, Rx, Ry].
        5. Validates and saves landmark (.npz) and token (.npz) files.
        """
        video_path = video_entry["video_path"]
        sign_class = video_entry["sign_class"]
        signer_id = video_entry["signer_id"]
        video_id = video_entry["video_id"]

        result = {
            "video_path": video_path,
            "sign_class": sign_class,
            "signer_id": signer_id,
            "video_id": video_id,
            "fps": 0.0,
            "total_frames": 0,
            "valid_landmark_frames": 0,
            "selected_frame_count": 0,
            "selection_ratio": 0.0,
            "token_sequence_length": 0,
            "landmark_file": "",
            "token_file": "",
            "status": "failed",
            "error": ""
        }

        if not os.path.exists(video_path):
            result["error"] = "Video file does not exist."
            return result

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = "Corrupt or unreadable video file."
            return result

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        result["fps"] = round(float(fps), 2)

        if hasattr(self.landmark_extractor, "reset"):
            self.landmark_extractor.reset()

        frame_selector = MotionFrameSelector()
        gesture_tokenizer = GestureTokenizer()

        all_landmarks = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1

            try:
                landmark_data = self.landmark_extractor.process_frame(frame)
                all_landmarks.append(landmark_data)
                frame_selector.process_frame(frame_idx, landmark_data)
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx} of {video_path}: {e}")

        cap.release()

        result["total_frames"] = frame_idx
        result["valid_landmark_frames"] = len(all_landmarks)

        if frame_idx == 0:
            result["error"] = "Zero frames read from video."
            return result

        # Selected landmarks sequence and gesture tokens
        selected_landmarks = frame_selector.get_selected_sequence()
        tokens = gesture_tokenizer.tokenize_sequence(selected_landmarks)

        result["selected_frame_count"] = len(selected_landmarks)
        result["selection_ratio"] = round(len(selected_landmarks) / max(1, frame_idx), 4)
        result["token_sequence_length"] = len(tokens)

        # Minimum Token Validation
        if tokens.ndim != 2 or tokens.shape[1] != TOKEN_DIM:
            result["error"] = f"Invalid token shape {tokens.shape}. Expected (N, 6)."
            return result

        if not np.isfinite(tokens).all():
            result["error"] = "Tokens contain non-finite values (NaN/Inf)."
            return result

        if len(tokens) < self.min_tokens:
            result["error"] = f"Too few selected motion tokens ({len(tokens)} < min {self.min_tokens})."
            return result

        # Save Landmark Sequence (.npz)
        lm_out_dir = self.landmarks_dir / sign_class / signer_id
        lm_out_dir.mkdir(parents=True, exist_ok=True)
        lm_file = lm_out_dir / f"{video_id}.npz"
        np.savez_compressed(
            lm_file,
            video_path=video_path,
            sign_class=sign_class,
            signer_id=signer_id,
            video_id=video_id,
            total_frames=frame_idx,
            selected_frames=len(selected_landmarks),
            landmarks=np.array(all_landmarks, dtype=object)
        )
        result["landmark_file"] = str(lm_file)

        # Save 6D Gesture Tokens (.npz)
        tk_out_dir = self.tokens_dir / sign_class / signer_id
        tk_out_dir.mkdir(parents=True, exist_ok=True)
        tk_file = tk_out_dir / f"{video_id}.npz"
        np.savez_compressed(
            tk_file,
            tokens=tokens.astype(np.float32),
            sign_class=sign_class,
            signer_id=signer_id,
            video_id=video_id
        )
        result["token_file"] = str(tk_file)

        result["status"] = "success"
        result["error"] = ""
        return result

    def process_dataset(self):
        """
        Processes all discovered video files in raw_dir and updates master metadata CSV.
        Returns dataset summary metrics dict.
        """
        video_entries = self.discover_video_files()
        results = []

        logger.info(f"Discovered {len(video_entries)} video files in {self.raw_dir}")

        for entry in video_entries:
            res = self.process_video_file(entry)
            results.append(res)

        self.save_metadata_csv(results)
        summary = self.generate_summary(results)
        return summary

    def save_metadata_csv(self, records):
        """Saves metadata records list to dataset.csv."""
        fieldnames = [
            "video_path", "sign_class", "signer_id", "video_id", "fps",
            "total_frames", "valid_landmark_frames", "selected_frame_count",
            "selection_ratio", "token_sequence_length", "landmark_file",
            "token_file", "status", "error"
        ]

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        logger.info(f"Saved dataset metadata CSV to {self.csv_path}")

    def generate_summary(self, records):
        """Generates summary metrics from processing results."""
        total_discovered = len(records)
        successful = [r for r in records if r["status"] == "success"]
        failed = [r for r in records if r["status"] == "failed"]

        classes = sorted(list(set(r["sign_class"] for r in records)))
        signers = sorted(list(set(r["signer_id"] for r in records)))

        total_frames = sum(r["total_frames"] for r in records)
        valid_lm_frames = sum(r["valid_landmark_frames"] for r in records)
        selected_frames = sum(r["selected_frame_count"] for r in records)

        avg_frames = total_frames / max(1, total_discovered)
        avg_selected = selected_frames / max(1, total_discovered)
        overall_ratio = (selected_frames / max(1, total_frames)) * 100.0

        summary = {
            "videos_discovered": total_discovered,
            "videos_processed": total_discovered,
            "successful": len(successful),
            "failed": len(failed),
            "unique_classes": len(classes),
            "classes_list": classes,
            "unique_signers": len(signers),
            "signers_list": signers,
            "total_frames": total_frames,
            "valid_landmark_frames": valid_lm_frames,
            "selected_frames": selected_frames,
            "avg_frames_per_video": round(avg_frames, 2),
            "avg_selected_per_video": round(avg_selected, 2),
            "selection_ratio_pct": round(overall_ratio, 2),
            "token_dim": TOKEN_DIM
        }
        return summary

    def print_summary_report(self, summary):
        """Prints standard summary report."""
        report = f"""
REAL ISL DATASET SUMMARY
=========================
Videos discovered:       {summary['videos_discovered']}
Videos processed:        {summary['videos_processed']}
Successful:              {summary['successful']}
Failed:                  {summary['failed']}

Classes:                 {summary['unique_classes']}
Signers:                 {summary['unique_signers']}

Total frames:            {summary['total_frames']}
Valid landmark frames:   {summary['valid_landmark_frames']}
Selected frames:         {summary['selected_frames']}

Average frames/video:    {summary['avg_frames_per_video']}
Average selected/video:  {summary['avg_selected_per_video']}
Selection ratio:         {summary['selection_ratio_pct']}%

Token dimension:         {summary['token_dim']}
"""
        print(report)
        return report


if __name__ == "__main__":
    pipeline = RealISLDatasetPipeline()
    summary = pipeline.process_dataset()
    pipeline.print_summary_report(summary)
