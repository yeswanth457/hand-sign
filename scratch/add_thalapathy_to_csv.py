import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
from src.real_dataset_pipeline import RealISLDatasetPipeline

pipeline = RealISLDatasetPipeline()
raw_dir = os.path.join("dataset", "raw", "thalapathy", "hf_real")
records = []
for fname in sorted(os.listdir(raw_dir)):
    if not fname.endswith(".mp4"):
        continue
    fpath = os.path.join(raw_dir, fname)
    entry = {
        "video_path": fpath,
        "sign_class": "thalapathy",
        "signer_id": "hf_real",
        "video_id": os.path.splitext(fname)[0]
    }
    res = pipeline.process_video_file(entry)
    records.append(res)
    print(f"Processed {fname}: status={res.get('status')}, tokens={res.get('token_sequence_length')}")

csv_path = os.path.join("dataset", "metadata", "dataset.csv")
# Read existing rows
with open(csv_path, "r", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))
    fieldnames = reader[0].keys() if reader else list(records[0].keys())

# Remove any old thalapathy rows if present
filtered_rows = [r for r in reader if r.get("sign_class") != "thalapathy"]
# Add new records
for r in records:
    filtered_rows.append(r)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(filtered_rows)

print(f"Successfully updated {csv_path} with {len(records)} Thalapathy records. Total rows: {len(filtered_rows)}")
