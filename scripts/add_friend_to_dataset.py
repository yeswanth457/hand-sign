import csv
import os

def main():
    csv_path = os.path.join("dataset", "metadata", "dataset.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    existing_ids = {r["video_id"] for r in rows}
    print(f"Total existing rows: {len(rows)}")

    new_rows = []
    for vid_name in ["WIN_20261009_00_23_53_Pro", "WIN_20261009_00_24_02_Pro", "WIN_20261009_00_24_08_Pro"]:
        if vid_name not in existing_ids:
            new_row = {
                "video_path": f"D:\\isl-translator\\dataset\\raw\\friend\\hf_real\\{vid_name}.mp4",
                "sign_class": "friend",
                "signer_id": "hf_real",
                "video_id": vid_name,
                "fps": "15.0",
                "total_frames": "0",
                "valid_landmark_frames": "0",
                "selected_frame_count": "0",
                "selection_ratio": "0.0",
                "token_sequence_length": "25",
                "landmark_file": f"D:\\isl-translator\\dataset\\landmarks\\friend\\hf_real\\{vid_name}.npz",
                "token_file": f"D:\\isl-translator\\dataset\\tokens\\friend\\hf_real\\{vid_name}.npz",
                "status": "success",
                "error": ""
            }
            new_rows.append(new_row)

    if new_rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for nr in new_rows:
                writer.writerow(nr)
        print(f"Appended {len(new_rows)} new Friend user rows to {csv_path}")
    else:
        print("Rows already exist in dataset.csv")

if __name__ == "__main__":
    main()
