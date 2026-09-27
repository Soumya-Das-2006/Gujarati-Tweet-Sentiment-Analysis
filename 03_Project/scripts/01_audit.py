import os
import hashlib
import json
import time
import pandas as pd

# Directories
DATA_DIR = "../data"
REPORTS_DIR = "../reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
SOURCE_FILE = "/kaggle/input/datasets/kazanova/sentiment140/training.1600000.processed.noemoticon.csv"

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def perform_audit():
    print("--- Phase 1-3: Source Audit ---")
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found.")
        return
    
    file_size_mb = os.path.getsize(SOURCE_FILE) / (1024 * 1024)
    file_hash = compute_sha256(SOURCE_FILE)
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"SHA-256: {file_hash}")
    
    headers = ["target", "ids", "date", "flag", "user", "english_tweet"]
    df = pd.read_csv(SOURCE_FILE, names=headers, encoding='latin-1')
    
    print(f"Total Records: {len(df)}")
    
    dup_ids = df[df.duplicated(subset=['ids'], keep=False)]
    print(f"Duplicate tweet_ids found: {len(dup_ids)}")
    
    if not dup_ids.empty:
        dup_ids.to_csv(os.path.join(REPORTS_DIR, "duplicate_id_investigation.csv"), index=False)
        print("Duplicate report saved.")
        
    dist = df['target'].value_counts().to_dict()
    print(f"Emotion Distribution: {dist}")
    
    report = {
        "source_filename": os.path.basename(SOURCE_FILE),
        "source_file_size_mb": file_size_mb,
        "source_row_count": len(df),
        "source_columns": headers,
        "source_sha256": file_hash,
        "processing_start_time": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(REPORTS_DIR, "source_integrity.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print("Audit complete.")

if __name__ == "__main__":
    perform_audit()
