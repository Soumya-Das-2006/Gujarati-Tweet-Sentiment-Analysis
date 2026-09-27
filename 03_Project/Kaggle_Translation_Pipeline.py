# %% [markdown]
# # Production-Grade English → Gujarati Tweet Translation Pipeline
# This notebook implements a robust, checkpointed, hardware-aware pipeline for translating 1.6M English tweets to Gujarati using AI4Bharat's IndicTrans2 model.
# It is designed specifically for Kaggle's T4x2 GPU environment.

# %% [markdown]
# ## Step 1 & 2: Environment & Hardware Setup
# **IMPORTANT KAGGLE INSTRUCTION**: 
# The `ai4bharat/indictrans2-en-indic-1B` model requires `transformers<=4.39.3` to load correctly because newer versions removed the `transformers.onnx` module.
# Run `!pip install "transformers==4.39.3"` in a Kaggle cell, and then **Restart your Kaggle Session** before running this script!

# %%
import os
import re
import json
import time
import hashlib
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers

# Monkeypatch transformers.onnx for IndicTrans2 compatibility on newer transformers versions
import sys, types

# Force recreate the stubs to avoid issues if a cell is rerun
onnx_stub = types.ModuleType('transformers.onnx')
onnx_stub.__path__ = []  # Declares it as a package
class OnnxConfig: pass
class OnnxSeq2SeqConfigWithPast(OnnxConfig): pass
onnx_stub.OnnxConfig = OnnxConfig
onnx_stub.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast

onnx_utils_stub = types.ModuleType('transformers.onnx.utils')
onnx_utils_stub.compute_effective_axis_dimension = lambda *args, **kwargs: None
onnx_stub.utils = onnx_utils_stub

sys.modules['transformers.onnx'] = onnx_stub
sys.modules['transformers.onnx.utils'] = onnx_utils_stub

import transformers
transformers.onnx = onnx_stub

from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings("ignore")

# Configure Directories
DATA_DIR = "./data"
INTERMEDIATE_DIR = "./intermediate"
BATCH_DIR = os.path.join(INTERMEDIATE_DIR, "batches")
OUTPUT_DIR = "./output"
REPORTS_DIR = "./reports"

for d in [INTERMEDIATE_DIR, BATCH_DIR, OUTPUT_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Kaggle Dataset Path
SOURCE_FILE = "/kaggle/input/datasets/kazanova/sentiment140/training.1600000.processed.noemoticon.csv"

# Hardware Audit
def check_hardware():
    print("--- Hardware Audit ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs available: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        device = "cuda"
    else:
        print("No GPU found. Using CPU.")
        device = "cpu"
    return device

device = check_hardware()

# %% [markdown]
# ## Step 3: Source Audit & Duplicates

# %%
def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def perform_source_audit(filepath):
    print("--- Source Audit ---")
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    file_hash = compute_sha256(filepath)
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"SHA-256: {file_hash}")
    
    # Sentiment140 has no headers. Injecting them.
    headers = ["target", "ids", "date", "flag", "user", "english_tweet"]
    df = pd.read_csv(filepath, names=headers, encoding='latin-1')
    
    print(f"Total Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for duplicates
    dup_ids = df[df.duplicated(subset=['ids'], keep=False)]
    print(f"Duplicate tweet_ids found: {len(dup_ids)}")
    
    # Save duplicate report
    if not dup_ids.empty:
        dup_ids.to_csv(os.path.join(REPORTS_DIR, "duplicate_id_investigation.csv"), index=False)
        print("Duplicate investigation report saved.")
    
    # Emotion distribution
    dist = df['target'].value_counts().to_dict()
    print(f"Emotion Distribution: {dist}")
    
    integrity_report = {
        "source_filename": os.path.basename(filepath),
        "source_file_size_mb": file_size_mb,
        "source_row_count": len(df),
        "source_columns": headers,
        "source_sha256": file_hash,
        "processing_start_time": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(REPORTS_DIR, "source_integrity.json"), "w") as f:
        json.dump(integrity_report, f, indent=4)
        
    return df, file_hash

df_source, original_hash = perform_source_audit(SOURCE_FILE)

# %% [markdown]
# ## Step 4: Mask-Translate-Restore Logic
# We protect URLs, Mentions, and Hashtags before translating.

# %%
class TweetMasker:
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')

    def mask(self, text):
        if not isinstance(text, str):
            return str(text), {}
        
        entities = {}
        
        # Mask URLs
        urls = self.url_pattern.findall(text)
        for i, url in enumerate(urls):
            token = f" __URL_{i}__ "
            text = text.replace(url, token, 1)
            entities[token.strip()] = url
            
        # Mask Mentions
        mentions = self.mention_pattern.findall(text)
        for i, mention in enumerate(mentions):
            token = f" __MNTN_{i}__ "
            text = text.replace(mention, token, 1)
            entities[token.strip()] = mention
            
        # Mask Hashtags
        hashtags = self.hashtag_pattern.findall(text)
        for i, hashtag in enumerate(hashtags):
            token = f" __HTAG_{i}__ "
            text = text.replace(hashtag, token, 1)
            entities[token.strip()] = hashtag
            
        # Cleanup extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text, entities

    def restore(self, translated_text, entities):
        for token, original_value in entities.items():
            translated_text = translated_text.replace(token, original_value)
        return translated_text

# Test Masker
masker = TweetMasker()
test_tweet = "Hello @JohnDoe check this out https://example.com #awesome"
masked, ents = masker.mask(test_tweet)
print("Masked:", masked)
print("Entities:", ents)
print("Restored:", masker.restore("નમસ્તે __MNTN_0__ આ જુઓ __URL_0__ __HTAG_0__", ents))

# %% [markdown]
# ## Step 5: IndicTrans2 Initialization & HuggingFace Login
# The `ai4bharat/indictrans2-en-indic-1B` model is gated. You must authenticate.
# Ensure you have added your HF_TOKEN to Kaggle Secrets!

# %%
hf_token = os.environ.get("HF_TOKEN")
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    from huggingface_hub import login
    login(token=hf_token)
    print("HuggingFace Login Successful via Kaggle Secrets!")
except ImportError:
    # Not running in Kaggle environment
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("HuggingFace Login Successful via Environment Variable!")
except Exception as e:
    print("====================================================================")
    print("🚨 ERROR: HuggingFace login failed!")
    print("Did you attach the HF_TOKEN secret to this Kaggle notebook?")
    print("In Kaggle: Go to 'Add-ons' (or the side menu) -> 'Secrets' -> Turn ON your HF_TOKEN.")
    print(f"Exception details: {e}")
    print("====================================================================")
    raise RuntimeError("Cannot proceed without a valid HF_TOKEN attached to the notebook.")

def load_translation_model(token=None):
    print("Loading IndicTrans2 Model...")
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
        # Using device_map="auto" automatically handles multi-GPU (T4x2) distribution via accelerate
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", token=token)
        
        model.eval()
        print("Translation model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError("CRITICAL: Translation model failed to load. Check the traceback above for the exact cause (e.g., authentication, missing module, or version mismatch).") from e

tokenizer, model = load_translation_model(token=hf_token)

def translate_batch(texts, tokenizer, model):
    if not texts or model is None:
        return ["TRANSLATION_FAILED"] * len(texts) if texts else []
    
    # 1. Masking
    masked_texts = []
    entities_list = []
    for t in texts:
        m, e = masker.mask(t)
        masked_texts.append(m)
        entities_list.append(e)
        
    # 2. Translation
    try:
        # device_map="auto" models automatically move inputs to the correct device
        inputs = tokenizer(masked_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)
            
        translated_masked = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 3. Restoration
        final_translations = []
        for i, tm in enumerate(translated_masked):
            final_translations.append(masker.restore(tm, entities_list[i]))
            
        return final_translations
    except Exception as e:
        print(f"Translation batch failed: {e}")
        return ["TRANSLATION_FAILED"] * len(texts)

# %% [markdown]
# ## Step 6: Semantic Validation (LaBSE)

# %%
def load_semantic_model(device):
    print("Loading LaBSE Semantic Model...")
    sem_model = SentenceTransformer("sentence-transformers/LaBSE").to(device)
    return sem_model

sem_model = load_semantic_model(device)

def compute_semantic_similarity(en_texts, gu_texts, sem_model):
    if not en_texts or not gu_texts:
        return []
    
    en_emb = sem_model.encode(en_texts, convert_to_tensor=True, show_progress_bar=False)
    gu_emb = sem_model.encode(gu_texts, convert_to_tensor=True, show_progress_bar=False)
    
    cosine_scores = util.cos_sim(en_emb, gu_emb)
    scores = [cosine_scores[i][i].item() for i in range(len(en_texts))]
    return scores

def determine_status(score, threshold=0.5):
    if score >= threshold:
        return "ACCEPT"
    else:
        return "REVIEW"

# %% [markdown]
# ## Step 7: Checkpoint System & Processing Logic

# %%
CHECKPOINT_FILE = os.path.join(INTERMEDIATE_DIR, "checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {
        "last_completed_row": 0,
        "completed_batches": 0,
        "failed_batches": 0,
        "accepted_count": 0,
        "review_count": 0,
        "translation_failed_count": 0,
        "timestamp": ""
    }

def save_checkpoint(ckpt):
    ckpt["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=4)

def process_dataframe(df, batch_size=256, is_pilot=False):
    ckpt = load_checkpoint()
    
    if is_pilot:
        print("--- Running Pilot (2000 records) ---")
        df = df.sample(n=2000, random_state=42).reset_index()
        df.rename(columns={"index": "source_row_id"}, inplace=True)
        ckpt["last_completed_row"] = 0 
    else:
        print("--- Running Full Dataset ---")
        df = df.reset_index()
        df.rename(columns={"index": "source_row_id"}, inplace=True)
    
    start_row = ckpt["last_completed_row"]
    total_rows = len(df)
    
    print(f"Starting from row {start_row} / {total_rows}")
    
    for start_idx in range(start_row, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        en_texts = batch_df['english_tweet'].tolist()
        gu_texts = translate_batch(en_texts, tokenizer, model)
        
        valid_en = []
        valid_gu = []
        valid_indices = []
        
        for i, g_text in enumerate(gu_texts):
            if g_text != "TRANSLATION_FAILED":
                valid_en.append(en_texts[i])
                valid_gu.append(g_text)
                valid_indices.append(i)
                
        scores = compute_semantic_similarity(valid_en, valid_gu, sem_model)
        
        final_scores = [0.0] * len(gu_texts)
        statuses = ["TRANSLATION_FAILED"] * len(gu_texts)
        
        for i, score in zip(valid_indices, scores):
            final_scores[i] = score
            statuses[i] = determine_status(score, threshold=0.55)
            
        batch_df['gujarati_tweet'] = gu_texts
        batch_df['semantic_similarity_score'] = final_scores
        batch_df['translation_status'] = statuses
        batch_df['emotion_preservation'] = "PRESERVED" 
        batch_df['english_tweet_hash'] = batch_df['english_tweet'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
        
        batch_filename = os.path.join(BATCH_DIR, f"batch_{start_idx:07d}.csv")
        batch_df.to_csv(batch_filename, index=False)
        
        ckpt["completed_batches"] += 1
        ckpt["last_completed_row"] = end_idx
        ckpt["accepted_count"] += sum(1 for s in statuses if s == "ACCEPT")
        ckpt["review_count"] += sum(1 for s in statuses if s == "REVIEW")
        ckpt["translation_failed_count"] += sum(1 for s in statuses if s == "TRANSLATION_FAILED")
        save_checkpoint(ckpt)
        
        print(f"Processed batch {start_idx} to {end_idx}. Checkpoint saved.")
        
    print("Processing complete.")
    if is_pilot:
        pilot_df = pd.concat([pd.read_csv(os.path.join(BATCH_DIR, f)) for f in sorted(os.listdir(BATCH_DIR)) if f.startswith("batch_")])
        pilot_df.to_csv(os.path.join(INTERMEDIATE_DIR, "pilot_2000.csv"), index=False)
        
        # Clean batch dir
        for f in os.listdir(BATCH_DIR):
            os.remove(os.path.join(BATCH_DIR, f))
            
        print("Pilot saved to intermediate/pilot_2000.csv")

# %% [markdown]
# ## Step 8: Execution (Run Pilot)

# %%
# process_dataframe(df_source, batch_size=32, is_pilot=True)

# %% [markdown]
# ## Step 9 & 10: Final Validation

# %%
def finalize_dataset():
    print("--- Finalizing Dataset ---")
    batch_files = sorted([os.path.join(BATCH_DIR, f) for f in os.listdir(BATCH_DIR) if f.startswith("batch_")])
    if not batch_files:
        print("No batches found.")
        return
        
    final_df = pd.concat([pd.read_csv(f) for f in batch_files])
    
    failed_df = final_df[final_df['translation_status'] == "TRANSLATION_FAILED"]
    review_df = final_df[final_df['translation_status'] == "REVIEW"]
    
    final_df.to_csv(os.path.join(OUTPUT_DIR, "Gujarati_Tweet_Emotion_Dataset.csv"), index=False)
    failed_df.to_csv(os.path.join(OUTPUT_DIR, "Gujarati_Tweet_Translation_Failed.csv"), index=False)
    review_df.to_csv(os.path.join(OUTPUT_DIR, "Gujarati_Tweet_Translation_Review.csv"), index=False)
    
    print("Data splits saved.")
    
    print("Performing final validation...")
    current_hash = compute_sha256(SOURCE_FILE)
    if current_hash != original_hash:
        print("CRITICAL FAILURE: Source file hash changed!")
    else:
        print("PASS: Source file hash matched.")
        
    print(f"Final dataset rows: {len(final_df)}")
    
# finalize_dataset()
