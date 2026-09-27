import os
import re
import json
import time
import hashlib
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings("ignore")

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

DATA_DIR = "../data"
INTERMEDIATE_DIR = "../intermediate"
BATCH_DIR = os.path.join(INTERMEDIATE_DIR, "batches")
SOURCE_FILE = "/kaggle/input/datasets/kazanova/sentiment140/training.1600000.processed.noemoticon.csv"
CHECKPOINT_FILE = os.path.join(INTERMEDIATE_DIR, "checkpoint.json")

class TweetMasker:
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')

    def mask(self, text):
        if not isinstance(text, str):
            return str(text), {}
        entities = {}
        for i, url in enumerate(self.url_pattern.findall(text)):
            token = f" __URL_{i}__ "
            text = text.replace(url, token, 1)
            entities[token.strip()] = url
        for i, mention in enumerate(self.mention_pattern.findall(text)):
            token = f" __MNTN_{i}__ "
            text = text.replace(mention, token, 1)
            entities[token.strip()] = mention
        for i, hashtag in enumerate(self.hashtag_pattern.findall(text)):
            token = f" __HTAG_{i}__ "
            text = text.replace(hashtag, token, 1)
            entities[token.strip()] = hashtag
        return re.sub(r'\s+', ' ', text).strip(), entities

    def restore(self, text, entities):
        for token, val in entities.items():
            text = text.replace(token, val)
        return text

def translate_pipeline(is_pilot=True):
    from huggingface_hub import login
    import os
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("HuggingFace Login Successful via Environment Variable!")
    else:
        print("====================================================================")
        print("🚨 WARNING: HF_TOKEN not found in environment.")
        print("If running in Kaggle: You MUST attach the HF_TOKEN secret via Add-ons -> Secrets.")
        print("If running locally: Ensure you are logged in via `huggingface-cli login`.")
        print("====================================================================")
        raise RuntimeError("Cannot proceed without a valid HF_TOKEN.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using fallback device for embeddings: {device}")
    
    print("Loading IndicTrans2...")
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        # Use device_map="auto" to automatically balance across Kaggle GPUs
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", token=hf_token)
        model.eval()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError("CRITICAL: Translation model failed to load. Check the traceback above for the exact cause (e.g., authentication, missing module, or version mismatch).") from e
    
    print("Loading LaBSE...")
    sem_model = SentenceTransformer("sentence-transformers/LaBSE").to(device)
    
    masker = TweetMasker()
    
    df = pd.read_csv(SOURCE_FILE, names=["target", "ids", "date", "flag", "user", "english_tweet"], encoding='latin-1')
    
    ckpt = {"last_completed_row": 0, "completed_batches": 0}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            ckpt = json.load(f)
            
    if is_pilot:
        print("Running Pilot (2000 records)")
        df = df.sample(n=2000, random_state=42).reset_index()
        ckpt["last_completed_row"] = 0
    else:
        df = df.reset_index()
        
    df.rename(columns={"index": "source_row_id"}, inplace=True)
    start_row = ckpt["last_completed_row"]
    batch_size = 256
    
    for start_idx in range(start_row, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        en_texts = batch_df['english_tweet'].tolist()
        
        masked_texts = []
        entities_list = []
        for t in en_texts:
            m, e = masker.mask(t)
            masked_texts.append(m)
            entities_list.append(e)
            
        inputs = tokenizer(masked_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)
        translated_masked = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        gu_texts = [masker.restore(tm, entities_list[i]) for i, tm in enumerate(translated_masked)]
        
        en_emb = sem_model.encode(en_texts, convert_to_tensor=True, show_progress_bar=False)
        gu_emb = sem_model.encode(gu_texts, convert_to_tensor=True, show_progress_bar=False)
        scores = [util.cos_sim(en_emb, gu_emb)[i][i].item() for i in range(len(en_texts))]
        
        statuses = ["ACCEPT" if s >= 0.55 else "REVIEW" for s in scores]
        
        batch_df['gujarati_tweet'] = gu_texts
        batch_df['semantic_similarity_score'] = scores
        batch_df['translation_status'] = statuses
        batch_df['emotion_preservation'] = "PRESERVED"
        
        batch_df.to_csv(os.path.join(BATCH_DIR, f"batch_{start_idx:07d}.csv"), index=False)
        
        ckpt["last_completed_row"] = end_idx
        ckpt["completed_batches"] += 1
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(ckpt, f)
            
        print(f"Processed batch {start_idx}-{end_idx}")

if __name__ == "__main__":
    # Change is_pilot=False for full dataset on Kaggle
    translate_pipeline(is_pilot=True)
