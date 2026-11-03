"""
Token-length distribution analysis on 50,000 sentences of legalDataset/train.csv
using InLegalBERT tokenizer (no padding).
"""

import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def analyze_token_lengths(csv_path="legalDataset/train.csv", sample_size=50000, max_len=128):
    print(f"Loading {sample_size:,} random sample from {csv_path}...")
    t0 = time.time()
    
    # Read CSV with keep_default_na=False
    df = pd.read_csv(csv_path, dtype={"Text": str, "Label": str}, keep_default_na=False)
    # Filter valid rows
    valid_mask = (df["Text"].str.strip() != "") & (df["Label"].str.strip() != "")
    df_clean = df[valid_mask].reset_index(drop=True)
    print(f"Loaded {len(df_clean):,} valid sentences in {time.time() - t0:.2f}s.")
    
    # Random sample of 50,000
    sample_df = df_clean.sample(n=min(sample_size, len(df_clean)), random_state=42).reset_index(drop=True)
    sample_texts = sample_df["Text"].tolist()
    
    print(f"Loading InLegalBERT tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    print(f"Tokenizer loaded in {time.time() - t0:.2f}s.")
    
    print(f"Tokenizing {len(sample_texts):,} sentences (no padding, no truncation) to get raw lengths...")
    t0 = time.time()
    # Tokenize without truncation to find actual raw lengths
    raw_encodings = tokenizer(sample_texts, padding=False, truncation=False, add_special_tokens=True)
    tok_time = time.time() - t0
    print(f"Tokenization completed in {tok_time:.2f}s ({len(sample_texts) / tok_time:.1f} sent/sec).")
    
    lengths = np.array([len(ids) for ids in raw_encodings["input_ids"]])
    
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    p95_len = float(np.percentile(lengths, 95))
    p99_len = float(np.percentile(lengths, 99))
    max_raw_len = int(np.max(lengths))
    min_raw_len = int(np.min(lengths))
    
    # Sentences truncated at max_len=128
    truncated_count = int(np.sum(lengths > max_len))
    truncated_pct = (truncated_count / len(lengths)) * 100.0
    
    # Also check padded-to-multiple-of-8 average length
    # In dynamic padding, batch max length is rounded up to multiple of 8
    # For a typical batch of 32:
    batch_size = 32
    batch_maxes = []
    for i in range(0, len(lengths), batch_size):
        b = np.minimum(lengths[i : i + batch_size], max_len)
        max_in_b = int(np.max(b))
        # pad to multiple of 8
        padded_b = int(np.ceil(max_in_b / 8.0) * 8)
        batch_maxes.append(padded_b)
    avg_dynamic_batch_len = float(np.mean(batch_maxes))
    
    print("\n" + "="*70)
    print("TOKEN LENGTH DISTRIBUTION RESULTS (50,000 Legal Sentences):")
    print("="*70)
    print(f"  Sample size:                     {len(lengths):,} sentences")
    print(f"  Mean token length:               {mean_len:.2f} tokens")
    print(f"  Median token length:             {median_len:.1f} tokens")
    print(f"  95th Percentile (p95):           {p95_len:.1f} tokens")
    print(f"  99th Percentile (p99):           {p99_len:.1f} tokens")
    print(f"  Min / Max raw length:            {min_raw_len} / {max_raw_len} tokens")
    print(f"  Sentences exceeding {max_len} tokens:  {truncated_count:,} ({truncated_pct:.2f}%)")
    print(f"  Avg Dynamic Batch Length (pad=8):{avg_dynamic_batch_len:.2f} tokens")
    print("="*70)
    
    return {
        "sample_size": len(lengths),
        "mean": mean_len,
        "median": median_len,
        "p95": p95_len,
        "p99": p99_len,
        "min": min_raw_len,
        "max": max_raw_len,
        "truncated_count": truncated_count,
        "truncated_pct": truncated_pct,
        "avg_dynamic_batch_len": avg_dynamic_batch_len,
    }

if __name__ == "__main__":
    analyze_token_lengths()
