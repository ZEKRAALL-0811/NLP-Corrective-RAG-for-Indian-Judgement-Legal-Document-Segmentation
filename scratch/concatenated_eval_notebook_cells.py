# ======================================================================
# CELL 1
# ======================================================================
# ==============================================================================
# Cell 1: Environment & GPU Verification
# ==============================================================================
import os
import sys
import torch
import psutil

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

if not torch.cuda.is_available():
    print("[WARNING]: CUDA GPU is not available! Running on CPU will be slower.")
    device = torch.device("cpu")
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n[GPU DETECTED]: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
    device = torch.device("cuda")

# Install required dependencies


# ======================================================================
# CELL 2
# ======================================================================
# ==============================================================================
# Cell 2: Run Configuration & Google Drive Path Setup
# ==============================================================================
from google.colab import drive
from pathlib import Path
import shutil

# ── RUN CONFIGURATION ─────────────────────────────────────────────────────────
# DRY_RUN = False: Evaluates the complete test.csv (~150k) and val.csv (~293k) splits.
# DRY_RUN = True:  Evaluates 1% stratified sample for rapid pipeline testing.
DRY_RUN = False
DRY_RUN_FRACTION = 0.01

# Mount Google Drive
drive.mount('/content/drive')

DRIVE_PROJECT_ROOT = Path('/content/drive/MyDrive/JurisGraph')
DRIVE_DATA_DIR = DRIVE_PROJECT_ROOT / 'legalDataset'
DRIVE_CHECKPOINT_DIR = DRIVE_PROJECT_ROOT / 'artifacts/model_checkpoint'
DRIVE_REPORTS_DIR = DRIVE_PROJECT_ROOT / 'artifacts/eval_reports'

DRIVE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Fast local NVMe cache for CSV datasets
LOCAL_DATA_DIR = Path('/content/data')
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

for f in ['test.csv', 'val.csv']:
    src = DRIVE_DATA_DIR / f
    dst = LOCAL_DATA_DIR / f
    if src.exists() and not dst.exists():
        print(f"Copying {f} from Drive to fast local NVMe cache...")
        shutil.copy(src, dst)

DATA_DIR = LOCAL_DATA_DIR if (LOCAL_DATA_DIR / 'test.csv').exists() else DRIVE_DATA_DIR

print("\nConfiguration Summary:")
print(f"  Mode:                {'DRY RUN (1% sample)' if DRY_RUN else 'FULL EVALUATION (100% splits)'}")
print(f"  Data Directory:      {DATA_DIR}")
print(f"  Checkpoint Dir:      {DRIVE_CHECKPOINT_DIR}")
print(f"  Drive Reports Dir:   {DRIVE_REPORTS_DIR}")


# ======================================================================
# CELL 3
# ======================================================================
# ==============================================================================
# Cell 3: Canonical Rhetorical Roles & Hyperparameters
# ==============================================================================
CANONICAL_LABELS = [
    "Facts",
    "Issue",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Reasoning",
    "Decision",
    "None",
]

LABEL2ID = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(CANONICAL_LABELS)}
NUM_LABELS = len(CANONICAL_LABELS)

BASE_MODEL_NAME = "law-ai/InLegalBERT"
MAX_SEQ_LENGTH = 128
EVAL_BATCH_SIZE = 64  # Higher batch size for pure evaluation throughput

print(f"Total Canonical Labels: {NUM_LABELS}")
for idx, lbl in ID2LABEL.items():
    print(f"  [{idx}] {lbl}")


# ======================================================================
# CELL 4
# ======================================================================
# ==============================================================================
# Cell 4: Strict Dataset Loading (with keep_default_na=False)
# ==============================================================================
import os
import psutil
import pandas as pd

def log_ram_usage(stage: str = ""):
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)
    tag = f"[{stage}] " if stage else ""
    print(f"{tag}RAM: Process RSS = {rss_gb:.2f} GB | System Available = {avail_gb:.2f} GB / {total_gb:.2f} GB")

def load_and_validate_split(csv_path: Path, split_name: str) -> pd.DataFrame:
    print(f"Loading {split_name} from: {csv_path}...")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}. Please ensure it is present in {csv_path.parent}.")

    # keep_default_na=False prevents 'None' label from becoming NaN
    # Keep Index (document id), Text, and Label columns
    df = pd.read_csv(csv_path, dtype={'Index': str, 'Text': str, 'Label': str}, keep_default_na=False)
    raw_count = len(df)

    # Clean empty/blank text rows
    valid_mask = (df['Text'].str.strip() != '') & (df['Label'].str.strip() != '')
    dropped = (~valid_mask).sum()
    df_clean = df[valid_mask].reset_index(drop=True)

    print(f"  {split_name}: {raw_count:,} raw rows -> {len(df_clean):,} clean rows (dropped {dropped:,} blank rows)")

    unique_labels = set(df_clean['Label'].unique())
    diff = unique_labels - set(CANONICAL_LABELS)
    if diff:
        raise ValueError(f"Found unexpected labels in {split_name}: {diff}")

    return df_clean

# Load splits
test_csv_path = DATA_DIR / 'test.csv'
if not test_csv_path.exists() and (DRIVE_DATA_DIR / 'test.csv').exists():
    test_csv_path = DRIVE_DATA_DIR / 'test.csv'

val_csv_path = DATA_DIR / 'val.csv'
if not val_csv_path.exists() and (DRIVE_DATA_DIR / 'val.csv').exists():
    val_csv_path = DRIVE_DATA_DIR / 'val.csv'

df_test = load_and_validate_split(test_csv_path, 'Test')
df_val = load_and_validate_split(val_csv_path, 'Validation')

if DRY_RUN:
    print(f"\n[DRY RUN ACTIVE]: Subsampling {DRY_RUN_FRACTION*100:.1f}% stratified data for test and val...")
    df_test = df_test.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(frac=DRY_RUN_FRACTION, random_state=42)
    ).reset_index(drop=True)
    df_val = df_val.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(frac=DRY_RUN_FRACTION, random_state=42)
    ).reset_index(drop=True)
    print(f"  Dry-run Test rows: {len(df_test):,}")
    print(f"  Dry-run Val rows:  {len(df_val):,}")

log_ram_usage("After Data Loading")


# ======================================================================
# CELL 5
# ======================================================================
# ==============================================================================
# Cell 5: Load Fine-Tuned InLegalBERT Checkpoint (best_model)
# ==============================================================================
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Search for best_model in Drive checkpoint directory
best_model_path = DRIVE_CHECKPOINT_DIR / "best_model"
if not best_model_path.exists() or not (best_model_path / "config.json").exists():
    alt_paths = [
        DRIVE_PROJECT_ROOT / "artifacts/model_checkpoint/best_model",
        DRIVE_PROJECT_ROOT / "artifacts/model_checkpoint/_dry_run/best_model",
    ]
    for alt in alt_paths:
        if alt.exists() and (alt / "config.json").exists():
            best_model_path = alt
            break

if not best_model_path.exists():
    raise FileNotFoundError(f"CRITICAL: Could not find fine-tuned best_model at {best_model_path}!")

print(f"Loading best_model from Drive: {best_model_path}...")
model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
model.to(device)
model.eval()

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    print("Loaded tokenizer from checkpoint directory.")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    print("Loaded tokenizer from base model.")

# Inspect best_meta.json if present
best_meta_file = best_model_path / "best_meta.json"
if best_meta_file.exists():
    with open(best_meta_file) as f:
        meta_info = json.load(f)
    print(f"Checkpoint Metadata: {meta_info}")

param_count = sum(p.numel() for p in model.parameters())
print(f"[MODEL READY]: Total parameters = {param_count:,} on {device}")


# ======================================================================
# CELL 6
# ======================================================================
# ==============================================================================
# Cell 6: Memory-Lean Tokenization & Dynamic Padding Infrastructure
# ==============================================================================
import gc
import itertools
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def tokenize_texts_to_flat(texts, tokenizer, max_seq_length=MAX_SEQ_LENGTH, chunk_size=50000):
    """
    Tokenizes sentences in chunks to flat uint16 array + int64 offsets.
    Guarantees minimal RAM footprint.
    """
    total_texts = len(texts)
    token_dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.int32
    offsets = np.zeros(total_texts + 1, dtype=np.int64)
    flat_chunks = []

    for chunk_start in range(0, total_texts, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_texts)
        chunk_texts = texts[chunk_start:chunk_end]
        enc = tokenizer(
            chunk_texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        chunk_ids = enc["input_ids"]
        chunk_lens = [len(x) for x in chunk_ids]
        offsets[chunk_start + 1 : chunk_end + 1] = chunk_lens
        total_chunk_tokens = sum(chunk_lens)
        chunk_flat = np.fromiter(
            itertools.chain.from_iterable(chunk_ids),
            dtype=token_dtype,
            count=total_chunk_tokens,
        )
        flat_chunks.append(chunk_flat)
        del enc, chunk_ids, chunk_texts, chunk_lens

    np.cumsum(offsets, out=offsets)
    flat_input_ids = np.concatenate(flat_chunks) if flat_chunks else np.empty(0, dtype=token_dtype)
    del flat_chunks
    return flat_input_ids, offsets

class PretokenizedLegalDataset(Dataset):
    def __init__(self, flat_input_ids, offsets, labels, indices=None):
        self.flat_input_ids = flat_input_ids
        self.offsets = offsets
        self.labels = labels if isinstance(labels, np.ndarray) else np.array(labels, dtype=np.int64)
        if indices is not None:
            self.indices = np.array(indices, dtype=np.int64)
            self.lengths = (offsets[self.indices + 1] - offsets[self.indices]).astype(np.int32)
        else:
            self.indices = None
            self.lengths = (offsets[1:] - offsets[:-1]).astype(np.int32)

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.labels)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        start = self.offsets[real_idx]
        end = self.offsets[real_idx + 1]
        return {
            "input_ids": self.flat_input_ids[start:end],
            "labels": self.labels[real_idx],
        }

class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int = 0, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        batch_size = len(batch)
        lengths = [len(item["input_ids"]) for item in batch]
        max_len = max(lengths)
        if self.pad_to_multiple_of > 0:
            pad_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        else:
            pad_len = max_len

        batch_input_ids = np.full((batch_size, pad_len), self.pad_token_id, dtype=np.int64)
        batch_attention_mask = np.zeros((batch_size, pad_len), dtype=np.int64)
        batch_labels = np.empty(batch_size, dtype=np.int64)

        for i, item in enumerate(batch):
            seq_len = lengths[i]
            batch_input_ids[i, :seq_len] = item["input_ids"]
            batch_attention_mask[i, :seq_len] = 1
            batch_labels[i] = item["labels"]

        return {
            "input_ids": torch.from_numpy(batch_input_ids),
            "attention_mask": torch.from_numpy(batch_attention_mask),
            "labels": torch.from_numpy(batch_labels),
        }

data_collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8)
print("[INFRASTRUCTURE INITIALIZED]: tokenize_texts_to_flat, PretokenizedLegalDataset, DynamicPaddingCollator ready.")


# ======================================================================
# CELL 7
# ======================================================================
# ==============================================================================
# Cell 7: Length-Sorted High-Throughput Evaluation Engine & Watch Metric
# ==============================================================================
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

def track_petitioner_watch_metric(report_dict: dict):
    """
    Bidirectional analysis of Arguments of Petitioner vs. Arguments of Respondent.
    Uses absolute gap and explicitly identifies which class is lagging.
    """
    p_metrics = report_dict.get("Arguments of Petitioner", {})
    r_metrics = report_dict.get("Arguments of Respondent", {})

    p_prec = p_metrics.get("precision", 0.0)
    p_rec = p_metrics.get("recall", 0.0)
    p_f1 = p_metrics.get("f1-score", 0.0)

    r_prec = r_metrics.get("precision", 0.0)
    r_rec = r_metrics.get("recall", 0.0)
    r_f1 = r_metrics.get("f1-score", 0.0)

    abs_rec_gap = abs(r_rec - p_rec)
    abs_f1_gap = abs(r_f1 - p_f1)

    if p_rec < r_rec:
        lagging_class = "Arguments of Petitioner"
    elif r_rec < p_rec:
        lagging_class = "Arguments of Respondent"
    else:
        lagging_class = "Neither (Tied)"

    print("\n" + "="*75)
    print("WATCH METRIC: Arguments of Petitioner vs. Arguments of Respondent")
    print("="*75)
    print(f"  Arguments of Petitioner: Precision={p_prec:.4f} | Recall={p_rec:.4f} | F1={p_f1:.4f}")
    print(f"  Arguments of Respondent: Precision={r_prec:.4f} | Recall={r_rec:.4f} | F1={r_f1:.4f}")
    print(f"  Absolute Recall Gap: {abs_rec_gap:.4f}")
    print(f"  Absolute F1 Gap:     {abs_f1_gap:.4f}")
    print(f"  Lagging Class:       {lagging_class} (by {abs_rec_gap*100:.2f}%)")

    if abs_rec_gap > 0.15:
        if lagging_class == "Arguments of Petitioner":
            print("""
  [DIAGNOSTIC ADVISORY]:
  Arguments of Petitioner recall is lagging significantly behind Respondent (>15% gap).
  Structural advocate-argument overlap detected where Petitioner arguments are absorbed into Respondent.
""")
        else:
            print("""
  [DIAGNOSTIC ADVISORY]:
  Arguments of Respondent recall is lagging significantly behind Petitioner (>15% gap).
""")
    else:
        print("\n  [STATUS]: Petitioner and Respondent recalls are well-balanced within 15% tolerance.")
    print("="*75)


def print_normalized_confusion_matrix(cm_norm: np.ndarray, labels: list):
    """
    Print an aligned, formatted row-normalized confusion matrix (percentages).
    Rows: True Class | Columns: Predicted Class (Diagonal = Recall)
    """
    print("\n" + "="*85)
    print("ROW-NORMALIZED CONFUSION MATRIX (% of True Class Predicted as Each Role)")
    print("Rows: True Class  |  Columns: Predicted Class (Diagonal = Recall)")
    print("="*85)

    header = f"{'True Class':<28s} " + " ".join([f"[{i}]" for i in range(len(labels))])
    print(header)
    col_legend = "  " + " | ".join([f"[{i}]: {lbl[:10]}" for i, lbl in enumerate(labels)])
    print(col_legend)
    print("-" * 85)

    for i, lbl in enumerate(labels):
        row_vals = " ".join([f"{cm_norm[i, j]*100:4.1f}%" for j in range(len(labels))])
        print(f"[{i}] {lbl:<24s} {row_vals}")
    print("="*85)


def evaluate_split(df_split: pd.DataFrame, split_name: str, model, tokenizer, device, batch_size=EVAL_BATCH_SIZE):
    """
    Evaluates a split with length-sorted batching for speed, restores original CSV order,
    and returns metrics, softmax probabilities, predictions, and true labels.
    """
    N = len(df_split)
    print("\n" + "#"*75)
    print(f"# EVALUATING {split_name.upper()} SPLIT ({N:,} sentences)")
    print("#"*75)

    texts = df_split['Text'].tolist()
    true_labels = np.array([LABEL2ID[l] for l in df_split['Label']], dtype=np.int64)

    # 1. Pre-tokenize
    t_tok_start = time.time()
    flat_ids, offsets = tokenize_texts_to_flat(texts, tokenizer, chunk_size=50000)
    tok_duration = time.time() - t_tok_start
    print(f"Pre-tokenized {N:,} sentences in {tok_duration:.2f}s ({N/max(tok_duration, 1e-4):.1f} sent/s)")
    print(f"Flat array size: {flat_ids.nbytes / (1024**2):.2f} MB, Offsets: {offsets.nbytes / (1024**2):.2f} MB")

    # 2. Length-sort indices strictly for inference speed
    lengths = (offsets[1:] - offsets[:-1]).astype(np.int32)
    sort_indices = sorted(range(N), key=lambda i: lengths[i])
    sort_indices = np.array(sort_indices, dtype=np.int64)

    dataset = PretokenizedLegalDataset(flat_ids, offsets, true_labels, indices=sort_indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )

    # Free raw text list
    del texts
    gc.collect()

    # 3. Model forward pass
    all_sorted_probs = []
    t_eval_start = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference ({split_name})", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().to(torch.float32).numpy()

            all_sorted_probs.append(probs)

    eval_duration = time.time() - t_eval_start
    eval_speed = N / max(eval_duration, 1e-4)
    print(f"Inference completed in {eval_duration:.2f}s ({eval_speed:.1f} sent/s)")

    sorted_probs = np.concatenate(all_sorted_probs, axis=0)

    # 4. RESTORE ORIGINAL CSV ROW ORDER
    orig_probs = np.empty_like(sorted_probs)
    orig_probs[sort_indices] = sorted_probs
    orig_preds = np.argmax(orig_probs, axis=-1)

    # 5. Row count verification check
    assert len(orig_probs) == N, f"CRITICAL: Restored prob count ({len(orig_probs)}) != CSV row count ({N})"
    assert len(orig_preds) == N, f"CRITICAL: Restored pred count ({len(orig_preds)}) != CSV row count ({N})"
    print(f"[VERIFIED]: Restored {len(orig_probs):,} probabilities match original CSV row count ({N:,}).")

    # 6. Compute Metrics
    macro_f1 = float(f1_score(true_labels, orig_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_labels, orig_preds, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(true_labels, orig_preds))

    report_str = classification_report(
        true_labels,
        orig_preds,
        target_names=CANONICAL_LABELS,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        true_labels,
        orig_preds,
        target_names=CANONICAL_LABELS,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    raw_cm = confusion_matrix(true_labels, orig_preds)
    norm_cm = confusion_matrix(true_labels, orig_preds, normalize="true")

    # 7. Print Reports
    print("\n" + "="*75)
    print(f"EVALUATION RESULTS: {split_name.upper()} SPLIT")
    print("="*75)
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Accuracy:    {accuracy*100:.2f}% ({accuracy:.4f})")
    print("\nPer-Class Breakdown:")
    print(report_str)

    print_normalized_confusion_matrix(norm_cm, CANONICAL_LABELS)
    track_petitioner_watch_metric(report_dict)

    results = {
        "split": split_name,
        "sample_count": N,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "report_dict": report_dict,
        "raw_confusion_matrix": raw_cm.tolist(),
        "normalized_confusion_matrix": norm_cm.tolist(),
        "eval_speed_sent_sec": eval_speed,
        "eval_duration_sec": eval_duration,
    }

    return results, orig_probs, orig_preds, true_labels


# ======================================================================
# CELL 8
# ======================================================================
# ==============================================================================
# Cell 8: Evaluate on Test Set & Export Probabilities and Report
# ==============================================================================
test_results, test_probs, test_preds, test_labels = evaluate_split(
    df_split=df_test,
    split_name="Test",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

# Extract document IDs from the Index column (as fixed-width strings for seamless npz loading)
test_doc_ids = np.array(df_test['Index'].values, dtype=str) if 'Index' in df_test.columns else np.arange(len(df_test)).astype(str)

# Verify counts before saving
assert len(test_probs) == len(df_test), "Mismatch in test probs and df_test length!"
assert len(test_doc_ids) == len(df_test), "Mismatch in test doc_ids and df_test length!"

# Save probs_test.npz to Google Drive
test_npz_path = DRIVE_REPORTS_DIR / "probs_test.npz"
print(f"\nSaving per-sentence test probabilities to: {test_npz_path}...")
np.savez_compressed(
    test_npz_path,
    probs=test_probs,
    doc_ids=test_doc_ids,
    Index=test_doc_ids,
    labels=test_labels,
    label_names=np.array(CANONICAL_LABELS, dtype=str),
)
test_npz_size_mb = os.path.getsize(test_npz_path) / (1024**2)
print(f"[SAVED]: {test_npz_path.name} ({test_npz_size_mb:.2f} MB, {len(test_probs):,} sentences)")

# Save test report JSON
test_report_path = DRIVE_REPORTS_DIR / "test_evaluation_report.json"
with open(test_report_path, "w", encoding="utf-8") as f:
    json.dump(test_results, f, indent=2)
print(f"[SAVED]: {test_report_path.name} to {DRIVE_REPORTS_DIR}")


# ======================================================================
# CELL 9
# ======================================================================
# ==============================================================================
# Cell 9: Re-run Evaluation on Validation Set & Export Probabilities
# ==============================================================================
val_results, val_probs, val_preds, val_labels = evaluate_split(
    df_split=df_val,
    split_name="Validation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

# Extract document IDs from the Index column (as fixed-width strings for seamless npz loading)
val_doc_ids = np.array(df_val['Index'].values, dtype=str) if 'Index' in df_val.columns else np.arange(len(df_val)).astype(str)

# Verify counts before saving
assert len(val_probs) == len(df_val), "Mismatch in val probs and df_val length!"
assert len(val_doc_ids) == len(df_val), "Mismatch in val doc_ids and df_val length!"

# Save probs_val.npz to Google Drive
val_npz_path = DRIVE_REPORTS_DIR / "probs_val.npz"
print(f"\nSaving per-sentence validation probabilities to: {val_npz_path}...")
np.savez_compressed(
    val_npz_path,
    probs=val_probs,
    doc_ids=val_doc_ids,
    Index=val_doc_ids,
    labels=val_labels,
    label_names=np.array(CANONICAL_LABELS, dtype=str),
)
val_npz_size_mb = os.path.getsize(val_npz_path) / (1024**2)
print(f"[SAVED]: {val_npz_path.name} ({val_npz_size_mb:.2f} MB, {len(val_probs):,} sentences)")

# Save val report JSON
val_report_path = DRIVE_REPORTS_DIR / "val_evaluation_report.json"
with open(val_report_path, "w", encoding="utf-8") as f:
    json.dump(val_results, f, indent=2)
print(f"[SAVED]: {val_report_path.name} to {DRIVE_REPORTS_DIR}")


# ======================================================================
# CELL 10
# ======================================================================
# ==============================================================================
# Cell 10: Comparison Summary & Viterbi Feature Verification
# ==============================================================================
print("\n" + "="*85)
print("SUMMARY COMPARISON: VALIDATION SPLIT VS. TEST SPLIT")
print("="*85)
print(f"{'Split':<15s} {'Sentences':>12s} {'Macro F1':>12s} {'Weighted F1':>14s} {'Accuracy':>12s}")
print("-"*85)
print(f"{'Validation':<15s} {len(df_val):12,d} {val_results['macro_f1']:12.4f} {val_results['weighted_f1']:14.4f} {val_results['accuracy']*100:11.2f}%")
print(f"{'Test':<15s} {len(df_test):12,d} {test_results['macro_f1']:12.4f} {test_results['weighted_f1']:14.4f} {test_results['accuracy']*100:11.2f}%")
print("="*85)

print("\nPer-Class F1 Comparison:")
print(f"{'Class':<28s} {'Val F1':>12s} {'Test F1':>12s} {'Delta F1':>12s}")
print("-" * 68)
for lbl in CANONICAL_LABELS:
    v_f1 = val_results['report_dict'].get(lbl, {}).get('f1-score', 0.0)
    t_f1 = test_results['report_dict'].get(lbl, {}).get('f1-score', 0.0)
    d_f1 = t_f1 - v_f1
    print(f"{lbl:<28s} {v_f1:12.4f} {t_f1:12.4f} {d_f1:+12.4f}")
print("="*68)

# Verify saved .npz files can be reloaded seamlessly
print("\nVerifying exported .npz archives in Drive:")
for npz_p in [test_npz_path, val_npz_path]:
    with np.load(npz_p) as data:
        print(f"\nArchive: {npz_p.name}")
        print(f"  Keys:            {list(data.keys())}")
        print(f"  probs shape:     {data['probs'].shape} (dtype: {data['probs'].dtype})")
        print(f"  doc_ids count:   {len(data['doc_ids']):,}")
        print(f"  labels count:    {len(data['labels']):,}")
        print(f"  label_names:     {list(data['label_names'])}")
        # Verify probabilities sum to 1.0
        row_sums = data['probs'].sum(axis=-1)
        valid_sums = np.allclose(row_sums, 1.0, atol=1e-5)
        print(f"  Probabilities sum to 1.0: {valid_sums}")

print("\n" + "="*75)
print("[SUCCESS]: Evaluation complete. Softmax probability matrices are saved")
print("and ready for offline CPU Viterbi transition smoothing!")
print("="*75)

