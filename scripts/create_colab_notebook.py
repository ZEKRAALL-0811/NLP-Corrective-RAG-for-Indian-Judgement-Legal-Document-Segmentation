"""
Script to generate the Google Colab training notebook:
notebooks/InLegalBERT_FineTuning_Colab.ipynb

Incorporate improvements:
1. Dry-run checkpoint isolation (_dry_run/ folder) and config mismatch validation.
2. BATCHING_MODE flag: 'grouped10' (default), 'grouped50', and 'random' with checkpoint compatibility check.
3. Fused AdamW optimizer (fused=True on CUDA) to eliminate per-step kernel launch overhead.
4. SimulatedInterruption exception cleanly halts on interruption, guarding epoch-boundary saves.
5. Early-warning evaluation (~300/class fixed stratified slice) at mid-epoch step checkpoints.
6. Step-only speed accounting (excludes checkpoint save I/O and validation) without double counting.
7. Length-sorted validation set (no shuffle) for zero-padding evaluation.
8. Padding efficiency tracking (mean padded length, pad/real ratio in epoch 1).
9. Deterministic shuffle and mid-epoch resume.
10. Absolute gap watch metric with explicit lagging class indicator.
11. EPOCHS = 2 for full run.
"""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def add_markdown(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 1: Header Markdown
    # ──────────────────────────────────────────────────────────────────────────
    add_markdown(r"""# JurisGraph — Module 2: InLegalBERT Fine-Tuning (Full GPU Run)
### Rhetorical Role Classification on the LegalSeg Corpus (~1.12M Sentences)

This notebook executes the fine-tuning of `law-ai/InLegalBERT` on Google Colab (GPU: T4 / V100 / A100).

---

### Pipeline Specifications & Architecture:
* **Model Checkpoint:** `law-ai/InLegalBERT` (110M parameters)
* **7 Rhetorical Roles:** `Facts`, `Issue`, `Arguments of Petitioner`, `Arguments of Respondent`, `Reasoning`, `Decision`, `None`
* **Loss Function:** Sqrt-dampened, ratio-capped CrossEntropyLoss (`--weight_mode full`):
  $$w_c = \\sqrt{N_{\\text{total}} / (7 \\times N_c)}, \\quad \\text{normalized } \\min(w) = 1.0, \\quad \\text{capped at } 6.0\\times$$
* **High-Throughput Optimization:**
  * **Pre-tokenized datasets** (tokenized once upfront, eliminating DataLoader tokenization bottlenecks).
  * **Configurable Batching Mode (`BATCHING_MODE = "grouped10"`):** 10-batch mega-chunks ($10 \times \text{BATCH\_SIZE}$) sorted by token length, then shuffled by batch order. Cuts mean padded length to $\approx 40\text{–}44$ tokens ($\approx 2.5\times$ speedup over random shuffling, $\approx 3\times$ over static padding). Also supports `"grouped50"` and `"random"`.
  * **Fused CUDA Optimizer:** `torch.optim.AdamW(..., fused=True)` to minimize fixed per-step kernel launch overhead.
  * **Length-Sorted Validation:** Validation set pre-sorted by length without shuffling for zero-padding evaluation overhead.
  * **Memory-Lean Dynamic Padding Collator:** `DynamicPaddingCollator(pad_to_multiple_of=8)` with flat numpy arrays (uint16) for Tensor Core alignment.
* **Fault-Tolerant Checkpointing & Auto-Resume:**
  * Mid-epoch step checkpointing (every 3,000 steps on full run; every 100 on dry run), automatically keeping only the **2 most recent** step checkpoints to conserve Drive storage.
  * **Early-Warning Step Evaluation:** Evaluates a fixed stratified slice (~300/class, ~2,100 samples) in ~1–2 seconds at every step checkpoint to track Macro F1 and per-class recall progress in real time.
  * **Deterministic per-epoch shuffling** (seeded pure function of epoch + token lengths + mode) allowing exact step-level resume without repeating or skipping samples.
  * **Strict Config Isolation:** Dry-run artifacts are isolated in `_dry_run/`. Checkpoints from different sample sizes, dry-run states, or batching modes are rejected to prevent contamination.
* **Watch Metric Analysis:** Bidirectional absolute recall/F1 gap tracker between `Arguments of Petitioner` and `Arguments of Respondent`.
* **Baseline Comparison:** Direct side-by-side comparison table against the 2-epoch CPU smoke test (Macro F1 = 0.4768).""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 2: Environment Setup & GPU Verification
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 1: Environment & GPU Verification
# ==============================================================================
import os
import sys
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

if not torch.cuda.is_available():
    raise SystemError("CRITICAL: CUDA GPU is not available! Please enable GPU in Colab: Runtime -> Change runtime type -> T4 / A100 GPU.")

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"\\n[GPU DETECTED]: {gpu_name} ({gpu_mem:.2f} GB VRAM)")

# Install required dependencies
!pip install -q transformers>=4.40.0 accelerate scikit-learn tqdm pandas psutil""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 3: Configuration & Drive Mount (with Checkpoint Isolation)
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 2: Run Configuration & Google Drive Path Setup (Isolated Checkpoints)
# ==============================================================================
from google.colab import drive
from pathlib import Path
import shutil

# ── RUN CONFIGURATION ─────────────────────────────────────────────────────────
# DRY_RUN = True:  Runs 1% stratified subset (~11,200 train, ~2,933 val) for 1 epoch (~3-5 mins).
#                  Saves checkpoints & reports to '_dry_run/' subfolder.
# DRY_RUN = False: Launches full multi-hour training across all 1.12M sentences (2 epochs).
#                  Saves to main 'artifacts/' folders.
DRY_RUN = True
DRY_RUN_FRACTION = 0.01

# Full run is configured for 2 epochs (sufficient for 1.12M sentences with dynamic padding)
EPOCHS = 1 if DRY_RUN else 2

# Mid-epoch checkpoint frequency:
# In dry run: every 100 steps (~3,200 sentences) to test mid-epoch save & resume
# In full run: every 3,000 steps (~96,000 sentences, ~every 15-20 minutes)
CHECKPOINT_EVERY_STEPS = 100 if DRY_RUN else 3000

# Batching Mode:
#  - "random":    Dynamic padding with standard random shuffle (no grouping)
#  - "grouped50": 50-batch mega-chunks (50 * BATCH_SIZE = 1,600 samples)
#  - "grouped10": 10-batch mega-chunks (10 * BATCH_SIZE = 320 samples) [Default for full run]
BATCHING_MODE = "grouped10"

# Optional test flag: Set to an integer (e.g. 150) during dry run to simulate an interruption
# at that step. When triggered, it raises SimulatedInterruption to immediately halt execution.
# CRITICAL: You MUST reset SIMULATE_INTERRUPT_STEP = None before re-running to resume after restart,
# otherwise execution will immediately halt again!
SIMULATE_INTERRUPT_STEP = None

# Mount Google Drive
drive.mount('/content/drive')

DRIVE_PROJECT_ROOT = Path('/content/drive/MyDrive/JurisGraph')
DRIVE_DATA_DIR = DRIVE_PROJECT_ROOT / 'legalDataset'

# ISOLATED DIRECTORIES: Dry-run and full-run checkpoints never mix!
if DRY_RUN:
    DRIVE_CHECKPOINT_DIR = DRIVE_PROJECT_ROOT / 'artifacts/model_checkpoint/_dry_run'
    DRIVE_REPORTS_DIR = DRIVE_PROJECT_ROOT / 'artifacts/eval_reports/_dry_run'
else:
    DRIVE_CHECKPOINT_DIR = DRIVE_PROJECT_ROOT / 'artifacts/model_checkpoint'
    DRIVE_REPORTS_DIR = DRIVE_PROJECT_ROOT / 'artifacts/eval_reports'

DRIVE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Fast local NVMe cache for CSV datasets
LOCAL_DATA_DIR = Path('/content/data')
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

if (DRIVE_DATA_DIR / 'train.csv').exists():
    print("Found legalDataset in Google Drive. Copying to local NVMe disk for maximum speed...")
    for f in ['train.csv', 'val.csv', 'test.csv']:
        src = DRIVE_DATA_DIR / f
        dst = LOCAL_DATA_DIR / f
        if src.exists() and not dst.exists():
            print(f"  Copying {f}...")
            shutil.copy(src, dst)
    DATA_DIR = LOCAL_DATA_DIR
    print("[SUCCESS] Local data cache ready.")
else:
    print(f"[WARNING] Could not find {DRIVE_DATA_DIR}. Using LOCAL_DATA_DIR.")
    DATA_DIR = LOCAL_DATA_DIR

print(f"\\nConfiguration Summary:")
print(f"  Mode:                   {'DRY RUN (1% sample, 1 epoch)' if DRY_RUN else 'FULL RUN (100% corpus, 2 epochs)'}")
print(f"  Batching Mode:          {BATCHING_MODE}")
print(f"  Epochs:                 {EPOCHS}")
print(f"  Checkpoint Every Steps: {CHECKPOINT_EVERY_STEPS}")
print(f"  Drive Checkpoint Dir:   {DRIVE_CHECKPOINT_DIR}")
print(f"  Drive Reports Dir:      {DRIVE_REPORTS_DIR}")""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 4: Canonical Labels & Model Hyperparameters
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
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

# Model & Training Hyperparameters
BASE_MODEL_NAME = "law-ai/InLegalBERT"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRADIENT_CLIP = 1.0

print(f"Total Canonical Labels: {NUM_LABELS}")
for idx, lbl in ID2LABEL.items():
    print(f"  [{idx}] {lbl}")""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 5: Strict Data Loading
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
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
        raise FileNotFoundError(f"Missing file: {csv_path}")

    # keep_default_na=False prevents 'None' label from becoming NaN
    df = pd.read_csv(csv_path, dtype={'Text': str, 'Label': str}, keep_default_na=False)
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

# Load full splits
df_train = load_and_validate_split(DATA_DIR / 'train.csv', 'Train')
df_val = load_and_validate_split(DATA_DIR / 'val.csv', 'Val')
df_test = load_and_validate_split(DATA_DIR / 'test.csv', 'Test') if (DATA_DIR / 'test.csv').exists() else None

# Stratified subsampling if DRY_RUN is active
if DRY_RUN:
    print(f"\\n[DRY RUN ACTIVE]: Subsampling {DRY_RUN_FRACTION*100:.1f}% stratified data for pre-flight validation...")
    df_train = df_train.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(frac=DRY_RUN_FRACTION, random_state=42)
    ).reset_index(drop=True)
    df_val = df_val.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(frac=DRY_RUN_FRACTION, random_state=42)
    ).reset_index(drop=True)
    print(f"  Dry-run Train rows: {len(df_train):,}")
    print(f"  Dry-run Val rows:   {len(df_val):,}")

print("\\n[SUCCESS] Splits loaded and validated.")
log_ram_usage("After Data Loading")""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 6: Sqrt-Dampened Class Weights
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 5: Class-Weighted Loss Calculation (--weight_mode full)
# ==============================================================================
import math
import torch

def compute_class_weights_full(
    df: pd.DataFrame,
    max_ratio: float = 6.0,
    petitioner_manual_floor: float = None,
) -> torch.Tensor:
    \"\"\"
    Compute sqrt-dampened, ratio-capped class weights from the corpus distribution.
    Formula:
        raw_w_c = total / (7 * n_c)
        dampened_w_c = sqrt(raw_w_c)
    Normalised so min(w) = 1.0, capped at max_ratio = 6.0.
    \"\"\"
    total = len(df)
    counts = df['Label'].value_counts()

    raw_weights = []
    for label in CANONICAL_LABELS:
        c = counts.get(label, 1)
        raw_w = total / (NUM_LABELS * c)
        raw_weights.append(raw_w)

    dampened = [math.sqrt(w) for w in raw_weights]
    min_w = min(dampened)
    normed = [w / min_w for w in dampened]
    capped = [min(w, max_ratio) for w in normed]

    if petitioner_manual_floor is not None:
        pet_idx = LABEL2ID["Arguments of Petitioner"]
        if capped[pet_idx] < petitioner_manual_floor:
            print(f"[NOTE] Applying manual floor of {petitioner_manual_floor} to Arguments of Petitioner (was {capped[pet_idx]:.4f})")
            capped[pet_idx] = petitioner_manual_floor

    weight_tensor = torch.tensor(capped, dtype=torch.float32)

    print(f"\\nClass Weights (Sqrt-Dampened, max_ratio={max_ratio}):")
    print(f"  {'Label':<28s} {'Count':>9s} {'Pct':>7s} {'Raw':>8s} {'Final Weight':>14s}")
    print(f"  {'-'*70}")
    for label, rw, cw in zip(CANONICAL_LABELS, raw_weights, capped):
        cnt = counts.get(label, 0)
        pct = (cnt / total) * 100
        print(f"  {label:<28s} {cnt:9,d} {pct:6.2f}% {rw:8.4f} {cw:14.4f}")

    return weight_tensor

class_weights = compute_class_weights_full(df_train, max_ratio=6.0)""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 7: Pre-tokenization & Dynamic Padding (Throughput Optimization)
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 6: High-Throughput Memory-Lean Pre-Tokenization & Dynamic Padding
# ==============================================================================
import gc
import itertools
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

print(f"Loading tokenizer: {BASE_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

train_texts = df_train['Text'].tolist()
train_labels = np.array([LABEL2ID[l] for l in df_train['Label']], dtype=np.int64)

val_texts = df_val['Text'].tolist()
val_labels = np.array([LABEL2ID[l] for l in df_val['Label']], dtype=np.int64)

# Chunked pre-tokenization into flat arrays (uint16/int32 + int64 offsets).
# Eliminates Rust Encoding objects and lists-of-lists memory bottleneck.
def tokenize_texts_to_flat(texts, tokenizer, max_seq_length=MAX_SEQ_LENGTH, chunk_size=50000):
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

t_pretok_start = time.time()
print(f"\\nPre-tokenizing {len(train_texts):,} train sentences in chunks of ~50,000...")
train_flat_ids, train_offsets = tokenize_texts_to_flat(train_texts, tokenizer)

print(f"Pre-tokenizing {len(val_texts):,} val sentences in chunks of ~50,000...")
val_flat_ids, val_offsets = tokenize_texts_to_flat(val_texts, tokenizer)

pretok_duration = time.time() - t_pretok_start
total_pretok_sentences = len(train_texts) + len(val_texts)
pretok_speed = total_pretok_sentences / max(pretok_duration, 1e-4)

print(f"\\n[TIMING: PRE-TOKENIZATION COMPLETED]")
print(f"  Sentences Pre-Tokenized: {total_pretok_sentences:,}")
print(f"  Time Taken:              {pretok_duration:.2f}s ({pretok_speed:.1f} sentences/sec)")
print(f"  Train flat token array:  {train_flat_ids.nbytes / (1024**2):.2f} MB ({len(train_flat_ids):,} tokens, {train_flat_ids.dtype})")
print(f"  Val flat token array:    {val_flat_ids.nbytes / (1024**2):.2f} MB ({len(val_flat_ids):,} tokens, {val_flat_ids.dtype})")

# Free raw text columns, text lists, and unneeded memory
if 'Text' in df_train.columns:
    df_train.drop(columns=['Text'], inplace=True)
if 'Text' in df_val.columns:
    df_val.drop(columns=['Text'], inplace=True)
if df_test is not None and 'Text' in df_test.columns:
    df_test.drop(columns=['Text'], inplace=True)

del train_texts, val_texts
gc.collect()

log_ram_usage("After Tokenization")

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

train_dataset = PretokenizedLegalDataset(train_flat_ids, train_offsets, train_labels)

# Sort validation set strictly by length (no shuffle) to minimize padding overhead during evaluation.
val_lengths = (val_offsets[1:] - val_offsets[:-1]).astype(np.int32)
val_sort_indices = sorted(range(len(val_lengths)), key=lambda i: val_lengths[i])
val_dataset = PretokenizedLegalDataset(val_flat_ids, val_offsets, val_labels, indices=val_sort_indices)

# Dynamic numpy collator pads each batch to its longest sequence rounded up to a multiple of 8
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

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=True,
)

# Fixed, class-stratified early-warning validation slice (up to 300 per class)
# Evaluated at every mid-epoch step checkpoint in ~1-2 seconds for immediate visibility!
quick_indices = []
rng_quick = np.random.default_rng(42)
for label_name in CANONICAL_LABELS:
    l_id = LABEL2ID[label_name]
    matching = [i for i, lbl in enumerate(val_labels) if lbl == l_id]
    sample_size = min(len(matching), 300)
    if sample_size > 0:
        chosen = rng_quick.choice(matching, size=sample_size, replace=False).tolist()
        quick_indices.extend(chosen)

# Sort quick slice strictly by length to minimize padding during step checkpoint evaluations
quick_indices.sort(key=lambda i: val_lengths[i])
val_quick_dataset = PretokenizedLegalDataset(val_flat_ids, val_offsets, val_labels, indices=quick_indices)

val_quick_loader = DataLoader(
    val_quick_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=True,
)

log_ram_usage("After DataLoader Creation")
print(f"\\n[SUCCESS] Datasets initialized with Dynamic Padding collator.")
print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,} (sorted by length)")
print(f"Early-warning slice: {len(val_quick_dataset):,} samples (~300/class, sorted by length)")""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 8: Model Loading & Mixed Precision
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 7: InLegalBERT Model & Mixed Precision (AMP)
# ==============================================================================
from transformers import AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model

model = create_model()
model.to(device)

scaler = torch.cuda.amp.GradScaler()

param_count = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[MODEL LOADED]: {BASE_MODEL_NAME}")
print(f"Total parameters: {param_count:,} | Trainable: {trainable:,}")""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 9: Checkpointing & Step Resume Logic
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 8: Mid-Epoch Step Checkpointing & Deterministic Resume Logic
# ==============================================================================
import json
import time
import torch

checkpoint_save_times = []

def save_checkpoint(
    epoch: int,
    step: int,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    macro_f1,
    all_metrics,
    checkpoint_dir: Path,
    is_step_checkpoint: bool = False,
    best_full_val_macro_f1: float = -1.0,
):
    \"\"\"
    Save checkpoint to Google Drive.
    - If is_step_checkpoint: saves checkpoint_epoch_{epoch}_step_{step} and prunes
      so only the 2 most recent step checkpoints remain.
    - If epoch boundary: saves checkpoint_epoch_{epoch} and removes obsolete step checkpoints.
    \"\"\"
    t_save_start = time.time()
    if is_step_checkpoint:
        ckpt_dir = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}"
    else:
        ckpt_dir = checkpoint_dir / f"checkpoint_epoch_{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model weights and tokenizer
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # 2. Save PyTorch state dictionary
    torch.save({
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "macro_f1": macro_f1,
    }, ckpt_dir / "training_state.pt")

    save_duration = time.time() - t_save_start
    checkpoint_save_times.append(save_duration)

    # 3. Save metadata JSON with run configuration
    meta = {
        "epoch": epoch,
        "step": step,
        "is_step_checkpoint": is_step_checkpoint,
        "dry_run": bool(DRY_RUN),
        "batching_mode": BATCHING_MODE,
        "epochs_configured": int(EPOCHS),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batch_size": int(BATCH_SIZE),
        "quick_macro_f1": float(macro_f1) if macro_f1 is not None else None,
        "best_full_val_macro_f1": float(best_full_val_macro_f1),
        "metrics": all_metrics,
        "save_duration_sec": round(save_duration, 2),
        "saved_at": str(ckpt_dir),
    }
    with open(ckpt_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    tag = f"Step {step} (Epoch {epoch})" if is_step_checkpoint else f"Epoch {epoch}"
    print(f"\\n[CHECKPOINT SAVED]: {tag} -> {ckpt_dir} (Drive write time: {save_duration:.2f}s)")

    # Prune step checkpoints: keep only 2 most recent
    if is_step_checkpoint:
        step_folders = sorted(
            [d for d in checkpoint_dir.glob("checkpoint_epoch_*_step_*") if d.is_dir()],
            key=lambda d: d.stat().st_mtime
        )
        if len(step_folders) > 2:
            for old in step_folders[:-2]:
                shutil.rmtree(old)
                print(f"[PRUNED OLD CHECKPOINT]: Removed {old.name} to conserve Drive storage")
    else:
        # Epoch boundary checkpoint saved: clean up mid-epoch checkpoints for this epoch
        for old in checkpoint_dir.glob(f"checkpoint_epoch_{epoch}_step_*"):
            shutil.rmtree(old, ignore_errors=True)
            print(f"[CLEANUP]: Removed obsolete mid-epoch checkpoint {old.name}")


def find_latest_checkpoint(checkpoint_dir: Path):
    \"\"\"
    Search for valid checkpoints in Drive.
    Validates configuration (dry_run flag, sample count, epochs, batching_mode) and refuses
    to resume from contaminated or mismatched checkpoints!
    Returns (latest_dir, epoch, step, best_full_val_macro_f1).
    best_full_val_macro_f1 defaults to -1.0 so the first epoch-end eval always saves best_model.
    \"\"\"
    if not checkpoint_dir.exists():
        return None, 0, 0, -1.0

    # First, check best_model/best_meta.json for authoritative best F1
    best_meta_path = checkpoint_dir / "best_model" / "best_meta.json"
    authoritative_best_f1 = -1.0
    if best_meta_path.exists():
        try:
            with open(best_meta_path) as f:
                best_meta = json.load(f)
            authoritative_best_f1 = float(best_meta.get("best_full_val_macro_f1", -1.0))
            print(f"[FOUND best_meta.json]: best_full_val_macro_f1 = {authoritative_best_f1:.4f}")
        except Exception:
            pass

    valid_checkpoints = []
    for d in checkpoint_dir.glob("checkpoint_*"):
        if not d.is_dir() or not (d / "training_state.pt").exists() or not (d / "checkpoint_meta.json").exists():
            continue
        try:
            with open(d / "checkpoint_meta.json") as f:
                meta = json.load(f)

            # Refuse to resume if config doesn't match!
            if meta.get("dry_run") != DRY_RUN:
                print(f"[REJECTED CHECKPOINT]: {d.name} has dry_run={meta.get('dry_run')} (current run has dry_run={DRY_RUN}). Skipping.")
                continue
            if meta.get("batching_mode") is not None and meta.get("batching_mode") != BATCHING_MODE:
                print(f"[REJECTED CHECKPOINT]: {d.name} has batching_mode='{meta.get('batching_mode')}' (current run has BATCHING_MODE='{BATCHING_MODE}'). Skipping.")
                continue
            if abs(meta.get("train_samples", 0) - len(train_dataset)) > 10:
                print(f"[REJECTED CHECKPOINT]: {d.name} trained on {meta.get('train_samples')} samples (current dataset has {len(train_dataset)}). Skipping.")
                continue
            if meta.get("epochs_configured") is not None and meta.get("epochs_configured") != EPOCHS:
                print(f"[REJECTED CHECKPOINT]: {d.name} configured for {meta.get('epochs_configured')} epochs (current run has EPOCHS={EPOCHS}). Skipping.")
                continue

            ep = meta.get("epoch", 0)
            st = meta.get("step", 0)
            is_step = "step_" in d.name
            progress_key = (ep, 0 if is_step else 1, st)
            valid_checkpoints.append((progress_key, ep, st, d))
        except Exception:
            continue

    if not valid_checkpoints:
        return None, 0, 0, -1.0

    # Sort by progress key descending
    valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
    _, latest_epoch, latest_step, latest_dir = valid_checkpoints[0]

    # Use authoritative best F1 from best_meta.json; fall back to checkpoint's field
    if authoritative_best_f1 > -1.0:
        best_f1 = authoritative_best_f1
    else:
        try:
            with open(latest_dir / "checkpoint_meta.json") as f:
                meta = json.load(f)
            best_f1 = float(meta.get("best_full_val_macro_f1", -1.0))
        except Exception:
            best_f1 = -1.0

    return latest_dir, latest_epoch, latest_step, best_f1


def load_checkpoint(checkpoint_path: Path, model, optimizer=None, scheduler=None, scaler=None):
    \"\"\"Restore full training state from a saved Drive checkpoint.\"\"\"
    print(f"\\n[RESUMING TRAINING]: Loading state from {checkpoint_path}...")
    
    from transformers import AutoModelForSequenceClassification
    restored = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.load_state_dict(restored.state_dict())
    
    state_file = checkpoint_path / "training_state.pt"
    state = torch.load(state_file, map_location='cpu')
    
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state and state["scheduler_state_dict"]:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in state and state["scaler_state_dict"]:
        scaler.load_state_dict(state["scaler_state_dict"])
        
    epoch = state.get("epoch", 0)
    step = state.get("step", 0)
    print(f"[RESUME SUCCESSFUL]: Restored Epoch {epoch}, Step {step}")
    return epoch, step""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 10: Evaluation & Watch Metric Function (Absolute Gap Bug Fixed)
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 9: Evaluation Function & Bidirectional Petitioner Watch Metric Tracker
# ==============================================================================
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

def evaluate_model(model, dataloader, device, class_weights=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    loss_fn = (
        torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else torch.nn.CrossEntropyLoss()
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    avg_loss = total_loss / max(n_batches, 1)

    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=CANONICAL_LABELS,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=CANONICAL_LABELS,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_loss": avg_loss,
        "report_str": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def track_petitioner_watch_metric(eval_results: dict):
    \"\"\"
    Bidirectional analysis of Arguments of Petitioner vs. Arguments of Respondent.
    Uses absolute gap and explicitly identifies which class is lagging.
    \"\"\"
    p_metrics = eval_results["report_dict"].get("Arguments of Petitioner", {})
    r_metrics = eval_results["report_dict"].get("Arguments of Respondent", {})

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

    print("\\n" + "="*70)
    print("WATCH METRIC: Arguments of Petitioner vs. Arguments of Respondent")
    print("="*70)
    print(f"  Arguments of Petitioner: Precision={p_prec:.4f} | Recall={p_rec:.4f} | F1={p_f1:.4f}")
    print(f"  Arguments of Respondent: Precision={r_prec:.4f} | Recall={r_rec:.4f} | F1={r_f1:.4f}")
    print(f"  Absolute Recall Gap: {abs_rec_gap:.4f}")
    print(f"  Absolute F1 Gap:     {abs_f1_gap:.4f}")
    print(f"  Lagging Class:       {lagging_class} (by {abs_rec_gap*100:.2f}%)")

    if abs_rec_gap > 0.15:
        if lagging_class == "Arguments of Petitioner":
            print(\"\"\"
  [DIAGNOSTIC ADVISORY]:
  Arguments of Petitioner recall is lagging significantly behind Respondent (>15% gap).
  This confirms structural advocate-argument overlap where Petitioner arguments
  are being absorbed into Respondent.
  RECOMMENDATION: Apply an explicit manual weight floor (>= 4.5) for Petitioner in training.
\"\"\")
        else:
            print(\"\"\"
  [DIAGNOSTIC ADVISORY]:
  Arguments of Respondent recall is lagging significantly behind Petitioner (>15% gap).
  RECOMMENDATION: Consider raising Respondent's weight floor in training.
\"\"\")
    else:
        print(\"\\n  [STATUS]: Petitioner and Respondent recalls are well-balanced within 15% tolerance.\")
    print("="*70)


def early_warning_evaluate(model, dataloader, device, interval_loss, step, epoch):
    \"\"\"
    Fast mid-epoch evaluation on a fixed stratified slice (~300/class, ~2,100 samples).
    Executes in ~1-2 seconds on GPU.
    Prints a compact table of Macro F1, interval train loss, and per-class recall.
    \"\"\"
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=CANONICAL_LABELS,
        output_dict=True,
        zero_division=0,
    )

    print(f"\\n" + "-"*75)
    print(f"[EARLY-WARNING EVALUATION - Epoch {epoch}, Step {step}]")
    log_ram_usage("Early-Warning Step Checkpoint")
    print(f"  Last Interval Avg Train Loss: {interval_loss:.4f} | Quick Macro F1: {macro_f1:.4f}")
    print(f"  {'Class':<28s} {'Recall':>9s} {'F1':>9s} {'Support':>9s}")
    print("  " + "-"*58)
    for lbl in CANONICAL_LABELS:
        c_info = report_dict.get(lbl, {})
        rec = c_info.get("recall", 0.0)
        f1 = c_info.get("f1-score", 0.0)
        sup = c_info.get("support", 0)
        print(f"  {lbl:<28s} {rec:9.4f} {f1:9.4f} {int(sup):9d}")
    print("-" * 75)

    return {
        "macro_f1": macro_f1,
        "interval_loss": interval_loss,
        "report_dict": report_dict,
    }""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 11: Training Loop with Deterministic Shuffle & Mid-Epoch Resume
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 10: Multi-Epoch Training Loop with Mid-Epoch Resume & Step Checkpointing
# ==============================================================================
import time
from torch.utils.data import Subset
from transformers import get_linear_schedule_with_warmup

class SimulatedInterruption(Exception):
    \"\"\"Raised when SIMULATE_INTERRUPT_STEP triggers to halt execution immediately.\"\"\"
    pass

total_steps_total = (len(train_dataset) // BATCH_SIZE) * EPOCHS
warmup_steps = int(total_steps_total * WARMUP_RATIO)

use_fused = torch.cuda.is_available()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fused=use_fused,
)
print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, fused={use_fused})")

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps_total,
)

# ── CHECK FOR EXISTING CHECKPOINT IN DRIVE ───────────────────────────────────
latest_dir, resume_epoch, resume_step, best_macro_f1 = find_latest_checkpoint(DRIVE_CHECKPOINT_DIR)
start_epoch = 1
start_step = 0
print(f"best_macro_f1 (full-val only, from checkpoint): {best_macro_f1:.4f}")

if latest_dir:
    print(f"\\n[FOUND VALID CHECKPOINT]: {latest_dir.name} (Epoch {resume_epoch}, Step {resume_step})")
    load_checkpoint(latest_dir, model, optimizer, scheduler, scaler)
    if resume_step > 0 and resume_step < (len(train_dataset) // BATCH_SIZE):
        start_epoch = resume_epoch
        start_step = resume_step
        print(f"Resuming MID-EPOCH at Epoch {start_epoch}, Step {start_step + 1}...")
    else:
        start_epoch = resume_epoch + 1
        start_step = 0
        print(f"Resuming at start of Epoch {start_epoch}...")

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

total_sentences_trained = 0
total_step_time = 0.0          # Pure forward + backward + optimizer step time (excludes Drive save I/O)
total_steps_executed = 0

def generate_epoch_indices(dataset, epoch, batch_size, mode):
    rng = np.random.default_rng(seed=42 + epoch)
    if mode == "random":
        return rng.permutation(len(dataset)).tolist()

    if mode == "grouped50":
        mega_chunk_mult = 50
    elif mode == "grouped10":
        mega_chunk_mult = 10
    else:
        raise ValueError(f"Unknown BATCHING_MODE: '{mode}'. Must be 'random', 'grouped50', or 'grouped10'.")

    indices = rng.permutation(len(dataset)).tolist()
    mega_chunk_size = mega_chunk_mult * batch_size

    batches = []
    tail = []
    for i in range(0, len(indices), mega_chunk_size):
        chunk = indices[i : i + mega_chunk_size]
        chunk.sort(key=lambda idx: dataset.lengths[idx])
        for j in range(0, len(chunk), batch_size):
            b = chunk[j : j + batch_size]
            if len(b) == batch_size:
                batches.append(b)
            else:
                tail.extend(b)

    rng.shuffle(batches)
    return [idx for b in batches for idx in b] + tail

# ── EPOCH LOOP ────────────────────────────────────────────────────────────────
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    n_train_batches = 0
    t_epoch_start = time.time()

    epoch_indices = generate_epoch_indices(train_dataset, epoch, BATCH_SIZE, BATCHING_MODE)
    total_epoch_steps = len(epoch_indices) // BATCH_SIZE

    # If resuming mid-epoch on start_epoch:
    if epoch == start_epoch and start_step > 0:
        print(f"\\n[DETERMINISTIC SKIP]: Skipping first {start_step:,} steps ({start_step * BATCH_SIZE:,} samples) of Epoch {epoch}...")
        epoch_indices = epoch_indices[start_step * BATCH_SIZE :]
        initial_step = start_step
    else:
        initial_step = 0

    # Build DataLoader for remaining epoch indices
    epoch_subset = Subset(train_dataset, epoch_indices)
    epoch_loader = DataLoader(
        epoch_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Already deterministically ordered!
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )

    progress = tqdm(epoch_loader, desc=f"Epoch {epoch}/{EPOCHS}", initial=initial_step, total=total_epoch_steps)
    current_step = initial_step

    # Track padding efficiency during Epoch 1
    epoch_batch_padded_lens = []
    epoch_real_tokens = 0
    epoch_pad_tokens = 0

    interval_train_loss = 0.0
    interval_step_count = 0

    for batch in progress:
        current_step += 1
        t_step_start = time.time()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if epoch == 1:
            batch_seq_len = input_ids.shape[1]
            epoch_batch_padded_lens.append(batch_seq_len)
            real_cnt = int(attention_mask.sum().item())
            pad_cnt = int((attention_mask == 0).sum().item())
            epoch_real_tokens += real_cnt
            epoch_pad_tokens += pad_cnt

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        t_step_end = time.time()
        step_duration = t_step_end - t_step_start
        total_step_time += step_duration
        total_steps_executed += 1

        loss_val = loss.item()
        total_train_loss += loss_val
        n_train_batches += 1
        interval_train_loss += loss_val
        interval_step_count += 1

        progress.set_postfix(loss=f"{loss_val:.4f}", step=current_step)

        # ── MID-EPOCH STEP CHECKPOINTING & EARLY-WARNING EVALUATION ────────
        if current_step % CHECKPOINT_EVERY_STEPS == 0 and current_step < total_epoch_steps:
            avg_interval_loss = interval_train_loss / max(interval_step_count, 1)

            # Fast early-warning evaluation on ~300/class slice (~1-2 seconds)
            quick_metrics = early_warning_evaluate(
                model=model,
                dataloader=val_quick_loader,
                device=device,
                interval_loss=avg_interval_loss,
                step=current_step,
                epoch=epoch,
            )

            save_checkpoint(
                epoch=epoch,
                step=current_step,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                macro_f1=quick_metrics["macro_f1"],
                all_metrics=quick_metrics,
                checkpoint_dir=DRIVE_CHECKPOINT_DIR,
                is_step_checkpoint=True,
                best_full_val_macro_f1=best_macro_f1,
            )

            # Reset interval loss counters
            interval_train_loss = 0.0
            interval_step_count = 0

            # Put model back in train mode
            model.train()

        # ── OPTIONAL SIMULATED INTERRUPTION (FOR RESUME TESTING) ─────────────
        if SIMULATE_INTERRUPT_STEP is not None and current_step == SIMULATE_INTERRUPT_STEP:
            target_ckpt_step = current_step - (current_step % CHECKPOINT_EVERY_STEPS)
            print(f"\\n" + "="*75)
            print(f"[SIMULATED INTERRUPTION TRIGGERED]: Halting execution at Step {current_step}.")
            print(f"Step checkpoint 'checkpoint_epoch_{epoch}_step_{target_ckpt_step}' is saved in Drive.")
            print(f"CRITICAL: Set SIMULATE_INTERRUPT_STEP = None in Cell 2 before re-running to resume!")
            print(f"When re-run, training will resume seamlessly from Step {target_ckpt_step + 1}.")
            print("="*75)
            raise SimulatedInterruption(
                f"Simulated interruption triggered at Step {current_step}. Reset SIMULATE_INTERRUPT_STEP = None in Cell 2 before resuming!"
            )

    # Guard: only proceed to epoch-boundary save if all steps completed
    if current_step < total_epoch_steps:
        print(f"\\n[ABORTED]: Epoch {epoch} stopped prematurely at Step {current_step}/{total_epoch_steps}. Skipping epoch boundary save.")
        break

    epoch_elapsed = time.time() - t_epoch_start
    sentences_in_epoch = len(epoch_subset)
    total_sentences_trained += sentences_in_epoch
    avg_train_loss = total_train_loss / max(n_train_batches, 1)

    epoch_pure_speed = sentences_in_epoch / max(total_step_time, 1e-4)
    epoch_pure_steps_s = total_steps_executed / max(total_step_time, 1e-4)
    print(f"\\n--- Epoch {epoch}/{EPOCHS} Finished ---")
    print(f"Average Train Loss:     {avg_train_loss:.4f} | Total Wall-Clock: {epoch_elapsed:.1f}s")
    print(f"Step-Only Compute Time: {total_step_time:.1f}s ({epoch_pure_speed:.1f} sentences/sec, {epoch_pure_steps_s:.2f} steps/sec)")

    # ── PRINT PADDING EFFICIENCY (EPOCH 1) ──────────────────────────────────
    if epoch == 1 and epoch_batch_padded_lens:
        mean_padded_len = float(np.mean(epoch_batch_padded_lens))
        pad_to_real_ratio = epoch_pad_tokens / max(epoch_real_tokens, 1)
        pad_pct = (epoch_pad_tokens / max(epoch_real_tokens + epoch_pad_tokens, 1)) * 100
        print(f"\\n" + "="*75)
        print(f"PADDING EFFICIENCY (Epoch 1 with BATCHING_MODE='{BATCHING_MODE}'):")
        print(f"="*75)
        print(f"  Mean Padded Sequence Length: {mean_padded_len:.1f} tokens / batch (Max limit: {MAX_SEQ_LENGTH})")
        print(f"  Total Real Tokens:           {epoch_real_tokens:,}")
        print(f"  Total Pad Tokens:            {epoch_pad_tokens:,} ({pad_pct:.1f}% of all tokens)")
        print(f"  Pad-to-Real Token Ratio:     {pad_to_real_ratio:.3f}x")
        print("="*75)

    # ── VALIDATION EVALUATION ─────────────────────────────────────────────────
    print(f"Running validation evaluation for Epoch {epoch}...")
    t_val_start = time.time()
    eval_results = evaluate_model(model, val_loader, device, class_weights)
    val_elapsed = time.time() - t_val_start
    val_speed = len(val_dataset) / max(val_elapsed, 1e-4)
    print(f"Val Loss:     {eval_results['avg_loss']:.4f}")
    print(f"Macro F1:     {eval_results['macro_f1']:.4f}")
    print(f"Weighted F1:  {eval_results['weighted_f1']:.4f}")
    print(f"Val Duration: {val_elapsed:.1f}s ({val_speed:.1f} sentences/sec)")
    print(f"\\nPer-class report:\\n{eval_results['report_str']}")

    track_petitioner_watch_metric(eval_results)

    # ── SAVE EPOCH BOUNDARY CHECKPOINT ────────────────────────────────────────
    if eval_results["macro_f1"] > best_macro_f1:
        best_macro_f1 = eval_results["macro_f1"]
        best_dir = DRIVE_CHECKPOINT_DIR / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        # Save authoritative best_meta.json alongside best_model
        with open(best_dir / "best_meta.json", "w") as f:
            json.dump({"best_full_val_macro_f1": best_macro_f1, "epoch": epoch}, f, indent=2)
        print(f"[NEW BEST MODEL]: Saved to {best_dir} (Full-Val Macro F1 = {best_macro_f1:.4f})")

    save_checkpoint(
        epoch=epoch,
        step=0,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        macro_f1=eval_results['macro_f1'],
        all_metrics=eval_results,
        checkpoint_dir=DRIVE_CHECKPOINT_DIR,
        is_step_checkpoint=False,
        best_full_val_macro_f1=best_macro_f1,
    )

    # Reset start_step after first completed epoch
    start_step = 0

# ── STEP-ONLY SPEED & UNBIASED FULL-RUN PROJECTION ───────────────────────────
pure_train_speed = total_sentences_trained / max(total_step_time, 1e-4)
pure_steps_per_sec = total_steps_executed / max(total_step_time, 1e-4)
avg_ckpt_save_sec = (sum(checkpoint_save_times) / len(checkpoint_save_times)) if len(checkpoint_save_times) > 0 else 0.0

FULL_TRAIN_SAMPLES = 1120448
FULL_VAL_SAMPLES = 293362
FULL_EPOCHS = 2
FULL_CHECKPOINT_EVERY_STEPS = 3000

# Total checkpoints in full run:
# 1,120,448 // 32 = 35,014 steps per epoch.
# 35,014 // 3,000 = 11 mid-epoch step checkpoints + 1 epoch boundary checkpoint = 12 checkpoints per epoch.
total_ckpts_full = ( (FULL_TRAIN_SAMPLES // (BATCH_SIZE * FULL_CHECKPOINT_EVERY_STEPS)) + 1 ) * FULL_EPOCHS

# Unbiased projection: step-only training time + validation + pretokenization + checkpoint I/O counted ONCE
proj_pretok_sec = (FULL_TRAIN_SAMPLES + FULL_VAL_SAMPLES) / max(pretok_speed, 1.0)
proj_pure_train_sec = (FULL_TRAIN_SAMPLES * FULL_EPOCHS) / max(pure_train_speed, 1.0)
proj_val_sec = (FULL_VAL_SAMPLES * FULL_EPOCHS) / max(val_speed, 1.0)
proj_ckpt_io_sec = total_ckpts_full * avg_ckpt_save_sec
proj_total_sec = proj_pretok_sec + proj_pure_train_sec + proj_val_sec + proj_ckpt_io_sec
proj_total_hours = proj_total_sec / 3600

print(f"\\n" + "="*75)
print(f"TIMING INSTRUMENTATION SUMMARY (Step-Only Throughput & Pure Projection):")
print(f"="*75)
print(f"  Training Sentences Processed:  {total_sentences_trained:,}")
print(f"  Step-Only Compute Time:        {total_step_time:.1f}s (excludes Drive save I/O and val)")
print(f"  STEP-ONLY TRAINING SPEED:      {pure_train_speed:.1f} sentences/sec ({pure_steps_per_sec:.2f} steps/sec)")
print(f"  MEASURED VALIDATION SPEED:     {val_speed:.1f} sentences/sec")
print(f"  MEASURED PRE-TOKENIZATION:     {pretok_speed:.1f} sentences/sec ({pretok_duration:.2f}s for sample)")
print(f"  CHECKPOINT DRIVE WRITE TIME:   {avg_ckpt_save_sec:.2f}s average per checkpoint (n={len(checkpoint_save_times)})")
print(f"\\n" + "-"*75)
print(f"PROJECTED FULL-RUN WALL-CLOCK (2 Epochs across {FULL_TRAIN_SAMPLES:,} sentences):")
print(f"  Computed as: pure step training + validation + pre-tokenization + Drive I/O (counted once)")
print(f"-"*75)
print(f"  Pre-tokenization (1.41M sent): {proj_pretok_sec/60:.1f} min ({proj_pretok_sec:.0f}s)")
print(f"  Pure Training (2 Epochs):      {proj_pure_train_sec/60:.1f} min ({proj_pure_train_sec/3600:.2f} hours)")
print(f"  Validation (2 Epochs):         {proj_val_sec/60:.1f} min ({proj_val_sec/3600:.2f} hours)")
print(f"  Drive Checkpoint I/O ({total_ckpts_full} saves): {proj_ckpt_io_sec/60:.1f} min ({proj_ckpt_io_sec:.0f}s)")
print(f"  -------------------------------------------------------------------------")
print(f"  TOTAL ESTIMATED FULL RUN:      {proj_total_hours:.2f} HOURS ({proj_total_sec/60:.1f} MINUTES)")
print(f"="*75)""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 12: Final Comparison Table vs Smoke Test Run 3
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 11: Final Evaluation & Direct Comparison Against Smoke Test Run 3
# ==============================================================================
best_dir = DRIVE_CHECKPOINT_DIR / "best_model"
if best_dir.exists() and (best_dir / "config.json").exists():
    print(f"Loading best checkpoint from: {best_dir}...")
    final_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
else:
    # Fallback: find the latest epoch checkpoint
    print(f"[WARNING]: best_model not found at {best_dir}. Falling back to latest epoch checkpoint.")
    epoch_ckpts = sorted(
        [d for d in DRIVE_CHECKPOINT_DIR.glob("checkpoint_epoch_*") if d.is_dir() and "step" not in d.name],
        key=lambda d: int(d.name.split("_")[-1]),
        reverse=True,
    )
    if epoch_ckpts:
        fallback_dir = epoch_ckpts[0]
        print(f"Falling back to: {fallback_dir.name}")
        final_model = AutoModelForSequenceClassification.from_pretrained(fallback_dir).to(device)
    else:
        raise FileNotFoundError(f"No best_model or epoch checkpoint found in {DRIVE_CHECKPOINT_DIR}")

final_eval = evaluate_model(final_model, val_loader, device, class_weights)

SMOKE_TEST_RUN3_F1 = {
    "Facts": 0.5031,
    "Issue": 0.7544,
    "Arguments of Petitioner": 0.1586,
    "Arguments of Respondent": 0.4876,
    "Reasoning": 0.3927,
    "Decision": 0.7692,
    "None": 0.2718,
    "Macro F1": 0.4768,
}

print("\\n" + "="*85)
print("FINAL PERFORMANCE COMPARISON: Smoke Test (Run 3) vs. Colab Training")
print("="*85)
print(f"{'Class':<28s} {'Smoke F1':>10s} {'Current Prec':>13s} {'Current Rec':>12s} {'Current F1':>11s} {'Delta F1':>11s}")
print("-"*85)

for label in CANONICAL_LABELS:
    smoke_f1 = SMOKE_TEST_RUN3_F1.get(label, 0.0)
    p_info = final_eval["report_dict"].get(label, {})
    full_prec = p_info.get("precision", 0.0)
    full_rec = p_info.get("recall", 0.0)
    full_f1 = p_info.get("f1-score", 0.0)
    delta = full_f1 - smoke_f1
    delta_str = f"{delta:+.4f}"

    print(f"{label:<28s} {smoke_f1:10.4f} {full_prec:13.4f} {full_rec:12.4f} {full_f1:11.4f} {delta_str:>11s}")

print("-"*85)
smoke_macro = SMOKE_TEST_RUN3_F1["Macro F1"]
full_macro = final_eval["macro_f1"]
macro_delta = full_macro - smoke_macro
print(f"{'Macro F1':<28s} {smoke_macro:10.4f} {'-':>13s} {'-':>12s} {full_macro:11.4f} {macro_delta:>+11.4f}")
print("="*85)

final_report_data = {
    "dry_run": bool(DRY_RUN),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_seq_length": MAX_SEQ_LENGTH,
    "learning_rate": LEARNING_RATE,
    "best_macro_f1": best_macro_f1,
    "final_eval": final_eval,
    "smoke_test_comparison": {
        "smoke_test_run3_macro_f1": smoke_macro,
        "current_run_macro_f1": full_macro,
        "macro_f1_gain": macro_delta,
    },
    "batching_mode": BATCHING_MODE,
    "pure_step_train_speed_sent_sec": float(pure_train_speed),
    "pure_steps_per_sec": float(pure_steps_per_sec),
    "measured_val_speed_sent_sec": float(val_speed),
    "measured_pretok_speed_sent_sec": float(pretok_speed),
    "measured_avg_ckpt_save_sec": float(avg_ckpt_save_sec),
    "projected_full_run_hours": float(proj_total_hours),
    "class_weights": class_weights.tolist(),
}

report_save_path = DRIVE_REPORTS_DIR / "training_report.json"
with open(report_save_path, "w", encoding="utf-8") as f:
    json.dump(final_report_data, f, indent=2)
print(f"\\n[SUCCESS] Training Report saved to Drive: {report_save_path}")""")

    # ──────────────────────────────────────────────────────────────────────────
    # CELL 13: Inference Demo & Sanity Test
    # ──────────────────────────────────────────────────────────────────────────
    add_code("""# ==============================================================================
# Cell 12: Interactive Inference Demo & Real Judgment Test
# ==============================================================================
def predict_sentence_role(sentence: str, model, tokenizer):
    model.eval()
    encoding = tokenizer(
        sentence,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_idx]
        confidence = probs[pred_idx]

    return pred_label, confidence, probs

test_samples = [
    ("The appellant was charged under Section 302 of the Indian Penal Code for the alleged murder of the deceased on the night of 14th August 2012.", "Facts"),
    ("Whether the High Court was justified in setting aside the concurrent findings of fact recorded by the trial court and the first appellate court.", "Issue"),
    ("Learned senior counsel appearing for the appellant vehemently argued that the prosecution failed to prove the chain of circumstantial evidence beyond reasonable doubt.", "Arguments of Petitioner"),
    ("On the other hand, the learned standing counsel for the State of Maharashtra submitted that the recovery of the weapon and the forensic report establish guilt.", "Arguments of Respondent"),
    ("We have carefully considered the rival submissions and scrutinised the evidence on record including the ocular testimony of PW-1 and PW-3.", "Reasoning"),
    ("For the reasons aforesaid, the appeal is allowed and the judgment of the High Court is hereby set aside.", "Decision"),
    ("CRIMINAL APPEAL NO. 1421 OF 2019", "None"),
]

print("\\n" + "="*75)
print("INFERENCE DEMO: Real Judgment Sentence Predictions")
print("="*75)
for sent, true_role in test_samples:
    pred_role, conf, _ = predict_sentence_role(sent, final_model, tokenizer)
    match = "[CORRECT]" if pred_role == true_role else "[MISMATCH]"
    preview = sent[:90]
    print(f"\\nText: '{preview}...'")
    print(f"  True Role:      {true_role}")
    print(f"  Predicted Role: {pred_role} (Confidence: {conf*100:.1f}%) {match}")
print("="*75)""")

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": cells
    }

    output_path = Path("notebooks/InLegalBERT_FineTuning_Colab.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[SUCCESS] Generated Colab notebook at: {output_path} ({len(cells)} cells)")

if __name__ == "__main__":
    create_notebook()
