# Google Colab Fine-Tuning Guide: InLegalBERT

This guide explains how to run the full GPU fine-tuning for **Module 2: InLegalBERT Rhetorical Role Classification** using the prepared notebook: [`notebooks/InLegalBERT_FineTuning_Colab.ipynb`](file:///d:/NLP%20Project/notebooks/InLegalBERT_FineTuning_Colab.ipynb).

---

## 1. Prerequisites & Google Drive Setup

1. **Upload Dataset to Google Drive:**
   In your Google Drive root, create a folder named `JurisGraph` and upload your dataset files into a `legalDataset/` subfolder:
   ```
   My Drive/
   └── JurisGraph/
       └── legalDataset/
           ├── train.csv  (1,120,448 rows)
           ├── val.csv    (293,362 rows)
           └── test.csv   (149,868 rows)
   ```
2. **Output Artifacts Location & Isolation:**
   Dry-run and full-run checkpoints are strictly isolated:
   ```
   My Drive/
   └── JurisGraph/
       └── artifacts/
           ├── model_checkpoint/
           │   ├── _dry_run/                     <-- Isolated dry-run checkpoints (1% sample)
           │   │   ├── checkpoint_epoch_1_step_100/
           │   │   ├── checkpoint_epoch_1/
           │   │   └── best_model/
           │   ├── checkpoint_epoch_1_step_3000/ <-- Mid-epoch step checkpoints (only 2 kept)
           │   ├── checkpoint_epoch_1/
           │   ├── checkpoint_epoch_2/
           │   └── best_model/
           └── eval_reports/
               ├── _dry_run/
               │   └── training_report.json
               └── training_report.json
   ```

---

## 2. Drive Cleanup Instructions (Before Full Run)

> [!IMPORTANT]
> **Manual Drive Deletion Required:**
> If you previously ran an unisolated dry run, delete the following contaminated folders from your Google Drive before starting a fresh run:
> 1. `My Drive/JurisGraph/artifacts/model_checkpoint/checkpoint_epoch_1/`
> 2. `My Drive/JurisGraph/artifacts/model_checkpoint/best_model/`
> 3. `My Drive/JurisGraph/artifacts/eval_reports/training_report_full.json` (or `training_report.json`)
>
> Checkpoints now store run configuration (`dry_run`, `train_samples`, `epochs_configured`) in `checkpoint_meta.json`. Any checkpoint whose configuration does not match the active run will be automatically rejected.

---

## 3. Launching in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **File -> Upload notebook** and select `InLegalBERT_FineTuning_Colab.ipynb`.
3. Set Hardware Accelerator:
   * Navigate to **Runtime -> Change runtime type**.
   * Select **Hardware accelerator: GPU** (T4, V100, or A100).
   * Click **Save**.

---

## 4. Notebook Structure & Flow

| Cell | Purpose | Key Details |
|---|---|---|
| **Cell 1** | Header & Architecture | Problem formulation, 7 canonical rhetorical roles |
| **Cell 2** | Environment & GPU Check | Verifies CUDA device, installs dependencies |
| **Cell 3** | Drive Mount & Path Setup | Mounts `/content/drive`, isolates `_dry_run/` folder, caches CSVs to NVMe |
| **Cell 4** | Hyperparameters & Labels | Sets `BATCH_SIZE=32`, `LR=2e-5`, `MAX_SEQ_LENGTH=128`, `EPOCHS=2` |
| **Cell 5** | Strict Data Loading | Uses `keep_default_na=False` to prevent `None` label corruption |
| **Cell 6** | Sqrt-Dampened Loss Weights | Computes `--weight_mode full` ($w_c = \sqrt{N / (7 \cdot N_c)}$, capped at $6.0\times$) |
| **Cell 7** | **Pre-Tokenization & Collator** | Pre-tokenizes once; sorts validation set by length (no shuffle) for zero-padding eval |
| **Cell 8** | Model & Mixed Precision | Loads `law-ai/InLegalBERT` with `torch.cuda.amp.GradScaler()` |
| **Cell 9** | **Step Checkpoint & Resume** | Saves every 3,000 steps (keeps 2), rejects mismatched configs |
| **Cell 10**| Evaluation & Watch Metric | Classification report + bidirectional **Petitioner vs Respondent** tracker |
| **Cell 11**| **Training Loop & Length-Grouping**| $50 \times \text{BATCH\_SIZE}$ mega-chunk length grouping; prints epoch 1 padding efficiency; step resume |
| **Cell 12**| **Baseline Comparison** | Direct side-by-side comparison table against Smoke Test Run 3 |
| **Cell 13**| Inference Demo | Tests real Supreme Court sentences across all 7 roles |

---

## 5. Pre-Flight Dry Run vs. Full Run

The notebook is configured with a built-in `DRY_RUN` toggle in **Cell 2**:

```python
DRY_RUN = True           # Kept as True until you confirm launching the full run
DRY_RUN_FRACTION = 0.01  # 1% stratified subset (~11,200 train, ~2,933 val)
BATCHING_MODE = "grouped10"  # Options: "random", "grouped50", "grouped10" (default)
EPOCHS = 1 if DRY_RUN else 2
CHECKPOINT_EVERY_STEPS = 100 if DRY_RUN else 3000
SIMULATE_INTERRUPT_STEP = None  # Set e.g. 150 to test interrupt, reset to None before resuming
```

### Dry Run Protocol:
1. Keep `DRY_RUN = True` and run all cells.
2. The dry run executes on ~11,200 train sentences:
   - Checkpoints and reports are written strictly into `artifacts/model_checkpoint/_dry_run/`.
   - Mid-epoch checkpoints are saved every 100 steps, keeping only the 2 most recent.
   - **Early-Warning Evaluation:** At every mid-epoch step checkpoint, evaluates on a fixed, class-stratified slice of the validation set (up to 300 per class, ~2,100 total) in ~2-3 seconds, printing a compact recall table and interval train loss.
   - **Step-Only Throughput:** Measures compute-only throughput (excluding Drive checkpoint writes and validation time) to project full-run wall clock accurately without double counting I/O.
   - Uses `torch.optim.AdamW(..., fused=True)` on CUDA for reduced kernel launch overhead.
   - Petitioner watch metric reports the absolute recall gap and flags any lagging class.

### Exercising Mid-Epoch Step Resume on Colab:
To test real step-level recovery on Colab:
* **Option A (Manual Interrupt):** Run Cell 10. Once you see `[CHECKPOINT SAVED]: Step 100` printed, click **Runtime -> Interrupt execution** (or Stop) around Step 130–150. Restart runtime (`Runtime -> Restart runtime`), re-mount Drive, and re-run Cell 10. It will find `checkpoint_epoch_1_step_100`, restore training state, skip the first 100 steps deterministically, and resume from Step 101.
* **Option B (Configured Interrupt):** In Cell 2, set `SIMULATE_INTERRUPT_STEP = 150`. Training will raise `SimulatedInterruption` and halt immediately at Step 150, strictly skipping epoch-boundary saves. **Crucial:** Reset `SIMULATE_INTERRUPT_STEP = None` before re-running Cell 10 to allow training to complete.

### Launching the Full Run:
Once confirmed:
1. In **Cell 2**, set `DRY_RUN = False` (and ensure `SIMULATE_INTERRUPT_STEP = None`).
2. Run all cells.

---

## 6. Throughput, Token-Lengths & Wall-Clock Estimates

### Real Token-Length Distribution (Measured on 50,000 Train Sentences):
* **Mean Token Length:** **39.32 tokens**
* **Median Token Length:** **33.0 tokens**
* **95th Percentile (p95):** **89.0 tokens**
* **99th Percentile (p99):** **139.0 tokens**
* **Truncation Rate at 128 Tokens:** **1.35%** (675 out of 50,000 sentences)
* **Recommendation:** Since truncation is only 1.35% (well below the 2% threshold), **`MAX_SEQ_LENGTH = 128` is optimal**. Over 98.65% of legal sentences fit completely without any cutoff.

### Why Length-Grouped Batching is Essential:
* **Naive Random Shuffling:** With batch size 32, almost every randomly assembled batch contains at least one sentence near $p95$ ($89$ tokens) or $p99$ ($139$ tokens, clipped to 128). Because `DataCollatorWithPadding` pads each batch to its longest sentence, random batching yields an average padded length of $\approx 111$ tokens (only $\approx 1.15\times$ speedup over static 128).
* **Length-Grouped Batching ($50 \times \text{BATCH\_SIZE}$ mega-chunks):** By sorting within 1,600-sample chunks before cutting into batches and shuffling batch order, sentences of similar lengths are grouped together. This brings the mean padded length down to $\approx 40\text{–}44$ tokens, delivering an actual **$\approx 2.5\times$ speedup over random shuffling** and **$\approx 2.9\text{–}3.2\times$ speedup over static 128**.
* **Length-Sorted Validation (No Shuffle):** Sorting the validation set by length yields near-perfect padding efficiency during evaluation with zero impact on order-independent metrics.

### Wall-Clock Projection Formula (Empirical Rebase):
The full-run wall-clock time is dynamically projected from dry-run measurements rather than hardcoded assumptions:
$$\text{Time}_{\text{total}} = \frac{N_{\text{total, pretok}}}{S_{\text{pretok}}} + \frac{N_{\text{train}} \times 2}{S_{\text{train, measured}}} + \frac{N_{\text{val}} \times 2}{S_{\text{val, measured}}} + (N_{\text{ckpts}} \times T_{\text{ckpt\_save, measured}})$$
* $S_{\text{train, measured}}$: Empirical sentences/sec measured during GPU training with length grouping.
* $S_{\text{val, measured}}$: Empirical sentences/sec measured during validation.
* $T_{\text{ckpt\_save, measured}}$: Empirical Drive write time per checkpoint measured over the dry-run checkpoints.

---

## 7. Google Drive Storage Footprint

Each InLegalBERT checkpoint directory contains:
* Model weights (`model.safetensors` / `pytorch_model.bin`): ~440 MB
* PyTorch training state (`training_state.pt`, AdamW fp32 optimizer states): ~880 MB
* Tokenizer, config, & metadata: ~1 MB
* **Total per checkpoint folder:** $\approx \mathbf{1.32\text{ GB}}$

### Storage Consumption During Full Run (`EPOCHS = 2`):
* **2 Mid-Epoch Step Checkpoints** (auto-pruned to 2 most recent): $2 \times 1.32\text{ GB} = \mathbf{2.64\text{ GB}}$
* **2 Epoch Boundary Checkpoints** (`checkpoint_epoch_1`, `checkpoint_epoch_2`): $2 \times 1.32\text{ GB} = \mathbf{2.64\text{ GB}}$
* **Best Model Checkpoint** (model weights only, no optimizer states): $\approx \mathbf{0.44\text{ GB}}$
* **Reports and JSON logs:** $<5\text{ MB}$
* **Total Peak Drive Storage:** $\approx \mathbf{5.72\text{ GB}}$

*(Leaves > 9 GB free on a standard 15 GB Google Drive account.)*

---

## 8. Key Metrics to Watch

### A. Arguments of Petitioner vs. Arguments of Respondent
* Evaluates the absolute recall gap: `abs(Recall_Respondent - Recall_Petitioner)`.
* If gap $> 15\%$, flags whether Petitioner or Respondent is lagging and prints a diagnostic advisory.

### B. Comparison Baseline: Run 3 Smoke Test

| Rhetorical Role | Smoke Test (Run 3, CPU Subsample) | Full GPU Run Target |
|---|---|---|
| **Decision** | 0.7692 | $\ge 0.82$ |
| **Issue** | 0.7544 | $\ge 0.80$ |
| **Facts** | 0.5031 | $\ge 0.70$ |
| **Arguments of Respondent** | 0.4876 | $\ge 0.65$ |
| **Reasoning** | 0.3927 | $\ge 0.65$ |
| **None** | 0.2718 | $\ge 0.80$ |
| **Arguments of Petitioner** | 0.1586 | $\ge 0.55$ *(Watch Metric)* |
| **Macro F1** | **0.4768** | $\ge \mathbf{0.72}$ |
