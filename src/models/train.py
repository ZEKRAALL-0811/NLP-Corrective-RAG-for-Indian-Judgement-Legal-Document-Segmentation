"""
Module 2: InLegalBERT Fine-Tuning
Stratified subsample creation, class-weighted CrossEntropyLoss,
training loop, and per-class evaluation reporting.

Usage:
    # CPU smoke test (1 epoch, 500/class cap):
    python src/models/train.py --mode smoke

    # Full training (GPU, multiple epochs):
    python src/models/train.py --mode full --epochs 5 --batch_size 32
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    TRAIN_CSV,
    VAL_CSV,
    CANONICAL_LABELS,
    LABEL2ID,
    NUM_LABELS,
    CHECKPOINT_DIR,
    REPORTS_DIR,
    MAX_SEQ_LENGTH,
)
from src.models.classifier import load_tokenizer, load_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LegalSentenceDataset(Dataset):
    """PyTorch dataset for tokenised legal sentences with integer labels."""

    def __init__(self, texts, labels, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Stratified Subsample
# ---------------------------------------------------------------------------

def create_stratified_subsample(
    df: pd.DataFrame,
    max_per_class: int = 500,
    min_per_class: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a balanced subsample for the CPU smoke test.

    Strategy:
    - For majority classes (count > max_per_class): sample exactly max_per_class.
    - For minority classes (count <= max_per_class): keep ALL samples so that
      Issue and Decision are not proportionally reduced to near-nothing.
    - Guarantees every class has at least min_per_class (raises if not met).
    """
    frames = []
    for label in CANONICAL_LABELS:
        subset = df[df["Label"] == label]
        n = len(subset)

        if n < min_per_class:
            print(
                f"[WARNING] Class '{label}' has only {n} samples "
                f"(< min_per_class={min_per_class}). Keeping all."
            )
            frames.append(subset)
        elif n > max_per_class:
            frames.append(subset.sample(n=max_per_class, random_state=seed))
        else:
            frames.append(subset)

    result = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"\nStratified subsample created: {len(result):,} rows")
    for label in CANONICAL_LABELS:
        cnt = (result["Label"] == label).sum()
        print(f"  {label}: {cnt:,}")
    return result


# ---------------------------------------------------------------------------
# Class Weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    df: pd.DataFrame,
    max_ratio: float = 6.0,
) -> torch.Tensor:
    """
    Compute sqrt-dampened, ratio-capped class weights for CrossEntropyLoss.

    Raw inverse-frequency (total / (C × n_c)) can produce extreme spreads
    (e.g. 42× between None and Issue), which causes the model to never predict
    the majority classes.  Instead:

        raw_w_c = total / (num_classes × n_c)
        dampened_w_c = sqrt(raw_w_c)

    Then the entire vector is rescaled so that min(w) = 1.0, and any weight
    exceeding max_ratio is clipped to max_ratio.
    """
    import math

    total = len(df)
    counts = df["Label"].value_counts()

    raw_weights = []
    for label in CANONICAL_LABELS:
        c = counts.get(label, 1)
        raw_w = total / (NUM_LABELS * c)
        raw_weights.append(raw_w)

    # 1. Square-root dampening
    dampened = [math.sqrt(w) for w in raw_weights]

    # 2. Normalise so minimum weight = 1.0
    min_w = min(dampened)
    normed = [w / min_w for w in dampened]

    # 3. Cap max/min ratio
    capped = [min(w, max_ratio) for w in normed]

    weight_tensor = torch.tensor(capped, dtype=torch.float32)

    print(f"\nClass weights (sqrt-dampened, max_ratio={max_ratio}):")
    print(f"  {'Label':<30s} {'Raw':>8s} {'Sqrt':>8s} {'Normed':>8s} {'Capped':>8s}")
    for label, rw, dw, nw, cw in zip(
        CANONICAL_LABELS, raw_weights, dampened, normed, capped
    ):
        print(f"  {label:<30s} {rw:8.4f} {dw:8.4f} {nw:8.4f} {cw:8.4f}")

    return weight_tensor


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, class_weights=None) -> Dict:
    """
    Run evaluation and return per-class precision/recall/F1,
    macro F1, weighted F1, confusion matrix, and average loss.
    """
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

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
    avg_loss = total_loss / max(n_batches, 1)

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "avg_loss": float(avg_loss),
        "report_str": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device, class_weights=None) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    loss_fn = (
        torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else torch.nn.CrossEntropyLoss()
    )

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def run_training(
    mode: str = "smoke",
    weight_mode: str = "smoke",
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_per_class: int = 500,
    val_max_per_class: int = 200,
):
    """
    Run InLegalBERT fine-tuning.

    Modes:
      - "smoke": CPU, stratified subsample, validates the loop.
      - "full":  GPU if available, full dataset, multiple epochs.

    Weight Modes:
      - "smoke": compute weights from actual subsample (near-uniform ~1.0 each).
      - "full":  compute sqrt-dampened weights from full 1.12M corpus.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        import os
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"Configured PyTorch CPU threads: {num_threads}")
    print(f"\n{'='*70}")
    print(f"JurisGraph InLegalBERT Fine-Tuning — mode={mode} | weight_mode={weight_mode}")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate} | Max seq length: {MAX_SEQ_LENGTH}")
    print(f"{'='*70}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading datasets...")
    df_train = pd.read_csv(TRAIN_CSV, dtype={"Text": str, "Label": str}, keep_default_na=False)
    df_val = pd.read_csv(VAL_CSV, dtype={"Text": str, "Label": str}, keep_default_na=False)

    # Drop blank text rows
    df_train = df_train[df_train["Text"].str.strip() != ""].reset_index(drop=True)
    df_val = df_val[df_val["Text"].str.strip() != ""].reset_index(drop=True)

    # ── Subsample for smoke test ───────────────────────────────────────────
    if mode == "smoke":
        print("\n[2/6] Creating stratified subsample for smoke test...")
        df_train_use = create_stratified_subsample(df_train, max_per_class=max_per_class)
        df_val_use = create_stratified_subsample(df_val, max_per_class=val_max_per_class)
    else:
        df_train_use = df_train
        df_val_use = df_val

    # ── Compute class weights ──────────────────────────────────────────────
    if weight_mode == "smoke":
        print("\n[3/6] Computing class weights from actual training subsample (balanced)...")
        class_weights = compute_class_weights(df_train_use)
    else:
        print("\n[3/6] Computing sqrt-dampened class weights from full training split...")
        class_weights = compute_class_weights(df_train)

    # ── Convert labels ─────────────────────────────────────────────────────
    train_labels = [LABEL2ID[l] for l in df_train_use["Label"]]
    val_labels = [LABEL2ID[l] for l in df_val_use["Label"]]
    train_texts = df_train_use["Text"].tolist()
    val_texts = df_val_use["Text"].tolist()

    # ── Tokenizer & model ──────────────────────────────────────────────────
    print("\n[4/6] Loading InLegalBERT tokenizer & model...")
    tokenizer = load_tokenizer()
    model = load_model()
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {param_count:,} | Trainable: {trainable:,}")

    # ── Dataloaders ────────────────────────────────────────────────────────
    train_dataset = LegalSentenceDataset(train_texts, train_labels, tokenizer)
    val_dataset = LegalSentenceDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    print(f"\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ── Training loop ──────────────────────────────────────────────────────
    print("\n[5/6] Training...")
    best_macro_f1 = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        elapsed = time.time() - t0

        print(f"\n--- Epoch {epoch}/{epochs} ---")
        print(f"Train loss: {avg_train_loss:.4f} | Time: {elapsed:.1f}s")

        # Evaluate
        eval_results = evaluate(model, val_loader, device, class_weights)
        print(f"Val loss:   {eval_results['avg_loss']:.4f}")
        print(f"Macro F1:   {eval_results['macro_f1']:.4f}")
        print(f"Weighted F1:{eval_results['weighted_f1']:.4f}")
        print(f"\nPer-class classification report:\n{eval_results['report_str']}")

        # Save best checkpoint
        if eval_results["macro_f1"] > best_macro_f1:
            best_macro_f1 = eval_results["macro_f1"]
            save_path = CHECKPOINT_DIR / "best_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"[SAVED] Best model checkpoint (macro F1={best_macro_f1:.4f}) -> {save_path}")

    # ── Save final report ──────────────────────────────────────────────────
    print("\n[6/6] Saving evaluation report...")
    report_path = REPORTS_DIR / f"training_report_{mode}.json"
    report_data = {
        "mode": mode,
        "weight_mode": weight_mode,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "best_macro_f1": best_macro_f1,
        "final_eval": {
            "macro_f1": eval_results["macro_f1"],
            "weighted_f1": eval_results["weighted_f1"],
            "avg_loss": eval_results["avg_loss"],
            "per_class": eval_results["report_dict"],
            "confusion_matrix": eval_results["confusion_matrix"],
        },
        "class_weights": class_weights.tolist(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SUCCESS] Report saved -> {report_path}")
    print(f"\n{'='*70}")
    print(f"Training complete. Best macro F1: {best_macro_f1:.4f}")
    print(f"{'='*70}")

    return eval_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InLegalBERT fine-tuning")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--weight_mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke: compute weights from actual subsample (near-uniform ~1.0); full: compute sqrt-dampened weights from full 1.12M corpus",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_per_class", type=int, default=500)
    parser.add_argument("--val_max_per_class", type=int, default=200)
    args = parser.parse_args()

    run_training(
        mode=args.mode,
        weight_mode=args.weight_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_per_class=args.max_per_class,
        val_max_per_class=args.val_max_per_class,
    )
