import os
import sys
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(r"d:\NLP Project")
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model_checkpoint" / "best_model"
SMOKE_DIR = PROJECT_ROOT / "artifacts" / "model_checkpoint" / "best_model_smoke_test"
LABEL_MAPPING_PATH = PROJECT_ROOT / "artifacts" / "label_mapping.json"

print("="*80)
print("STEP 1: LISTING FILES IN artifacts/model_checkpoint/best_model")
print("="*80)

def list_dir_recursive(path: Path, indent=0):
    for p in sorted(path.iterdir()):
        prefix = "  " * indent
        if p.is_dir():
            print(f"{prefix}[DIR]  {p.name}/")
            list_dir_recursive(p, indent + 1)
        else:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"{prefix}[FILE] {p.name:<25} {p.stat().st_size:>12,} bytes ({size_mb:>8.2f} MB)")

list_dir_recursive(MODEL_DIR)

# Detect if nested
if (MODEL_DIR / "best_model" / "model.safetensors").exists():
    ACTUAL_MODEL_PATH = MODEL_DIR / "best_model"
    IS_NESTED = True
    print("\n[NOTE]: Files are nested in subfolder:", ACTUAL_MODEL_PATH)
else:
    ACTUAL_MODEL_PATH = MODEL_DIR
    IS_NESTED = False
    print("\n[NOTE]: Files are located directly at:", ACTUAL_MODEL_PATH)

print("\n" + "="*80)
print("STEP 3: CONFIRM id2label in config.json MATCHES label_mapping.json")
print("="*80)

with open(ACTUAL_MODEL_PATH / "config.json", "r", encoding="utf-8") as f:
    config_data = json.load(f)

with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
    label_map_data = json.load(f)

cfg_id2label = config_data.get("id2label", {})
ref_id2label = label_map_data.get("id2label", {})

print(f"config.json id2label:        {cfg_id2label}")
print(f"label_mapping.json id2label: {ref_id2label}")

keys_match = list(cfg_id2label.keys()) == list(ref_id2label.keys())
vals_match = list(cfg_id2label.values()) == list(ref_id2label.values())
order_match = list(cfg_id2label.items()) == list(ref_id2label.items())

print(f"Keys match:   {keys_match}")
print(f"Values match: {vals_match}")
print(f"Exact match (including ordering): {order_match}")
assert order_match, "CRITICAL ERROR: id2label does not match label_mapping.json!"
print("[CONFIRMED]: config.json id2label matches artifacts/label_mapping.json EXACTLY.")

print("\n" + "="*80)
print("STEP 4: SHA-256 COMPARISON (best_model vs. best_model_smoke_test)")
print("="*80)

def compute_sha256(filepath: Path):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

best_weights_path = ACTUAL_MODEL_PATH / "model.safetensors"
smoke_weights_path = SMOKE_DIR / "model.safetensors"

print(f"Hashing {best_weights_path} ...")
best_sha256 = compute_sha256(best_weights_path)
print(f"Hashing {smoke_weights_path} ...")
smoke_sha256 = compute_sha256(smoke_weights_path)

print(f"best_model SHA-256:            {best_sha256}")
print(f"best_model_smoke_test SHA-256: {smoke_sha256}")
differ = best_sha256 != smoke_sha256
print(f"Hashes differ: {differ}")
assert differ, "CRITICAL ERROR: SHA-256 hashes are identical!"
print("[CONFIRMED]: Weights differ completely from the smoke-test checkpoint.")

print("\n" + "="*80)
print("STEP 5: CHECK best_meta.json CONTENTS")
print("="*80)
best_meta_path = ACTUAL_MODEL_PATH / "best_meta.json"
if best_meta_path.exists():
    with open(best_meta_path, "r", encoding="utf-8") as f:
        best_meta_content = json.load(f)
    print("best_meta.json found:")
    print(json.dumps(best_meta_content, indent=2))
else:
    print("best_meta.json not found.")

print("\n" + "="*80)
print("STEP 2 & STEP 4: CPU INFERENCE ON 7 TEST SENTENCES & LOGIT COMPARISON")
print("="*80)

# 7 test sentences, one per role
test_sentences = [
    # Facts
    ("Facts", "The appellant and the deceased were married on 12th May 1998 according to Hindu rites and ceremonies."),
    # Issue
    ("Issue", "The substantial question of law that arises for consideration is whether the High Court was justified in reversing the finding of acquittal."),
    # Arguments of Petitioner
    ("Arguments of Petitioner", "Learned senior counsel appearing for the petitioner contended that the respondent authorities acted arbitrarily and in violation of Article 14 of the Constitution."),
    # Arguments of Respondent
    ("Arguments of Respondent", "Learned Additional Solicitor General appearing on behalf of the respondent submitted that the tender process was conducted in a transparent and fair manner."),
    # Reasoning
    ("Reasoning", "Having carefully examined the oral testimonies and the forensic report, we find that the chain of circumstantial evidence is complete and points conclusively to the guilt of the accused."),
    # Decision
    ("Decision", "In the result, the appeal is dismissed with costs and the judgment of the trial court is hereby affirmed."),
    # None
    ("None", "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION CIVIL APPEAL NO. 4512 OF 2019"),
]

print("Loading tokenizer and model from:", ACTUAL_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(str(ACTUAL_MODEL_PATH))
model_best = AutoModelForSequenceClassification.from_pretrained(str(ACTUAL_MODEL_PATH))
model_best.eval()

print("Loading smoke test model from:", SMOKE_DIR)
tokenizer_smoke = AutoTokenizer.from_pretrained(str(SMOKE_DIR))
model_smoke = AutoModelForSequenceClassification.from_pretrained(str(SMOKE_DIR))
model_smoke.eval()

texts = [s[1] for s in test_sentences]
expected_roles = [s[0] for s in test_sentences]

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
inputs_smoke = tokenizer_smoke(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

with torch.no_grad():
    outputs_best = model_best(**inputs)
    logits_best = outputs_best.logits
    probs_best = F.softmax(logits_best, dim=-1)
    preds_best = torch.argmax(logits_best, dim=-1).tolist()

    outputs_smoke = model_smoke(**inputs_smoke)
    logits_smoke = outputs_smoke.logits
    diff_logits = torch.abs(logits_best - logits_smoke)
    max_logit_diff = torch.max(diff_logits).item()

print(f"\nMax Absolute Logit Difference (best_model vs. smoke_test) across all 7 sentences: {max_logit_diff:.4f}")

print("\n" + "-"*100)
print(f"{'Target/Expected Role':<26} | {'Predicted Role':<26} | {'Confidence':<10} | {'Match?':<6} | Sentence Preview")
print("-"*100)

for i in range(len(test_sentences)):
    exp_role = expected_roles[i]
    pred_id = preds_best[i]
    pred_role = cfg_id2label[str(pred_id)]
    conf = probs_best[i][pred_id].item()
    match_str = "YES" if pred_role == exp_role else "NO"
    sent_preview = texts[i][:60] + ("..." if len(texts[i]) > 60 else "")
    print(f"{exp_role:<26} | {pred_role:<26} | {conf*100:6.2f}%    | {match_str:<6} | {sent_preview}")

print("-"*100)
