"""
Final Evaluation Script for Module 5: Query Role Mapper.

Runs ONCE on the held-out test split (35 questions) and gold_questions.csv (33 questions).
Computes:
1. Top-1 and Top-2 accuracy across variants (rules_only, llm_only, hybrid)
2. Per-role accuracy breakdown
3. Full confusion table
4. Out_of_scope precision, recall, and F1
5. Determinism verification (15 questions x 3 runs)
6. Latency measurement (cold vs warm) in ms
"""

import sys
from pathlib import Path
import json
import csv
import time
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")
from src.rag.role_mapper import map_query_to_roles, check_ollama
from src.config import CANONICAL_LABELS
import src.config as cfg

# Verify Ollama
check_ollama()

BENCH_FILE = Path("artifacts/role_mapper_benchmark.json")
GOLD_CSV = Path("legalDataset/samples/gold_questions.csv")
REPORT_FILE = Path("artifacts/role_mapper_evaluation_report.json")

with open(BENCH_FILE, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

test_set = bench_data["test"]
print(f"Loaded {len(test_set)} test benchmark questions.")

# Canonical 9 evaluation classes
ALL_CLASSES = CANONICAL_LABELS + ["whole_document", "out_of_scope"]

def get_gold_class(item):
    if item["gold_kind"] in {"out_of_scope", "whole_document"}:
        return item["gold_kind"]
    return item["gold_role"]

def get_pred_class(mapping):
    if mapping.kind in {"out_of_scope", "whole_document"}:
        return mapping.kind
    if mapping.roles:
        return mapping.roles[0]
    return "whole_document"

def check_top1(item, mapping):
    g_class = get_gold_class(item)
    p_class = get_pred_class(mapping)
    return g_class == p_class

def check_top2(item, mapping):
    g_kind = item["gold_kind"]
    g_role = item["gold_role"]
    if g_kind in {"out_of_scope", "whole_document"}:
        return mapping.kind == g_kind
    return bool(mapping.roles and g_role in mapping.roles)

# ==============================================================================
# 1. EVALUATION ACROSS THREE VARIANTS ON TEST BENCHMARK
# ==============================================================================
print("\n" + "="*80)
print("1. EVALUATING TEST BENCHMARK (35 QUESTIONS) ACROSS 3 VARIANTS")
print("="*80)

variants_results = {}
for mode in ["rules_only", "llm_only", "hybrid"]:
    t0 = time.time()
    preds = []
    top1_hits = 0
    top2_hits = 0
    
    for item in test_set:
        m = map_query_to_roles(item["question"], mode=mode)
        t1 = check_top1(item, m)
        t2 = check_top2(item, m)
        if t1: top1_hits += 1
        if t2: top2_hits += 1
        preds.append({
            "id": item["id"],
            "question": item["question"],
            "gold": get_gold_class(item),
            "pred_top1": get_pred_class(m),
            "roles": m.roles,
            "kind": m.kind,
            "confidence": m.confidence,
            "top1_correct": t1,
            "top2_correct": t2,
        })
        
    elapsed = time.time() - t0
    top1_acc = top1_hits / len(test_set)
    top2_acc = top2_hits / len(test_set)
    variants_results[mode] = {
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "top1_hits": top1_hits,
        "top2_hits": top2_hits,
        "total": len(test_set),
        "elapsed_sec": round(elapsed, 2),
        "predictions": preds,
    }
    print(f"Mode: {mode:<12} | Top-1: {top1_acc*100:5.1f}% ({top1_hits}/{len(test_set)}) | Top-2: {top2_acc*100:5.1f}% ({top2_hits}/{len(test_set)}) | Time: {elapsed:5.1f}s")

# Primary chosen mode is 'hybrid'
hybrid_preds = variants_results["hybrid"]["predictions"]

# ==============================================================================
# 2. PER-ROLE ACCURACY & CONFUSION TABLE FOR HYBRID MODE
# ==============================================================================
print("\n" + "="*80)
print("2. PER-ROLE ACCURACY & CONFUSION TABLE (HYBRID MODE)")
print("="*80)

per_role_stats = defaultdict(lambda: {"total": 0, "correct_top1": 0, "correct_top2": 0})
confusion = {g: {p: 0 for p in ALL_CLASSES} for g in ALL_CLASSES}

for p in hybrid_preds:
    g = p["gold"]
    pred = p["pred_top1"]
    per_role_stats[g]["total"] += 1
    if p["top1_correct"]:
        per_role_stats[g]["correct_top1"] += 1
    if p["top2_correct"]:
        per_role_stats[g]["correct_top2"] += 1
    confusion[g][pred] += 1

print(f"{'Class':<26} | {'Total':<6} | {'Top-1 Correct':<14} | {'Top-1 Acc':<10} | {'Top-2 Acc':<10}")
print("-" * 75)
for cls in ALL_CLASSES:
    st = per_role_stats[cls]
    tot = st["total"]
    if tot > 0:
        c1 = st["correct_top1"]
        c2 = st["correct_top2"]
        print(f"{cls:<26} | {tot:<6} | {c1:<14} | {c1/tot*100:6.1f}%    | {c2/tot*100:6.1f}%")

print("\nCONFUSION TABLE (Rows: Gold, Columns: Predicted):")
label_hdr = "Gold \\ Pred"
header_str = f"{label_hdr:<24} | " + " | ".join(f"{c[:4]:<4}" for c in ALL_CLASSES)
print(header_str)
print("-" * len(header_str))
for g in ALL_CLASSES:
    row_counts = [confusion[g][p] for p in ALL_CLASSES]
    if sum(row_counts) > 0:
        row_str = f"{g:<24} | " + " | ".join(f"{cnt:<4}" for cnt in row_counts)
        print(row_str)

# Out-of-scope precision, recall, F1
oos_tp = confusion["out_of_scope"]["out_of_scope"]
oos_fp = sum(confusion[g]["out_of_scope"] for g in ALL_CLASSES if g != "out_of_scope")
oos_fn = sum(confusion["out_of_scope"][p] for p in ALL_CLASSES if p != "out_of_scope")

oos_prec = oos_tp / max(oos_tp + oos_fp, 1e-9)
oos_rec = oos_tp / max(oos_tp + oos_fn, 1e-9)
oos_f1 = 2 * oos_prec * oos_rec / max(oos_prec + oos_rec, 1e-9)

print("\n" + "="*80)
print(f"OUT_OF_SCOPE METRICS: Precision = {oos_prec*100:.1f}%, Recall = {oos_rec*100:.1f}%, F1 = {oos_f1*100:.1f}%")
print("="*80)

# ==============================================================================
# 3. ADDITIONAL HELD-OUT EVALUATION: GOLD_QUESTIONS.CSV (33 QUESTIONS)
# ==============================================================================
print("\n" + "="*80)
print("3. ADDITIONAL HELD-OUT EVALUATION: GOLD_QUESTIONS.CSV (33 QUESTIONS)")
print("="*80)

gold_items = []
with open(GOLD_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gold_items.append(row)

gold_eval = []
gold_top1_hits = 0
gold_top2_hits = 0

for item in gold_items:
    q = item["question"]
    raw_role = item["role"].strip()
    
    if raw_role == "OUT_OF_SCOPE":
        gold_cls = "out_of_scope"
        gold_kind = "out_of_scope"
        gold_role_name = None
    else:
        gold_cls = raw_role
        gold_kind = "role"
        gold_role_name = raw_role

    m = map_query_to_roles(q, mode="hybrid")
    pred_cls = get_pred_class(m)
    
    t1 = (pred_cls == gold_cls)
    if gold_kind == "out_of_scope":
        t2 = (m.kind == "out_of_scope")
    else:
        t2 = bool(m.roles and gold_role_name in m.roles)
        
    if t1: gold_top1_hits += 1
    if t2: gold_top2_hits += 1
    
    gold_eval.append({
        "sample": item["sample"],
        "question": q,
        "gold": gold_cls,
        "pred": pred_cls,
        "roles": m.roles,
        "kind": m.kind,
        "confidence": m.confidence,
        "top1_correct": t1,
        "top2_correct": t2,
    })

gold_top1_acc = gold_top1_hits / len(gold_items)
gold_top2_acc = gold_top2_hits / len(gold_items)
print(f"Gold Questions Top-1 Accuracy: {gold_top1_acc*100:.1f}% ({gold_top1_hits}/{len(gold_items)})")
print(f"Gold Questions Top-2 Accuracy: {gold_top2_acc*100:.1f}% ({gold_top2_hits}/{len(gold_items)})")

# ==============================================================================
# 4. DETERMINISM TEST: 15 QUESTIONS RUN 3 TIMES
# ==============================================================================
print("\n" + "="*80)
print("4. DETERMINISM TEST (15 QUESTIONS x 3 RUNS)")
print("="*80)

sample_15_questions = [item["question"] for item in test_set[:15]]
run1 = [map_query_to_roles(q, mode="hybrid") for q in sample_15_questions]
run2 = [map_query_to_roles(q, mode="hybrid") for q in sample_15_questions]
run3 = [map_query_to_roles(q, mode="hybrid") for q in sample_15_questions]

identical_matches = 0
for i in range(15):
    m1, m2, m3 = run1[i], run2[i], run3[i]
    is_same = (m1.roles == m2.roles == m3.roles) and (m1.kind == m2.kind == m3.kind)
    if is_same:
        identical_matches += 1
    else:
        print(f"Determinism mismatch at Q{i+1}: '{sample_15_questions[i]}'")
        print(f"  Run 1: {m1}")
        print(f"  Run 2: {m2}")
        print(f"  Run 3: {m3}")

determinism_pct = (identical_matches / 15) * 100
print(f"Determinism: {identical_matches}/15 ({determinism_pct:.1f}%) questions gave 100% identical outputs across 3 runs.")

# ==============================================================================
# 5. LATENCY MEASUREMENT (COLD VS WARM)
# ==============================================================================
print("\n" + "="*80)
print("5. LATENCY MEASUREMENT (COLD VS WARM IN MS)")
print("="*80)

# Unload model by keep_alive 0 to measure true cold call
unload_payload = {
    "model": getattr(cfg, "OLLAMA_MODEL", "llama3.2:3b"),
    "keep_alive": 0,
}
import urllib.request
try:
    req = urllib.request.Request(
        f"{getattr(cfg, 'OLLAMA_HOST', 'http://localhost:11434').rstrip('/')}/api/generate",
        data=json.dumps(unload_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        pass
except Exception:
    pass

time.sleep(1)

test_q_cold = "What transactions led to the dispute between the builder and the allottees?"
t_cold_start = time.time()
m_cold = map_query_to_roles(test_q_cold, mode="llm_only")
t_cold_ms = (time.time() - t_cold_start) * 1000

# Warm calls
warm_times = []
for _ in range(3):
    t_warm_start = time.time()
    m_warm = map_query_to_roles(test_q_cold, mode="llm_only")
    warm_times.append((time.time() - t_warm_start) * 1000)

t_warm_avg_ms = float(np.mean(warm_times))

print(f"Cold Call Latency (Model load + inference): {t_cold_ms:8.1f} ms")
print(f"Warm Call Latency (Model in RAM, average):  {t_warm_avg_ms:8.1f} ms")

# Measure rule-layer latency
t_rule_start = time.time()
for _ in range(100):
    map_query_to_roles("What did the petitioner argue regarding section 132?", mode="rules_only")
t_rule_avg_ms = ((time.time() - t_rule_start) / 100) * 1000
print(f"Rule Layer Latency (Average over 100 calls): {t_rule_avg_ms:8.3f} ms")

# ==============================================================================
# SAVE COMPLETE FINAL REPORT
# ==============================================================================
report_payload = {
    "test_benchmark": {
        "total_questions": len(test_set),
        "variants": {
            k: {
                "top1_acc": v["top1_acc"],
                "top2_acc": v["top2_acc"],
                "top1_hits": v["top1_hits"],
                "top2_hits": v["top2_hits"],
                "elapsed_sec": v["elapsed_sec"],
            } for k, v in variants_results.items()
        },
        "hybrid_per_role_stats": {
            cls: per_role_stats[cls] for cls in ALL_CLASSES if per_role_stats[cls]["total"] > 0
        },
        "confusion_matrix": {
            g: {p: confusion[g][p] for p in ALL_CLASSES if confusion[g][p] > 0}
            for g in ALL_CLASSES if sum(confusion[g].values()) > 0
        },
        "out_of_scope": {
            "precision": oos_prec,
            "recall": oos_rec,
            "f1": oos_f1,
            "tp": oos_tp,
            "fp": oos_fp,
            "fn": oos_fn,
        },
    },
    "gold_questions_evaluation": {
        "total_questions": len(gold_items),
        "top1_acc": gold_top1_acc,
        "top2_acc": gold_top2_acc,
        "top1_hits": gold_top1_hits,
        "top2_hits": gold_top2_hits,
        "predictions": gold_eval,
    },
    "determinism": {
        "num_questions": 15,
        "num_runs": 3,
        "identical_pct": determinism_pct,
    },
    "latency_ms": {
        "cold_ms": round(t_cold_ms, 1),
        "warm_avg_ms": round(t_warm_avg_ms, 1),
        "rule_avg_ms": round(t_rule_avg_ms, 3),
    },
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report_payload, f, indent=2)

print(f"\nFinal report successfully written to {REPORT_FILE}")
