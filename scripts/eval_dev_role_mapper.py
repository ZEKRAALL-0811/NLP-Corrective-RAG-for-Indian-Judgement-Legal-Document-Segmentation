"""
Dev-set iteration script for query role mapper.
Evaluates rules_only, llm_only, and hybrid on the 35 dev questions.
"""

import sys
from pathlib import Path
import json
import time

sys.path.insert(0, ".")
from src.rag.role_mapper import map_query_to_roles, check_ollama

check_ollama()

bench_file = Path("artifacts/role_mapper_benchmark.json")
with open(bench_file, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

dev_set = benchmark["dev"]
print(f"Loaded {len(dev_set)} dev questions.")

def is_top1_correct(item, mapping):
    g_kind = item["gold_kind"]
    g_role = item["gold_role"]
    if g_kind == "out_of_scope":
        return mapping.kind == "out_of_scope"
    if g_kind == "whole_document":
        return mapping.kind == "whole_document"
    # Role question
    if mapping.roles and mapping.roles[0] == g_role:
        return True
    return False

def is_top2_correct(item, mapping):
    g_kind = item["gold_kind"]
    g_role = item["gold_role"]
    if g_kind in {"out_of_scope", "whole_document"}:
        return mapping.kind == g_kind
    return bool(mapping.roles and g_role in mapping.roles)

# Evaluate each variant on dev
variants = ["rules_only", "llm_only", "hybrid"]
results = {}

for mode in variants:
    print(f"\n--- Evaluating {mode} on dev ---")
    correct_top1 = 0
    correct_top2 = 0
    t0 = time.time()
    
    for i, item in enumerate(dev_set):
        mapping = map_query_to_roles(item["question"], mode=mode)
        top1 = is_top1_correct(item, mapping)
        top2 = is_top2_correct(item, mapping)
        if top1:
            correct_top1 += 1
        if top2:
            correct_top2 += 1
        if not top1:
            print(f"  [MISS] ({item['id']}) Gold: {item['gold_role'] or item['gold_kind']} | Pred: {mapping.roles} ({mapping.kind}) | Q: {item['question']}")

    elapsed = time.time() - t0
    acc_top1 = correct_top1 / len(dev_set)
    acc_top2 = correct_top2 / len(dev_set)
    results[mode] = {
        "top1_acc": acc_top1,
        "top2_acc": acc_top2,
        "elapsed_sec": round(elapsed, 2),
    }
    print(f"{mode} Dev Top-1 Accuracy: {acc_top1*100:.1f}% ({correct_top1}/{len(dev_set)}) | Top-2: {acc_top2*100:.1f}% in {elapsed:.1f}s")
