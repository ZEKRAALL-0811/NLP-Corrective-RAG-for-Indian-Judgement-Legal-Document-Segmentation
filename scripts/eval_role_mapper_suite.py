"""
Comprehensive evaluation script for Module 5 Query Role Mapper with:
1. Updated metadata cues (role None)
2. Optional reason field OFF (default, debug=False)
3. Evaluates on:
   - dev split (35 questions)
   - test split (35 questions, marked 'no longer pristine')
   - gold_questions.csv (32 questions)
4. Reports accuracy and warm latency (with reason OFF)
5. Lists all misses on the 32 gold questions
6. Saves raw output to artifacts/logs/part_a_role_mapper_eval.txt
"""

import sys
from pathlib import Path
import json
import csv
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, ".")
from src.rag.role_mapper import map_query_to_roles, check_ollama
from src.config import CANONICAL_LABELS

check_ollama()

BENCH_FILE = Path("artifacts/role_mapper_benchmark.json")
GOLD_CSV = Path("legalDataset/samples/gold_questions.csv")
OUT_LOG = Path("artifacts/logs/part_a_role_mapper_eval.txt")

with open(BENCH_FILE, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

dev_set = bench_data["dev"]
test_set = bench_data["test"]

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

def evaluate_split(split_name, items, debug=False, mode="hybrid"):
    top1_hits = 0
    top2_hits = 0
    times_ms = []
    per_class = defaultdict(lambda: {"total": 0, "correct_top1": 0})
    misses = []

    for item in items:
        q = item["question"]
        g_cls = get_gold_class(item)
        g_kind = item["gold_kind"]
        g_role = item.get("gold_role")

        t0 = time.time()
        m = map_query_to_roles(q, mode=mode, debug=debug)
        t_elapsed_ms = (time.time() - t0) * 1000
        times_ms.append(t_elapsed_ms)

        p_cls = get_pred_class(m)
        t1 = (p_cls == g_cls)

        if g_kind in {"out_of_scope", "whole_document"}:
            t2 = (m.kind == g_kind)
        else:
            t2 = bool(m.roles and g_role in m.roles)

        if t1: top1_hits += 1
        if t2: top2_hits += 1

        per_class[g_cls]["total"] += 1
        if t1:
            per_class[g_cls]["correct_top1"] += 1
        else:
            misses.append({
                "id": item.get("id", ""),
                "question": q,
                "gold": g_cls,
                "pred": p_cls,
                "roles": m.roles,
                "kind": m.kind,
            })

    total = len(items)
    top1_acc = (top1_hits / total) * 100
    top2_acc = (top2_hits / total) * 100
    avg_latency = float(np.mean(times_ms))
    median_latency = float(np.median(times_ms))

    return {
        "split_name": split_name,
        "total": total,
        "top1_hits": top1_hits,
        "top2_hits": top2_hits,
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "per_class": dict(per_class),
        "misses": misses,
    }

def evaluate_gold_csv(debug=False, mode="hybrid"):
    gold_rows = []
    with open(GOLD_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold_rows.append(row)

    top1_hits = 0
    top2_hits = 0
    times_ms = []
    misses = []
    per_role = defaultdict(lambda: {"total": 0, "correct_top1": 0})

    for row in gold_rows:
        q = row["question"].strip()
        raw_role = row["role"].strip()
        if raw_role == "OUT_OF_SCOPE":
            g_cls = "out_of_scope"
            g_kind = "out_of_scope"
            g_role = None
        else:
            g_cls = raw_role
            g_kind = "role"
            g_role = raw_role

        t0 = time.time()
        m = map_query_to_roles(q, mode=mode, debug=debug)
        t_elapsed_ms = (time.time() - t0) * 1000
        times_ms.append(t_elapsed_ms)

        p_cls = get_pred_class(m)
        t1 = (p_cls == g_cls)
        if g_kind == "out_of_scope":
            t2 = (m.kind == "out_of_scope")
        else:
            t2 = bool(m.roles and g_role in m.roles)

        if t1: top1_hits += 1
        if t2: top2_hits += 1

        per_role[g_cls]["total"] += 1
        if t1:
            per_role[g_cls]["correct_top1"] += 1
        else:
            misses.append({
                "sample": row.get("sample", ""),
                "question": q,
                "gold": g_cls,
                "pred": p_cls,
                "roles": m.roles,
                "kind": m.kind,
                "answer_hint": row.get("answer_hint", ""),
            })

    total = len(gold_rows)
    top1_acc = (top1_hits / total) * 100
    top2_acc = (top2_hits / total) * 100
    avg_latency = float(np.mean(times_ms))

    return {
        "total": total,
        "top1_hits": top1_hits,
        "top2_hits": top2_hits,
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "avg_latency_ms": avg_latency,
        "per_role": dict(per_role),
        "misses": misses,
    }

print("Running evaluations with reason OFF (debug=False)...")
dev_res = evaluate_split("Dev Split (35 questions)", dev_set, debug=False)
test_res = evaluate_split("Test Split (35 questions) [NOTE: no longer pristine]", test_set, debug=False)
gold_res = evaluate_gold_csv(debug=False)

# Build log output
lines = []
lines.append("=" * 80)
lines.append("PART A: ROLE MAPPER SUITE EVALUATION (REASON OFF / debug=False)")
lines.append("=" * 80)
lines.append("")

# 1. Dev Split
lines.append(f"1. DEV SPLIT (Total: {dev_res['total']})")
lines.append(f"   Top-1 Accuracy: {dev_res['top1_acc']:.1f}% ({dev_res['top1_hits']}/{dev_res['total']})")
lines.append(f"   Top-2 Accuracy: {dev_res['top2_acc']:.1f}% ({dev_res['top2_hits']}/{dev_res['total']})")
lines.append(f"   Warm Latency (Mean): {dev_res['avg_latency_ms']:.1f} ms | Median: {dev_res['median_latency_ms']:.1f} ms")
lines.append("   Per-Class Breakdown:")
for cls, st in sorted(dev_res["per_class"].items()):
    c_tot = st["total"]
    c_hits = st["correct_top1"]
    lines.append(f"     - {cls:<25}: {c_hits}/{c_tot} ({c_hits/c_tot*100:5.1f}%)")
lines.append("")

# 2. Test Split
lines.append(f"2. TEST SPLIT [NOTE: no longer pristine] (Total: {test_res['total']})")
lines.append(f"   Top-1 Accuracy: {test_res['top1_acc']:.1f}% ({test_res['top1_hits']}/{test_res['total']})")
lines.append(f"   Top-2 Accuracy: {test_res['top2_acc']:.1f}% ({test_res['top2_hits']}/{test_res['total']})")
lines.append(f"   Warm Latency (Mean): {test_res['avg_latency_ms']:.1f} ms | Median: {test_res['median_latency_ms']:.1f} ms")
lines.append("   Per-Class Breakdown:")
for cls, st in sorted(test_res["per_class"].items()):
    c_tot = st["total"]
    c_hits = st["correct_top1"]
    lines.append(f"     - {cls:<25}: {c_hits}/{c_tot} ({c_hits/c_tot*100:5.1f}%)")
lines.append("")

# 3. Gold Questions CSV
lines.append(f"3. GOLD QUESTIONS CSV (Total: {gold_res['total']})")
lines.append(f"   Top-1 Accuracy: {gold_res['top1_acc']:.1f}% ({gold_res['top1_hits']}/{gold_res['total']})")
lines.append(f"   Top-2 Accuracy: {gold_res['top2_acc']:.1f}% ({gold_res['top2_hits']}/{gold_res['total']})")
lines.append(f"   Warm Latency (Mean): {gold_res['avg_latency_ms']:.1f} ms")
lines.append("   Per-Role Breakdown:")
for cls, st in sorted(gold_res["per_role"].items()):
    c_tot = st["total"]
    c_hits = st["correct_top1"]
    lines.append(f"     - {cls:<25}: {c_hits}/{c_tot} ({c_hits/c_tot*100:5.1f}%)")
lines.append("")

lines.append("-" * 80)
lines.append(f"LIST OF MISSED QUESTIONS ON GOLD_QUESTIONS.CSV ({len(gold_res['misses'])} missed out of {gold_res['total']}):")
lines.append("-" * 80)
for i, m in enumerate(gold_res["misses"], 1):
    lines.append(f"Miss #{i:02d} [{m['sample']}]:")
    lines.append(f"  Question:    \"{m['question']}\"")
    lines.append(f"  Gold Class:  {m['gold']}")
    lines.append(f"  Predicted:   {m['pred']} (Roles: {m['roles']}, Kind: {m['kind']})")
    lines.append(f"  Answer Hint: {m['answer_hint']}")
    lines.append("")

out_text = "\n".join(lines)
OUT_LOG.write_text(out_text, encoding="utf-8")
print(f"\nSaved evaluation to {OUT_LOG} ({len(out_text)} bytes)")
print(out_text)
