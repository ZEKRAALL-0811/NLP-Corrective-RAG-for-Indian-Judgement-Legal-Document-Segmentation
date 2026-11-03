import json
import csv
import re
import numpy as np
from pathlib import Path

# 1. Load pipeline runs and generation results
with open("artifacts/logs/pipeline_runs.jsonl", "r", encoding="utf-8") as f:
    pipe_runs = [json.loads(line) for line in f if line.strip()]

with open("artifacts/logs/generation_results.jsonl", "r", encoding="utf-8") as f:
    gen_runs = [json.loads(line) for line in f if line.strip()]

gen_by_id = {r["id"]: r for r in gen_runs}

with open("legalDataset/samples/gold_keyfacts.csv", "r", encoding="utf-8") as f:
    kf_rows = list(csv.DictReader(f))
keyfacts = {r["gold_id"]: [x.strip() for x in r["key_facts"].split(";;") if x.strip()] for r in kf_rows}

# Load gold paragraphs
with open("legalDataset/samples/gold_paragraphs.json", "r", encoding="utf-8") as f:
    gold_paras = json.load(f)

# Load cross-encoder model
from src.rag.reranker import score_chunks
from scripts.run_day2_benchmark import check_overlap

print("All libraries loaded successfully!")

# Test Key Fact Recall on a couple of samples
def compute_keyfact_recall(answer: str, facts: list) -> float:
    if not facts or not answer:
        return 0.0
    ans_clean = re.sub(r"\s+", " ", answer.lower().strip())
    matched = 0
    for fact in facts:
        alts = [a.strip().lower() for a in fact.split("|") if a.strip()]
        if any(alt in ans_clean for alt in alts):
            matched += 1
    return matched / len(facts)

for r in gen_runs[:5]:
    qid = r["id"]
    ans = r["answer"]
    kf = keyfacts.get(qid, [])
    rec = compute_keyfact_recall(ans, kf)
    print(f"{qid} keyfact recall: {rec:.2f} (facts: {len(kf)})")
