import json
import csv
from pathlib import Path

# 1. Load pipeline_runs.jsonl
with open("artifacts/logs/pipeline_runs.jsonl", "r", encoding="utf-8") as f:
    pipe_runs = [json.loads(line) for line in f if line.strip()]

print(f"Total pipeline runs: {len(pipe_runs)}")
configs = sorted(list(set(r["config"] for r in pipe_runs)))
print(f"Configs: {configs}")

for c in configs:
    c_runs = [r for r in pipe_runs if r["config"] == c]
    statuses = {}
    for r in c_runs:
        s = r.get("final_status")
        statuses[s] = statuses.get(s, 0) + 1
    gold_runs = [r for r in c_runs if r["question_id"].startswith("gold_")]
    unans_runs = [r for r in c_runs if r["question_id"].startswith("unans_")]
    print(f"Config {c}: total={len(c_runs)} (gold={len(gold_runs)}, unans={len(unans_runs)}), statuses: {statuses}")

# 2. Check generation_results.jsonl
with open("artifacts/logs/generation_results.jsonl", "r", encoding="utf-8") as f:
    gen_results = [json.loads(line) for line in f if line.strip()]
print(f"\nTotal generation_results: {len(gen_results)}")
gen_statuses = {}
for r in gen_results:
    s = r.get("final_status")
    gen_statuses[s] = gen_statuses.get(s, 0) + 1
print(f"generation_results statuses: {gen_statuses}")

# Gold ready in generation_results
gen_gold = [r for r in gen_results if r["id"].startswith("gold_")]
s2_gold_ready = [r for r in gen_gold if r["sample"] == "sample2" and r["final_status"] == "ready"]
s3_gold_ready = [r for r in gen_gold if r["sample"] == "sample3" and r["final_status"] == "ready"]
print(f"Sample 2 ready gold ids: {[r['id'] for r in s2_gold_ready]}")
print(f"Sample 3 ready gold ids: {[r['id'] for r in s3_gold_ready]}")

# 3. Check CSV gold questions
csv_path = Path("legalDataset/samples/gold_questions.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    gold_csv = {row["id"]: row for row in reader}

print(f"\nCSV gold ids: {list(gold_csv.keys())}")

# Check text equality for sample2 and sample3 candidates
for r in (s2_gold_ready[:3] + s3_gold_ready[:3]):
    qid = r["id"]
    csv_row = gold_csv.get(qid)
    csv_q = csv_row["question"] if csv_row else None
    gen_q = r["question"]
    match = (csv_q == gen_q)
    chunks = [c.get("chunk_id") for c in r.get("context_chunks", [])]
    print(f"ID {qid}: CSV match={match}, chunk_ids={chunks}")
