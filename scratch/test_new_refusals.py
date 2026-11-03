import json
import re

with open("artifacts/logs/pipeline_runs.jsonl", "r", encoding="utf-8") as f:
    runs = [json.loads(line) for line in f if line.strip()]

# The user explicitly listed:
# "do not provide (sufficient) information", "no mention of", "does not mention", "not specified", "does not address"
# plus standard refusal indicators like "no reliable answer", "abstained", "no_answer", "not answer", etc.

EXPLICIT_PHRASES = [
    "do not provide",
    "does not provide",
    "no mention of",
    "does not mention",
    "do not mention",
    "not specified",
    "does not address",
    "do not address",
    "cannot answer",
    "unable to answer",
    "not answer",
    "no reliable answer",
]

def check_refusal(r):
    status = r.get("final_status", "")
    ans = r.get("answer", "")
    ans_l = ans.lower().strip()
    
    # 1. Pipeline status
    if status in ("abstained", "no_answer"):
        return True, f"status={status}"
    
    # 2. Key phrases
    for p in EXPLICIT_PHRASES:
        if p in ans_l:
            return True, f"phrase='{p}'"
            
    return False, "answered"

for cfg in ["naive_rag", "role_rag", "full_pipeline"]:
    print(f"\n=================== CONFIG: {cfg} ===================")
    c_runs = [r for r in runs if r["config"] == cfg]
    gold_runs = [r for r in c_runs if r["question_id"].startswith("gold_")]
    unans_runs = [r for r in c_runs if r["question_id"].startswith("unans_")]
    
    gold_refusals = []
    gold_answered = []
    for r in gold_runs:
        is_ref, reason = check_refusal(r)
        if is_ref:
            gold_refusals.append((r["question_id"], r["answer"], reason))
        else:
            gold_answered.append(r["question_id"])
            
    unans_refusals = []
    unans_answered = []
    for r in unans_runs:
        is_ref, reason = check_refusal(r)
        if is_ref:
            unans_refusals.append((r["question_id"], r["answer"], reason))
        else:
            unans_answered.append((r["question_id"], r["answer"]))
            
    print(f"Gold Total: {len(gold_runs)}")
    print(f"  Wrong Refusals ({len(gold_refusals)}): {[x[0] for x in gold_refusals]}")
    for qid, ans, reason in gold_refusals:
        print(f"    - {qid} ({reason}): {ans[:100]}...")
    print(f"  Answered ({len(gold_answered)}): {gold_answered}")
    
    print(f"Unanswerable Total: {len(unans_runs)}")
    print(f"  Refusals ({len(unans_refusals)}): {[x[0] for x in unans_refusals]}")
    print(f"  Non-Refusals (Answered) ({len(unans_answered)}): {[x[0] for x in unans_answered]}")
    for qid, ans in unans_answered:
        print(f"    - Non-refusal {qid}: {ans}")
