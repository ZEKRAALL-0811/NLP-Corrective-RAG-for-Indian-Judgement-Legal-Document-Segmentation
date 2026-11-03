"""
Phase 3 Live Gemini DeepEval Verification Script
Runs real Gemini-backed verify_generation_faithfulness() without mocking.
"""

import sys
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure root is in path
sys.path.insert(0, ".")
load_dotenv()

import src.config as cfg
from src.evaluation.quality_gate import verify_generation_faithfulness

def count_ledger_entries(ledger_path: Path) -> int:
    if not ledger_path.exists():
        return 0
    with open(ledger_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def read_new_ledger_entries(ledger_path: Path, start_idx: int):
    entries = []
    if not ledger_path.exists():
        return entries
    with open(ledger_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= start_idx and line.strip():
                entries.append(json.loads(line))
    return entries

def main():
    ledger_path = cfg.ARTIFACTS_DIR / "logs" / "judge_usage.jsonl"
    
    # Verify environment
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print(json.dumps({"error": "GEMINI_API_KEY missing from environment"}, indent=2))
        return

    orig_gate_state = cfg.ENABLE_ONLINE_GATE
    print(f"[Phase 3 Setup] Original ENABLE_ONLINE_GATE: {orig_gate_state}", flush=True)
    print(f"[Phase 3 Setup] Configured JUDGE_PROVIDER: {cfg.JUDGE_PROVIDER}", flush=True)

    results = {}

    try:
        # In-memory temporary override
        cfg.ENABLE_ONLINE_GATE = True
        print(f"[Phase 3 Setup] Temporarily set ENABLE_ONLINE_GATE: {cfg.ENABLE_ONLINE_GATE}", flush=True)

        question = "Under what section was the Contempt of Courts Act invoked?"
        context_chunks = [
            {"text": "The judgment states that Section 12 of the Contempt of Courts Act, 1971 provides the punishment for contempt."}
        ]
        positive_answer = "The judgment identifies Section 12 of the Contempt of Courts Act, 1971 as the provision addressing punishment for contempt."

        # Ledger count before Test 1
        ledger_count_before_1 = count_ledger_entries(ledger_path)
        t0 = time.time()

        print("[Phase 3 Test 1] Executing live positive test...", flush=True)
        res_1 = verify_generation_faithfulness(
            question=question,
            context_chunks=context_chunks,
            answer=positive_answer,
            judge=None, # instantiate GeminiJudge directly
            threshold=0.7,
            relevancy_threshold=0.7,
        )
        t_elapsed_1 = time.time() - t0
        entries_1 = read_new_ledger_entries(ledger_path, ledger_count_before_1)

        results["test_1_positive"] = {
            "question": question,
            "answer": positive_answer,
            "result": res_1,
            "latency_seconds": round(t_elapsed_1, 2),
            "new_ledger_calls_count": len(entries_1),
            "ledger_entries": entries_1,
        }
        print(f"[Phase 3 Test 1 Complete] Result: {res_1['status']}, Verified: {res_1['verified']}, Faithfulness: {res_1.get('faithfulness_score')}, Relevancy: {res_1.get('relevancy_score')}", flush=True)

        # If Test 1 succeeded, run Test 2 (contradictory answer)
        if res_1.get("status") == "passed":
            contradictory_answer = "The judgment says Section 15 applies and the court awarded ₹50,000 in damages."
            ledger_count_before_2 = count_ledger_entries(ledger_path)
            t0_2 = time.time()

            print("[Phase 3 Test 2] Executing live contradictory test...", flush=True)
            res_2 = verify_generation_faithfulness(
                question=question,
                context_chunks=context_chunks,
                answer=contradictory_answer,
                judge=None,
                threshold=0.7,
                relevancy_threshold=0.7,
            )
            t_elapsed_2 = time.time() - t0_2
            entries_2 = read_new_ledger_entries(ledger_path, ledger_count_before_2)

            results["test_2_contradictory"] = {
                "question": question,
                "answer": contradictory_answer,
                "result": res_2,
                "latency_seconds": round(t_elapsed_2, 2),
                "new_ledger_calls_count": len(entries_2),
                "ledger_entries": entries_2,
            }
            print(f"[Phase 3 Test 2 Complete] Result: {res_2['status']}, Verified: {res_2['verified']}, Faithfulness: {res_2.get('faithfulness_score')}, Relevancy: {res_2.get('relevancy_score')}", flush=True)

    finally:
        # Guarantee restoration
        cfg.ENABLE_ONLINE_GATE = orig_gate_state
        print(f"[Phase 3 Teardown] Restored ENABLE_ONLINE_GATE: {cfg.ENABLE_ONLINE_GATE}", flush=True)

    print("\n--- PHASE 3 FINAL RESULTS ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
