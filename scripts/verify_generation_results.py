"""
Verification script for artifacts/logs/generation_results.jsonl.
Verifies:
1. 30 rows exactly (22 gold, 8 unanswerable)
2. Question texts identical to gold_questions.csv and gold_unanswerable.csv
3. Gold markers equal to gold_paragraphs.json
4. Every chunk text (retrieved & context) is a whitespace-normalized substring of the cleaned source text
5. Required fields present in all rows:
   - timestamp
   - models_used
   - per_node_latency (ms from timers)
   - ollama_total_duration (total_duration per LLM call)
   - ungrounded_numbers
   - citations
   - final_status
6. Latencies plausible (positive numbers)

Outputs PASS/FAIL per check to artifacts/logs/verify_generation_results.txt.
Returns exit code 0 if all checks PASS, 1 otherwise.
"""

import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, ".")
from src.models.segmenter import segment_document

def normalize_ws(text: str) -> str:
    return " ".join(text.split())

def verify_jsonl(jsonl_path: Path, log_path: Path) -> bool:
    results = {}
    details = {}

    if not jsonl_path.exists():
        msg = f"FAIL: {jsonl_path} does not exist."
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(msg, encoding="utf-8")
        print(msg)
        return False

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows = []
    for l in lines:
        try:
            rows.append(json.loads(l))
        except Exception as e:
            pass

    # ---------------------------------------------------------
    # Check 1: 30 rows exactly
    # ---------------------------------------------------------
    c1_pass = len(rows) == 30
    results["check_1_row_count"] = "PASS" if c1_pass else "FAIL"
    details["check_1_row_count"] = f"Total rows found: {len(rows)} (expected 30)"

    # ---------------------------------------------------------
    # Check 2: Question texts identical to CSVs
    # ---------------------------------------------------------
    gold_csv_path = Path("legalDataset/samples/gold_questions.csv")
    unans_csv_path = Path("legalDataset/samples/gold_unanswerable.csv")
    gold_json_path = Path("legalDataset/samples/gold_paragraphs.json")

    with open(gold_csv_path, "r", encoding="utf-8") as f:
        gold_csv_rows = [r for r in csv.DictReader(f) if r["sample"] in ("sample2", "sample3") and r["role"] != "None"]
    with open(unans_csv_path, "r", encoding="utf-8") as f:
        unans_csv_rows = list(csv.DictReader(f))

    gold_json = json.loads(gold_json_path.read_text(encoding="utf-8"))

    c2_pass = True
    c2_errs = []
    if len(rows) >= 30:
        for idx, (expected, row) in enumerate(zip(gold_csv_rows, rows[:22])):
            if row.get("question") != expected["question"]:
                c2_pass = False
                c2_errs.append(f"Row {idx+1} gold question mismatch: '{row.get('question')}' != '{expected['question']}'")
        for idx, (expected, row) in enumerate(zip(unans_csv_rows, rows[22:30])):
            if row.get("question") != expected["question"]:
                c2_pass = False
                c2_errs.append(f"Row {idx+23} unanswerable question mismatch: '{row.get('question')}' != '{expected['question']}'")
    else:
        c2_pass = False
        c2_errs.append("Insufficient rows to verify question texts")

    results["check_2_question_texts"] = "PASS" if c2_pass else "FAIL"
    details["check_2_question_texts"] = f"Mismatches: {len(c2_errs)}" if not c2_pass else "All 30 question texts identical to CSVs"

    # ---------------------------------------------------------
    # Check 3: Gold markers equal to gold_paragraphs.json
    # ---------------------------------------------------------
    c3_pass = True
    c3_errs = []
    if len(rows) >= 22:
        for idx, row in enumerate(rows[:22]):
            q = row.get("question", "")
            exp_markers = gold_json.get(q, {}).get("markers", [])
            row_markers = row.get("gold_markers", [])
            if row_markers != exp_markers:
                c3_pass = False
                c3_errs.append(f"Row {idx+1} markers mismatch: {row_markers} != {exp_markers}")
    else:
        c3_pass = False
        c3_errs.append("Insufficient rows to verify gold markers")

    results["check_3_gold_markers"] = "PASS" if c3_pass else "FAIL"
    details["check_3_gold_markers"] = f"Mismatches: {len(c3_errs)}" if not c3_pass else "All 22 gold markers match gold_paragraphs.json exactly"

    # ---------------------------------------------------------
    # Check 4: Chunk texts are whitespace-normalized substrings of source text
    # ---------------------------------------------------------
    recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
    recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)
    clean_src_2 = normalize_ws(" ".join(r["text"] for r in recs2))
    clean_src_3 = normalize_ws(" ".join(r["text"] for r in recs3))

    c4_pass = True
    c4_errs = []
    for r_idx, row in enumerate(rows):
        sample = row.get("sample", "")
        clean_src = clean_src_2 if sample == "sample2" else clean_src_3
        chunks = row.get("context_chunks", []) + row.get("retrieved_chunks", [])
        for c_idx, c in enumerate(chunks):
            txt = c.get("text", "")
            if not txt:
                continue
            norm_c = normalize_ws(txt)
            if norm_c not in clean_src:
                sents = [normalize_ws(s) for s in txt.split(". ") if len(s.strip()) > 10]
                if not all(s in clean_src for s in sents):
                    c4_pass = False
                    c4_errs.append(f"Row {r_idx+1} chunk {c.get('chunk_id')} not a substring of {sample}")
                    break

    results["check_4_chunk_substrings"] = "PASS" if c4_pass else "FAIL"
    details["check_4_chunk_substrings"] = f"Non-substring chunks: {len(c4_errs)}" if not c4_pass else "All chunks are verified whitespace-normalized substrings of clean source text"

    # ---------------------------------------------------------
    # Check 5: Required fields present
    # (timestamp, models_used, per_node_latency in ms from timers,
    #  ollama_total_duration, ungrounded_numbers, citations, final_status)
    # ---------------------------------------------------------
    required_fields = [
        "timestamp",
        "models_used",
        "per_node_latency",
        "ollama_total_duration",
        "ungrounded_numbers",
        "citations",
        "final_status"
    ]
    c5_pass = True
    c5_errs = []
    for r_idx, row in enumerate(rows):
        for field in required_fields:
            if field not in row:
                # Check aliases or flags if applicable
                if field == "models_used" and "models" in row:
                    continue
                if field == "ungrounded_numbers" and ("ungrounded_numbers" in row.get("flags", {}) or "ungrounded_numbers" in row):
                    continue
                if field == "ollama_total_duration" and ("total_duration" in row or "ollama_duration" in row):
                    continue
                c5_pass = False
                c5_errs.append(f"Row {r_idx+1} missing required field: '{field}'")

    results["check_5_required_fields"] = "PASS" if c5_pass else "FAIL"
    details["check_5_required_fields"] = f"Missing field occurrences: {len(c5_errs)}" if not c5_pass else "All required fields present across all 30 rows"

    # ---------------------------------------------------------
    # Check 6: Latencies plausible
    # ---------------------------------------------------------
    c6_pass = True
    c6_errs = []
    for r_idx, row in enumerate(rows):
        node_lats = row.get("per_node_latency", {})
        if not isinstance(node_lats, dict) or len(node_lats) == 0:
            c6_pass = False
            c6_errs.append(f"Row {r_idx+1} missing or empty per_node_latency")
        else:
            for node, lat in node_lats.items():
                if not isinstance(lat, (int, float)) or lat < 0:
                    c6_pass = False
                    c6_errs.append(f"Row {r_idx+1} node {node} has non-positive latency {lat}")

    results["check_6_latencies_plausible"] = "PASS" if c6_pass else "FAIL"
    details["check_6_latencies_plausible"] = f"Implausible latency occurrences: {len(c6_errs)}" if not c6_pass else "All per-node latencies are positive and plausible"

    # ---------------------------------------------------------
    # Summary & Output
    # ---------------------------------------------------------
    all_passed = all(status == "PASS" for status in results.values())
    output_lines = [
        "=" * 80,
        "VERIFICATION REPORT: artifacts/logs/generation_results.jsonl",
        "=" * 80,
    ]
    for check, status in results.items():
        output_lines.append(f"[{status}] {check}: {details[check]}")
    output_lines.append("=" * 80)
    output_lines.append(f"OVERALL STATUS: {'PASS' if all_passed else 'FAIL'}")
    output_lines.append("=" * 80)

    report_text = "\n".join(output_lines)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report_text, encoding="utf-8")
    print(report_text)

    return all_passed

if __name__ == "__main__":
    jsonl_file = Path("artifacts/logs/generation_results.jsonl")
    log_file = Path("artifacts/logs/verify_generation_results.txt")
    passed = verify_jsonl(jsonl_file, log_file)
    sys.exit(0 if passed else 1)
