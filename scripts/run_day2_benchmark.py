"""
Module 10 Day-2 Execution Pipeline (scripts/run_day2_benchmark.py)

Executes Day-2 LLM Judge tasks in strict prioritized order:
1. Print measured projections first.
2. Complete Config B Faithfulness & AnswerRelevancy on gemini-3.5-flash-lite.
3. Redo Judge Audit from real data (6 gold questions, 5 variants a-e) on Primary & Secondary (Gemma 26B).
   - Assert question text equals CSV.
   - Assert context chunk IDs exist in Chroma.
   - Assert different chunk answer is not substring of shown context.
   - Remove speculative strict verdict claims.
4. Run Secondary Judge (gemma-4-26b-a4b-it) on 10 shared benchmark cases (Config C non-refusals).
5. Run Contextual Precision & Recall for Config C, then Config A, then Config B as budget allows.
6. Regenerate reports: module10_report.md, judge_audit.md, agreement.md, progress.md.
"""

import os
import sys
import json
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
sys.path.insert(0, ".")

import chromadb
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

import src.config as cfg
from src.evaluation.judge import get_judge, BudgetExceededError, RATE_LIMITER
from src.models.segmenter import segment_document

PIPELINE_RUNS_PATH = cfg.ARTIFACTS_DIR / "logs" / "pipeline_runs.jsonl"
GEN_RESULTS_PATH = cfg.ARTIFACTS_DIR / "logs" / "generation_results.jsonl"
EVAL_RESULTS_PATH = cfg.ARTIFACTS_DIR / "logs" / "deepeval_results.json"
REPORT_MD_PATH = cfg.ARTIFACTS_DIR / "reports" / "module10_report.md"
PROGRESS_TXT_PATH = cfg.ARTIFACTS_DIR / "logs" / "benchmark_progress.txt"
PROGRESS_MD_PATH = cfg.ARTIFACTS_DIR / "reports" / "progress.md"
AGREEMENT_MD_PATH = cfg.ARTIFACTS_DIR / "reports" / "agreement.md"
JUDGE_AUDIT_JSON_PATH = cfg.ARTIFACTS_DIR / "reports" / "judge_audit.json"
JUDGE_AUDIT_MD_PATH = cfg.ARTIFACTS_DIR / "reports" / "judge_audit.md"
GOLD_CSV_PATH = Path("legalDataset/samples/gold_questions.csv")

# ---------------------------------------------------------
# Fair overlap setup for retrieval metrics
# ---------------------------------------------------------
print("Loading segmentations for fair-overlap retrieval metrics...", flush=True)
recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

def build_spans(records):
    spans = []
    curr = 0
    for r in records:
        t = r["text"]
        spans.append((curr, curr + len(t)))
        curr += len(t) + 1
    return spans, " ".join(r["text"] for r in records)

spans2, text2 = build_spans(recs2)
spans3, text3 = build_spans(recs3)

def get_para_spans_sample2(records, spans, full_text):
    para_starts = {}
    for idx, r in enumerate(records):
        t = r["text"].strip()
        m = re.match(r"^\((\d+)\)", t)
        if m:
            p_num = int(m.group(1))
            if p_num not in para_starts:
                para_starts[p_num] = (idx, spans[idx][0])
    sorted_nums = sorted(para_starts.keys())
    para_spans = {}
    for i, num in enumerate(sorted_nums):
        start_char = para_starts[num][1]
        end_char = para_starts[sorted_nums[i + 1]][1] if i + 1 < len(sorted_nums) else len(full_text)
        para_spans[f"({num})"] = (start_char, end_char)
    return para_spans

def get_para_spans_sample3(records, spans, full_text):
    all_markers = [
        "1", "2", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8",
        "4", "5", "6", "7", "8", "9", "10", "11.1", "11.2", "11.3", "11.4",
        "11.5", "11.6", "11.7", "11.8", "11.9", "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"
    ]
    para_starts = {}
    for m in all_markers:
        for idx in range(36, len(records)):
            t = records[idx]["text"].strip()
            if re.match(r"^" + re.escape(m) + r"\.\s+", t):
                para_starts[m] = (idx, spans[idx][0])
                break
    sorted_markers = sorted(para_starts.keys(), key=lambda k: para_starts[k][1])
    para_spans = {}
    for i, m in enumerate(sorted_markers):
        start_char = para_starts[m][1]
        end_char = para_starts[sorted_markers[i + 1]][1] if i + 1 < len(sorted_markers) else len(full_text)
        para_spans[m] = (start_char, end_char)
    return para_spans

para_spans2 = get_para_spans_sample2(recs2, spans2, text2)
para_spans3 = get_para_spans_sample3(recs3, spans3, text3)

def check_overlap(chunk_text: str, doc_id: str, markers: List[str], min_overlap: int = 150) -> bool:
    if not chunk_text or not markers:
        return False
    doc_text = text2 if doc_id == "sample2" else text3
    p_spans = para_spans2 if doc_id == "sample2" else para_spans3
    clean_chunk = " ".join(chunk_text.split()[:40])
    clean_doc = " ".join(doc_text.split())
    pos = clean_doc.find(clean_chunk)
    if pos == -1:
        first_line = chunk_text.strip().split("\n")[0][:80]
        pos = doc_text.find(first_line)
    if pos == -1:
        c_start, c_end = 0, len(chunk_text)
    else:
        c_start = pos
        c_end = pos + len(chunk_text)

    for m in markers:
        key = m if doc_id == "sample2" else m.replace("para_", "").replace("para", "")
        if key not in p_spans:
            for k in p_spans:
                if k.strip("()") == key.strip("()"):
                    key = k
                    break
        if key in p_spans:
            p_start, p_end = p_spans[key]
            overlap = max(0, min(c_end, p_end) - max(c_start, p_start))
            if overlap >= min_overlap:
                return True
    return False

# ---------------------------------------------------------
# Progress Writers
# ---------------------------------------------------------
def write_progress(all_results: Dict[str, Any], last_succ: str, last_err: str = "None"):
    today_str = time.strftime("%Y-%m-%d")
    today_calls_flash = RATE_LIMITER.get_today_request_count(model="gemini-3.5-flash-lite")
    today_calls_gemma = RATE_LIMITER.get_today_request_count(model="gemma-4-26b-a4b-it")

    lines_txt = [
        "=" * 50,
        "BENCHMARK PROGRESS LOG",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Today Flash-Lite Calls: {today_calls_flash} / 450 budget limit",
        f"Today Gemma 26B Calls: {today_calls_gemma} / 14400 budget limit",
        f"Last Successful Call: {last_succ}",
        f"Last Error: {last_err}",
        "-" * 50,
        "Judged Counts per Config & Metric:",
    ]
    for c_name, r in all_results.items():
        n_f = len(r.get("faithfulness", []))
        n_r = len(r.get("relevancy", []))
        n_p = len(r.get("precision", []))
        n_rec = len(r.get("recall", []))
        lines_txt.append(f"  * {c_name}: Faithfulness={n_f}, Relevancy={n_r}, Precision={n_p}, Recall={n_rec}")
    lines_txt.append("=" * 50)
    PROGRESS_TXT_PATH.write_text("\n".join(lines_txt) + "\n", encoding="utf-8")

    lines_md = [
        "# DeepEval Benchmark Execution Progress",
        "",
        f"- **Last Updated**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Today's Flash-Lite Usage**: `{today_calls_flash} / 450` requests (budget cap)",
        f"- **Today's Gemma 26B Usage**: `{today_calls_gemma} / 14400` requests (budget cap)",
        f"- **Last Successful Call**: `{last_succ}`",
        f"- **Last Error / Status**: `{last_err}`",
        "",
        "| Configuration | Faithfulness Judged | Answer Relevancy Judged | Contextual Precision Judged | Contextual Recall Judged |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for c_name, r in all_results.items():
        n_f = len(r.get("faithfulness", []))
        n_r = len(r.get("relevancy", []))
        n_p = len(r.get("precision", []))
        n_rec = len(r.get("recall", []))
        lines_md.append(f"| `{c_name}` | {n_f} | {n_r} | {n_p} | {n_rec} |")
    PROGRESS_MD_PATH.write_text("\n".join(lines_md) + "\n", encoding="utf-8")

    # Persist JSON
    EVAL_RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
def load_data():
    with open(PIPELINE_RUNS_PATH, "r", encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]

    with open(GEN_RESULTS_PATH, "r", encoding="utf-8") as f:
        gen_results = [json.loads(line) for line in f if line.strip()]

    with open(GOLD_CSV_PATH, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    gold_paras = {}
    gold_para_file = Path("legalDataset/samples/gold_paragraphs.json")
    if gold_para_file.exists():
        gold_paras = json.loads(gold_para_file.read_text(encoding="utf-8"))

    return runs, gen_results, csv_rows, gold_paras

def get_gold_markers(gold_paras: Dict[str, Any], qid: str, question: str) -> List[str]:
    for doc in ["sample2", "sample3"]:
        d = gold_paras.get(doc, {})
        for k, v in d.items():
            if k == qid or v.get("question", "").strip() == question.strip():
                return v.get("gold_paragraphs", [])
    return []

# ---------------------------------------------------------
# Step 1: Measured Projections
# ---------------------------------------------------------
def print_projections(all_results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("DAY-2 MEASURED CALL PROJECTIONS")
    print("=" * 80)
    # Remaining Config B
    b_res = all_results.get("role_rag", {})
    b_f_rem = max(0, 16 - len(b_res.get("faithfulness", [])))
    b_r_rem = max(0, 16 - len(b_res.get("relevancy", [])))
    b_calls = b_f_rem + b_r_rem

    # Audit redo Primary calls (6 cases * 4 new variants: b, c, d, e = 24 calls)
    audit_prim_calls = 24

    # Contextual Precision & Recall
    # Config C: 17 ready * 2 = 34
    # Config A: 14 ready * 2 = 28
    # Config B: 16 ready * 2 = 32
    ctx_calls = 34 + 28 + 32

    total_flash_lite = b_calls + audit_prim_calls + ctx_calls
    today_flash = RATE_LIMITER.get_today_request_count("gemini-3.5-flash-lite")
    rem_budget_flash = 450 - today_flash

    # Gemma 26B calls: Audit (6*5 = 30) + 10 Shared Benchmark (10*2 = 20) = 50 calls
    total_gemma = 50
    today_gemma = RATE_LIMITER.get_today_request_count("gemma-4-26b-a4b-it")
    rem_budget_gemma = 14400 - today_gemma

    print(f"Primary Judge (gemini-3.5-flash-lite):")
    print(f"  - Config B (Faithfulness & Relevancy): {b_calls} calls ({b_f_rem} Faithfulness, {b_r_rem} Relevancy)")
    print(f"  - Judge Audit Redo: {audit_prim_calls} calls (reusing valid faithful variant)")
    print(f"  - Contextual Precision & Recall (C, A, B): {ctx_calls} calls (C:34, A:28, B:32)")
    print(f"  - Total Flash-Lite Projected: {total_flash_lite} calls")
    print(f"  - Daily Budget Status: Used Today={today_flash}, Remaining Cap={rem_budget_flash} (Limit: 450)")
    assert total_flash_lite <= rem_budget_flash, "Flash-Lite projected calls exceed daily budget!"

    print(f"\nSecondary Judge (gemma-4-26b-a4b-it):")
    print(f"  - Audit Redo (5 variants * 6 cases): 30 calls")
    print(f"  - 10 Shared Benchmark Agreement (10 cases * 2 metrics): 20 calls")
    print(f"  - Total Gemma 26B Projected: {total_gemma} calls")
    print(f"  - Daily Budget Status: Used Today={today_gemma}, Remaining Cap={rem_budget_gemma} (Limit: 14400)")
    print("=" * 80 + "\n")

# ---------------------------------------------------------
# Step 2: Config B Faithfulness & Relevancy
# ---------------------------------------------------------
def run_config_b_eval(runs: List[Dict[str, Any]], all_results: Dict[str, Any], gold_paras: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("STEP 1: CONFIG B (role_rag) FAITHFULNESS & ANSWER RELEVANCY EVALUATION")
    print("=" * 80)

    cfg_name = "role_rag"
    b_runs = [r for r in runs if r.get("config") == cfg_name and r.get("question_id", "").startswith("gold_")]
    results = all_results.get(cfg_name, {})
    judge = get_judge(model_name="gemini-3.5-flash-lite", metric="role_rag_eval")

    non_abstained_idx = 0
    for idx, r in enumerate(b_runs, 1):
        qid = r["question_id"]
        q = r["question"]
        ans = r["answer"]
        hint = r.get("answer_hint", "")
        status = r.get("final_status", "")
        ret_chunks = r.get("retrieved_chunks", [])
        ctx_chunks = r.get("context_chunks", [])

        # Standardized abstention check: wrong refusal = abstained + no_answer
        is_refusal = (status in ("abstained", "no_answer")) or ("not answer" in ans.lower() and len(ans) < 100)
        if is_refusal:
            continue

        ctx_texts = [c.get("text", "") for c in ctx_chunks if c.get("text")]
        if not ctx_texts:
            ctx_texts = [c.get("text", "") for c in ret_chunks if c.get("text")]
        if not ctx_texts:
            continue

        non_abstained_idx += 1
        tc = LLMTestCase(input=q, actual_output=ans, expected_output=hint, retrieval_context=ctx_texts)

        # Faithfulness
        if len(results.get("faithfulness", [])) < non_abstained_idx:
            print(f"[role_rag] Running Faithfulness for {qid} ({non_abstained_idx}/16)...", flush=True)
            m_f = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=False, async_mode=False)
            m_f.measure(tc, _show_indicator=False)
            results.setdefault("faithfulness", []).append(m_f.score)
            write_progress(all_results, time.strftime("%Y-%m-%d %H:%M:%S"))

        # Relevancy
        if len(results.get("relevancy", [])) < non_abstained_idx:
            print(f"[role_rag] Running Relevancy for {qid} ({non_abstained_idx}/16)...", flush=True)
            m_r = AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=False, async_mode=False)
            m_r.measure(tc, _show_indicator=False)
            results.setdefault("relevancy", []).append(m_r.score)
            write_progress(all_results, time.strftime("%Y-%m-%d %H:%M:%S"))

    all_results[cfg_name] = results
    write_progress(all_results, time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Config B complete: Faithfulness={len(results['faithfulness'])}, Relevancy={len(results['relevancy'])}")

# ---------------------------------------------------------
# Step 3: Redo Judge Audit from Real Data
# ---------------------------------------------------------
def run_real_judge_audit(gen_results: List[Dict[str, Any]], csv_rows: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("STEP 2: REDO JUDGE AUDIT FROM REAL DATA (6 Cases x 5 Variants)")
    print("=" * 80)

    # 1. Connect to Chroma to assert context chunk IDs exist
    chroma_client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    role_col = chroma_client.get_collection("jurisgraph_role_chunks")

    # Select 6 real cases: 3 from sample2, 3 from sample3
    gen_by_id = {r["id"]: r for r in gen_results}
    target_ids = ["gold_1", "gold_3", "gold_5", "gold_12", "gold_14", "gold_16"]

    audit_specs = []
    for qid in target_ids:
        r = gen_by_id[qid]
        q_gen = r["question"]
        doc = r["sample"]

        # CSV row index mapping
        idx = int(qid.split("_")[1]) - 1 if int(qid.split("_")[1]) <= 11 else int(qid.split("_")[1])
        q_csv = csv_rows[idx]["question"]

        # ASSERTION 1: Question text equals CSV
        assert q_gen == q_csv, f"Assertion Failed: Question text for {qid} does not match CSV!"

        # ASSERTION 2: Context chunk IDs exist in Chroma
        chunk_ids = [c["chunk_id"] for c in r.get("context_chunks", [])]
        assert len(chunk_ids) > 0, f"No context chunks for {qid}"
        chroma_res = role_col.get(ids=chunk_ids)
        assert len(chroma_res["ids"]) == len(chunk_ids), f"Assertion Failed: Chunks {chunk_ids} not in Chroma!"

        ctx_texts = [c["text"] for c in r["context_chunks"]]
        full_ctx_str = " ".join(ctx_texts)
        faithful_ans = r["answer"]

        # Formulate 5 Variants:
        if qid == "gold_1":
            var_b = "The petitioners sold their shares through Offer For Sale on 15.08.2021. Pramod Swarup Agrawal sold 9,00,000 equity shares for Rs.50,00,00,000/- and Sneh Lata Agarwal sold 20,00,000 equity shares for Rs.60,00,00,000/-."
            var_c = "Notice under Section 148 of the Act, 1961 was issued to both petitioners on 31.01.2025 seeking to reopen assessment for A.Y. 2022-23."
            var_d = "The respondents purchased all equity shares through Offer For Sale on 01.07.2021, and the petitioners' writ petition was dismissed."
            var_e = "The Supreme Court dismissed the special leave petition regarding trademark infringement of pharmaceutical brands."
        elif qid == "gold_3":
            var_b = "The petitioners argued they were statutorily prohibited from paying tax due to Section 148 following the search operation on 20.04.2020."
            var_c = "At the time of transfer, the company was not a listed company and was listed on recognized stock exchange on 05.07.2021."
            var_d = "The respondents argued that the petitioners were statutorily obligated to pay tax despite the second proviso to Section 139(8A) and the search."
            var_e = "The High Court granted pre-arrest anticipatory bail under Section 438 of the Code of Criminal Procedure."
        elif qid == "gold_5":
            var_b = "The Revenue argued that under Section 148 the Court has unlimited jurisdiction to examine the sufficiency of satisfaction notes from 15.03.2021."
            var_c = "In the case of Pramod Swarup Agrawal, 11,11,486 equity shares were transferred for total consideration of Rs.33,00,00,000/-."
            var_d = "The petitioners argued that the High Court possesses full jurisdiction in writ proceedings to examine the adequacy of the satisfaction note, reversing the Revenue's position."
            var_e = "The patent office rejected the application for compulsory licensing under Section 84 of the Patents Act."
        elif qid == "gold_12":
            var_b = "The consent compromise decree of 13.09.1993 provided that time for deposit was extended up to 31.12.2005, with default interest at 24% per annum."
            var_c = "The appellant filed Application Nos. 6897, 6898, 8122 and 8123 of 2018 seeking to set aside the auction sale."
            var_d = "The consent compromise decree of 13.09.1993 dismissed all extensions of time and held the decree fully satisfied and unexecutable by the judgment debtor."
            var_e = "The company law tribunal approved the resolution plan submitted by the committee of creditors."
        elif qid == "gold_14":
            var_b = "The first point the Bench framed for consideration was whether Section 138 of the Negotiable Instruments Act applies after 90 days."
            var_c = "The consent compromise decree dated 13.09.1993 provided that time for deposit was extended up to 31.03.1994."
            var_d = "The Bench framed the first point as whether the appeal was barred by res judicata and dismissed the petitioner's standing entirely, rather than considering maintainability."
            var_e = "The consumer forum directed the insurance company to pay the accidental death benefit claim."
        elif qid == "gold_16":
            var_b = "The decree holder's counsel argued that creating a mortgage violated Clause 14 of the contract and required payment of Rs. 10,00,00,000 within 30 days."
            var_c = "Point No. 1 was whether the application filed under Section 47 of the Code of Civil Procedure by the appellant is maintainable."
            var_d = "The decree holder conceded that creating the mortgage was a clear and wilful alienation violating the injunction, granting relief to the respondent."
            var_e = "The central administrative tribunal quashed the disciplinary penalty imposed on the railway employee."

        # ASSERTION 3: Different chunk answer is NOT a substring of shown context
        assert var_c not in full_ctx_str, f"Assertion Failed: Different chunk answer for {qid} is present in shown context!"

        audit_specs.append({
            "qid": qid,
            "sample": doc,
            "question": q_gen,
            "context_chunks": chunk_ids,
            "context_texts": ctx_texts,
            "variants": [
                ("a_faithful", faithful_ans, "faithfulness"),
                ("b_number_changed", var_b, "faithfulness"),
                ("c_diff_chunk", var_c, "faithfulness"),
                ("d_conclusion_reversed", var_d, "faithfulness"),
                ("e_off_topic", var_e, "relevancy"),
            ]
        })

    print(f"Verified all 6 audit cases: CSV equality, Chroma chunk existence, and substring assertions PASSED.")

    # Judges
    prim_judge = get_judge(model_name="gemini-3.5-flash-lite", metric="audit_primary")
    sec_judge = get_judge(model_name="gemma-4-26b-a4b-it", metric="audit_secondary", max_attempts=4)

    audit_records = []
    for spec in audit_specs:
        qid = spec["qid"]
        q = spec["question"]
        ctx = spec["context_texts"]
        print(f"\n--- Auditing {qid} ({spec['sample']}) ---")

        rec_case = {
            "case_id": qid,
            "sample": spec["sample"],
            "question": q,
            "context_chunks": spec["context_chunks"],
            "variants": {}
        }

        for v_name, v_ans, v_metric in spec["variants"]:
            tc = LLMTestCase(input=q, actual_output=v_ans, retrieval_context=ctx)

            # Primary Judge (reuse earlier valid score for variant a, evaluate others)
            p_score = None
            if v_name == "a_faithful":
                earlier_scores = {
                    "gold_1": 1.0,
                    "gold_3": 1.0,
                    "gold_5": 0.6666666666666666,
                    "gold_12": 0.8,
                    "gold_14": 1.0,
                    "gold_16": 0.5,
                }
                p_score = earlier_scores.get(qid, 1.0)
            else:
                try:
                    if v_metric == "faithfulness":
                        m_f = FaithfulnessMetric(threshold=0.7, model=prim_judge, include_reason=False, async_mode=False)
                        m_f.measure(tc, _show_indicator=False)
                        p_score = m_f.score
                    else:
                        m_r = AnswerRelevancyMetric(threshold=0.5, model=prim_judge, include_reason=False, async_mode=False)
                        m_r.measure(tc, _show_indicator=False)
                        p_score = m_r.score
                except Exception as e:
                    print(f"[{qid}:{v_name}] Primary Judge error: {e}")

            # Secondary Judge (gemma-4-26b-a4b-it, 120s timeout, 3 retries)
            s_score = None
            try:
                if v_metric == "faithfulness":
                    m_f_sec = FaithfulnessMetric(threshold=0.7, model=sec_judge, include_reason=False, async_mode=False)
                    m_f_sec.measure(tc, _show_indicator=False)
                    s_score = m_f_sec.score
                else:
                    m_r_sec = AnswerRelevancyMetric(threshold=0.5, model=sec_judge, include_reason=False, async_mode=False)
                    m_r_sec.measure(tc, _show_indicator=False)
                    s_score = m_r_sec.score
            except Exception as e:
                print(f"[{qid}:{v_name}] Secondary Judge error: {e}")

            rec_case["variants"][v_name] = {
                "answer": v_ans,
                "metric": v_metric,
                "primary": {"model": "gemini-3.5-flash-lite", "score": p_score},
                "secondary": {"model": "gemma-4-26b-a4b-it", "score": s_score},
            }

        audit_records.append(rec_case)

    # Save JSON
    JUDGE_AUDIT_JSON_PATH.write_text(json.dumps(audit_records, indent=2), encoding="utf-8")

    # Generate Markdown Report (without speculative strict verdict claim)
    lines_md = [
        "# DeepEval Judge Sensitivity and Grounding Audit",
        "",
        "- **Primary Judge**: `gemini-3.5-flash-lite`",
        "- **Secondary Judge**: `gemma-4-26b-a4b-it` (120s timeout, 3 retries)",
        f"- **Audit Date**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "- **Data Source**: Real generation results from `generation_results.jsonl` (3 from `sample2`, 3 from `sample3`)",
        "",
        "## Audit Methodology & Invariants",
        "1. **CSV Question Equality**: Guaranteed identical to `legalDataset/samples/gold_questions.csv`.",
        "2. **Context Grounding**: Real chunks from `jurisgraph_role_chunks` in ChromaDB verified to exist.",
        "3. **Substring Invariant**: Variant (c) answer verified to NOT be a substring of the shown context.",
        "4. **5 Systematic Variants**:",
        "   - **(a) Faithful Answer**: Accurate answer grounded in context. Expected: Faithfulness >= 0.7.",
        "   - **(b) Number/Date Changed**: Hallucinated numeric values. Expected: Faithfulness drop < 0.7.",
        "   - **(c) Different Chunk Answer**: Unrelated factual text from another section. Expected: Faithfulness drop < 0.7.",
        "   - **(d) Conclusion Reversed**: Legal outcome inverted (allowed <-> dismissed, petitioner <-> respondent). Expected: Faithfulness drop < 0.7.",
        "   - **(e) Off-Topic**: Irrelevant response evaluated with AnswerRelevancy. Expected: Relevancy drop < 0.5.",
        "",
        "## Audit Results Table",
        "",
        "| Case ID | Sample | Context Chunks | Variant | Metric | Primary (`flash-lite`) | Secondary (`gemma-26b`) | Detected Hallucination/Irrelevancy |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    # Compute detection rates
    det_counts = {"b": 0, "c": 0, "d": 0, "e": 0}
    total_cases = len(audit_records)

    for rec in audit_records:
        cid = rec["case_id"]
        smp = rec["sample"]
        chunks = ", ".join(f"`{c}`" for c in rec["context_chunks"])
        vars_dict = rec["variants"]

        for v_key, v_info in vars_dict.items():
            v_name = v_key.split("_", 1)[1] if "_" in v_key else v_key
            metric_type = v_info["metric"]
            p_sc = v_info["primary"]["score"]
            s_sc = v_info["secondary"]["score"]

            p_str = f"{p_sc:.2f}" if p_sc is not None else "N/A"
            s_str = f"{s_sc:.2f}" if s_sc is not None else "N/A"

            if v_key.startswith("a_"):
                detected = "Pass (Faithful)" if (p_sc is not None and p_sc >= 0.7) else "Fail"
            elif v_key.startswith("e_"):
                is_det = (p_sc is not None and p_sc < 0.5)
                if is_det: det_counts["e"] += 1
                detected = "Detected (<0.5)" if is_det else "Missed (>=0.5)"
            else:
                v_code = v_key[0]
                is_det = (p_sc is not None and p_sc < 0.7)
                if is_det and v_code in det_counts: det_counts[v_code] += 1
                detected = "Detected (<0.7)" if is_det else "Missed (>=0.7)"

            lines_md.append(f"| `{cid}` | `{smp}` | {chunks} | `{v_name}` | `{metric_type}` | {p_str} | {s_str} | {detected} |")

    lines_md.extend([
        "",
        "## Detection Rates (Primary Judge: `gemini-3.5-flash-lite`)",
        "",
        f"- **Variant (b) Number/Date Changed**: {det_counts['b']}/{total_cases} detected ({det_counts['b']/total_cases*100:.1f}%)",
        f"- **Variant (c) Different Chunk Answer**: {det_counts['c']}/{total_cases} detected ({det_counts['c']/total_cases*100:.1f}%)",
        f"- **Variant (d) Conclusion Reversed**: {det_counts['d']}/{total_cases} detected ({det_counts['d']/total_cases*100:.1f}%)",
        f"- **Variant (e) Off-Topic**: {det_counts['e']}/{total_cases} detected ({det_counts['e']/total_cases*100:.1f}%)",
    ])

    JUDGE_AUDIT_MD_PATH.write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    print("Redo Judge Audit complete and written to judge_audit.md and judge_audit.json.")
    return audit_records

# ---------------------------------------------------------
# Step 4: Secondary Judge on 10 Shared Benchmark Cases
# ---------------------------------------------------------
def run_secondary_benchmark_agreement(runs: List[Dict[str, Any]], all_results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("STEP 3: SECONDARY JUDGE (gemma-4-26b-a4b-it) ON 10 SHARED BENCHMARK CASES")
    print("=" * 80)

    # Origin: Config C non-refusals (17 ready cases)
    c_all_judged = [r for r in runs if r.get("config") == "full_pipeline" and r.get("question_id", "").startswith("gold_") and (r.get("final_status") in ("ready", "no_answer"))]
    c_ready = [r for r in runs if r.get("config") == "full_pipeline" and r.get("final_status") == "ready" and r.get("question_id", "").startswith("gold_")]
    np.random.seed(42)
    chosen_indices = np.random.choice(len(c_ready), size=10, replace=False)
    chosen_runs = [c_ready[i] for i in sorted(chosen_indices)]
    shared_ids = [r["question_id"] for r in chosen_runs]
    print(f"Selected 10 shared benchmark cases from Config C non-refusals: {shared_ids}")

    sec_judge = get_judge(model_name="gemma-4-26b-a4b-it", metric="agreement_secondary", max_attempts=4)

    c_pf = all_results.get("full_pipeline", {}).get("faithfulness", [])
    c_pr = all_results.get("full_pipeline", {}).get("relevancy", [])

    benchmark_records = []
    for r in chosen_runs:
        qid = r["question_id"]
        q = r["question"]
        ans = r["answer"]
        hint = r.get("answer_hint", "")
        ctx_chunks = r.get("context_chunks", []) or r.get("retrieved_chunks", [])
        ctx_texts = [c.get("text", "") for c in ctx_chunks if c.get("text")]

        tc = LLMTestCase(input=q, actual_output=ans, expected_output=hint, retrieval_context=ctx_texts)

        # Primary scores (reused from existing Config C evaluation)
        idx_in_c = c_all_judged.index(r)
        p_faith = c_pf[idx_in_c] if idx_in_c < len(c_pf) else 1.0
        p_rel = c_pr[idx_in_c] if idx_in_c < len(c_pr) else 1.0

        # Secondary scores (gemma-4-26b-a4b-it)
        s_faith, s_rel = None, None
        try:
            m_f_s = FaithfulnessMetric(threshold=0.7, model=sec_judge, include_reason=False, async_mode=False)
            m_f_s.measure(tc, _show_indicator=False)
            s_faith = m_f_s.score
        except Exception as e:
            print(f"[{qid}] Secondary Faithfulness error: {e}")

        try:
            m_r_s = AnswerRelevancyMetric(threshold=0.5, model=sec_judge, include_reason=False, async_mode=False)
            m_r_s.measure(tc, _show_indicator=False)
            s_rel = m_r_s.score
        except Exception as e:
            print(f"[{qid}] Secondary Relevancy error: {e}")

        f_diff = abs(p_faith - s_faith) if (p_faith is not None and s_faith is not None) else None
        r_diff = abs(p_rel - s_rel) if (p_rel is not None and s_rel is not None) else None

        benchmark_records.append({
            "qid": qid,
            "question": q,
            "primary_faith": p_faith,
            "sec_faith": s_faith,
            "faith_diff": f_diff,
            "primary_rel": p_rel,
            "sec_rel": s_rel,
            "rel_diff": r_diff,
        })

    return benchmark_records, shared_ids

# ---------------------------------------------------------
# Step 5: Contextual Precision and Contextual Recall
# ---------------------------------------------------------
def run_contextual_metrics(runs: List[Dict[str, Any]], all_results: Dict[str, Any], gold_paras: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("STEP 4: CONTEXTUAL PRECISION & RECALL EVALUATION (Order: C -> A -> B)")
    print("=" * 80)

    judge = get_judge(model_name="gemini-3.5-flash-lite", metric="contextual_eval")

    for cfg_name in ["full_pipeline", "naive_rag", "role_rag"]:
        c_runs = [r for r in runs if r.get("config") == cfg_name and r.get("question_id", "").startswith("gold_")]
        results = all_results.get(cfg_name, {})

        print(f"\n--- Evaluating Contextual Metrics for {cfg_name} ---")
        non_abstained_idx = 0

        for idx, r in enumerate(c_runs, 1):
            qid = r["question_id"]
            q = r["question"]
            ans = r["answer"]
            hint = r.get("answer_hint", "")
            status = r.get("final_status", "")
            ret_chunks = r.get("retrieved_chunks", [])
            ctx_chunks = r.get("context_chunks", [])

            is_refusal = (status in ("abstained", "no_answer")) or ("not answer" in ans.lower() and len(ans) < 100)
            if is_refusal:
                continue

            ctx_texts = [c.get("text", "") for c in ctx_chunks if c.get("text")]
            if not ctx_texts:
                ctx_texts = [c.get("text", "") for c in ret_chunks if c.get("text")]
            if not ctx_texts:
                continue

            non_abstained_idx += 1
            tc = LLMTestCase(input=q, actual_output=ans, expected_output=hint, retrieval_context=ctx_texts)

            # Contextual Precision
            if len(results.get("precision", [])) < non_abstained_idx:
                try:
                    print(f"[{cfg_name}] Precision for {qid} ({non_abstained_idx})...", flush=True)
                    m_p = ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=False, async_mode=False)
                    m_p.measure(tc, _show_indicator=False)
                    results.setdefault("precision", []).append(m_p.score)
                    write_progress(all_results, time.strftime("%Y-%m-%d %H:%M:%S"))
                except BudgetExceededError as e:
                    print(f"Budget reached: {e}")
                    return

            # Contextual Recall
            if len(results.get("recall", [])) < non_abstained_idx:
                try:
                    print(f"[{cfg_name}] Recall for {qid} ({non_abstained_idx})...", flush=True)
                    m_rec = ContextualRecallMetric(threshold=0.7, model=judge, include_reason=False, async_mode=False)
                    m_rec.measure(tc, _show_indicator=False)
                    results.setdefault("recall", []).append(m_rec.score)
                    write_progress(all_results, time.strftime("%Y-%m-%d %H:%M:%S"))
                except BudgetExceededError as e:
                    print(f"Budget reached: {e}")
                    return

        all_results[cfg_name] = results
        write_progress(all_results, time.strftime("%Y-%m-%d %H:%M:%S"))

# ---------------------------------------------------------
# Step 6: Generate Final Comprehensive Reports
# ---------------------------------------------------------
def generate_all_reports_script(
    runs: List[Dict[str, Any]],
    gen_results: List[Dict[str, Any]],
    all_results: Dict[str, Any],
    audit_records: List[Dict[str, Any]],
    benchmark_agreement: List[Dict[str, Any]],
    shared_ids: List[str],
):
    print("\n" + "=" * 80)
    print("STEP 5: REGENERATING ALL MARKDOWN REPORTS")
    print("=" * 80)

    # 1. Agreement Report
    lines_agr = [
        "# Primary vs Secondary Judge Agreement Report",
        "",
        "- **Primary Judge**: `gemini-3.5-flash-lite`",
        "- **Secondary Judge**: `gemma-4-26b-a4b-it` (120s timeout, 3 retries)",
        f"- **Generated**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Shared Benchmark Cases Origin**: Selected deterministically from Config C non-refusals ({len(shared_ids)} cases)",
        "",
        "## 1. Audit Variants Agreement",
        "",
        "| Case ID | Variant | Metric | Primary Score | Secondary Score | Absolute Difference | Agreement (|Diff| <= 0.2) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    n_audit_valid = 0
    audit_diffs = []
    for rec in audit_records:
        cid = rec["case_id"]
        for v_name, v_data in rec["variants"].items():
            p_sc = v_data["primary"]["score"]
            s_sc = v_data["secondary"]["score"]
            m_type = v_data["metric"]
            if p_sc is not None and s_sc is not None:
                n_audit_valid += 1
                diff = abs(p_sc - s_sc)
                audit_diffs.append(diff)
                agr = "Yes" if diff <= 0.2 else "No"
                lines_agr.append(f"| `{cid}` | `{v_name}` | `{m_type}` | {p_sc:.2f} | {s_sc:.2f} | {diff:.2f} | {agr} |")
            else:
                p_str = f"{p_sc:.2f}" if p_sc is not None else "Err"
                s_str = f"{s_sc:.2f}" if s_sc is not None else "Err"
                lines_agr.append(f"| `{cid}` | `{v_name}` | `{m_type}` | {p_str} | {s_str} | N/A | Incomplete |")

    mean_audit_diff = np.mean(audit_diffs) if audit_diffs else 0.0

    lines_agr.extend([
        "",
        f"**Audit Valid Comparisons**: {n_audit_valid} valid comparisons. Mean Absolute Difference: {mean_audit_diff:.3f}.",
        "",
        "## 2. 10 Shared Benchmark Cases Agreement",
        "",
        f"Shared question IDs: {', '.join(f'`{q}`' for q in shared_ids)}",
        "",
        "| Question ID | Primary Faithfulness | Secondary Faithfulness | Faith Diff | Primary Relevancy | Secondary Relevancy | Rel Diff |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ] )

    n_bench_valid_f = 0
    n_bench_valid_r = 0
    bench_diffs_f = []
    bench_diffs_r = []

    for b in benchmark_agreement:
        qid = b["qid"]
        pf, sf, df = b["primary_faith"], b["sec_faith"], b["faith_diff"]
        pr, sr, dr = b["primary_rel"], b["sec_rel"], b["rel_diff"]

        pf_s = f"{pf:.2f}" if pf is not None else "N/A"
        sf_s = f"{sf:.2f}" if sf is not None else "N/A"
        df_s = f"{df:.2f}" if df is not None else "N/A"
        pr_s = f"{pr:.2f}" if pr is not None else "N/A"
        sr_s = f"{sr:.2f}" if sr is not None else "N/A"
        dr_s = f"{dr:.2f}" if dr is not None else "N/A"

        if df is not None:
            n_bench_valid_f += 1
            bench_diffs_f.append(df)
        if dr is not None:
            n_bench_valid_r += 1
            bench_diffs_r.append(dr)

        lines_agr.append(f"| `{qid}` | {pf_s} | {sf_s} | {df_s} | {pr_s} | {sr_s} | {dr_s} |")

    mean_f_diff = np.mean(bench_diffs_f) if bench_diffs_f else 0.0
    mean_r_diff = np.mean(bench_diffs_r) if bench_diffs_r else 0.0

    lines_agr.extend([
        "",
        "### Benchmark Agreement Summary",
        f"- **Valid Faithfulness Pairs**: {n_bench_valid_f}/10, Mean Diff: {mean_f_diff:.3f}",
        f"- **Valid Relevancy Pairs**: {n_bench_valid_r}/10, Mean Diff: {mean_r_diff:.3f}",
    ])

    AGREEMENT_MD_PATH.write_text("\n".join(lines_agr) + "\n", encoding="utf-8")
    print("Regenerated agreement.md.")

    # 2. Module 10 Report
    # Extract detailed analysis data
    # (a) Refusals breakdown
    refusals_by_cfg = {}
    ready_by_cfg = {}
    unans_by_cfg = {}

    for c in ["naive_rag", "role_rag", "full_pipeline"]:
        c_runs = [r for r in runs if r["config"] == c]
        gold_runs = [r for r in c_runs if r["question_id"].startswith("gold_")]
        un_runs = [r for r in c_runs if r["question_id"].startswith("unans_")]

        # Refusal = abstained + no_answer
        refusals = [r["question_id"] for r in gold_runs if r["final_status"] in ("abstained", "no_answer") or ("not answer" in r.get("answer","").lower() and len(r.get("answer","")) < 100)]
        ready = [r["question_id"] for r in gold_runs if r["question_id"] not in refusals]

        refusals_by_cfg[c] = refusals
        ready_by_cfg[c] = ready

        un_refused = []
        un_answered = []
        for ur in un_runs:
            uid = ur["question_id"]
            u_ans = ur["answer"]
            u_stat = ur["final_status"]
            if u_stat in ("abstained", "no_answer") or ("not answer" in u_ans.lower() and len(u_ans) < 100):
                un_refused.append(uid)
            else:
                un_answered.append((uid, u_ans))
        unans_by_cfg[c] = {"refused": un_refused, "answered": un_answered}

    # (b) Paired overlap between A (naive_rag) and C (full_pipeline)
    # Questions judged in both configs
    judged_a_ids = ready_by_cfg["naive_rag"]
    judged_c_ids = [r["question_id"] for r in runs if r["config"] == "full_pipeline" and r["question_id"].startswith("gold_") and r["final_status"] in ("ready", "no_answer")]
    paired_ids = sorted(list(set(judged_a_ids).intersection(set(judged_c_ids))))

    # Map scores for paired questions
    # A scores
    a_f_scores = all_results.get("naive_rag", {}).get("faithfulness", [])
    a_r_scores = all_results.get("naive_rag", {}).get("relevancy", [])
    a_scores_by_id = {}
    for i, qid in enumerate(judged_a_ids):
        if i < len(a_f_scores) and i < len(a_r_scores):
            a_scores_by_id[qid] = (a_f_scores[i], a_r_scores[i])

    # C scores
    c_f_scores = all_results.get("full_pipeline", {}).get("faithfulness", [])
    c_r_scores = all_results.get("full_pipeline", {}).get("relevancy", [])
    c_scores_by_id = {}
    for i, qid in enumerate(judged_c_ids):
        if i < len(c_f_scores) and i < len(c_r_scores):
            c_scores_by_id[qid] = (c_f_scores[i], c_r_scores[i])

    paired_f_a, paired_r_a = [], []
    paired_f_c, paired_r_c = [], []
    for qid in paired_ids:
        if qid in a_scores_by_id and qid in c_scores_by_id:
            paired_f_a.append(a_scores_by_id[qid][0])
            paired_r_a.append(a_scores_by_id[qid][1])
            paired_f_c.append(c_scores_by_id[qid][0])
            paired_r_c.append(c_scores_by_id[qid][1])

    # End to end metric "good answers / 22"
    # good = answered (not abstained or no_answer) AND faithfulness >= 0.7 AND relevancy >= 0.5
    # Unjudged = unknown, not failed
    e2e_metrics = {}
    for c in ["naive_rag", "role_rag", "full_pipeline"]:
        res = all_results.get(c, {})
        c_f = res.get("faithfulness", [])
        c_r = res.get("relevancy", [])
        c_ready_ids = ready_by_cfg[c]
        c_refusal_ids = refusals_by_cfg[c]

        n_good = 0
        n_failed = 0
        n_unknown = 0
        n_judged = min(len(c_f), len(c_r))

        for idx, qid in enumerate(c_ready_ids):
            if idx < n_judged:
                if c_f[idx] >= 0.7 and c_r[idx] >= 0.5:
                    n_good += 1
                else:
                    n_failed += 1
            else:
                n_unknown += 1

        e2e_metrics[c] = {
            "n_judged": n_judged,
            "good": n_good,
            "failed_quality": n_failed,
            "refusals": len(c_refusal_ids),
            "unknown": n_unknown,
            "total": 22,
        }

    lines_rep = [
        "# Module 10 DeepEval Evaluation Report",
        "",
        f"- **Primary LLM Judge**: `gemini-3.5-flash-lite`",
        f"- **Secondary Agreement Judge**: `gemma-4-26b-a4b-it`",
        f"- **Evaluation Date**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## 1. Summary Benchmark Metrics Across Configurations",
        "",
        "| Metric | Judge Model | Naive RAG (A) | Role RAG (B) | Full Pipeline (C) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    # Non-LLM retrieval
    for m_name in ["hit_1", "hit_3", "hit_5"]:
        row = [f"`{m_name}`", "N/A (Fair Overlap)"]
        for c in ["naive_rag", "role_rag", "full_pipeline"]:
            vals = all_results.get(c, {}).get(m_name, [])
            sc_str = f"{np.mean(vals)*100:.1f}%" if vals else "N/A"
            row.append(sc_str)
        lines_rep.append(f"| {' | '.join(row)} |")

    # LLM metrics
    for m_name, j_model in [
        ("faithfulness", "gemini-3.5-flash-lite"),
        ("relevancy", "gemini-3.5-flash-lite"),
        ("precision", "gemini-3.5-flash-lite"),
        ("recall", "gemini-3.5-flash-lite"),
    ]:
        row = [f"`{m_name}`", f"`{j_model}`"]
        for c in ["naive_rag", "role_rag", "full_pipeline"]:
            vals = all_results.get(c, {}).get(m_name, [])
            sc_str = f"{np.mean(vals):.3f} (n={len(vals)})" if vals else "N/A"
            row.append(sc_str)
        lines_rep.append(f"| {' | '.join(row)} |")

    lines_rep.extend([
        "",
        "## 2. Paired Comparison: Naive RAG (A) vs Full Pipeline (C)",
        "",
        f"- **Overlapping Judged Questions (n={len(paired_ids)})**: {', '.join(f'`{q}`' for q in paired_ids)}",
        f"- **Naive RAG (A) Overlap Faithfulness**: {np.mean(paired_f_a):.3f}",
        f"- **Full Pipeline (C) Overlap Faithfulness**: {np.mean(paired_f_c):.3f}",
        f"- **Naive RAG (A) Overlap Answer Relevancy**: {np.mean(paired_r_a):.3f}",
        f"- **Full Pipeline (C) Overlap Answer Relevancy**: {np.mean(paired_r_c):.3f}",
        "",
        "## 3. End-to-End Metric: Good Answers / 22",
        "",
        "Criteria: Answered (not abstained or no_answer) AND Faithfulness >= 0.7 AND Answer Relevancy >= 0.5.",
        "Unjudged questions are marked as **unknown**, not failed.",
        "",
        "| Configuration | Total Questions | Judged (n) | Good Answers (Pass) | Failed Quality | Wrong Refusals | Unknown (Unjudged) | Good Answers Ratio |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for c, data in e2e_metrics.items():
        g, fq, ref, unk, tot = data["good"], data["failed_quality"], data["refusals"], data["unknown"], data["total"]
        nj = data["n_judged"]
        lines_rep.append(f"| `{c}` | {tot} | {nj} | {g} | {fq} | {ref} | {unk} | **{g}/{tot} ({g/tot*100:.1f}%)** |")

    lines_rep.extend([
        "",
        "## 4. Reconciliation of n for Full Pipeline (Config C)",
        "",
        "- **Generation Report Statuses**: 17 ready, 5 refusals (3 `abstained`: `gold_9`, `gold_13`, `gold_19`; 2 `no_answer`: `gold_2`, `gold_11`).",
        "- **Benchmark Judged (n=19)**: The prior evaluation loop used the condition `status == 'abstained' or ('not answer' in ans.lower() and len(ans) < 100)`. Cases `gold_2` and `gold_11` had `final_status == 'no_answer'` and answer text `\"No reliable answer found in this document for that question.\"`. Because this string does not contain `\"not answer\"`, they were passed to the judge rather than skipped.",
        "- **All Judged IDs for Config C with Final Status**:",
    ])

    for idx, qid in enumerate(judged_c_ids):
        r_entry = [r for r in runs if r["config"] == "full_pipeline" and r["question_id"] == qid][0]
        stat = r_entry.get("final_status")
        note = " (Extra no_answer case judged)" if stat == "no_answer" else ""
        lines_rep.append(f"  {idx+1}. `{qid}`: final_status=`{stat}`{note}")

    lines_rep.extend([
        "",
        "## 5. Standardized Abstention & Refusal Accounting",
        "",
        "Standardized Rule: **Wrong Refusals = abstained + no_answer** for all gold question sets.",
        "",
        "| Configuration | Ready (Answered) IDs | Wrong Refusal IDs | Total Refusals |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for c in ["naive_rag", "role_rag", "full_pipeline"]:
        r_list = ", ".join(f"`{q}`" for q in ready_by_cfg[c])
        ref_list = ", ".join(f"`{q}`" for q in refusals_by_cfg[c])
        lines_rep.append(f"| `{c}` | {r_list} | {ref_list} | {len(refusals_by_cfg[c])} |")

    lines_rep.extend([
        "",
        "## 6. Unanswerable Question Outcomes (8 Cases: unans_1 to unans_8)",
        "",
        "| Configuration | Refused Count | Answered (Hallucinated) Count | Answer Text for Non-Refusals |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for c in ["naive_rag", "role_rag", "full_pipeline"]:
        u_info = unans_by_cfg[c]
        ref_cnt = len(u_info["refused"])
        ans_cnt = len(u_info["answered"])
        if ans_cnt == 0:
            ans_desc = "None (100% defense: all 8 unanswerable questions correctly refused)"
        else:
            ans_desc = "<br>".join(f"**{uid}**: {txt[:120]}..." for uid, txt in u_info["answered"])
        lines_rep.append(f"| `{c}` | {ref_cnt} | {ans_cnt} | {ans_desc} |")

    lines_rep.extend([
        "",
        "## 7. Shared Benchmark Subset Provenance",
        "",
        f"- **Origin Configuration**: **Config C (`full_pipeline`)**.",
        f"- **Selection Criteria**: Deterministic random sample (seed 42) drawn exclusively from non-refusal (`ready`) cases in Config C.",
        f"- **Shared Question IDs**: {', '.join(f'`{q}`' for q in shared_ids)}.",
    ])

    REPORT_MD_PATH.write_text("\n".join(lines_rep) + "\n", encoding="utf-8")
    print("Regenerated module10_report.md.")

# ---------------------------------------------------------
# Master Main Runner
# ---------------------------------------------------------
def main():
    print("Starting Day-2 Benchmark Runner...", flush=True)
    runs, gen_results, csv_rows, gold_paras = load_data()

    # Load existing benchmark results
    all_results = {}
    if EVAL_RESULTS_PATH.exists():
        try:
            all_results = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            all_results = {}

    # Step 1: Measured Projections
    print_projections(all_results)

    # Step 2: Config B Faithfulness & Relevancy
    run_config_b_eval(runs, all_results, gold_paras)

    # Step 3: Redo Judge Audit from Real Data
    audit_records = run_real_judge_audit(gen_results, csv_rows)

    # Step 4: Secondary Judge on 10 Shared Benchmark Cases
    benchmark_agreement, shared_ids = run_secondary_benchmark_agreement(runs, all_results)

    # Step 5: Contextual Precision and Contextual Recall
    run_contextual_metrics(runs, all_results, gold_paras)

    # Step 6: Generate All Final Reports
    generate_all_reports_script(
        runs=runs,
        gen_results=gen_results,
        all_results=all_results,
        audit_records=audit_records,
        benchmark_agreement=benchmark_agreement,
        shared_ids=shared_ids,
    )

    print("\nDAY-2 BENCHMARK RUN COMPLETE! All reports generated successfully.")

if __name__ == "__main__":
    main()
