"""
Comprehensive evaluation script for Part A (Fair Retrieval Metric)
and Part B (Module 7: Relevance Grader Evaluation & Latency/Determinism).

Saves outputs to:
- artifacts/logs/fair_retrieval_eval.txt
- artifacts/logs/grader_eval.txt
"""

import sys
from pathlib import Path
import json
import re
from typing import Dict, Tuple, Any

sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore
from src.rag.role_mapper import map_query_to_roles
from src.rag.graph import auto_escalate_retrieval
from src.rag.grader import grade_chunks

FAIR_RETRIEVAL_LOG = Path("artifacts/logs/fair_retrieval_eval.txt")
GRADER_EVAL_LOG = Path("artifacts/logs/grader_eval.txt")
GOLD_JSON_PATH = Path("legalDataset/samples/gold_paragraphs.json")

# 1. Load data and setup document spans
recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

def build_doc_spans(records):
    sent_spans = []
    curr = 0
    full_text_parts = []
    for r in records:
        text = r["text"]
        full_text_parts.append(text)
        start = curr
        end = curr + len(text)
        sent_spans.append((start, end))
        curr = end + 1
    return sent_spans, " ".join(full_text_parts)

sent_spans2, text2 = build_doc_spans(recs2)
sent_spans3, text3 = build_doc_spans(recs3)

def find_all_paragraphs_sample2(records, sent_spans, full_text):
    para_starts = {}
    for idx, r in enumerate(records):
        t = r["text"].strip()
        m = re.match(r"^\((\d+)\)", t)
        if m:
            p_num = int(m.group(1))
            if p_num not in para_starts:
                para_starts[p_num] = (idx, sent_spans[idx][0])
    sorted_nums = sorted(para_starts.keys())
    para_spans = {}
    for i, num in enumerate(sorted_nums):
        start_char = para_starts[num][1]
        if i + 1 < len(sorted_nums):
            next_num = sorted_nums[i + 1]
            end_char = para_starts[next_num][1]
        else:
            end_char = len(full_text)
        para_spans[f"({num})"] = (start_char, end_char)
    return para_spans

def find_all_paragraphs_sample3(records, sent_spans, full_text):
    markers = [
        "1", "2", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8",
        "4", "5", "6", "7", "8", "9", "10", "11.1", "11.2", "11.3", "11.4",
        "11.5", "11.6", "11.7", "11.8", "11.9", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"
    ]
    para_starts = {}
    for m in markers:
        for idx, r in enumerate(records):
            t = r["text"].strip()
            if re.match(r"^" + re.escape(m) + r"[\.\)\s]", t):
                para_starts[m] = (idx, sent_spans[idx][0])
                break
    sorted_markers = sorted(para_starts.keys(), key=lambda k: para_starts[k][1])
    para_spans = {}
    for i, m in enumerate(sorted_markers):
        start_char = para_starts[m][1]
        if i + 1 < len(sorted_markers):
            next_m = sorted_markers[i + 1]
            end_char = para_starts[next_m][1]
        else:
            end_char = len(full_text)
        para_spans[m] = (start_char, end_char)
    return para_spans

para_spans_2 = find_all_paragraphs_sample2(recs2, sent_spans2, text2)
para_spans_3 = find_all_paragraphs_sample3(recs3, sent_spans3, text3)

gold_data = json.loads(GOLD_JSON_PATH.read_text(encoding="utf-8"))
store = LegalVectorStore()

def get_chunk_char_span(chunk: Dict[str, Any], doc_id: str) -> Tuple[int, int]:
    meta = chunk.get("metadata", {})
    if "start_char_idx" in meta and "end_char_idx" in meta:
        return int(meta["start_char_idx"]), int(meta["end_char_idx"])
    sent_spans = sent_spans2 if doc_id == "sample2" else sent_spans3
    s_start = int(meta.get("start_sentence_idx", 0))
    s_end = int(meta.get("end_sentence_idx", s_start))
    s_start = min(s_start, len(sent_spans) - 1)
    s_end = min(s_end, len(sent_spans) - 1)
    return sent_spans[s_start][0], sent_spans[s_end][1]

def check_old_marker_hit(chunk_text: str, markers: list) -> bool:
    text = chunk_text.strip()
    for m in markers:
        m_str = str(m).strip()
        if m_str.startswith("(") and m_str.endswith(")"):
            num = m_str[1:-1]
            pat = re.compile(
                r"(?:^|\n|[\.\?\!]\s+)(?:\(" + re.escape(num) + r"\)|\b" + re.escape(num) + r"\.)(?!\d)",
                re.IGNORECASE,
            )
            if pat.search(text):
                return True
        else:
            pat = re.compile(
                r"(?:^|\n|[\.\?\!]\s+)(?:" + re.escape(m_str) + r"[\.\)\s])(?!\d{1,2}\.\d{4})",
                re.IGNORECASE,
            )
            if pat.search(text):
                return True
            if text.startswith(m_str):
                after = text[len(m_str):]
                if not re.match(r"^\.\d{1,2}\.\d{4}", after):
                    return True
    return False

def check_fair_overlap_hit(chunk: Dict[str, Any], doc_id: str, markers: list, min_overlap: int = 150) -> bool:
    c_start, c_end = get_chunk_char_span(chunk, doc_id)
    doc_paras = para_spans_2 if doc_id == "sample2" else para_spans_3
    for m in markers:
        m_key = str(m).strip()
        # Handle markers like 7(g) or 7(b) by matching base marker 7
        if m_key not in doc_paras:
            if m_key.startswith("7("):
                m_key = "7"
        if m_key in doc_paras:
            p_start, p_end = doc_paras[m_key]
            overlap = max(0, min(c_end, p_end) - max(c_start, p_start))
            if overlap >= min_overlap:
                return True
    return False

CONDITIONS = [
    ("naive", "Naive Chunks (1000/200)"),
    ("role_relax3", "Role Chunks (No Filter / Relax 3)"),
    ("oracle_relax0", "Oracle Role (Relax 0)"),
    ("oracle_relax1", "Oracle Role (Relax 1)"),
    ("oracle_relax2", "Oracle Role (Relax 2)"),
    ("pred_relax0", "Predicted Role (Relax 0)"),
    ("pred_auto_escalate", "Predicted Role (Auto-Escalate)"),
]

print("Running Fair Retrieval Evaluation across 7 conditions...")

old_ranks = {c[0]: {} for c in CONDITIONS}
new_ranks = {c[0]: {} for c in CONDITIONS}
retrieved_pool = {} # q -> cond -> chunks

for q, info in gold_data.items():
    doc_id = info["sample"]
    gold_role = info["role"]
    markers = info["markers"]

    m = map_query_to_roles(q, mode="hybrid", debug=False)
    pred_roles = m.roles
    pos_pred = (0.85, 1.0) if "Decision" in pred_roles else None
    pos_oracle = (0.85, 1.0) if gold_role == "Decision" else None

    # Retrieve for each condition
    c_naive = store.retrieve(question=q, document_id=doc_id, mode="naive", top_k=5)
    c_r3 = store.retrieve(question=q, document_id=doc_id, mode="role", relax=3, top_k=5)
    c_o0 = store.retrieve(question=q, document_id=doc_id, mode="role", roles=[gold_role], relax=0, rel_pos_range=pos_oracle, top_k=5)
    c_o1 = store.retrieve(question=q, document_id=doc_id, mode="role", roles=[gold_role], relax=1, rel_pos_range=pos_oracle, top_k=5)
    c_o2 = store.retrieve(question=q, document_id=doc_id, mode="role", roles=[gold_role], relax=2, rel_pos_range=pos_oracle, top_k=5)
    c_p0 = store.retrieve(question=q, document_id=doc_id, mode="role", roles=pred_roles, relax=0, rel_pos_range=pos_pred, top_k=5)
    c_pae, _ = auto_escalate_retrieval(question=q, document_id=doc_id, store=store, mapped_roles=pred_roles, min_chunks=3, rel_pos_range=pos_pred, top_k=5)

    cond_chunks = {
        "naive": c_naive,
        "role_relax3": c_r3,
        "oracle_relax0": c_o0,
        "oracle_relax1": c_o1,
        "oracle_relax2": c_o2,
        "pred_relax0": c_p0,
        "pred_auto_escalate": c_pae,
    }
    retrieved_pool[q] = cond_chunks

    for c_key, chunks in cond_chunks.items():
        # Old rank
        r_old = 0
        for rank, chk in enumerate(chunks, 1):
            if check_old_marker_hit(chk.get("text", ""), markers):
                r_old = rank
                break
        old_ranks[c_key][q] = r_old

        # New fair rank (>= 150 chars overlap)
        r_new = 0
        for rank, chk in enumerate(chunks, 1):
            if check_fair_overlap_hit(chk, doc_id, markers, min_overlap=150):
                r_new = rank
                break
        new_ranks[c_key][q] = r_new

def compute_metrics(ranks: list):
    n = len(ranks)
    if n == 0:
        return {"hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr": 0.0, "h1_cnt": 0, "h3_cnt": 0, "h5_cnt": 0}
    h1 = sum(1 for r in ranks if r == 1)
    h3 = sum(1 for r in ranks if 1 <= r <= 3)
    h5 = sum(1 for r in ranks if 1 <= r <= 5)
    mrr = sum(1.0 / r for r in ranks if r > 0) / n
    return {
        "hit1": (h1 / n) * 100,
        "hit3": (h3 / n) * 100,
        "hit5": (h5 / n) * 100,
        "mrr": mrr,
        "h1_cnt": h1,
        "h3_cnt": h3,
        "h5_cnt": h5,
    }

# Build fair retrieval comparison log
lines_r = []
lines_r.append("=" * 115)
lines_r.append("FAIR RETRIEVAL EVALUATION REPORT: OLD (MARKER) VS NEW (>=150 CHAR OVERLAP)")
lines_r.append("=" * 115)
lines_r.append("Model: BAAI/bge-small-en-v1.5 | Top-K: 5 | Total Evaluated Questions: 22")
lines_r.append("Overlap Rule: Chunk character span overlap with gold paragraph span >= 150 characters.")
lines_r.append("NOTE: With N=22 questions, differences of 1-2 questions are within sample variance.")
lines_r.append("")
lines_r.append("-" * 115)
lines_r.append(f"{'Condition':<32} | {'Old Hit@1':<10} | {'New Hit@1':<10} | {'Old Hit@5':<10} | {'New Hit@5':<10} | {'Old MRR':<7} | {'New MRR':<7}")
lines_r.append("-" * 115)

for c_key, c_label in CONDITIONS:
    m_old = compute_metrics(list(old_ranks[c_key].values()))
    m_new = compute_metrics(list(new_ranks[c_key].values()))
    lines_r.append(
        f"{c_label:<32} | "
        f"{m_old['hit1']:4.1f}% ({m_old['h1_cnt']:2d}) | "
        f"{m_new['hit1']:4.1f}% ({m_new['h1_cnt']:2d}) | "
        f"{m_old['hit5']:4.1f}% ({m_old['h5_cnt']:2d}) | "
        f"{m_new['hit5']:4.1f}% ({m_new['h5_cnt']:2d}) | "
        f"{m_old['mrr']:6.3f} | "
        f"{m_new['mrr']:6.3f}"
    )

lines_r.append("-" * 115)
lines_r.append("")
lines_r.append("=" * 115)
lines_r.append("PER-QUESTION RANK COMPARISON (OLD MARKER HIT vs NEW FAIR OVERLAP HIT):")
lines_r.append("Format: Old# / New# (e.g. #1 / #1, or - / #2)")
lines_r.append("=" * 115)
lines_r.append(f"{'#':<3} | {'Sample':<7} | {'Role':<18} | {'Naive':<9} | {'Role-R3':<9} | {'Orc-R0':<9} | {'Pred-R0':<9} | {'Pred-AE':<9} | {'Question':<28}")
lines_r.append("-" * 115)

for i, (q, info) in enumerate(gold_data.items(), 1):
    def fmt_pair(r_o, r_n):
        so = f"#{r_o}" if r_o > 0 else "-"
        sn = f"#{r_n}" if r_n > 0 else "-"
        return f"{so}/{sn}"

    p_naive = fmt_pair(old_ranks["naive"][q], new_ranks["naive"][q])
    p_r3 = fmt_pair(old_ranks["role_relax3"][q], new_ranks["role_relax3"][q])
    p_o0 = fmt_pair(old_ranks["oracle_relax0"][q], new_ranks["oracle_relax0"][q])
    p_p0 = fmt_pair(old_ranks["pred_relax0"][q], new_ranks["pred_relax0"][q])
    p_pae = fmt_pair(old_ranks["pred_auto_escalate"][q], new_ranks["pred_auto_escalate"][q])
    q_short = q[:26] + ".." if len(q) > 28 else q

    lines_r.append(
        f"{i:<3} | {info['sample']:<7} | {info['role']:<18} | "
        f"{p_naive:<9} | {p_r3:<9} | {p_o0:<9} | {p_p0:<9} | {p_pae:<9} | {q_short:<28}"
    )

lines_r.append("=" * 115)
fair_log_text = "\n".join(lines_r)
FAIR_RETRIEVAL_LOG.write_text(fair_log_text, encoding="utf-8")
print(f"Fair retrieval evaluation saved to {FAIR_RETRIEVAL_LOG} ({len(fair_log_text)} bytes)")

# ==============================================================================
# 2. RELEVANCE GRADER EVALUATION (MODULE 7)
# ==============================================================================
print("\nRunning Relevance Grader Evaluation on Top-3 Chunks (66 pred_relax0 + 66 role_relax3)...")

eval_queries = []
# Prepare 132 chunks: 22 questions x 3 chunks from (a) pred_relax0 and (b) role_relax3
for q, info in gold_data.items():
    doc_id = info["sample"]
    markers = info["markers"]
    
    top3_pred = retrieved_pool[q]["pred_relax0"][:3]
    top3_r3 = retrieved_pool[q]["role_relax3"][:3]

    eval_queries.append({
        "question": q,
        "doc_id": doc_id,
        "markers": markers,
        "pred_chunks": top3_pred,
        "r3_chunks": top3_r3,
    })

# Run grading for 800 chars and 500 chars
def evaluate_grader(max_chars: int):
    call_times = []
    
    all_chunks_evaluated = [] # list of dict: {chunk, truth, grade, dist}
    false_reject_questions = 0
    total_questions_with_relevant = 0

    for item in eval_queries:
        q = item["question"]
        doc_id = item["doc_id"]
        markers = item["markers"]

        # Evaluate both pred and r3 sets
        for c_set in [item["pred_chunks"], item["r3_chunks"]]:
            if not c_set:
                continue
            graded, route, duration = grade_chunks(q, c_set, max_chars=max_chars)
            call_times.append(duration)

            set_has_relevant = False
            set_all_incorrect = (route == "rewrite")

            for idx, chk in enumerate(c_set):
                truth = check_fair_overlap_hit(chk, doc_id, markers, min_overlap=150)
                grade = graded[idx]["grade"] if idx < len(graded) else "incorrect"
                dist = chk.get("distance", 1.0)
                
                if truth:
                    set_has_relevant = True

                all_chunks_evaluated.append({
                    "truth": truth,
                    "grade": grade,
                    "distance": dist,
                })

            if set_has_relevant:
                total_questions_with_relevant += 1
                if set_all_incorrect:
                    false_reject_questions += 1

    # Compute metrics
    # Confusion matrix
    # Truth: True / False
    # Grade: correct / ambiguous / incorrect
    c_tp = sum(1 for x in all_chunks_evaluated if x["truth"] and x["grade"] == "correct")
    c_fp = sum(1 for x in all_chunks_evaluated if not x["truth"] and x["grade"] == "correct")
    a_tp = sum(1 for x in all_chunks_evaluated if x["truth"] and x["grade"] == "ambiguous")
    a_fp = sum(1 for x in all_chunks_evaluated if not x["truth"] and x["grade"] == "ambiguous")
    i_fn = sum(1 for x in all_chunks_evaluated if x["truth"] and x["grade"] == "incorrect")
    i_tn = sum(1 for x in all_chunks_evaluated if not x["truth"] and x["grade"] == "incorrect")

    total_relevant = sum(1 for x in all_chunks_evaluated if x["truth"])
    total_irrelevant = sum(1 for x in all_chunks_evaluated if not x["truth"])

    # Strict (only correct is positive)
    p_strict = (c_tp / max(c_tp + c_fp, 1)) * 100
    r_strict = (c_tp / max(total_relevant, 1)) * 100

    # Lenient (correct or ambiguous is positive)
    lenient_pred = c_tp + c_fp + a_tp + a_fp
    p_lenient = ((c_tp + a_tp) / max(lenient_pred, 1)) * 100
    r_lenient = ((c_tp + a_tp) / max(total_relevant, 1)) * 100

    false_reject_rate = (false_reject_questions / max(total_questions_with_relevant, 1)) * 100
    false_keep_rate = (c_fp / max(c_tp + c_fp, 1)) * 100

    mean_latency = sum(call_times) / max(len(call_times), 1)

    return {
        "max_chars": max_chars,
        "n_chunks": len(all_chunks_evaluated),
        "total_relevant": total_relevant,
        "total_irrelevant": total_irrelevant,
        "cm": {
            "correct_rel": c_tp, "correct_irrel": c_fp,
            "ambig_rel": a_tp, "ambig_irrel": a_fp,
            "incor_rel": i_fn, "incor_irrel": i_tn,
        },
        "p_strict": p_strict, "r_strict": r_strict,
        "p_lenient": p_lenient, "r_lenient": r_lenient,
        "false_reject_rate": false_reject_rate,
        "false_keep_rate": false_keep_rate,
        "mean_latency_ms": mean_latency,
        "min_latency_ms": min(call_times) if call_times else 0.0,
        "max_latency_ms": max(call_times) if call_times else 0.0,
        "raw_items": all_chunks_evaluated,
    }

print("Grading 132 chunks with max_chars = 800...")
res_800 = evaluate_grader(max_chars=800)

print("Grading 132 chunks with max_chars = 500...")
res_500 = evaluate_grader(max_chars=500)

# Evaluate No-LLM Cosine Distance Baseline on the same chunks
baseline_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
baseline_results = []
items = res_800["raw_items"]

for th in baseline_thresholds:
    b_tp = sum(1 for x in items if x["truth"] and x["distance"] <= th)
    b_fp = sum(1 for x in items if not x["truth"] and x["distance"] <= th)
    b_fn = sum(1 for x in items if x["truth"] and x["distance"] > th)
    b_tn = sum(1 for x in items if not x["truth"] and x["distance"] > th)
    
    prec = (b_tp / max(b_tp + b_fp, 1)) * 100
    rec = (b_tp / max(b_tp + b_fn, 1)) * 100
    f1 = (2 * prec * rec) / max(prec + rec, 1e-4)

    baseline_results.append({
        "threshold": th,
        "tp": b_tp, "fp": b_fp, "fn": b_fn, "tn": b_tn,
        "precision": prec, "recall": rec, "f1": f1,
    })

# Run Determinism Test: 10 questions x 3 runs
print("Running Determinism Test (10 questions x 3 runs)...")
det_queries = eval_queries[:10]
det_runs = []
det_latencies = []

for run_i in range(3):
    run_grades = []
    for item in det_queries:
        graded, _, wall = grade_chunks(item["question"], item["pred_chunks"][:3], max_chars=800)
        run_grades.append([g["grade"] for g in graded])
        det_latencies.append(wall)
    det_runs.append(run_grades)

# Check if run 0, 1, 2 are identical
identical_count = sum(1 for q_idx in range(len(det_queries)) if det_runs[0][q_idx] == det_runs[1][q_idx] == det_runs[2][q_idx])
is_fully_deterministic = (identical_count == len(det_queries))

# Format Grader Evaluation Report
lines_g = []
lines_g.append("=" * 80)
lines_g.append("MODULE 7: RELEVANCE GRADER EVALUATION REPORT")
lines_g.append("=" * 80)
lines_g.append("Evaluation Corpus: 132 Chunks (22 Gold Qs x Top-3 from pred_relax0 and role_relax3)")
lines_g.append(f"Ground Truth: Fair Paragraph Overlap >= 150 chars (Total Relevant: {res_800['total_relevant']}, Irrelevant: {res_800['total_irrelevant']})")
lines_g.append("")

lines_g.append("1. CONFUSION MATRIX & ACCURACY: 800 CHARS VS 500 CHARS TRUNCATION:")
lines_g.append("-" * 80)
lines_g.append(f"{'Metric':<36} | {'800 Chars':<18} | {'500 Chars':<18}")
lines_g.append("-" * 80)
lines_g.append(f"{'True Relevant Chunks':<36} | {res_800['total_relevant']:<18} | {res_500['total_relevant']:<18}")
lines_g.append(f"{'True Irrelevant Chunks':<36} | {res_800['total_irrelevant']:<18} | {res_500['total_irrelevant']:<18}")
lines_g.append(f"{'Graded Correct (Relevant)':<36} | {res_800['cm']['correct_rel']:<18} | {res_500['cm']['correct_rel']:<18}")
lines_g.append(f"{'Graded Correct (Irrelevant)':<36} | {res_800['cm']['correct_irrel']:<18} | {res_500['cm']['correct_irrel']:<18}")
lines_g.append(f"{'Graded Ambiguous (Relevant)':<36} | {res_800['cm']['ambig_rel']:<18} | {res_500['cm']['ambig_rel']:<18}")
lines_g.append(f"{'Graded Ambiguous (Irrelevant)':<36} | {res_800['cm']['ambig_irrel']:<18} | {res_500['cm']['ambig_irrel']:<18}")
lines_g.append(f"{'Graded Incorrect (Relevant)':<36} | {res_800['cm']['incor_rel']:<18} | {res_500['cm']['incor_rel']:<18}")
lines_g.append(f"{'Graded Incorrect (Irrelevant)':<36} | {res_800['cm']['incor_irrel']:<18} | {res_500['cm']['incor_irrel']:<18}")
lines_g.append("-" * 80)
lines_g.append(f"{'Precision (Strict, Correct only)':<36} | {res_800['p_strict']:5.1f}%             | {res_500['p_strict']:5.1f}%")
lines_g.append(f"{'Recall (Strict, Correct only)':<36} | {res_800['r_strict']:5.1f}%             | {res_500['r_strict']:5.1f}%")
lines_g.append(f"{'Precision (Lenient, Correct/Ambig)':<36} | {res_800['p_lenient']:5.1f}%             | {res_500['p_lenient']:5.1f}%")
lines_g.append(f"{'Recall (Lenient, Correct/Ambig)':<36} | {res_800['r_lenient']:5.1f}%             | {res_500['r_lenient']:5.1f}%")
lines_g.append(f"{'False-Reject Rate (Needless Rewrite)':<36} | {res_800['false_reject_rate']:5.1f}%             | {res_500['false_reject_rate']:5.1f}%")
lines_g.append(f"{'False-Keep Rate (Irrelevant as Correct)':<36} | {res_800['false_keep_rate']:5.1f}%             | {res_500['false_keep_rate']:5.1f}%")
lines_g.append(f"{'Mean Latency per 3-Chunk Call':<36} | {res_800['mean_latency_ms']:6.1f} ms          | {res_500['mean_latency_ms']:6.1f} ms")
lines_g.append("-" * 80)
lines_g.append("")

lines_g.append("2. COMPARISON AGAINST NO-LLM COSINE DISTANCE BASELINE:")
lines_g.append("-" * 80)
lines_g.append(f"{'Distance Threshold':<20} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<10} | {'TP / FP / FN'}")
lines_g.append("-" * 80)
for b in baseline_results:
    lines_g.append(
        f"Threshold <= {b['threshold']:<9.2f} | "
        f"{b['precision']:5.1f}%      | "
        f"{b['recall']:5.1f}%      | "
        f"{b['f1']:5.1f}%     | "
        f"{b['tp']:2d} / {b['fp']:2d} / {b['fn']:2d}"
    )
lines_g.append("-" * 80)
# Analysis of baseline vs LLM grader
best_b = max(baseline_results, key=lambda x: x["f1"])
lines_g.append(f"Baseline Assessment: Best distance threshold ({best_b['threshold']:.2f}) achieves F1: {best_b['f1']:.1f}%")
if best_b['f1'] >= res_800['p_strict']:
    lines_g.append("Plain Finding: The cosine-distance baseline is highly competitive with raw strict grading.")
else:
    lines_g.append(f"Plain Finding: The LLM grader achieves higher precision ({res_800['p_strict']:.1f}% vs {best_b['precision']:.1f}%) by semantically validating subtle relevance beyond vector distance.")
lines_g.append("")

lines_g.append("3. LATENCY AND DETERMINISM BENCHMARK:")
lines_g.append("-" * 80)
lines_g.append(f"Mean Wall Time per Grading Call: {res_800['mean_latency_ms']:8.1f} ms")
lines_g.append(f"Min Wall Time per Grading Call:  {res_800['min_latency_ms']:8.1f} ms")
lines_g.append(f"Max Wall Time per Grading Call:  {res_800['max_latency_ms']:8.1f} ms")
lines_g.append(f"Determinism Test (10 Qs x 3 Runs): {identical_count} / {len(det_queries)} runs 100% identical ({'PERFECT 100% DETERMINISM' if is_fully_deterministic else 'MINOR VARIANCE'})")
lines_g.append("=" * 80)

grader_log_text = "\n".join(lines_g)
GRADER_EVAL_LOG.write_text(grader_log_text, encoding="utf-8")
print(f"Grader evaluation saved to {GRADER_EVAL_LOG} ({len(grader_log_text)} bytes)")
print(grader_log_text)
