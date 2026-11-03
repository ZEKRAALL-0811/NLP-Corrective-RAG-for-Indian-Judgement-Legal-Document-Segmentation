"""
Script to evaluate dense retrieval across conditions:
(a) naive chunks (1000/200 characters)
(b) role chunks with no role filter (relax 3)
(c) role chunks with gold role (oracle) at relax 0, 1, and 2
(d) role chunks with mapper-predicted role, relax 0, and with auto-escalation

Evaluates on the 22 substantive questions from gold_paragraphs.json.
Reports hit@1, hit@3, hit@5, and MRR overall, per sample, and per role.
Reports number of questions where strict filter returned fewer than 3 chunks.
Generates per-question rank table.
Saves raw output to artifacts/logs/retrieval_eval.txt.
"""

import sys
from pathlib import Path
import json
import re

sys.path.insert(0, ".")
from src.rag.vectorstore import LegalVectorStore
from src.rag.role_mapper import map_query_to_roles
from src.rag.graph import auto_escalate_retrieval

GOLD_JSON = Path("legalDataset/samples/gold_paragraphs.json")
LOG_OUT = Path("artifacts/logs/retrieval_eval.txt")

with open(GOLD_JSON, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

print(f"Loaded {len(gold_data)} evaluation questions.")
store = LegalVectorStore()

def check_hit(chunk_text: str, markers: list) -> bool:
    """
    Checks if a chunk contains any of the gold paragraph markers at a sentence boundary.
    Carefully avoids matching dates like '19.06.2009' or '19.04.2005'.
    """
    text = chunk_text.strip()
    for m in markers:
        m_str = str(m).strip()
        if m_str.startswith("(") and m_str.endswith(")"):
            num = m_str[1:-1]
            pat = re.compile(
                r"(?:^|\n|[\.\?\!]\s+)(?:\(" + re.escape(num) + r"\)|\b" + re.escape(num) + r"\.)(?!\d)",
                re.IGNORECASE
            )
            if pat.search(text):
                return True
        else:
            pat = re.compile(
                r"(?:^|\n|[\.\?\!]\s+)(?:" + re.escape(m_str) + r"[\.\)\s])(?!\d{1,2}\.\d{4})",
                re.IGNORECASE
            )
            if pat.search(text):
                return True
            if text.startswith(m_str):
                after = text[len(m_str):]
                if not re.match(r"^\.\d{1,2}\.\d{4}", after):
                    return True
    return False

def find_first_hit_rank(chunks: list, markers: list) -> int:
    """Returns 1-indexed rank of first hit, or 0 if none found."""
    for rank, chunk in enumerate(chunks, 1):
        if check_hit(chunk.get("text", ""), markers):
            return rank
    return 0

CONDITIONS = [
    ("naive", "Naive Chunks (1000/200)"),
    ("role_relax3", "Role Chunks (No Filter / Relax 3)"),
    ("oracle_relax0", "Oracle Role (Relax 0)"),
    ("oracle_relax1", "Oracle Role (Relax 1)"),
    ("oracle_relax2", "Oracle Role (Relax 2)"),
    ("pred_relax0", "Predicted Role (Relax 0)"),
    ("pred_auto_escalate", "Predicted Role (Auto-Escalate)"),
]

# Track results per condition
# q_results[cond_key][q] = rank (0 if miss)
q_results = {c[0]: {} for c in CONDITIONS}
q_chunk_counts = {c[0]: {} for c in CONDITIONS}

print("Running retrieval evaluation across 7 conditions...")
for q_idx, (question, info) in enumerate(gold_data.items(), 1):
    doc_id = info["sample"]
    gold_role = info["role"]
    markers = info["markers"]

    # Map role using role mapper (hybrid mode)
    m = map_query_to_roles(question, mode="hybrid", debug=False)
    pred_roles = m.roles

    # Positional range for Decision queries
    pos_range = (0.85, 1.0) if "Decision" in (pred_roles or [gold_role]) else None

    # Condition 1: Naive chunks (top 5)
    c_naive = store.retrieve(question=question, document_id=doc_id, mode="naive", top_k=5)
    q_results["naive"][question] = find_first_hit_rank(c_naive, markers)
    q_chunk_counts["naive"][question] = len(c_naive)

    # Condition 2: Role chunks no filter (relax 3)
    c_r3 = store.retrieve(question=question, document_id=doc_id, mode="role", relax=3, top_k=5)
    q_results["role_relax3"][question] = find_first_hit_rank(c_r3, markers)
    q_chunk_counts["role_relax3"][question] = len(c_r3)

    # Condition 3: Oracle role relax 0
    pos_oracle = (0.85, 1.0) if gold_role == "Decision" else None
    c_o0 = store.retrieve(question=question, document_id=doc_id, mode="role", roles=[gold_role], relax=0, rel_pos_range=pos_oracle, top_k=5)
    q_results["oracle_relax0"][question] = find_first_hit_rank(c_o0, markers)
    q_chunk_counts["oracle_relax0"][question] = len(c_o0)

    # Condition 4: Oracle role relax 1
    c_o1 = store.retrieve(question=question, document_id=doc_id, mode="role", roles=[gold_role], relax=1, rel_pos_range=pos_oracle, top_k=5)
    q_results["oracle_relax1"][question] = find_first_hit_rank(c_o1, markers)
    q_chunk_counts["oracle_relax1"][question] = len(c_o1)

    # Condition 5: Oracle role relax 2
    c_o2 = store.retrieve(question=question, document_id=doc_id, mode="role", roles=[gold_role], relax=2, rel_pos_range=pos_oracle, top_k=5)
    q_results["oracle_relax2"][question] = find_first_hit_rank(c_o2, markers)
    q_chunk_counts["oracle_relax2"][question] = len(c_o2)

    # Condition 6: Predicted role relax 0
    pos_pred = (0.85, 1.0) if "Decision" in pred_roles else None
    c_p0 = store.retrieve(question=question, document_id=doc_id, mode="role", roles=pred_roles, relax=0, rel_pos_range=pos_pred, top_k=5)
    q_results["pred_relax0"][question] = find_first_hit_rank(c_p0, markers)
    q_chunk_counts["pred_relax0"][question] = len(c_p0)

    # Condition 7: Predicted role auto-escalate
    c_pae, final_relax = auto_escalate_retrieval(question=question, document_id=doc_id, store=store, mapped_roles=pred_roles, min_chunks=3, rel_pos_range=pos_pred, top_k=5)
    q_results["pred_auto_escalate"][question] = find_first_hit_rank(c_pae, markers)
    q_chunk_counts["pred_auto_escalate"][question] = len(c_pae)

print("Retrieval queries complete. Computing metric aggregations...")

def compute_metrics(ranks: list):
    n = len(ranks)
    if n == 0:
        return {"hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr": 0.0, "count": 0}
    h1 = sum(1 for r in ranks if r == 1) / n
    h3 = sum(1 for r in ranks if 1 <= r <= 3) / n
    h5 = sum(1 for r in ranks if 1 <= r <= 5) / n
    mrr = sum(1.0 / r for r in ranks if r > 0) / n
    return {
        "hit1": h1 * 100,
        "hit3": h3 * 100,
        "hit5": h5 * 100,
        "mrr": mrr,
        "count": n,
        "hits1_cnt": sum(1 for r in ranks if r == 1),
        "hits3_cnt": sum(1 for r in ranks if 1 <= r <= 3),
        "hits5_cnt": sum(1 for r in ranks if 1 <= r <= 5),
    }

lines = []
lines.append("=" * 115)
lines.append("MODULE 6: RETRIEVAL EVALUATION REPORT (22 GOLD QUESTIONS)")
lines.append("=" * 115)
lines.append("Dense Retrieval Model: BAAI/bge-small-en-v1.5 | Top-K: 5 | Total Evaluated Questions: 22")
lines.append("NOTE: With N=22 questions, differences of 1-2 questions (~4.5-9%) are within sample variance")
lines.append("and should not be over-interpreted as statistically significant.")
lines.append("")

# Table 1: Overall Comparison
lines.append("-" * 115)
lines.append("1. OVERALL RETRIEVAL PERFORMANCE ACROSS CONDITIONS (N=22):")
lines.append("-" * 115)
lines.append(f"{'Condition':<36} | {'Hit@1':<12} | {'Hit@3':<12} | {'Hit@5':<12} | {'MRR':<8} | {'<3 Chunks':<10}")
lines.append("-" * 115)

for cond_key, cond_label in CONDITIONS:
    ranks = list(q_results[cond_key].values())
    m = compute_metrics(ranks)
    few_chunks = sum(1 for cnt in q_chunk_counts[cond_key].values() if cnt < 3)
    lines.append(
        f"{cond_label:<36} | "
        f"{m['hit1']:5.1f}% ({m['hits1_cnt']:2d}) | "
        f"{m['hit3']:5.1f}% ({m['hits3_cnt']:2d}) | "
        f"{m['hit5']:5.1f}% ({m['hits5_cnt']:2d}) | "
        f"{m['mrr']:6.3f}   | "
        f"{few_chunks:2d} / 22"
    )
lines.append("-" * 115)
lines.append("")

# Table 2: Breakdown per Sample
lines.append("-" * 115)
lines.append("2. BREAKDOWN BY SAMPLE (Sample 2: N=11, Sample 3: N=11):")
lines.append("-" * 115)
for s_name in ["sample2", "sample3"]:
    lines.append(f"--- {s_name.upper()} (N=11) ---")
    lines.append(f"{'Condition':<36} | {'Hit@1':<12} | {'Hit@3':<12} | {'Hit@5':<12} | {'MRR':<8}")
    lines.append("-" * 88)
    for cond_key, cond_label in CONDITIONS:
        s_ranks = [q_results[cond_key][q] for q, info in gold_data.items() if info["sample"] == s_name]
        m = compute_metrics(s_ranks)
        lines.append(
            f"{cond_label:<36} | "
            f"{m['hit1']:5.1f}% ({m['hits1_cnt']:2d}) | "
            f"{m['hit3']:5.1f}% ({m['hits3_cnt']:2d}) | "
            f"{m['hit5']:5.1f}% ({m['hits5_cnt']:2d}) | "
            f"{m['mrr']:6.3f}"
        )
    lines.append("")

# Table 3: Breakdown per Role
lines.append("-" * 115)
lines.append("3. BREAKDOWN BY RHETORICAL ROLE:")
lines.append("-" * 115)
all_roles = sorted(list(set(info["role"] for info in gold_data.values())))
for role in all_roles:
    role_questions = [q for q, info in gold_data.items() if info["role"] == role]
    r_n = len(role_questions)
    lines.append(f"--- ROLE: {role} (N={r_n}) ---")
    lines.append(f"{'Condition':<36} | {'Hit@1':<12} | {'Hit@3':<12} | {'Hit@5':<12} | {'MRR':<8}")
    lines.append("-" * 88)
    for cond_key, cond_label in CONDITIONS:
        r_ranks = [q_results[cond_key][q] for q in role_questions]
        m = compute_metrics(r_ranks)
        lines.append(
            f"{cond_label:<36} | "
            f"{m['hit1']:5.1f}% ({m['hits1_cnt']:2d}) | "
            f"{m['hit3']:5.1f}% ({m['hits3_cnt']:2d}) | "
            f"{m['hit5']:5.1f}% ({m['hits5_cnt']:2d}) | "
            f"{m['mrr']:6.3f}"
        )
    lines.append("")

# Table 4: Per-Question Rank Comparison Table
lines.append("=" * 115)
lines.append("4. PER-QUESTION RANK OF FIRST HIT FOR EACH CONDITION:")
lines.append("=" * 115)
rank_hdr = f"{'#':<3} | {'Sample':<7} | {'Role':<18} | {'Naive':<5} | {'R-Rel3':<6} | {'Orc-R0':<6} | {'Orc-R1':<6} | {'Orc-R2':<6} | {'Prd-R0':<6} | {'Prd-AE':<6} | {'Question':<32}"
lines.append(rank_hdr)
lines.append("-" * 115)

for i, (q, info) in enumerate(gold_data.items(), 1):
    def fmt_r(val):
        return f" #{val}" if val > 0 else "  - "

    r_naive = fmt_r(q_results["naive"][q])
    r_r3 = fmt_r(q_results["role_relax3"][q])
    r_o0 = fmt_r(q_results["oracle_relax0"][q])
    r_o1 = fmt_r(q_results["oracle_relax1"][q])
    r_o2 = fmt_r(q_results["oracle_relax2"][q])
    r_p0 = fmt_r(q_results["pred_relax0"][q])
    r_pae = fmt_r(q_results["pred_auto_escalate"][q])

    q_short = q if len(q) <= 30 else q[:28] + ".."
    lines.append(
        f"{i:<3} | {info['sample']:<7} | {info['role']:<18} | "
        f"{r_naive:<5} | {r_r3:<6} | {r_o0:<6} | {r_o1:<6} | {r_o2:<6} | {r_p0:<6} | {r_pae:<6} | "
        f"{q_short:<32}"
    )

lines.append("=" * 115)

out_text = "\n".join(lines)
LOG_OUT.write_text(out_text, encoding="utf-8")
print(f"Evaluation complete. Saved to {LOG_OUT} ({len(out_text)} bytes)")
print(out_text)
