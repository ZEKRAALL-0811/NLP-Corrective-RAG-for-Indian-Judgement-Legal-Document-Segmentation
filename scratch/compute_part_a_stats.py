import sys
import json
import csv
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks
from src.rag.graph import TAU_HIGH, TAU_LOW

# We will load gold_paragraphs.json and gold_unanswerable.csv
store = LegalVectorStore()
recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

# Build paragraph spans for relevance check
# We can import check_chunk_gold_hit from scripts.eval_generation
from scripts.eval_generation import check_chunk_gold_hit

gold_data = json.loads(Path("legalDataset/samples/gold_paragraphs.json").read_text(encoding="utf-8"))
unans_rows = list(csv.DictReader(open("legalDataset/samples/gold_unanswerable.csv", "r", encoding="utf-8")))

print("Computing stats for 22 Gold + 8 Unanswerable questions...")

gold_stats = []
for idx, (q, info) in enumerate(gold_data.items(), 1):
    doc_id = info["sample"]
    markers = info["markers"]
    role = info["role"]
    # retrieve top-5 role chunks relax 0
    chunks = store.retrieve(question=q, document_id=doc_id, mode="role", roles=[role], relax=0, top_k=5)
    scored, _ = score_chunks(q, chunks)
    scores = [c["rerank_score"] for c in scored]
    best_score = max(scores) if scores else -99.0
    mean_top5 = float(np.mean(scores)) if scores else -99.0
    norm_signal = best_score - mean_top5
    
    # check if top-3 has truly relevant chunk
    has_rel_top3 = any(check_chunk_gold_hit(c, doc_id, markers) for c in chunks[:3])
    
    # Route under tau_high=-0.5, tau_low=-3.5
    if best_score >= TAU_HIGH:
        route = "generate"
    elif best_score >= TAU_LOW:
        route = "fallback"
    else:
        route = "rewrite"
        
    gold_stats.append({
        "num": idx,
        "sample": doc_id,
        "question": q,
        "role": role,
        "markers": markers,
        "scores": scores,
        "best_score": best_score,
        "mean_top5": mean_top5,
        "norm_signal": norm_signal,
        "has_rel_top3": has_rel_top3,
        "route": route,
    })

unans_stats = []
for idx, row in enumerate(unans_rows, 1):
    doc_id = row["sample"]
    q = row["question"]
    # retrieve top-5 role chunks relax 0 (roles=[] -> whole doc or all roles)
    chunks = store.retrieve(question=q, document_id=doc_id, mode="role", roles=[], relax=0, top_k=5)
    scored, _ = score_chunks(q, chunks)
    scores = [c["rerank_score"] for c in scored]
    best_score = max(scores) if scores else -99.0
    mean_top5 = float(np.mean(scores)) if scores else -99.0
    norm_signal = best_score - mean_top5
    
    if best_score >= TAU_HIGH:
        route = "generate"
    elif best_score >= TAU_LOW:
        route = "fallback"
    else:
        route = "rewrite"
        
    unans_stats.append({
        "num": idx,
        "sample": doc_id,
        "question": q,
        "scores": scores,
        "best_score": best_score,
        "mean_top5": mean_top5,
        "norm_signal": norm_signal,
        "has_rel_top3": False,
        "route": route,
    })

print(f"Done computing. Gold: {len(gold_stats)}, Unans: {len(unans_stats)}")

# Route tables per document (Sample 2 vs Sample 3)
for sample in ["sample2", "sample3"]:
    print(f"\n--- ROUTE TABLE FOR {sample.upper()} (Thresholds tau_high=-0.5, tau_low=-3.5) ---")
    s_gold = [g for g in gold_stats if g["sample"] == sample]
    
    # Truth: Present vs Absent
    present = [g for g in s_gold if g["has_rel_top3"]]
    absent = [g for g in s_gold if not g["has_rel_top3"]]
    
    def count_routes(items):
        gen = sum(1 for x in items if x["route"] == "generate")
        fb = sum(1 for x in items if x["route"] == "fallback")
        rw = sum(1 for x in items if x["route"] == "rewrite")
        return gen, fb, rw
    
    p_gen, p_fb, p_rw = count_routes(present)
    a_gen, a_fb, a_rw = count_routes(absent)
    
    print(f"Truth: Truly Relevant Chunk Present (N={len(present)}):")
    print(f"  Generate: {p_gen} ({p_gen/len(present)*100:.1f}%) | Fallback: {p_fb} ({p_fb/len(present)*100:.1f}%) | Rewrite: {p_rw} ({p_rw/len(present)*100:.1f}%)")
    if absent:
        print(f"Truth: Truly Relevant Chunk Absent (N={len(absent)}):")
        print(f"  Generate: {a_gen} ({a_gen/len(absent)*100:.1f}%) | Fallback: {a_fb} ({a_fb/len(absent)*100:.1f}%) | Rewrite: {a_rw} ({a_rw/len(absent)*100:.1f}%)")
    else:
        print("Truth: Truly Relevant Chunk Absent (N=0)")

# ROC-AUC calculations using sklearn
from sklearn.metrics import roc_auc_score

# 1. Present vs Absent among 22 gold questions (18 positive, 4 negative)
y_pres = [1 if g["has_rel_top3"] else 0 for g in gold_stats]
scores_raw_pres = [g["best_score"] for g in gold_stats]
scores_norm_pres = [g["norm_signal"] for g in gold_stats]

auc_raw_pres = roc_auc_score(y_pres, scores_raw_pres)
auc_norm_pres = roc_auc_score(y_pres, scores_norm_pres)

print("\n--- ROC-AUC: PRESENT VS ABSENT (N=22: 18 Positive, 4 Negative) ---")
print(f"Raw Best Cross-Encoder Score:       ROC-AUC = {auc_raw_pres:.4f}")
print(f"Normalized Signal (best - mean5):  ROC-AUC = {auc_norm_pres:.4f}")
print(f"Note: This AUC rests on only {sum(1 for y in y_pres if y == 0)} negative questions.")

# 2. Answerable vs Unanswerable (22 positive, 8 negative)
y_ans = [1] * len(gold_stats) + [0] * len(unans_stats)
scores_raw_ans = [g["best_score"] for g in gold_stats] + [u["best_score"] for u in unans_stats]
scores_norm_ans = [g["norm_signal"] for g in gold_stats] + [u["norm_signal"] for u in unans_stats]

auc_raw_ans = roc_auc_score(y_ans, scores_raw_ans)
auc_norm_ans = roc_auc_score(y_ans, scores_norm_ans)

print("\n--- ROC-AUC: ANSWERABLE VS UNANSWERABLE (N=30: 22 Positive, 8 Negative) ---")
print(f"Raw Best Cross-Encoder Score:       ROC-AUC = {auc_raw_ans:.4f}")
print(f"Normalized Signal (best - mean5):  ROC-AUC = {auc_norm_ans:.4f}")
print(f"Note: This AUC rests on only {sum(1 for y in y_ans if y == 0)} negative questions.")
