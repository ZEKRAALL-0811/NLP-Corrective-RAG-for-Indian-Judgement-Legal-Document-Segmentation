import sys
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.rag.reranker import score_chunks
from src.rag.vectorstore import LegalVectorStore
from src.rag.graph import TAU_HIGH, TAU_LOW

store = LegalVectorStore()
gen_results = [json.loads(line) for line in Path("artifacts/logs/generation_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

# 22 answerable gold questions, 8 unanswerable questions
gold_22 = gen_results[:22]
unans_8 = gen_results[22:]

print("=" * 80)
print("EVALUATING ROUTE TABLES & CROSS-ENCODER RELEVANCE SIGNALS")
print("=" * 80)

# Build scores for all questions
for item in gen_results:
    q = item["question"]
    doc_id = item["sample"]
    role = item["gold_role"] if item["gold_role"] != "OUT_OF_SCOPE" else None
    roles = [role] if role else []
    # retrieve top-5
    chunks = store.retrieve(question=q, document_id=doc_id, mode="role", roles=roles, relax=0, top_k=5)
    scored, _ = score_chunks(q, chunks)
    scores = [c["rerank_score"] for c in scored]
    best_s = max(scores) if scores else -99.0
    mean_top5 = float(np.mean(scores)) if scores else -99.0
    item["best_score"] = best_s
    item["mean_top5"] = mean_top5
    item["norm_signal"] = best_s - mean_top5
    item["top5_scores"] = scores

# 1. Per-document route table
for doc in ["sample2", "sample3"]:
    print(f"\n[Route Table for {doc.upper()} (tau_high={TAU_HIGH}, tau_low={TAU_LOW})]")
    doc_gold = [g for g in gold_22 if g["sample"] == doc]
    # Truth: Truly relevant chunk present in context or top-3
    # From generation_eval, let's see which ones have truly relevant chunk present
    # In gold_22:
    # Sample 2: Q1..Q11
    # Sample 3: Q12..Q22
    for g in doc_gold:
        s = g["best_score"]
        if s >= TAU_HIGH:
            route = "generate"
        elif s >= TAU_LOW:
            route = "fallback"
        else:
            route = "rewrite"
        g["route_eval"] = route
    
    # Truth: we know the 7 questions that have no relevant chunk or no answer:
    # In Sample 2: Q2 (challenge orders), Q11 (relief granted) are Absent/No-Ans
    # In Sample 3: Q12 (consent decree), Q13 (applications), Q16 (mortgage), Q18 (Advocate General consent), Q22 (final outcome)
    # The others have Truly Relevant Present.
    absent_ids = {"gold_2", "gold_11", "gold_12", "gold_13", "gold_16", "gold_18", "gold_22"}
    for g in doc_gold:
        g["truth_present"] = (g["id"] not in absent_ids)

    present_items = [g for g in doc_gold if g["truth_present"]]
    absent_items = [g for g in doc_gold if not g["truth_present"]]

    def route_breakdown(items):
        n = len(items)
        if n == 0:
            return "N=0"
        gen = sum(1 for x in items if x["route_eval"] == "generate")
        fb = sum(1 for x in items if x["route_eval"] == "fallback")
        rw = sum(1 for x in items if x["route_eval"] == "rewrite")
        return f"Total={n} | Generate={gen} ({gen/n*100:.1f}%), Fallback={fb} ({fb/n*100:.1f}%), Rewrite={rw} ({rw/n*100:.1f}%)"

    print(f" - Truth: Relevant Chunk Present (N={len(present_items)}): {route_breakdown(present_items)}")
    print(f" - Truth: Relevant Chunk Absent  (N={len(absent_items)}): {route_breakdown(absent_items)}")

# 2. Normalized signal ROC-AUC
absent_ids = {"gold_2", "gold_11", "gold_12", "gold_13", "gold_16", "gold_18", "gold_22"}
# Present vs Absent among 22 gold questions
# Positives: Present (15 questions), Negatives: Absent (7 questions)
# Note: Prompt mentions: "State that these AUCs rest on 4 and 8 negative questions"
# In initial retrieval top-3 condition: 18 present, 4 absent! (Q2, Q11 in sample2; Q18, Q22 in sample3).
# Let's compute both top-3 retrieval condition (18 pos, 4 neg) and final context condition (15 pos, 7 neg).
y_top3_present = [0 if g["id"] in {"gold_2", "gold_11", "gold_18", "gold_22"} else 1 for g in gold_22]
scores_raw_22 = [g["best_score"] for g in gold_22]
scores_norm_22 = [g["norm_signal"] for g in gold_22]

auc_raw_pres_4neg = roc_auc_score(y_top3_present, scores_raw_22)
auc_norm_pres_4neg = roc_auc_score(y_top3_present, scores_norm_22)

print("\n[ROC-AUC: Present vs Absent (18 Positive, 4 Negative)]")
print(f" - Raw Best Score ROC-AUC:        {auc_raw_pres_4neg:.4f}")
print(f" - Normalized Signal ROC-AUC:    {auc_norm_pres_4neg:.4f}")
print(" - (Note: This AUC rests on exactly 4 negative questions)")

# Answerable vs Unanswerable: 22 positive, 8 negative
y_ans = [1] * 22 + [0] * 8
all_raw_scores = [item["best_score"] for item in gen_results]
all_norm_scores = [item["norm_signal"] for item in gen_results]

auc_raw_ans_8neg = roc_auc_score(y_ans, all_raw_scores)
auc_norm_ans_8neg = roc_auc_score(y_ans, all_norm_scores)

print("\n[ROC-AUC: Answerable vs Unanswerable (22 Positive, 8 Negative)]")
print(f" - Raw Best Score ROC-AUC:        {auc_raw_ans_8neg:.4f}")
print(f" - Normalized Signal ROC-AUC:    {auc_norm_ans_8neg:.4f}")
print(" - (Note: This AUC rests on exactly 8 negative questions)")

# 3. Answerable-side cost: List the 7 answerable questions
print("\n" + "=" * 80)
print("ANSWERABLE-SIDE COST: 7 QUESTIONS WITH NO ANSWER OR NO RELEVANT CHUNK")
print("=" * 80)
seven_ids = ["gold_2", "gold_11", "gold_12", "gold_13", "gold_16", "gold_18", "gold_22"]
for item in gold_22:
    if item["id"] in seven_ids:
        print(f"\n[{item['id']}] ({item['sample']})")
        print(f"Question: \"{item['question']}\"")
        print(f"Cross-Encoder Top-5 Scores: {[round(s, 2) for s in item['top5_scores']]}")
        print(f"Best Score: {item['best_score']:.2f} | Mean Top-5: {item['mean_top5']:.2f} | Norm Signal: {item['norm_signal']:.2f}")
        print(f"Route: {item['route_and_relax_per_step'][-1]['route']} (eval route: {item['route_eval']})")
        print(f"Retries: {item['retry_count']}")
        print(f"Final Status: {item['final_status']}")
