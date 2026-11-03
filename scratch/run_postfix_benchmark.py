import sys
import json
import csv
import re
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore
from src.rag.role_mapper import map_query_to_roles

GOLD_JSON = Path("legalDataset/samples/gold_paragraphs.json")
UNANS_CSV = Path("legalDataset/samples/gold_unanswerable.csv")
BENCH_JSON = Path("artifacts/role_mapper_benchmark.json")

print("Loading documents and segmenting...")
recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

store = LegalVectorStore()

with open(GOLD_JSON, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

with open(UNANS_CSV, "r", encoding="utf-8") as f:
    unans_rows = list(csv.DictReader(f))

def check_hit(chunk_text: str, markers: list) -> bool:
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

print("\n--- 1. RETRIEVAL BENCHMARK EVALUATION (22 Gold Questions) ---")
retrieval_results = []
for idx, (q, info) in enumerate(gold_data.items(), 1):
    doc_id = info["sample"]
    markers = info["markers"]
    gold_role = info["role"]
    
    # Map role
    mapping = map_query_to_roles(q, mode="hybrid")
    pred_role = mapping.roles[0] if mapping.roles else "Reasoning"
    is_role_correct = (pred_role == gold_role)
    
    # Strict relax 0
    chunks = store.retrieve(question=q, document_id=doc_id, mode="role", roles=[pred_role], relax=0, top_k=5)
    
    ranks = []
    for rank, c in enumerate(chunks, 1):
        if check_hit(c["text"], markers):
            ranks.append(rank)
            
    hit1 = (1 in ranks)
    hit3 = any(r <= 3 for r in ranks)
    hit5 = any(r <= 5 for r in ranks)
    mrr = (1.0 / ranks[0]) if ranks else 0.0
    
    retrieval_results.append({
        "num": idx,
        "sample": doc_id,
        "question": q,
        "gold_role": gold_role,
        "pred_role": pred_role,
        "role_correct": is_role_correct,
        "chunks_count": len(chunks),
        "hit1": hit1,
        "hit3": hit3,
        "hit5": hit5,
        "mrr": mrr,
        "ranks": ranks,
    })

for sample in ["all", "sample2", "sample3"]:
    subset = retrieval_results if sample == "all" else [r for r in retrieval_results if r["sample"] == sample]
    n = len(subset)
    hit1_rate = sum(1 for r in subset if r["hit1"]) / n
    hit3_rate = sum(1 for r in subset if r["hit3"]) / n
    hit5_rate = sum(1 for r in subset if r["hit5"]) / n
    mean_mrr = sum(r["mrr"] for r in subset) / n
    role_acc = sum(1 for r in subset if r["role_correct"]) / n
    print(f"[{sample.upper()} N={n}] RoleAcc: {role_acc*100:.1f}% | Hit@1: {hit1_rate*100:.1f}% | Hit@3: {hit3_rate*100:.1f}% | Hit@5: {hit5_rate*100:.1f}% | MRR: {mean_mrr:.3f}")

output_path = Path("scratch/postfix_retrieval_results.json")
output_path.write_text(json.dumps(retrieval_results, indent=2), encoding="utf-8")
print(f"Saved retrieval metrics to {output_path}")
