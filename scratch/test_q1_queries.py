import sys
sys.path.insert(0, ".")
import re
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks

store = LegalVectorStore()
q1 = "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"

# Test different scoring queries
queries = [
    q1,
    "What were respondent-5 Bharathi arguments concerning mortgage transactions and prohibitory order?",
    "respondent-5 mortgage conditional transfer substitution Indian bank Sun TV shares",
]

chunks = store.retrieve(question=q1, document_id="sample3", mode="role", roles=["Arguments of Respondent"], relax=0, top_k=15)

for q_test in queries:
    print(f"\n--- Testing query: {q_test} ---")
    scored, _ = score_chunks(q_test, chunks)
    for c in sorted(scored, key=lambda x: x.get("rerank_score", -99), reverse=True)[:5]:
        cid = c.get("id") or c.get("chunk_id")
        print(f"  {cid} ({c.get('rerank_score'):.3f}): {c.get('text', '')[:100]}...")
