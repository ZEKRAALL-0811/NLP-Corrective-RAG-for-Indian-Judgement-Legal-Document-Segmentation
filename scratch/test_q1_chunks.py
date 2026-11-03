import sys
sys.path.insert(0, ".")
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks

store = LegalVectorStore()
q1 = "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"
chunks = store.retrieve(question=q1, document_id="sample3", mode="role", roles=["Arguments of Respondent"], relax=0, top_k=15)
print(f"Total chunks retrieved: {len(chunks)}")
for c in chunks:
    cid = c.get("id") or c.get("chunk_id")
    print(f"{cid} (role={c.get('role')}): {c.get('text', '')[:120]}...")

scored, _ = score_chunks(q1, chunks)
print("\nReranked chunks:")
for c in sorted(scored, key=lambda x: x.get("rerank_score", -99), reverse=True)[:5]:
    cid = c.get("id") or c.get("chunk_id")
    print(f"SCORE {c.get('rerank_score'):.3f}: {cid} -> {c.get('text', '')[:120]}...")
