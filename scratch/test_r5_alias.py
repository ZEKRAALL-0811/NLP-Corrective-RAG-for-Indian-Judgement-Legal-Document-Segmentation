import sys
sys.path.insert(0, ".")
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks

store = LegalVectorStore()
q_alias = "What were respondent-5’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"
chunks = store.retrieve(question=q_alias, document_id="sample3", mode="role", roles=["Arguments of Respondent"], relax=0, top_k=8)
print("Retrieved chunks:")
for c in chunks:
    cid = c.get("id") or c.get("chunk_id")
    print(f"  {cid}: {c.get('text', '')[:120]}...")

scored, _ = score_chunks(q_alias, chunks)
print("\nReranked:")
for c in sorted(scored, key=lambda x: x.get("rerank_score", -99), reverse=True)[:5]:
    cid = c.get("id") or c.get("chunk_id")
    print(f"  SCORE {c.get('rerank_score'):.3f}: {cid} -> {c.get('text', '')[:100]}...")
