import json
from src.rag.graph import run_query

q = "When did the petitioners sell their shares through the Offer For Sale?"
res = run_query(q, document_id="sample2", mode="role")
print("FINAL STATUS:", res.get("final_status"))
print("ANSWER:", res.get("generation"))
print("CITATIONS:", res.get("citations"))
print("VERIFIED CHUNKS COUNT:", len(res.get("verified_context", [])))
for i, c in enumerate(res.get("verified_context", [])):
    print(f"--- Chunk {i+1} ({c.get('chunk_id')}, grade={c.get('grade')}, score={c.get('rerank_score')}):")
    print(c.get("text", "")[:300])

print("\nTRACE:")
for tr in res.get("trace", []):
    print(tr.get("node"), tr.get("duration_ms"), tr.get("route"), tr.get("details"))
