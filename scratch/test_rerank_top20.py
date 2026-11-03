import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks

store = LegalVectorStore()
col = store._get_collection("role")

query = "Under what section was the Contempt of Courts Act invoked?"
q_vec = store.embed_query(query)

res = col.query(
    query_embeddings=[q_vec],
    n_results=20,
    where={"document_id": "sample3"},
    include=["documents", "metadatas", "distances"]
)

chunks = [{"id": cid, "chunk_id": cid, "text": doc, "metadata": meta, "distance": dist} 
          for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

scored, _ = score_chunks(query, chunks)

print(f"Top 20 vector chunks after Cross-Encoder reranking:")
for c in sorted(scored, key=lambda x: x["rerank_score"], reverse=True):
    is_target = " <--- SECTION 12" if "section 12" in c["text"].lower() else ""
    print(f"ID: {c['chunk_id']} | CE Score: {c['rerank_score']:.4f} | Dist: {c['distance']:.4f} | Role: {c['metadata'].get('role')}{is_target}")
