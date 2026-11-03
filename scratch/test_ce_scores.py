import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chromadb
from src.rag.reranker import score_chunks

client = chromadb.PersistentClient(path="artifacts/chroma")
col = client.get_collection("jurisgraph_role_chunks")

chunk_ids = ["sample3_role_81", "sample3_role_93", "sample3_role_53", "sample3_role_117", "sample3_role_91", "sample3_role_98", "sample3_role_100"]
res = col.get(ids=chunk_ids, include=["documents", "metadatas"])

chunks = [{"id": cid, "text": doc, "metadata": meta} for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

query = "Under what section was the Contempt of Courts Act invoked?"
scored, _ = score_chunks(query, chunks)

for c in sorted(scored, key=lambda x: x["rerank_score"], reverse=True):
    print(f"ID: {c['id']} | Score: {c['rerank_score']:.4f} | Role: {c['metadata'].get('role')}")
    print(f"Snippet: {c['text'][:150]}...\n")
