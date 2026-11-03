import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chromadb
from src.rag.grader import grade_chunks

client = chromadb.PersistentClient(path="artifacts/chroma")
col = client.get_collection("jurisgraph_role_chunks")

res = col.get(ids=["sample3_role_61", "sample3_role_100", "sample3_role_98"])
chunks = [{"id": cid, "chunk_id": cid, "text": doc, "metadata": meta} for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

query = "Under what section was the Contempt of Courts Act invoked?"
graded, route, _ = grade_chunks(query, chunks)

print(f"Grader Route: {route}")
for g in graded:
    print(f"ID: {g['chunk_id']} | Grade: {g['grade']} | Score: {g['score']}")
