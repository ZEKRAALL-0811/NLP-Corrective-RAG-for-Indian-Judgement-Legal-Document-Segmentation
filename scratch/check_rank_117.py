import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.vectorstore import LegalVectorStore

store = LegalVectorStore()
col = store._get_collection("role")

query = "Under what section was the Contempt of Courts Act invoked?"
q_vec = store.embed_query(query)

# Query top 30 chunks without role filter
res = col.query(
    query_embeddings=[q_vec],
    n_results=30,
    where={"document_id": "sample3"},
    include=["documents", "metadatas", "distances"]
)

ids = res["ids"][0]
docs = res["documents"][0]
metas = res["metadatas"][0]
dists = res["distances"][0]

print(f"Top 30 chunks for '{query}':")
for rank, (cid, dist, meta) in enumerate(zip(ids, dists, metas), start=1):
    role = meta.get("role")
    is_target = (cid == "sample3_role_117")
    target_mark = " <--- TARGET (Section 12)!" if is_target else ""
    if "section 12" in docs[rank-1].lower():
        target_mark = " <--- CONTAINS SECTION 12!"
    print(f"Rank {rank:2d}: {cid} | Dist: {dist:.4f} | Role: {role}{target_mark}")
