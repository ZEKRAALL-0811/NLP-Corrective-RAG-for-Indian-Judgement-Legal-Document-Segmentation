import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks

store = LegalVectorStore()
query = "Under what section was the Contempt of Courts Act invoked?"

for relax in [0, 1, 2, 3]:
    chunks = store.retrieve(
        question=query,
        document_id="sample3",
        mode="role",
        roles=["Arguments of Respondent"],
        relax=relax,
        top_k=5,
    )
    scored, _ = score_chunks(query, chunks)
    print(f"\n--- RELAX LEVEL {relax} (Returned {len(scored)} chunks) ---")
    for c in scored:
        meta = c.get("metadata", {})
        print(f"ID: {c['chunk_id']} | Role: {meta.get('role')} | CE Score: {c.get('rerank_score', 0):.4f} | Dist: {c.get('distance', 0):.4f}")
        print(f"Text snippet: {c['text'][:120]}...")
