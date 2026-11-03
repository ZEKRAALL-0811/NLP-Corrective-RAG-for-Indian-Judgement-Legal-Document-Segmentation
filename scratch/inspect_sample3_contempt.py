import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.vectorstore import LegalVectorStore

store = LegalVectorStore()
col_role = store._get_collection("role")
print("Total count in role collection:", col_role.count())

# Query all chunks for sample3
res = col_role.get(where={"document_id": "sample3"}, include=["documents", "metadatas"])
print(f"Total chunks for sample3: {len(res['ids'])}")

# Search for Contempt of Courts Act and Section 12
found = []
for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
    if "contempt of courts" in doc.lower() or "section 12" in doc.lower():
        found.append((cid, meta.get("role"), meta.get("role2"), meta.get("chunk_idx"), doc))

print(f"\nFound {len(found)} chunks matching 'contempt of courts' or 'section 12':")
for cid, r1, r2, cidx, doc in found:
    print("=" * 60)
    print(f"ID: {cid} | Role: {r1} | Role2: {r2} | Index: {cidx}")
    print(f"Snippet: {doc[:300]}...")
