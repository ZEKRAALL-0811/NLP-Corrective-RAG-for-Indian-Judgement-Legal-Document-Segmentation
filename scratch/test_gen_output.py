import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chromadb
from src.rag.generator import generate_grounded_answer

client = chromadb.PersistentClient(path="artifacts/chroma")
col = client.get_collection("jurisgraph_role_chunks")

res = col.get(ids=["sample3_role_93", "sample3_role_81"])
chunks = [{"id": cid, "text": doc, "metadata": meta} for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

query = "Under what section was the Contempt of Courts Act invoked?"
gen_res = generate_grounded_answer(query, chunks)

print("Answerable:", gen_res.get("answerable"))
print("Answer:", gen_res.get("answer"))
print("Citations:", gen_res.get("citations"))
print("Status:", gen_res.get("status"))
