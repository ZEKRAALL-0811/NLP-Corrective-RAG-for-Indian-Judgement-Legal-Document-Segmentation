import chromadb

client = chromadb.PersistentClient(path="artifacts/chroma")
col = client.get_collection("jurisgraph_role_chunks")

res = col.get(where={"document_id": "sample3"}, include=["documents", "metadatas"])

print(f"Total chunks in sample3: {len(res['ids'])}", flush=True)

for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
    doc_lower = doc.lower()
    if "section 12" in doc_lower or "contempt of courts act" in doc_lower:
        print("=" * 70, flush=True)
        print(f"ID: {cid} | Role: {meta.get('role')} | Role2: {meta.get('role2')} | rel_pos: {meta.get('rel_pos')}", flush=True)
        print(f"Text:\n{doc}\n", flush=True)
