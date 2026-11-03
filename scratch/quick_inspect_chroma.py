import chromadb

client = chromadb.PersistentClient(path="artifacts/chroma")
col = client.get_collection("jurisgraph_role_chunks")
print(f"Total count in jurisgraph_role_chunks: {col.count()}", flush=True)

res = col.get(where={"document_id": "sample3"}, include=["documents", "metadatas"])
print(f"sample3 chunks count: {len(res['ids'])}", flush=True)

found = []
for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
    doc_lower = doc.lower()
    if "contempt" in doc_lower or "section 12" in doc_lower or "1971" in doc_lower:
        found.append((cid, meta.get("role"), meta.get("role2"), meta.get("chunk_idx"), doc))

print(f"\nFound {len(found)} matching chunks for 'contempt' or 'section 12' or '1971':", flush=True)
for cid, r1, r2, cidx, doc in found:
    print("=" * 60, flush=True)
    print(f"ID: {cid} | Role: {r1} | Role2: {r2} | Index: {cidx}", flush=True)
    print(f"Snippet: {doc[:300]}...", flush=True)
