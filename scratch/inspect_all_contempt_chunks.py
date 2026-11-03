import chromadb

client = chromadb.PersistentClient(path="artifacts/chroma")
col = client.get_collection("jurisgraph_role_chunks")

cids = ["sample3_role_61", "sample3_role_100", "sample3_role_98", "sample3_role_91", "sample3_role_77", "sample3_role_117"]
res = col.get(ids=cids, include=["documents", "metadatas"])

for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
    print("=" * 60)
    print(f"ID: {cid} | Role: {meta.get('role')}")
    print(doc)
