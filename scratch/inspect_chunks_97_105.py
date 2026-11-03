from src.rag.vectorstore import LegalVectorStore

vs = LegalVectorStore()
chunks = vs.role_collection.get(ids=[f'sample3_role_{i}' for i in range(97, 105)])
for cid, meta, text in zip(chunks['ids'], chunks['metadatas'], chunks['documents']):
    print(f"=== {cid} ({meta.get('role')}) ===")
    print(text)
    print()
