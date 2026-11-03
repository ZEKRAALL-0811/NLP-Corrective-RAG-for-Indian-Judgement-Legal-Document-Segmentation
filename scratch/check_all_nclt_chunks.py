from src.rag.vectorstore import LegalVectorStore

vs = LegalVectorStore()
docs = vs.role_collection.get(where={'document_id': 'sample3'}, include=['documents', 'metadatas'])
for doc, meta, cid in zip(docs['documents'], docs['metadatas'], docs['ids']):
    if 'nclt' in doc.lower():
        role = meta.get('role')
        rel_pos = meta.get('rel_pos')
        print(f"=== {cid} [{role}] (rel_pos={rel_pos}) ===")
        print(doc)
        print('-'*50)
