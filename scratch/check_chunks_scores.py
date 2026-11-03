import chromadb
from src.config import CHROMA_DIR, ROLE_COLLECTION_NAME
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks
import numpy as np

vstore = LegalVectorStore()
q = 'Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?'

col = vstore.role_collection
res = col.get(ids=['sampleA_role_0', 'sampleA_role_1', 'sampleA_role_2', 'sampleA_role_3'], include=['documents', 'metadatas', 'embeddings'])

q_emb = vstore.embed_query(q)

for i, cid in enumerate(res['ids']):
    doc = res['documents'][i]
    meta = res['metadatas'][i]
    emb = res['embeddings'][i]
    cos_dist = 1.0 - float(np.dot(q_emb, emb))
    print(f"{cid} | Role: {meta.get('role')} | Cos Dist: {cos_dist:.4f}")

scored, _ = score_chunks(q, [{'id': cid, 'text': doc, 'metadata': meta} for cid, doc, meta in zip(res['ids'], res['documents'], res['metadatas'])])
for s in scored:
    print(f"CE score for {s['id']}: {s['rerank_score']:.4f}")
