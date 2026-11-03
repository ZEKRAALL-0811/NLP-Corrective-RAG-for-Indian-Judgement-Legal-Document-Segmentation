import json
from src.config import CHROMA_DIR, ROLE_COLLECTION_NAME
from src.rag.vectorstore import LegalVectorStore
from src.rag.role_mapper import map_query_to_roles
from src.rag.reranker import score_chunks

vstore = LegalVectorStore()
q = 'Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?'

mapping = map_query_to_roles(q)
print('Mapping roles:', mapping.roles)
print('Mapping kind:', mapping.kind)
print('Mapping confidence:', mapping.confidence)

# 1. Test retrieve with role filtering as graph.py does
res_role = vstore.retrieve(
    question=q,
    document_id='sampleA',
    mode='role',
    roles=mapping.roles,
    relax=0,
    top_k=5
)
print(f'\n--- Role retrieve (relax=0, roles={mapping.roles}) ---')
for i, c in enumerate(res_role):
    cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
    role = c.get('metadata', {}).get('role')
    dist = c.get('distance')
    txt = c.get('text', '')[:100].encode('ascii', errors='replace').decode('ascii')
    print(f'[{i}] {cid} | Role: {role} | Dist: {dist:.4f} | Snippet: {txt}')

# 2. Test cross-chunks (relax=3) as graph.py does at lines 283-300:
res_unconstrained = vstore.retrieve(
    question=q,
    document_id='sampleA',
    mode='role',
    roles=None,
    relax=3,
    top_k=15
)
print('\n--- Unconstrained retrieve (top 15) ---')
for i, c in enumerate(res_unconstrained):
    cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
    role = c.get('metadata', {}).get('role')
    dist = c.get('distance')
    txt = c.get('text', '')[:100].encode('ascii', errors='replace').decode('ascii')
    print(f'[{i}] {cid} | Role: {role} | Dist: {dist:.4f} | Snippet: {txt}')

# 3. Now let's see how reranker scores all of them
chunks = list(res_role)
existing_ids = {c.get('id') or c.get('chunk_id') for c in chunks}
for cc in res_unconstrained:
    cid = cc.get('id') or cc.get('chunk_id')
    if cid and cid not in existing_ids:
        chunks.append(cc)
        existing_ids.add(cid)

print(f'\nTotal candidate chunks to rerank: {len(chunks)}')
scored, ce_ms = score_chunks(q, chunks)
reranked = sorted(scored, key=lambda x: x.get('rerank_score', -99.0), reverse=True)
print('\n--- Top 10 Reranked ---')
for i, c in enumerate(reranked[:10]):
    cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
    role = c.get('metadata', {}).get('role')
    ce_score = c.get('rerank_score')
    txt = c.get('text', '')[:120].encode('ascii', errors='replace').decode('ascii')
    print(f'[{i}] {cid} | Role: {role} | CE Score: {ce_score:.4f} | Snippet: {txt}')
