from src.rag.vectorstore import LegalVectorStore
vstore = LegalVectorStore()
q = 'Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?'

res_none = vstore.retrieve(q, 'sampleA', mode='role', roles=['None'], relax=0, top_k=5)
print('--- Roles=[None] ---')
for i, c in enumerate(res_none):
    cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
    dist = c.get('distance')
    txt = c.get('text', '')[:100].encode('ascii', errors='replace').decode('ascii')
    print(f'[{i}] {cid} (dist {dist:.4f}): {txt}')

res_none_issue = vstore.retrieve(q, 'sampleA', mode='role', roles=['None', 'Issue'], relax=0, top_k=10)
print('\n--- Roles=[None, Issue] ---')
for i, c in enumerate(res_none_issue):
    cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
    dist = c.get('distance')
    txt = c.get('text', '')[:100].encode('ascii', errors='replace').decode('ascii')
    print(f'[{i}] {cid} (dist {dist:.4f}): {txt}')
