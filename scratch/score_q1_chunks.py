import sys
sys.path.insert(0, '.')
from src.rag.vectorstore import LegalVectorStore
from src.rag.graph import score_chunks

store = LegalVectorStore()
q = 'What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?'
res = store.retrieve(q, document_id='sample3_fixed', mode='role', roles=['Arguments of Respondent'], top_k=15)
scored, _ = score_chunks(q, res)
reranked = sorted(scored, key=lambda x: x.get('rerank_score', -99.0), reverse=True)
for r in reranked:
    cid = r.get('id')
    score = r.get('rerank_score')
    first_para = r.get('metadata', {}).get('first_para_marker')
    txt = r.get('text', '')[:100].replace('\n', ' ')
    print(f"{cid} (para: {first_para}) | score={score:.3f} | {txt}")
