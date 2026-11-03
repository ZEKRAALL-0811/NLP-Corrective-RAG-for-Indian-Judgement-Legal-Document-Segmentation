import re
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks

vstore = LegalVectorStore()
q = 'Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?'

chunks = vstore.retrieve(
    question=q,
    document_id='sampleA',
    mode='role',
    roles=None,
    relax=3,
    top_k=25,
)

scored, _ = score_chunks(q, chunks)

def compute_evidence_bonus(question: str, text: str, metadata: dict) -> float:
    bonus = 0.0
    q_lower = question.lower()
    t_lower = text.lower()
    
    # 1. Government Order / G.O. match
    if ('g.o' in q_lower or 'government order' in q_lower) and ('g.o' in t_lower or 'government order' in t_lower):
        if re.search(r'g\.?\s*o\.?(?:\s*\([a-z.]+\))?\s*(?:ms\.?\s*)?(?:no\.?\s*)?\d+', text, re.I):
            bonus += 4.0
        else:
            bonus += 1.5

    # 2. Specific Rules match
    if 'rule' in q_lower and 'rule' in t_lower:
        if re.search(r'\brules?\s+\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?\b', text, re.I):
            bonus += 3.0
        else:
            bonus += 1.0

    # 3. Party identification match
    if re.search(r'\b(?:who\s+(?:is|was|are|were)\s+(?:the\s+)?(?:petitioner|appellant|respondent)|who\s+filed|which\s+(?:party|association|person|company|entity)\s+filed|name\s+(?:of\s+)?(?:the\s+)?(?:petitioner|appellant))\b', q_lower):
        if re.search(r'\b(?:petition(?:er)?\s+has\s+been\s+filed\s+by|filed\s+by\s+(?:the\s+)?|writ\s+petition\s+has\s+been\s+filed|petitioner\(s\)|appellant\(s\))\b', text, re.I):
            bonus += 4.0
        elif 'petitioner' in t_lower or 'appellant' in t_lower:
            bonus += 1.0

    # 4. Introductory zone relevance for case identity
    rel_pos = metadata.get('rel_pos', 0.5)
    if rel_pos <= 0.08 and bonus > 0:
        bonus += 2.0

    return bonus

for c in scored:
    cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
    ce = c.get('rerank_score', 0.0)
    bonus = compute_evidence_bonus(q, c.get('text', ''), c.get('metadata', {}))
    c['composite_score'] = ce + bonus

ranked = sorted(scored, key=lambda x: x['composite_score'], reverse=True)
print('--- Evidence-First Ranked Chunks ---')
for i, c in enumerate(ranked[:6]):
    cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
    role = c.get('metadata', {}).get('role')
    ce = c.get('rerank_score', 0.0)
    bon = c['composite_score'] - ce
    tot = c['composite_score']
    snippet = c.get('text', '')[:100].encode('ascii', errors='replace').decode('ascii')
    print(f'[{i}] {cid} | Role: {role} | CE: {ce:.2f} + Bonus: {bon:.2f} = {tot:.2f} | Snippet: {snippet}')
