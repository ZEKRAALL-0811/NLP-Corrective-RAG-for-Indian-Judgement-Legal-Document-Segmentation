import re
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks
from src.rag.role_mapper import map_query_to_roles

vstore = LegalVectorStore()

test_queries = [
    "Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?",
    "Which association filed this writ petition, and which amendment is under challenge?",
    "What order introduced Rules 35-A to 35-H?",
    "Which Rules were added to the Tamil Nadu Minor Mineral Concession Rules to establish the Green Fund?",
]

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
        # Check if specific rule numbers match
        q_rules = re.findall(r'\b(?:rules?|rule)\s+(\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?)\b', q_lower)
        if q_rules:
            # If query has specific rule numbers, check if text has them
            if any(r in t_lower for r in q_rules):
                bonus += 5.0
            elif re.search(r'\brules?\s+\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?\b', text, re.I):
                bonus += 2.0
        else:
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

    # 4. Green Fund keyword match
    if 'green fund' in q_lower and 'green fund' in t_lower:
        bonus += 2.0

    # 5. Introductory zone relevance for case identity
    rel_pos = metadata.get('rel_pos', 0.5)
    if rel_pos <= 0.08 and bonus > 0:
        bonus += 2.0

    return bonus

for q in test_queries:
    print('='*70)
    print('Query:', q)
    chunks = vstore.retrieve(
        question=q,
        document_id='sampleA',
        mode='role',
        roles=None,
        relax=3,
        top_k=25,
    )
    scored, _ = score_chunks(q, chunks)
    for c in scored:
        ce = c.get('rerank_score', 0.0)
        bon = compute_evidence_bonus(q, c.get('text', ''), c.get('metadata', {}))
        c['composite_score'] = ce + bon
    
    ranked = sorted(scored, key=lambda x: x['composite_score'], reverse=True)
    for i, c in enumerate(ranked[:3]):
        cid = c.get('metadata', {}).get('chunk_id') or c.get('id')
        role = c.get('metadata', {}).get('role')
        tot = c['composite_score']
        snip = c.get('text', '')[:100].encode('ascii', errors='replace').decode('ascii')
        print(f'  [{i}] {cid} ({role}, score: {tot:.2f}): {snip}')
