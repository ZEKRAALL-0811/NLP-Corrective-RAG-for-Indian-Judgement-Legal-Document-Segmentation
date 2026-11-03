import sys
import re
import json
import urllib.request
from src.rag.vectorstore import LegalVectorStore
from src.rag.reranker import score_chunks
from src.rag.role_mapper import map_query_to_roles, PAT_METADATA, PAT_PETITIONER
from src.rag.generator import GENERATOR_SCHEMA, format_context_for_prompt

vstore = LegalVectorStore()
q = "Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?"

# 1. Test query mapping
m = map_query_to_roles(q)
print("Mapping:", m.roles, m.kind, m.confidence)

# 2. Test candidate retrieval (role + cross + intro)
roles = m.roles
chunks = vstore.retrieve(q, "sampleA", mode="role", roles=roles, relax=0, top_k=5)
existing_ids = {c.get("id") or c.get("chunk_id") for c in chunks}

cross_chunks = vstore.retrieve(q, "sampleA", mode="role", roles=None, relax=3, top_k=25)
for cc in cross_chunks:
    cid = cc.get("id") or cc.get("chunk_id")
    if cid and cid not in existing_ids:
        chunks.append(cc)
        existing_ids.add(cid)

intro_chunks = vstore.retrieve(q, "sampleA", mode="role", roles=None, relax=3, rel_pos_range=(0.0, 0.12), top_k=5)
for ic in intro_chunks:
    cid = ic.get("id") or ic.get("chunk_id")
    if cid and cid not in existing_ids:
        chunks.append(ic)
        existing_ids.add(cid)

print(f"Total candidate chunks retrieved: {len(chunks)}")
print("Has sampleA_role_2?", any((c.get("id") or c.get("chunk_id")) == "sampleA_role_2" for c in chunks))
print("Has sampleA_role_1?", any((c.get("id") or c.get("chunk_id")) == "sampleA_role_1" for c in chunks))
print("Has sampleA_role_0?", any((c.get("id") or c.get("chunk_id")) == "sampleA_role_0" for c in chunks))

# 3. Test evidence-first scoring
def compute_evidence_bonus(question: str, text: str, metadata: dict) -> float:
    bonus = 0.0
    q_lower = question.lower()
    t_lower = text.lower()
    
    if ("g.o" in q_lower or "government order" in q_lower) and ("g.o" in t_lower or "government order" in t_lower):
        if re.search(r"g\.?\s*o\.?(?:\s*\([a-z.]+\))?\s*(?:ms\.?\s*)?(?:no\.?\s*)?\d+", text, re.I):
            bonus += 4.0
        else:
            bonus += 1.5

    if "rule" in q_lower and "rule" in t_lower:
        q_rules = re.findall(r"\b(?:rules?|rule)\s+(\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?)\b", q_lower)
        if q_rules:
            if any(r in t_lower for r in q_rules):
                bonus += 5.0
            elif re.search(r"\brules?\s+\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?\b", text, re.I):
                bonus += 2.0
        else:
            if re.search(r"\brules?\s+\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?\b", text, re.I):
                bonus += 3.0
            else:
                bonus += 1.0

    if re.search(r"\b(?:who\s+(?:is|was|are|were)\s+(?:the\s+)?(?:petitioner|appellant|respondent)|who\s+filed|which\s+(?:party|association|person|company|entity)\s+filed|name\s+(?:of\s+)?(?:the\s+)?(?:petitioner|appellant))\b", q_lower):
        if re.search(r"\b(?:petition(?:er)?\s+has\s+been\s+filed\s+by|filed\s+by\s+(?:the\s+)?|writ\s+petition\s+has\s+been\s+filed|petitioner\(s\)|appellant\(s\))\b", text, re.I):
            bonus += 4.0
        elif "petitioner" in t_lower or "appellant" in t_lower:
            bonus += 1.0

    if "green fund" in q_lower and "green fund" in t_lower:
        bonus += 2.0

    rel_pos = metadata.get("rel_pos", 0.5)
    if rel_pos <= 0.08 and bonus > 0:
        bonus += 2.0

    return bonus

scored, _ = score_chunks(q, chunks)
for c in scored:
    ce = c.get("rerank_score", 0.0)
    bon = compute_evidence_bonus(q, c.get("text", ""), c.get("metadata", {}))
    c["composite_score"] = ce + bon

ranked = sorted(scored, key=lambda x: x["composite_score"], reverse=True)
print("\nTop 3 ranked chunks:")
for i, c in enumerate(ranked[:3]):
    cid = c.get("metadata", {}).get("chunk_id") or c.get("id")
    print(f"[{i}] {cid} (score {c['composite_score']:.2f})")

# 4. Test generator on top 2 chunks
top_chunks = ranked[:2]
f_ctx, _ = format_context_for_prompt(top_chunks)

sys_prompt = (
    "You are a strict, grounded legal assistant. Answer the user's question using ONLY the provided context chunks.\n"
    "Rules:\n"
    "1. Answer in at most 4 concise, factual sentences.\n"
    "2. Every factual statement must cite the supporting passage index, e.g. [1] or [2].\n"
    "3. STRICT PARTY IDENTIFICATION & ATTRIBUTION:\n"
    "   - Names ending in ', J.' or ', JJ.' or preceded by 'Justice' or 'Hon\\'ble' are JUDGES, NEVER the petitioner or appellant.\n"
    "   - The petitioner/appellant is the party filing the petition (e.g. 'filed by [Party Name]', or marked as Petitioner/Appellant in the cause title).\n"
    "   - Never confuse judges or counsel with the petitioner.\n"
    "4. MULTI-PART QUESTIONS: If the question asks multiple things (e.g. who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged), answer each part completely using the facts in the context.\n"
    "5. If the context does not contain the answer, set 'answerable': false and 'answer': 'The retrieved parts of this judgment do not answer this question.', 'citations': [].\n"
    "6. Do NOT invent or hallucinate any dates, case numbers, statutory sections, or factual claims not present in the context.\n"
    "Output JSON adhering strictly to the schema."
)

prompt = f"User Question: {q}\n\nContext:\n{f_ctx}\n\nProvide your grounded legal answer following all prompt rules."
payload = {
    "model": "llama3.2:3b",
    "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ],
    "format": GENERATOR_SCHEMA,
    "stream": False,
    "options": {"temperature": 0.0, "seed": 42, "num_predict": 250, "num_ctx": 2048},
    "keep_alive": "30m"
}
req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
ans = json.loads(data.get("message", {}).get("content"))
print("\nGenerated answer:")
print("Answerable:", ans.get("answerable"))
print("Answer:", ans.get("answer"))
print("Citations:", ans.get("citations"))
