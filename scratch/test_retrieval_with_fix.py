import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.chunker import chunk_document
from src.rag.vectorstore import LegalVectorStore
from src.rag.graph import score_chunks

# 1. Segment and apply block fix
recs, stats = segment_document('legalDataset/samples/sample3.pdf', use_cache=True)
fixed_recs = []
for idx, r in enumerate(recs):
    r_copy = dict(r)
    if 157 <= idx <= 249:
        if r_copy["role"] != "None":
            r_copy["role"] = "Arguments of Respondent"
    elif idx == 270:
        r_copy["role"] = "Arguments of Respondent"
    elif idx == 271:
        r_copy["role"] = "Arguments of Petitioner"
    fixed_recs.append(r_copy)

# 2. Ingest into a test collection
store = LegalVectorStore()
# Ingest with document_id="sample3_fixed"
store.ingest_document(fixed_recs, document_id="sample3_fixed", mode="role")

QUESTIONS = [
    ("q1", "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?", ["Arguments of Respondent"]),
    ("q2", "What was the legal effect of the prohibitory order dated 21 April 2005, and how did the parties disagree over whether it was violated?", ["Reasoning", "Arguments of Respondent"]),
    ("q3", "Why did the respondents contend that allegations of fraud and collusion required a proper trial?", ["Arguments of Respondent"]),
    ("q4", "How did the respondents distinguish the High Court’s powers under Article 215 of the Constitution from the procedures under the Contempt of Courts Act?", ["Arguments of Respondent"]),
]

for q_id, q_text, q_roles in QUESTIONS:
    print(f"\n==========================================")
    print(f"TESTING {q_id}: {q_text}")
    print(f"Target roles: {q_roles}")
    retrieved = store.retrieve(
        question=q_text,
        document_id="sample3_fixed",
        mode="role",
        roles=q_roles,
        relax=0,
        top_k=5,
    )
    scored, _ = score_chunks(q_text, retrieved)
    reranked = sorted(scored, key=lambda x: x.get("rerank_score", -99.0), reverse=True)
    for idx, c in enumerate(reranked[:3], start=1):
        cid = c.get("id") or c.get("chunk_id")
        crole = c.get("metadata", {}).get("role")
        score = c.get("rerank_score", 0.0)
        snippet = c.get("text", "")[:120].replace("\n", " ")
        print(f"  [{idx}] {cid} ({crole}) score={score:.2f} | {snippet}...")
