import json
import urllib.request
from src.rag.generator import GENERATOR_SCHEMA, format_context_for_prompt
import chromadb
from src.config import CHROMA_DIR, ROLE_COLLECTION_NAME

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_or_create_collection(ROLE_COLLECTION_NAME)

res = col.get(ids=['sampleA_role_2', 'sampleA_role_1'], include=['documents', 'metadatas'])
top_chunks = [{'id': cid, 'chunk_id': cid, 'text': doc, 'metadata': meta, 'role': meta.get('role')} for cid, doc, meta in zip(res['ids'], res['documents'], res['metadatas'])]
f_ctx, _ = format_context_for_prompt(top_chunks)

q = "Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?"

sys_prompt = (
    "You are a strict, grounded legal assistant. Answer the user's question using ONLY the provided context chunks.\n"
    "Rules:\n"
    "1. Answer in at most 4 concise, factual sentences.\n"
    "2. Every factual statement must cite the supporting passage index, e.g. [1] or [2].\n"
    "3. STRICT PARTY IDENTIFICATION & ATTRIBUTION:\n"
    "   - Names ending in ', J.' or ', JJ.' or preceded by 'Justice' or 'Hon\\'ble' are JUDGES, NEVER the petitioner or appellant.\n"
    "   - The petitioner/appellant is the party filing the petition (e.g. 'filed by [Party Name]', or marked as Petitioner/Appellant in the cause title).\n"
    "   - Never confuse judges or counsel with the petitioner.\n"
    "4. MULTI-PART & DETAILED FACTUAL QUESTIONS: If the question asks for multiple factual elements (such as party identity, challenged orders/notifications, specific rules or sections, and their subject matter/purpose), you MUST explicitly state each requested element supported by the context (including party name, complete order/G.O. identifier with department and date, specific rules amended/inserted, and the subject/fund involved).\n"
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
print(data.get("message", {}).get("content"))
