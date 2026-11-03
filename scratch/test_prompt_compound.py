import json
import urllib.request
from src.rag.generator import GENERATOR_SCHEMA, format_context_for_prompt
import chromadb
from src.config import CHROMA_DIR, ROLE_COLLECTION_NAME

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_or_create_collection(ROLE_COLLECTION_NAME)

res = col.get(ids=['sampleA_role_2'], include=['documents', 'metadatas'])
chunk2 = {'id': 'sampleA_role_2', 'chunk_id': 'sampleA_role_2', 'text': res['documents'][0], 'metadata': res['metadatas'][0], 'role': res['metadatas'][0].get('role')}

q = 'Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?'
f_ctx, _ = format_context_for_prompt([chunk2])

sys_prompt = (
    "You are a strict, grounded legal assistant. Answer the user's question using ONLY the provided context chunks.\n"
    "Rules:\n"
    "1. Answer in at most 4 concise, factual sentences.\n"
    "2. Every factual statement must cite the supporting passage index, e.g. [1] or [2].\n"
    "3. MULTI-PART & COMPOUND QUESTIONS: If the question asks multiple things (e.g. who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged), you MUST answer each part completely using the facts in the context.\n"
    "4. STRICT PARTY & INSTITUTIONAL ATTRIBUTION: Never attribute an argument made by one party to another.\n"
    "5. If the context does not contain the answer, set 'answerable': false and 'answer': 'The retrieved parts of this judgment do not answer this question.', 'citations': [].\n"
    "6. Do NOT invent or hallucinate any dates, case numbers, statutory sections, or factual claims not present in the context.\n"
    "Output JSON adhering strictly to the schema."
)

prompt = f"User Question: {q}\n\nContext:\n{f_ctx}\n\nProvide your grounded legal answer following all prompt rules."
payload = {
    'model': 'llama3.2:3b',
    'messages': [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': prompt}
    ],
    'format': GENERATOR_SCHEMA,
    'stream': False,
    'options': {'temperature': 0.0, 'seed': 42, 'num_predict': 250, 'num_ctx': 2048},
    'keep_alive': '30m'
}
req = urllib.request.Request('http://localhost:11434/api/chat', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
print(data.get('message', {}).get('content'))
