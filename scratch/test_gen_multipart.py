import json
import urllib.request
from src.rag.generator import GENERATOR_SYSTEM_PROMPT, GENERATOR_SCHEMA, format_context_for_prompt
import chromadb
from src.config import CHROMA_DIR, ROLE_COLLECTION_NAME

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_or_create_collection(ROLE_COLLECTION_NAME)

res = col.get(ids=['sampleA_role_2'], include=['documents', 'metadatas'])
chunk2 = {'id': 'sampleA_role_2', 'chunk_id': 'sampleA_role_2', 'text': res['documents'][0], 'metadata': res['metadatas'][0], 'role': res['metadatas'][0].get('role')}

q = 'Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?'
f_ctx, _ = format_context_for_prompt([chunk2])

# Test 1: with GENERATOR_SYSTEM_PROMPT with explicit multi-part instruction
sys_prompt = GENERATOR_SYSTEM_PROMPT

user_prompt = (
    f"User Question: {q}\n\n"
    f"Context:\n{f_ctx}\n\n"
    "Provide your grounded legal answer following all prompt rules. "
    "If the question asks multiple specific things (e.g. who is the party, and what specific order/rules are challenged), "
    "make sure to answer each specific aspect that is supported by the context."
)

payload = {
    'model': 'llama3.2:3b',
    'messages': [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt}
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
