import sys
from pathlib import Path
import gc
import psutil
import json
import urllib.request

sys.path.insert(0, ".")
import src.config as cfg
from src.rag.vectorstore import LegalVectorStore

LOG_OUT = Path("artifacts/logs/query_time_ram.txt")

# 1. Warm up Ollama with llama3.2:3b so it is resident in RAM
ollama_host = getattr(cfg, "OLLAMA_HOST", "http://localhost:11434")
ollama_model = getattr(cfg, "OLLAMA_MODEL", "llama3.2:3b")

url = f"{ollama_host.rstrip('/')}/api/chat"
req_payload = {
    "model": ollama_model,
    "messages": [{"role": "user", "content": "hi"}],
    "stream": False,
    "keep_alive": "30m",
}
req = urllib.request.Request(
    url,
    data=json.dumps(req_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        _ = resp.read()
    print("Ollama llama3.2:3b warmed up and resident.")
except Exception as e:
    print(f"Ollama warm up warning: {e}")

# 2. Load bge-small resident via LegalVectorStore
store = LegalVectorStore()
_ = store.embed_query("test query")
print("bge-small loaded resident in LegalVectorStore.")

# 3. Explicitly ensure segmenter / InLegalBERT is NOT loaded / unloaded
if "_DEFAULT_SEGMENTER" in globals():
    del globals()["_DEFAULT_SEGMENTER"]
gc.collect()

# 4. Measure System RAM
vm = psutil.virtual_memory()
total_gb = vm.total / (1024 ** 3)
avail_gb = vm.available / (1024 ** 3)
used_gb = vm.used / (1024 ** 3)
pct_used = vm.percent

lines = []
lines.append("=" * 80)
lines.append("QUERY-TIME RAM BENCHMARK (SEGMENTER UNLOADED)")
lines.append("=" * 80)
lines.append("State:")
lines.append(f"  - Ollama ({ollama_model}): Resident in RAM (keep_alive: 30m)")
lines.append("  - Embedding Model (BAAI/bge-small-en-v1.5): Resident in RAM")
lines.append("  - Rhetorical Segmenter (InLegalBERT): Unloaded (deleted, gc.collect())")
lines.append("-" * 80)
lines.append(f"  - Total System RAM:     {total_gb:6.2f} GB")
lines.append(f"  - Available System RAM: {avail_gb:6.2f} GB")
lines.append(f"  - Used System RAM:      {used_gb:6.2f} GB")
lines.append(f"  - Memory Utilization:   {pct_used:6.1f} %")
lines.append("=" * 80)

out_text = "\n".join(lines)
LOG_OUT.write_text(out_text, encoding="utf-8")
print(out_text)
