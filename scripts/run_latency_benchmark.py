"""
Latency benchmark script for Part A:
1. Tokens/sec for prompt processing and generation for llama3.2:3b using Ollama response fields.
2. Mock batched relevance-grading call:
   - K=3 and K=5 (truncated to 800 chars from real sample2/sample3 chunks)
   - Compare batched vs K separate single-chunk calls (wall time, tokens, speed)
3. Report system RAM available with Ollama (model loaded), bge-small, and InLegalBERT all loaded.
4. Save raw output to artifacts/logs/part_a_latency_benchmark.txt
"""

import sys
from pathlib import Path
import json
import time
import urllib.request
import psutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

sys.path.insert(0, ".")
import src.config as cfg
from src.data.preprocessor import preprocess_document

LOG_FILE = Path("artifacts/logs/part_a_latency_benchmark.txt")
OLLAMA_HOST = getattr(cfg, "OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = getattr(cfg, "OLLAMA_MODEL", "llama3.2:3b")

def call_ollama(messages, schema=None, stream=False):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_ctx": 2048,
        },
        "keep_alive": "30m",
    }
    if schema:
        payload["format"] = schema

    req = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.time() - t0) * 1000
    return res, wall_ms

# Extract 5 real chunks from sample2.pdf
print("Extracting real text chunks from sample2.pdf...")
sents_s2, _ = preprocess_document("legalDataset/samples/sample2.pdf")

# Assemble 5 distinct 800-character chunks
chunks_pool = []
curr = ""
for s in sents_s2[15:]:
    if len(curr) + len(s) < 800:
        curr += " " + s
    else:
        chunks_pool.append(curr.strip()[:800])
        curr = s
        if len(chunks_pool) >= 5:
            break
if len(chunks_pool) < 5 and curr:
    chunks_pool.append(curr.strip()[:800])

while len(chunks_pool) < 5:
    chunks_pool.append(f"Passage chunk {len(chunks_pool)+1}: Detailed legal context regarding tax dispute.")

print(f"Prepared {len(chunks_pool)} chunks (lengths: {[len(c) for c in chunks_pool]}).")

GRADER_SYSTEM_PROMPT = (
    "You are a legal passage relevance grader. Evaluate whether each chunk is relevant to the question. "
    "Output a JSON array of objects with id, grade ('correct' or 'incorrect'), and score (float between 0.0 and 1.0)."
)

BATCH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "grade": {"type": "string", "enum": ["correct", "incorrect"]},
            "score": {"type": "number"},
        },
        "required": ["id", "grade", "score"],
    },
}

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "grade": {"type": "string", "enum": ["correct", "incorrect"]},
        "score": {"type": "number"},
    },
    "required": ["id", "grade", "score"],
}

TEST_QUESTION = "What was the petitioners' argument about the second proviso to Section 139(8A)?"

lines = []
lines.append("=" * 80)
lines.append("PART A: LATENCY & EFFICIENCY BENCHMARK (llama3.2:3b on CPU)")
lines.append("=" * 80)
lines.append(f"Model: {OLLAMA_MODEL} | Host: {OLLAMA_HOST}")
lines.append("")

# 1. Warm-up and Baseline Tokens/Sec Measurement
print("Running warm-up and speed measurement call...")
warmup_messages = [
    {"role": "system", "content": "You are a legal assistant. Answer concisely."},
    {"role": "user", "content": "What is the difference between a decree holder and a judgment debtor in civil execution?"},
]
res_warm, wall_warm_ms = call_ollama(warmup_messages)
p_eval_cnt = res_warm.get("prompt_eval_count", 0)
p_eval_dur = res_warm.get("prompt_eval_duration", 1)  # nanoseconds
e_eval_cnt = res_warm.get("eval_count", 0)
e_eval_dur = res_warm.get("eval_duration", 1)  # nanoseconds

p_tps = p_eval_cnt / (p_eval_dur / 1e9) if p_eval_dur > 0 else 0.0
e_tps = e_eval_cnt / (e_eval_dur / 1e9) if e_eval_dur > 0 else 0.0

lines.append("1. OLLAMA INFERENCE SPEED METRICS (llama3.2:3b):")
lines.append(f"   Prompt tokens: {p_eval_cnt} | Prompt eval duration: {p_eval_dur/1e6:.1f} ms")
lines.append(f"   Prompt Processing Speed:  {p_tps:.2f} tokens/sec")
lines.append(f"   Generated tokens: {e_eval_cnt} | Generation duration: {e_eval_dur/1e6:.1f} ms")
lines.append(f"   Token Generation Speed:   {e_tps:.2f} tokens/sec")
lines.append(f"   Total Wall Time:          {wall_warm_ms:.1f} ms")
lines.append("")

# 2. Benchmark K=3
print("Benchmarking K=3 (batched vs 3 single calls)...")
chunks_3 = chunks_pool[:3]

# K=3 Batched Call
batch_prompt_3 = f"Question: {TEST_QUESTION}\n\nPassages:\n" + "\n\n".join(
    [f"[Passage {i+1}]: {c}" for i, c in enumerate(chunks_3)]
)
batch_msgs_3 = [
    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
    {"role": "user", "content": batch_prompt_3},
]
res_b3, wall_b3_ms = call_ollama(batch_msgs_3, schema=BATCH_SCHEMA)
b3_p_cnt = res_b3.get("prompt_eval_count", 0)
b3_e_cnt = res_b3.get("eval_count", 0)

# K=3 Single Calls
wall_s3_total_ms = 0.0
s3_p_cnt_total = 0
s3_e_cnt_total = 0
for i, c in enumerate(chunks_3):
    single_prompt = f"Question: {TEST_QUESTION}\n\n[Passage {i+1}]: {c}"
    s_msgs = [
        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": single_prompt},
    ]
    res_s, wall_s_ms = call_ollama(s_msgs, schema=SINGLE_SCHEMA)
    wall_s3_total_ms += wall_s_ms
    s3_p_cnt_total += res_s.get("prompt_eval_count", 0)
    s3_e_cnt_total += res_s.get("eval_count", 0)

lines.append("2. RELEVANCE GRADING COMPARISON FOR K=3 CHUNKS (800 chars each):")
lines.append("   A. Batched Call (1 call with 3 chunks):")
lines.append(f"      - Wall Time:       {wall_b3_ms:8.1f} ms")
lines.append(f"      - Prompt Tokens:   {b3_p_cnt:8d}")
lines.append(f"      - Eval Tokens:     {b3_e_cnt:8d}")
lines.append(f"      - Response:        {res_b3.get('message', {}).get('content', '')}")
lines.append("   B. Separate Single-Chunk Calls (3 individual calls):")
lines.append(f"      - Total Wall Time: {wall_s3_total_ms:8.1f} ms")
lines.append(f"      - Total Prompt Tks:{s3_p_cnt_total:8d}")
lines.append(f"      - Total Eval Tks:  {s3_e_cnt_total:8d}")
lines.append("   C. Speedup / Efficiency:")
lines.append(f"      - Wall Time Ratio (Single / Batched): {wall_s3_total_ms / max(wall_b3_ms, 1e-9):.2f}x")
lines.append(f"      - Time Saved via Batching:            {wall_s3_total_ms - wall_b3_ms:.1f} ms")
lines.append("")

# 3. Benchmark K=5
print("Benchmarking K=5 (batched vs 5 single calls)...")
chunks_5 = chunks_pool[:5]

# K=5 Batched Call
batch_prompt_5 = f"Question: {TEST_QUESTION}\n\nPassages:\n" + "\n\n".join(
    [f"[Passage {i+1}]: {c}" for i, c in enumerate(chunks_5)]
)
batch_msgs_5 = [
    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
    {"role": "user", "content": batch_prompt_5},
]
res_b5, wall_b5_ms = call_ollama(batch_msgs_5, schema=BATCH_SCHEMA)
b5_p_cnt = res_b5.get("prompt_eval_count", 0)
b5_e_cnt = res_b5.get("eval_count", 0)

# K=5 Single Calls
wall_s5_total_ms = 0.0
s5_p_cnt_total = 0
s5_e_cnt_total = 0
for i, c in enumerate(chunks_5):
    single_prompt = f"Question: {TEST_QUESTION}\n\n[Passage {i+1}]: {c}"
    s_msgs = [
        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": single_prompt},
    ]
    res_s, wall_s_ms = call_ollama(s_msgs, schema=SINGLE_SCHEMA)
    wall_s5_total_ms += wall_s_ms
    s5_p_cnt_total += res_s.get("prompt_eval_count", 0)
    s5_e_cnt_total += res_s.get("eval_count", 0)

lines.append("3. RELEVANCE GRADING COMPARISON FOR K=5 CHUNKS (800 chars each):")
lines.append("   A. Batched Call (1 call with 5 chunks):")
lines.append(f"      - Wall Time:       {wall_b5_ms:8.1f} ms")
lines.append(f"      - Prompt Tokens:   {b5_p_cnt:8d}")
lines.append(f"      - Eval Tokens:     {b5_e_cnt:8d}")
lines.append(f"      - Response:        {res_b5.get('message', {}).get('content', '')}")
lines.append("   B. Separate Single-Chunk Calls (5 individual calls):")
lines.append(f"      - Total Wall Time: {wall_s5_total_ms:8.1f} ms")
lines.append(f"      - Total Prompt Tks:{s5_p_cnt_total:8d}")
lines.append(f"      - Total Eval Tks:  {s5_e_cnt_total:8d}")
lines.append("   C. Speedup / Efficiency:")
lines.append(f"      - Wall Time Ratio (Single / Batched): {wall_s5_total_ms / max(wall_b5_ms, 1e-9):.2f}x")
lines.append(f"      - Time Saved via Batching:            {wall_s5_total_ms - wall_b5_ms:.1f} ms")
lines.append("")

# 4. System RAM Measurement with Ollama, bge-small, and InLegalBERT all loaded
print("Measuring system RAM with all models loaded...")
print("Loading InLegalBERT on CPU...")
inlegalbert_model = AutoModelForSequenceClassification.from_pretrained(cfg.BEST_MODEL_PATH)
inlegalbert_model.eval()

print("Loading bge-small on CPU...")
bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
bge_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
bge_model.eval()

# Check Ollama still loaded in memory
call_ollama([{"role": "user", "content": "ping"}])

mem = psutil.virtual_memory()
total_gb = mem.total / (1024**3)
avail_gb = mem.available / (1024**3)
used_gb = mem.used / (1024**3)
percent_used = mem.percent

lines.append("4. SYSTEM RAM CONCURRENCY BENCHMARK:")
lines.append("   (Ollama llama3.2:3b in RAM + BAAI/bge-small-en-v1.5 + InLegalBERT fine-tuned)")
lines.append(f"   - Total System RAM:     {total_gb:5.2f} GB")
lines.append(f"   - Available System RAM: {avail_gb:5.2f} GB")
lines.append(f"   - Used System RAM:      {used_gb:5.2f} GB")
lines.append(f"   - System RAM Used:      {percent_used:5.1f}%")
lines.append("")
lines.append("=" * 80)

# Write output to file
out_text = "\n".join(lines)
LOG_FILE.write_text(out_text, encoding="utf-8")
print(f"Benchmark finished. Results saved to {LOG_FILE} ({len(out_text)} bytes)")
print(out_text)
