import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

print("Loading .env...", flush=True)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY missing", flush=True)
    sys.exit(1)

print("Initializing genai.Client...", flush=True)
from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)
log_lines = []

# Test 1: system_instruction with gemma-4-31b-it
print("Running Test 1: system_instruction...", flush=True)
sys_ok = False
try:
    resp = client.models.generate_content(
        model="gemma-4-31b-it",
        contents='Return {"status": "ok"}',
        config=types.GenerateContentConfig(
            system_instruction="You are a strict JSON evaluator.",
            temperature=0.0
        )
    )
    sys_ok = True
    txt = (resp.text or "").strip().replace("\n", " ")
    log_lines.append(f"system_instruction support: SUCCESS (Output: {txt[:60]})")
    print(f"Test 1 SUCCESS: {txt[:60]}", flush=True)
except Exception as e:
    log_lines.append(f"system_instruction support: FAILED ({e})")
    print(f"Test 1 FAILED: {e}", flush=True)

# Test 2: response_mime_type='application/json' with gemma-4-31b-it
print("Running Test 2: response_mime_type=application/json...", flush=True)
json_mime_ok = False
try:
    resp2 = client.models.generate_content(
        model="gemma-4-31b-it",
        contents="Return a JSON object with key test and value 123",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    json_mime_ok = True
    txt2 = (resp2.text or "").strip().replace("\n", " ")
    log_lines.append(f"response_mime_type=application/json support: SUCCESS (Output: {txt2[:60]})")
    print(f"Test 2 SUCCESS: {txt2[:60]}", flush=True)
except Exception as e:
    log_lines.append(f"response_mime_type=application/json support: FAILED ({e})")
    print(f"Test 2 FAILED: {e}", flush=True)

mode = "API_NATIVE_JSON" if (json_mime_ok and sys_ok) else "PROMPT_INSTRUCTED_JSON_WITH_REPAIR"
log_lines.append(f"DETERMINED_MODE: {mode}")

out_path = Path("artifacts/logs/judge_json_mode.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(log_lines), encoding="utf-8")
print(f"DETERMINED_MODE: {mode}", flush=True)
