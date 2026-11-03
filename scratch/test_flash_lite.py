import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Calling gemini-3.1-flash-lite...", flush=True)
t0 = time.time()
try:
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="Say hello in JSON: {\"hello\": \"world\"}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    dt = time.time() - t0
    print(f"SUCCESS in {dt:.2f}s: {resp.text}", flush=True)
except Exception as e:
    dt = time.time() - t0
    print(f"FAILED in {dt:.2f}s: {e}", flush=True)
