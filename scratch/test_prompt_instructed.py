import os
import sys
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

prompt = """Based on the given text, please extract a comprehensive list of FACTUAL, undisputed truths, that can inferred from the provided text.
Text:
Pramod Swarup Agrawal transferred 11,11,486 equity shares for a total consideration of Rs.33,00,00,000/- on 01.07.2021.

Respond ONLY with a valid JSON object matching:
{"claims": ["claim 1", "claim 2"]}
Do not include any extra text.
"""

print("Calling gemma-4-31b-it with prompt-instructed JSON (no response_mime_type)...", flush=True)
t0 = time.time()
try:
    resp = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    dt = time.time() - t0
    txt = (resp.text or "").strip()
    print(f"SUCCESS in {dt:.2f}s:\n{txt}", flush=True)
    # Validate JSON
    parsed = json.loads(txt.strip("` \n").replace("json\n", ""))
    print("Parsed JSON successfully:", parsed, flush=True)
except Exception as e:
    print("FAILED:", e, flush=True)
