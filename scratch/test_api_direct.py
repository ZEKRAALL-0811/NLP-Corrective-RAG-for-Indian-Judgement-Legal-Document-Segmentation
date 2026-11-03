import os
import sys
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No key")
    sys.exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent?key={api_key}"

payload = {
    "contents": [{"parts": [{"text": "Hello, respond with: OK"}]}],
    "generationConfig": {"temperature": 0.0}
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    print("Sending request to gemma-4-31b-it via REST API...", flush=True)
    with urllib.request.urlopen(req, timeout=20) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Status:", resp.status)
        print("Response:", json.dumps(res)[:200])
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code, e.read().decode("utf-8"))
except Exception as e:
    print("Error:", e)
