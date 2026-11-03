import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Calling gemma-4-31b-it...", flush=True)
try:
    resp = client.models.generate_content(
        model="gemma-4-31b-it",
        contents="Say hello in JSON: {\"hello\": \"world\"}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    print("SUCCESS:", resp.text, flush=True)
except Exception as e:
    print("EXCEPTION TYPE:", type(e), flush=True)
    print("EXCEPTION STR:", str(e), flush=True)
    import traceback
    traceback.print_exc()
