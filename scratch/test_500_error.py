import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

prompt = """Based on the given text, please extract a comprehensive list of FACTUAL, undisputed truths, that can inferred from the provided text.
Example JSON:
{
    "claims": [
        "Einstein won the noble prize for his discovery of the photoelectric effect in 1968.",
        "The photoelectric effect is a phenomenon that laid the foundation for quantum mechanics."
    ]
}
Text:
Pramod Swarup Agrawal transferred 11,11,486 equity shares for a total consideration of Rs.33,00,00,000/- on 01.07.2021.
JSON:
"""

print("--- Test A: With system_instruction and response_mime_type=application/json ---", flush=True)
try:
    resp = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a strict JSON evaluator.",
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    print("Test A SUCCESS:", resp.text[:100], flush=True)
except Exception as e:
    print("Test A FAILED:", e, flush=True)

print("--- Test B: Without response_mime_type, instructing in prompt ---", flush=True)
try:
    resp = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt + "\nReturn ONLY valid JSON.",
        config=types.GenerateContentConfig(temperature=0.0)
    )
    print("Test B SUCCESS:", resp.text[:100], flush=True)
except Exception as e:
    print("Test B FAILED:", e, flush=True)

print("--- Test C: With gemini-3.1-flash-lite ---", flush=True)
try:
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    print("Test C SUCCESS:", resp.text[:100], flush=True)
except Exception as e:
    print("Test C FAILED:", e, flush=True)
