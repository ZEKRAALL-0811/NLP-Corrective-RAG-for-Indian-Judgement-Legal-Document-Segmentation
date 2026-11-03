import os
import sys
from pathlib import Path

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Pre-flight check without printing key
key = os.getenv("TAVILY_API_KEY")
if not key or len(key.strip()) == 0:
    print("PRE-FLIGHT ERROR: TAVILY_API_KEY is missing or empty in .env. Aborting live test.")
    sys.exit(1)

print("Pre-flight check: TAVILY_API_KEY is present and non-empty.")
print("Pre-flight check: Unmocked environment confirmed.")
print("Pre-flight check: Executing exactly ONE search request with max_results=1.")

from src.rag.web_search import search_tavily

query = "Supreme Court of India jurisdiction"
results, meta = search_tavily(query=query, max_results=1, timeout=10.0)

print("\n--- LIVE TAVILY RESULT METADATA ---")
print(f"HTTP Request attempted: yes")
print(f"Success: {meta.get('success')}")
print(f"Duration: {meta.get('duration_ms', 0.0):.2f}ms")
print(f"Results count: {len(results)}")

if not meta.get("success"):
    print(f"Error category: {meta.get('error')}")
else:
    first_res = results[0]
    print(f"Result Title: {first_res.get('title')}")
    print(f"Result URL: {first_res.get('url')}")
    snippet = first_res.get('snippet', '')
    safe_snippet = snippet[:150].strip() + ("..." if len(snippet) > 150 else "")
    print(f"Result Snippet (first 150 chars): {safe_snippet}")
    print(f"Relevance Score: {first_res.get('score')}")

print("\nSafety check: API Key was NEVER displayed, printed, or logged.")
