import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

key = os.getenv("TAVILY_API_KEY")
print(f"TAVILY_API_KEY present: {'yes' if key is not None else 'no'}")
print(f"TAVILY_API_KEY empty: {'yes' if not key or len(key.strip()) == 0 else 'no'}")

# Verify TavilyClient initialization (NO HTTP request)
try:
    from tavily import TavilyClient
    client = TavilyClient(api_key=key)
    print("TavilyClient instantiation: SUCCESS (valid object created)")
except Exception as e:
    print(f"TavilyClient instantiation FAILED: {type(e).__name__}")
