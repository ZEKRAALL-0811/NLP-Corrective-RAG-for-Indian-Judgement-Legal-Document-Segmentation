import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

print("=" * 60)
print("SAFE RUNTIME ENVIRONMENT DIAGNOSIS FOR TAVILY KEY")
print("=" * 60)

cwd = Path.cwd()
print(f"Working Directory: {cwd}")

# 1. Check BEFORE loading dotenv in fresh process
pre_env_val = os.environ.get("TAVILY_API_KEY")
print(f"OS Environment (pre-dotenv) TAVILY_API_KEY present: {'yes' if pre_env_val is not None else 'no'}")
if pre_env_val is not None:
    print(f"OS Environment (pre-dotenv) TAVILY_API_KEY empty: {'yes' if len(pre_env_val.strip()) == 0 else 'no'}")

# 2. Check dotenv loading
found_file = find_dotenv()
print(f"find_dotenv() resolved path: {found_file}")
loaded = load_dotenv(dotenv_path=Path("D:/NLP Project/.env"))
print(f"dotenv load attempted: yes")
print(f"dotenv load succeeded: {'yes' if loaded else 'no'}")

post_env_val = os.environ.get("TAVILY_API_KEY")
print(f"Resolved TAVILY_API_KEY present: {'yes' if post_env_val is not None else 'no'}")
if post_env_val is not None:
    print(f"Resolved TAVILY_API_KEY empty: {'yes' if len(post_env_val.strip()) == 0 else 'no'}")
    print(f"Resolved source: {'.env' if loaded else 'environment'}")

# 3. Check what src.rag.web_search sees
sys.path.insert(0, ".")
import src.rag.web_search as ws
# Test the key resolution logic in search_tavily without making any HTTP request
# search_tavily line 41: key = api_key or os.getenv("TAVILY_API_KEY")
resolved_in_module = os.getenv("TAVILY_API_KEY")
print(f"Module (src.rag.web_search) TAVILY_API_KEY present: {'yes' if resolved_in_module is not None else 'no'}")
if resolved_in_module is not None:
    print(f"Module (src.rag.web_search) TAVILY_API_KEY empty: {'yes' if len(resolved_in_module.strip()) == 0 else 'no'}")
    print(f"ENABLE_WEB_FALLBACK enabled: {ws.ENABLE_WEB_FALLBACK}")

# 4. Check whether app.py loads dotenv
app_text = Path("d:/NLP Project/app.py").read_text(encoding="utf-8")
print(f"app.py calls load_dotenv directly: {'yes' if 'load_dotenv' in app_text else 'no'}")

# 5. Check other potential variable names
for alt_name in ["CRAG_TAVILY_API_KEY", "TAVILY_KEY", "TAVILY_SEARCH_API_KEY"]:
    val = os.environ.get(alt_name)
    if val is not None:
        print(f"Alternative variable '{alt_name}' is also present in environment.")

print("=" * 60)
