"""
Module 8: Web Search Fallback (src/rag/web_search.py).
Integrates Tavily search as a secondary fallback when retrieved document chunks
contain only ambiguous passages and local retrieval relaxation yields no correct chunk.
Truncates snippets to 500 characters, respects 10s timeout, logs the exact query text,
and degrades gracefully when the key is missing or the search fails.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import time
from dotenv import load_dotenv
import src.config as cfg

load_dotenv()

ENABLE_WEB_FALLBACK: bool = getattr(cfg, "ENABLE_WEB_FALLBACK", True) and (
    os.getenv("ENABLE_WEB_FALLBACK", "True").lower() in ("true", "1", "yes")
)


def search_tavily(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 3,
    timeout: float = 10.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes a basic Tavily search for fallback context.

    Args:
        query: Search query string.
        api_key: Optional Tavily API key (defaults to os.getenv("TAVILY_API_KEY")).
        max_results: Number of results to return (default 3).
        timeout: Timeout in seconds (default 10s).

    Returns:
        results: List of result dicts with keys 'title', 'url', 'snippet', 'score'.
        meta: Metadata dict with 'success', 'query_sent', 'duration_ms', and optional 'error'.
    """
    key = api_key or os.getenv("TAVILY_API_KEY")
    query_text = query.strip()
    t0 = time.time()

    # Log query text sent to Tavily as required
    print(f"[Tavily Search] Query sent to Tavily: {query_text}")

    if not key:
        return [], {
            "success": False,
            "error": "TAVILY_API_KEY missing or not provided",
            "query_sent": query_text,
            "duration_ms": 0.0,
        }

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        # tavily client search
        raw_resp = client.search(
            query=query_text,
            search_depth="basic",
            max_results=max_results,
        )
        duration_ms = (time.time() - t0) * 1000.0

        raw_results = raw_resp.get("results", []) if isinstance(raw_resp, dict) else []
        formatted: List[Dict[str, Any]] = []
        for item in raw_results:
            raw_content = item.get("content", "")
            snippet = raw_content[:500] if raw_content else ""
            formatted.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "snippet": snippet,
                "score": item.get("score", 0.0),
            })

        return formatted, {
            "success": True,
            "query_sent": query_text,
            "duration_ms": duration_ms,
            "n_results": len(formatted),
        }

    except Exception as exc:
        duration_ms = (time.time() - t0) * 1000.0
        return [], {
            "success": False,
            "error": str(exc),
            "query_sent": query_text,
            "duration_ms": duration_ms,
        }
