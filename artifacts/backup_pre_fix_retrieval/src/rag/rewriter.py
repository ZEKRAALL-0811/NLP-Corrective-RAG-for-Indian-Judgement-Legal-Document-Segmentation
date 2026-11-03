"""
Module 8: Query Rewriter (src/rag/rewriter.py).
Generates a differently angled legal question using legal synonyms or the underlying principle
when retrieved chunks are graded incorrect.
Uses Ollama with temperature 0, seed 42, keep_alive '30m', and a JSON schema {"q": string}.
"""

from typing import Tuple, Dict, Any, Optional
import json
import time
import urllib.request
import urllib.error

import src.config as cfg


REWRITE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "q": {
            "type": "string",
            "description": "Rewritten query under 40 tokens with legal synonyms or principles.",
        }
    },
    "required": ["q"],
    "additionalProperties": False,
}

REWRITE_SYSTEM_PROMPT = (
    "You are an expert legal search assistant. "
    "When a legal query fails to retrieve relevant passages, rephrase the query from a different angle "
    "using formal legal terminology, statutory concepts, or core legal principles. "
    "Keep the rewritten query concise (under 40 tokens). "
    "Output must be valid JSON conforming to the schema with key 'q'."
)


def call_rewriter_ollama(
    query: str,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[Dict[str, Any], float]:
    """
    Executes a structured Ollama call for query rewriting.
    """
    ollama_host = (host or cfg.OLLAMA_HOST).rstrip("/")
    model_name = model or cfg.OLLAMA_MODEL
    url = f"{ollama_host}/api/chat"

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Original query to rewrite:\n{query}"},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "format": REWRITE_SCHEMA,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 64,
            "num_ctx": 2048,
        },
        "keep_alive": "30m",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            wall_ms = (time.time() - t0) * 1000.0
            return data, wall_ms
    except Exception as exc:
        wall_ms = (time.time() - t0) * 1000.0
        return {"error": str(exc)}, wall_ms


def rewrite_query(
    query: str,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[str, float]:
    """
    Rewrites a query into a differently angled question with legal synonyms or principles.

    Returns:
        rewritten_query: String rephrased query (or original query if call fails).
        wall_ms: Call wall-clock latency in milliseconds.
    """
    data, wall_ms = call_rewriter_ollama(
        query=query,
        host=host,
        model=model,
        timeout=timeout,
    )

    if "error" in data:
        return query, wall_ms

    try:
        content_str = data.get("message", {}).get("content", "{}")
        parsed = json.loads(content_str)
        rewritten = parsed.get("q", "").strip()
        if rewritten and len(rewritten) > 5:
            return rewritten, wall_ms
    except Exception:
        pass

    return query, wall_ms
