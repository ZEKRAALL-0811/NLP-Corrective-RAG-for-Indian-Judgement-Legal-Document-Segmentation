import os
import sys
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

env_path = project_root / ".env"
with open(env_path, "rb") as f:
    env_hash_before = hashlib.sha256(f.read()).hexdigest()

from src.rag.web_search import search_tavily
from src.rag.graph import fallback_node, JurisGraphState

print("=== Phase 2: Mocked Tavily Fallback & Integration Verification ===")

# Test 1: Missing TAVILY_API_KEY makes no external request
print("\n[Test 1] Missing TAVILY_API_KEY makes no external request...")
with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
    with patch("tavily.TavilyClient") as mock_client:
        results, meta = search_tavily("test query", api_key="")
        assert results == [], f"Expected empty results, got {results}"
        assert meta["success"] is False, "Expected success=False"
        assert "missing" in meta["error"].lower(), f"Unexpected error: {meta['error']}"
        assert not mock_client.called, "TavilyClient should NOT be called when key is missing"
print("PASS: Missing TAVILY_API_KEY makes no external request (TavilyClient never called).")

# Test 2: Successful search result is correctly represented as external context
print("\n[Test 2] Successful search result correctly represented as external context...")
mock_results = [{
    "title": "Landmark Ruling on Arbitration",
    "url": "https://sci.gov.in/arbitration",
    "snippet": "The Supreme Court held that arbitration clauses remain binding...",
    "score": 0.95
}]
mock_meta = {"success": True, "query_sent": "arbitration clause", "duration_ms": 120.0, "n_results": 1}

state_step2: JurisGraphState = {
    "question": "What is the rule on arbitration?",
    "document_id": "sample3",
    "fallback_step": 1,  # triggers Step 2
    "graded_chunks": [{"chunk_id": "c_ambig", "grade": "ambiguous", "text": "Ambiguous text"}],
    "trace": [],
}

with patch("src.rag.graph.search_tavily", return_value=(mock_results, mock_meta)) as mock_search:
    res = fallback_node(state_step2)
    assert res.get("external_context") is True, "external_context must be True for successful web search"
    assert res.get("fallback_context") == mock_results, "fallback_context must contain web results"
    assert res.get("route") == "generate", "route must be generate"
    assert res.get("fallback_step") == 2, "fallback_step must be 2"
    assert any(t.get("details", {}).get("strategy") == "web_fallback_tavily" for t in res.get("trace", []))
print("PASS: Successful search result sets external_context=True and populates fallback_context.")

# Test 3: Failed search degrades gracefully
print("\n[Test 3] Failed search degrades gracefully to ambiguous chunks...")
failed_meta = {"success": False, "error": "HTTP 500 Server Error", "duration_ms": 50.0}

with patch("src.rag.graph.search_tavily", return_value=([], failed_meta)) as mock_search:
    res_failed = fallback_node(state_step2)
    assert res_failed.get("external_context") is False, "external_context must be False on search failure"
    assert res_failed.get("low_confidence") is True, "low_confidence must be True on search failure"
    assert res_failed.get("route") == "generate", "route must be generate"
    assert res_failed.get("fallback_step") == 2
    # Confirms degraded context contains the ambiguous chunks from graded_chunks
    assert len(res_failed.get("fallback_context", [])) == 1
    assert res_failed.get("fallback_context")[0]["chunk_id"] == "c_ambig"
    assert any("graceful_degradation_ambiguous" in t.get("details", {}).get("strategy", "") for t in res_failed.get("trace", []))
print("PASS: Failed search degrades gracefully with low_confidence=True and retains ambiguous chunks.")

# Test 4: Graph does not claim external evidence when no results returned
print("\n[Test 4] Graph does not claim external evidence when 0 results returned...")
empty_meta = {"success": True, "n_results": 0, "duration_ms": 30.0}

with patch("src.rag.graph.search_tavily", return_value=([], empty_meta)) as mock_search:
    res_empty = fallback_node(state_step2)
    assert res_empty.get("external_context") is False, "external_context must be False when 0 results returned"
    assert res_empty.get("low_confidence") is True
    assert res_empty.get("fallback_context")[0]["chunk_id"] == "c_ambig"
print("PASS: When Tavily returns 0 results, external_context is False and no external evidence is claimed.")

# Test 5: Verify .env was not modified or removed
print("\n[Test 5] Verifying .env integrity...")
with open(env_path, "rb") as f:
    env_hash_after = hashlib.sha256(f.read()).hexdigest()
assert env_hash_before == env_hash_after, ".env file was altered during testing!"
print("PASS: .env integrity verified (SHA-256 match, file was not modified).")

print("\n=== ALL PHASE 2 TESTS PASSED SUCCESSFULLY ===")
