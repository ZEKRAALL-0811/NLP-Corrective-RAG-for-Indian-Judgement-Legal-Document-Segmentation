import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.web_search import search_tavily
from src.rag.graph import fallback_node, JurisGraphState

print("=" * 60)
print("PHASE 3: TAVILY MISSING-CREDENTIAL FALLBACK")
print("=" * 60)

# 1. Test search_tavily when TAVILY_API_KEY is empty/missing
with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
    with patch("tavily.TavilyClient") as mock_client:
        results, meta = search_tavily("arbitration clause legal precedent", api_key="")
        print("search_tavily with missing key:")
        print(f"  Results returned: {results}")
        print(f"  Success flag: {meta.get('success')}")
        print(f"  Error message: '{meta.get('error')}'")
        print(f"  TavilyClient instantiated: {mock_client.called}")
        
        assert results == []
        assert meta["success"] is False
        assert "TAVILY_API_KEY missing" in meta["error"]
        assert mock_client.called is False, "TavilyClient was called despite missing key!"

# 2. Test fallback_node behavior at step 2 when key is missing
state: JurisGraphState = {
    "question": "What is the penalty under Section 271?",
    "document_id": "sample3",
    "fallback_step": 1,  # Step 1 was tried, now entering Step 2
    "graded_chunks": [
        {"chunk_id": "c_ambig_1", "grade": "ambiguous", "text": "Ambiguous passage on penalty."}
    ],
    "trace": [],
}

with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
    with patch("src.rag.graph.search_tavily") as mock_search:
        # If ENABLE_WEB_FALLBACK is True but search_tavily returns failure
        mock_search.return_value = ([], {"success": False, "error": "TAVILY_API_KEY missing"})
        
        next_state = fallback_node(state)
        print("\nfallback_node with missing credentials:")
        print(f"  Route: {next_state.get('route')}")
        print(f"  External context (web evidence claimed): {next_state.get('external_context')}")
        print(f"  Low confidence: {next_state.get('low_confidence')}")
        print(f"  Fallback context chunks: {len(next_state.get('fallback_context', []))}")
        
        assert next_state.get("external_context") is False
        assert next_state.get("low_confidence") is True
        assert next_state.get("route") == "generate"
        # Verify fallback context contains the retained ambiguous chunks, NOT fabricated web results
        assert next_state.get("fallback_context")[0]["chunk_id"] == "c_ambig_1"

print("\nPhase 3: ALL ASSERTIONS PASSED (Tavily HTTP bypassed, no web evidence claimed).")
