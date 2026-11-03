"""
Unit and integration tests for Module 5: Query Rhetorical Role Mapper.

Includes comprehensive mocked tests for all rule cues, Ollama client behavior,
error handling, and fallback resilience, plus a real-call smoke test that skips
if the local Ollama server is unavailable.
"""

import json
from unittest.mock import patch, MagicMock
import urllib.error
import pytest

from src.rag.role_mapper import (
    map_query_to_roles,
    check_ollama,
    _call_ollama_llm,
    RoleMapping,
)
from src.config import CANONICAL_LABELS


# ==============================================================================
# 1. Health Check Tests (Mocked)
# ==============================================================================
def test_check_ollama_healthy():
    """Validates that check_ollama returns True when server and model are ready."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert check_ollama(host="http://localhost:11434", model="llama3.2:3b") is True


def test_check_ollama_unreachable():
    """Validates that check_ollama raises clear error when server is down."""
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(RuntimeError) as exc_info:
            check_ollama(host="http://localhost:11434", model="llama3.2:3b")
        assert "Ollama server is unreachable" in str(exc_info.value)


def test_check_ollama_missing_model():
    """Validates that check_ollama raises clear error when model is not pulled."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"models": [{"name": "mistral:7b"}]}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            check_ollama(host="http://localhost:11434", model="llama3.2:3b")
        assert "not installed in Ollama" in str(exc_info.value)


# ==============================================================================
# 2. Rule Layer Unit Tests
# ==============================================================================
@pytest.mark.parametrize(
    "question,expected_role",
    [
        ("What did the appellant contend regarding the mortgage?", "Arguments of Petitioner"),
        ("What did the petitioner submit in the writ petition?", "Arguments of Petitioner"),
        ("How did the respondent defend the impugned order?", "Arguments of Respondent"),
        ("What counter-submission was made by the revenue?", "Arguments of Respondent"),
        ("What was the final order and operative relief granted?", "Decision"),
        ("Were the original side appeals dismissed or allowed?", "Decision"),
        ("What was the central question of law framed by the bench?", "Issue"),
        ("What point for consideration was referred to the larger bench?", "Issue"),
        ("What was the factual background of the transaction in 1987?", "Facts"),
        ("What happened before the trial court prior to the appeal?", "Facts"),
        ("Why did the court hold that the search authorization was void?", "Reasoning"),
        ("On what reasoning did the bench distinguish the cited precedent?", "Reasoning"),
        ("Which judges decided the case and on what date was it delivered?", "None"),
        ("What was the neutral citation and case number of the appeal?", "None"),
    ],
)
def test_unambiguous_role_cues(question, expected_role):
    """Verifies that unambiguous rule cues map directly to the expected role."""
    mapping = map_query_to_roles(question, mode="rules_only")
    assert mapping.kind in {"role", "multi_role"}
    assert mapping.roles[0] == expected_role
    assert mapping.confidence >= 0.90


def test_out_of_scope_cues():
    """Verifies small-talk, weather, and general questions are caught as out_of_scope."""
    queries = [
        "What is the weather in Chennai today?",
        "Can you recommend a good biryani restaurant nearby?",
        "Write a Python script to reverse a string.",
        "Tell me a funny joke to brighten my morning.",
    ]
    for q in queries:
        m = map_query_to_roles(q, mode="rules_only")
        assert m.kind == "out_of_scope"
        assert m.roles == []


def test_whole_document_cues():
    """Verifies summary and overview questions map to whole_document with empty roles."""
    queries = [
        "Please provide a comprehensive summary of this entire judgment.",
        "Give me an executive brief and overview of this case.",
        "Explain the gist and overall outcome of the dispute.",
    ]
    for q in queries:
        m = map_query_to_roles(q, mode="rules_only")
        assert m.kind == "whole_document"
        assert m.roles == []


# ==============================================================================
# 3. LLM Parsing & Resilience Tests (Mocked)
# ==============================================================================
def test_mocked_llm_success():
    """Verifies structured parsing of LLM response."""
    payload = {
        "message": {
            "content": json.dumps({
                "roles": ["Facts"],
                "confidence": 0.96,
                "kind": "role",
                "reason": "Asks for case chronology",
            })
        }
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        m = _call_ollama_llm("When was the agreement entered into?")
        assert m.kind == "role"
        assert m.roles == ["Facts"]
        assert m.confidence == 0.96


def test_mocked_llm_fallback_on_network_error():
    """Verifies fallback to whole_document when Ollama call errors out."""
    with patch("urllib.request.urlopen", side_effect=Exception("Timeout")):
        m = map_query_to_roles("Complex legal question", mode="llm_only", debug=True)
        assert m.kind == "whole_document"
        assert m.roles == []
        assert "Fallback" in m.reason

        # Default debug=False has empty reason
        m_nodebug = map_query_to_roles("Complex legal question", mode="llm_only", debug=False)
        assert m_nodebug.kind == "whole_document"
        assert m_nodebug.reason == ""


def test_canonical_validation_normalization():
    """Verifies that non-canonical or lowercase roles are normalized."""
    payload = {
        "message": {
            "content": json.dumps({
                "roles": ["arguments of petitioner", "INVALID_ROLE"],
                "confidence": 0.92,
                "kind": "multi_role",
                "reason": "Test role normalization",
            })
        }
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        m = _call_ollama_llm("Query", debug=True)
        assert m.roles == ["Arguments of Petitioner"]
        assert m.roles[0] in CANONICAL_LABELS
        assert len(m.reason) > 0


# ==============================================================================
# 4. Real-Call Smoke Test (Live Ollama or Skip)
# ==============================================================================
def test_real_ollama_live_smoke():
    """Real smoke test executing live against Ollama (skips if server is offline)."""
    try:
        check_ollama()
    except Exception as e:
        pytest.skip(f"Ollama server is unavailable: {e}")

    question = "What relief did the High Court finally grant in the writ petitions?"
    # Test debug=False: reason is empty
    m_default = map_query_to_roles(question, mode="hybrid", debug=False)
    assert isinstance(m_default, RoleMapping)
    assert m_default.kind in {"role", "multi_role", "whole_document", "out_of_scope"}
    assert isinstance(m_default.roles, list)
    assert 0.0 <= m_default.confidence <= 1.0
    assert m_default.reason == ""

    # Test debug=True: reason is populated
    m_debug = map_query_to_roles(question, mode="hybrid", debug=True)
    assert isinstance(m_debug, RoleMapping)
    assert len(m_debug.reason) > 0
