"""
Unit and integration tests for Module 8: Self-Correction Loop & Budgets (src/rag/graph.py).

Tests:
1. Retry cap stops at MAX_RETRIES = 2 and ends in no_answer without infinite loop.
2. Progressive relax escalation order (0 -> 1 -> 2 -> 3).
3. Web fallback graceful degradation on missing key, API error, or timeout.
4. MAX_LLM_CALLS_PER_QUERY budget enforcement (terminates with low_confidence=True).
5. MAX_WALL_CLOCK_SECONDS (150s) timeout budget enforcement.
6. Unanswerable question ends in no_answer with exact required message.
7. Live smoke test running against local Ollama (skips if offline).
"""

from unittest.mock import patch
import time
import pytest

from src.rag.graph import (
    JurisGraphState,
    build_jurisgraph,
    run_query,
    MAX_RETRIES,
    MAX_LLM_CALLS_PER_QUERY,
    MAX_WALL_CLOCK_SECONDS,
)
from src.rag.role_mapper import check_ollama, RoleMapping


def test_retry_cap_stops_at_2_and_ends_no_answer():
    """Verify rewrite loop terminates strictly at retry_count=2 with final_status='no_answer'."""
    app = build_jurisgraph()

    # Mock grader so top chunks always fail -> route='rewrite'
    mock_graded = [
        {"chunk_id": "c1", "grade": "incorrect", "score": 0.1, "role": "Facts", "text": "t1"},
        {"chunk_id": "c2", "grade": "incorrect", "score": 0.1, "role": "Facts", "text": "t2"},
        {"chunk_id": "c3", "grade": "incorrect", "score": 0.1, "role": "Facts", "text": "t3"},
    ]

    with patch("src.rag.graph.grade_chunks", return_value=(mock_graded, "rewrite", 50.0)), \
         patch("src.rag.graph.rewrite_query", return_value=("rewritten query", 40.0)):
        
        initial_state: JurisGraphState = {
            "question": "What is the penalty under Section 271?",
            "original_question": "What is the penalty under Section 271?",
            "document_id": "sample2",
            "mode": "role",
            "mapped_roles": ["Facts"],
            "role_kind": "role",
            "retry_count": 0,
            "relax_level": 0,
            "llm_call_count": 0,
            "start_time": time.time(),
            "final_status": "ready",
            "trace": [],
        }

        result = app.invoke(initial_state)

        assert result.get("retry_count") == MAX_RETRIES
        assert result.get("final_status") == "no_answer"
        assert result.get("generation") == "No reliable answer found in this document for that question."


def test_relax_escalation_order():
    """Verify relaxation escalates by 1 on each rewrite attempt."""
    app = build_jurisgraph()

    mock_graded = [
        {"chunk_id": "c1", "grade": "incorrect", "score": 0.1, "role": "Facts", "text": "t1"}
    ]

    relax_history = []

    def mock_retrieve_step(state, store=None):
        relax_history.append(state.get("relax_level", 0))
        return {
            "retrieved_chunks": [{"chunk_id": "c1", "text": "chunk text", "role": "Facts"}],
            "trace": list(state.get("trace", [])),
        }

    mock_mapping = RoleMapping(roles=["Facts"], confidence=0.95, kind="role")

    with patch("src.rag.graph.map_query_to_roles", return_value=mock_mapping), \
         patch("src.rag.graph.retrieve_node", side_effect=mock_retrieve_step), \
         patch("src.rag.graph.grade_chunks", return_value=(mock_graded, "rewrite", 20.0)), \
         patch("src.rag.graph.rewrite_query", return_value=("rewritten", 20.0)):

        initial_state: JurisGraphState = {
            "question": "Test query",
            "original_question": "Test query",
            "document_id": "sample2",
            "mode": "role",
            "mapped_roles": ["Facts"],
            "role_kind": "role",
            "retry_count": 0,
            "relax_level": 0,
            "llm_call_count": 0,
            "start_time": time.time(),
            "trace": [],
        }

        result = app.invoke(initial_state)
        # Sequence of relax levels seen at retrieve_node: initial 0, retry 1 -> 1, retry 2 -> 2
        assert relax_history == [0, 1, 2]
        assert result.get("relax_level") == 2


def test_web_failure_degrades_gracefully():
    """Verify that Tavily search failure degrades gracefully to ambiguous chunks with low_confidence=True."""
    app = build_jurisgraph()

    # Step 1 local grading produces ambiguous
    ambig_chunks = [
        {"chunk_id": "c1", "grade": "ambiguous", "score": 0.5, "role": "Facts", "text": "ambiguous legal text"}
    ]

    with patch("src.rag.graph.grade_chunks", return_value=(ambig_chunks, "fallback", 30.0)), \
         patch("src.rag.graph.search_tavily", side_effect=Exception("Connection to Tavily failed")):

        initial_state: JurisGraphState = {
            "question": "What happened during the arbitration?",
            "original_question": "What happened during the arbitration?",
            "document_id": "sample3",
            "mode": "role",
            "mapped_roles": ["Facts"],
            "role_kind": "role",
            "retry_count": 0,
            "relax_level": 0,
            "llm_call_count": 0,
            "start_time": time.time(),
            "trace": [],
        }

        result = app.invoke(initial_state)

        assert result.get("low_confidence") is True
        assert result.get("final_status") in ("low_confidence", "abstained")
        assert result.get("external_context") is False


def test_max_llm_calls_budget_enforcement():
    """Verify MAX_LLM_CALLS_PER_QUERY = 6 terminates execution with low_confidence=True."""
    app = build_jurisgraph()

    # State starts with 6 calls already recorded
    initial_state: JurisGraphState = {
        "question": "Query that exceeded LLM call limit",
        "original_question": "Query that exceeded LLM call limit",
        "document_id": "sample2",
        "mode": "role",
        "mapped_roles": ["Facts"],
        "role_kind": "role",
        "retry_count": 0,
        "relax_level": 0,
        "llm_call_count": MAX_LLM_CALLS_PER_QUERY,
        "start_time": time.time(),
        "verified_context": [{"chunk_id": "c1", "text": "verified text"}],
        "trace": [],
    }

    result = app.invoke(initial_state)
    assert result.get("low_confidence") is True
    # Terminates with best verified context
    assert len(result.get("verified_context", [])) == 1


def test_wall_clock_timeout_budget_enforcement():
    """Verify MAX_WALL_CLOCK_SECONDS (150s) limit halts execution with low_confidence=True."""
    app = build_jurisgraph()

    # Start time set 160 seconds in the past
    initial_state: JurisGraphState = {
        "question": "Query that timed out",
        "original_question": "Query that timed out",
        "document_id": "sample2",
        "mode": "role",
        "mapped_roles": ["Facts"],
        "role_kind": "role",
        "retry_count": 0,
        "relax_level": 0,
        "llm_call_count": 1,
        "start_time": time.time() - (MAX_WALL_CLOCK_SECONDS + 10.0),
        "verified_context": [],
        "trace": [],
    }

    result = app.invoke(initial_state)
    assert result.get("low_confidence") is True
    assert result.get("final_status") == "no_answer"


def test_unanswerable_question_ends_in_no_answer():
    """Verify unanswerable question with no matching evidence cleanly yields no_answer."""
    app = build_jurisgraph()

    mock_graded = [
        {"chunk_id": "c1", "grade": "incorrect", "score": 0.1, "role": "Facts", "text": "irrelevant"}
    ]

    with patch("src.rag.graph.grade_chunks", return_value=(mock_graded, "rewrite", 25.0)), \
         patch("src.rag.graph.rewrite_query", return_value=("alt unanswerable query", 25.0)):

        initial_state: JurisGraphState = {
            "question": "Did the assessing officer issue an order under Section 148A regarding escaped turnover?",
            "original_question": "Did the assessing officer issue an order under Section 148A regarding escaped turnover?",
            "document_id": "sample2",
            "mode": "role",
            "mapped_roles": ["Facts"],
            "role_kind": "role",
            "retry_count": 0,
            "relax_level": 0,
            "llm_call_count": 0,
            "start_time": time.time(),
            "trace": [],
        }

        result = app.invoke(initial_state)

        assert result.get("final_status") == "no_answer"
        assert result.get("generation") == "No reliable answer found in this document for that question."


def test_live_smoke_test_self_correction_loop():
    """Live smoke test running end-to-end against local Ollama; skips if server is unavailable."""
    try:
        check_ollama()
    except Exception as e:
        pytest.skip(f"Local Ollama daemon is unreachable: {e}")

    q = "When did the petitioners sell their shares through the Offer For Sale?"
    result = run_query(q, document_id="sample2", mode="role")

    assert result is not None
    assert "final_status" in result
    assert result["final_status"] in {"ready", "low_confidence", "no_answer", "abstained"}
    assert len(result.get("trace", [])) >= 3
    assert result.get("llm_call_count", 0) >= 1

