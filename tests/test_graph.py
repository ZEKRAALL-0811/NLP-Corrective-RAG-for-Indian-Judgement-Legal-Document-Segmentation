"""
Unit tests for Module 6: LangGraph State & Retrieval Workflow (src/rag/graph.py).
Mocks the role mapper and vector store to verify graph transitions, routing,
trace generation, and auto-escalation.
"""

from unittest.mock import MagicMock, patch

from src.rag.graph import (
    JurisGraphState,
    map_role_node,
    route_after_role_mapping,
    route_from_start,
    retrieve_node,
    auto_escalate_retrieval,
    run_query,
)
from src.rag.role_mapper import RoleMapping


def test_out_of_scope_routes_to_end_with_canned_message():
    """Verify out_of_scope mapping generates canned message and routes to END."""
    mock_mapping = RoleMapping(
        roles=[],
        confidence=0.99,
        kind="out_of_scope",
        reason="Weather query",
    )

    with patch("src.rag.graph.map_query_to_roles", return_value=mock_mapping):
        initial_state: JurisGraphState = {
            "question": "What is the weather in Delhi?",
            "document_id": "sample2",
            "mode": "role",
        }
        res = map_role_node(initial_state)
        assert res["role_kind"] == "out_of_scope"
        assert res["eval_status"] == "out_of_scope"
        assert "outside the scope of legal document analysis" in res["generation"]
        assert len(res["trace"]) == 1
        assert res["trace"][0]["node"] == "map_role_node"

        # Verify routing
        updated_state = {**initial_state, **res}
        next_step = route_after_role_mapping(updated_state)
        assert next_step == "__end__"  # LangGraph END constant


def test_whole_document_sets_relax_3_and_routes_to_retrieve():
    """Verify whole_document mapping sets relax_level=3 and routes to retrieve_node."""
    mock_mapping = RoleMapping(
        roles=[],
        confidence=0.95,
        kind="whole_document",
        reason="Summary query",
    )

    with patch("src.rag.graph.map_query_to_roles", return_value=mock_mapping):
        initial_state: JurisGraphState = {
            "question": "Can you summarize the entire judgment?",
            "document_id": "sample2",
            "mode": "role",
        }
        res = map_role_node(initial_state)
        assert res["role_kind"] == "whole_document"
        assert res["relax_level"] == 3
        assert res["mapped_roles"] == []

        updated_state = {**initial_state, **res}
        next_step = route_after_role_mapping(updated_state)
        assert next_step == "retrieve_node"


def test_naive_mode_skips_mapper_and_queries_naive():
    """Verify naive mode routes directly to retrieve_node and queries naive collection."""
    state: JurisGraphState = {
        "question": "What did the court say about limitation?",
        "document_id": "sample2",
        "mode": "naive",
    }
    assert route_from_start(state) == "retrieve_node"

    mock_store = MagicMock()
    mock_store.retrieve.return_value = [
        {"text": "Chunk 1", "metadata": {"role": "Facts"}, "distance": 0.2}
    ]

    out = retrieve_node(state, store=mock_store)
    mock_store.retrieve.assert_called_once_with(
        question="What did the court say about limitation?",
        document_id="sample2",
        mode="naive",
        top_k=5,
    )
    assert len(out["retrieved_chunks"]) == 1
    assert len(out["trace"]) == 1
    assert out["trace"][0]["node"] == "retrieve_node"
    assert out["trace"][0]["details"]["mode"] == "naive"


def test_decision_role_defaults_rel_pos_range():
    """Verify queries with Decision role default rel_pos_range to (0.85, 1.0)."""
    state: JurisGraphState = {
        "question": "What was the final operative order?",
        "document_id": "sample3",
        "mode": "role",
        "mapped_roles": ["Decision"],
        "relax_level": 0,
    }

    mock_store = MagicMock()
    mock_store.retrieve.return_value = [{"text": "Operative order", "metadata": {}, "distance": 0.1}]

    out = retrieve_node(state, store=mock_store)
    mock_store.retrieve.assert_called_once_with(
        question="What was the final operative order?",
        document_id="sample3",
        mode="role",
        roles=["Decision"],
        relax=0,
        rel_pos_range=(0.85, 1.0),
        top_k=5,
    )
    assert out["trace"][0]["details"]["rel_pos_range"] == (0.85, 1.0)


def test_auto_escalation_helper():
    """Verify auto_escalate_retrieval steps through relax levels until >= min_chunks."""
    mock_store = MagicMock()
    # At relax 0: 1 chunk (too few)
    # At relax 1: 3 chunks (satisfies min_chunks=3)
    mock_store.retrieve.side_effect = [
        [{"text": "Chunk 1"}],
        [{"text": "Chunk 1"}, {"text": "Chunk 2"}, {"text": "Chunk 3"}],
    ]

    chunks, final_relax = auto_escalate_retrieval(
        question="What did the respondent argue?",
        document_id="sample2",
        store=mock_store,
        mapped_roles=["Arguments of Respondent"],
        min_chunks=3,
    )

    assert len(chunks) == 3
    assert final_relax == 1
    assert mock_store.retrieve.call_count == 2


def test_end_to_end_run_query_mocked():
    """Verify run_query executes through the LangGraph StateGraph pipeline."""
    mock_mapping = RoleMapping(
        roles=["Facts"],
        confidence=0.95,
        kind="role",
        reason="Factual question",
    )

    mock_store = MagicMock()
    mock_store.retrieve.return_value = [
        {"text": "Fact sentence", "metadata": {"role": "Facts"}, "distance": 0.15}
    ]

    with patch("src.rag.graph.map_query_to_roles", return_value=mock_mapping):
        final_state = run_query(
            question="When did the agreement take place?",
            document_id="sample2",
            mode="role",
            store=mock_store,
        )

        assert final_state["mapped_roles"] == ["Facts"]
        assert len(final_state["retrieved_chunks"]) == 1
        node_names = [t["node"] for t in final_state["trace"]]
        assert node_names[:2] == ["map_role_node", "retrieve_node"]
        assert all("duration_ms" in t for t in final_state["trace"])
