"""
Unit tests for Module 7: Relevance Grader (src/rag/grader.py).
Tests schema generation, label-to-score mapping, routing logic,
mocked Ollama calls, grade_node state updates, distance baseline,
and a live Ollama smoke test.
"""

from unittest.mock import patch
import json
import pytest

from src.rag.grader import (
    make_grader_schema,
    determine_route,
    grade_chunks,
    grade_node,
    grade_chunks_by_distance,
)
from src.rag.graph import JurisGraphState
from src.rag.role_mapper import check_ollama


def test_make_grader_schema():
    """Verify JSON schema generation for K=3 and K=5."""
    schema3 = make_grader_schema(3)
    assert schema3["type"] == "object"
    assert len(schema3["properties"]) == 3
    assert schema3["required"] == ["c1", "c2", "c3"]
    assert schema3["properties"]["c1"]["enum"] == ["correct", "ambiguous", "incorrect"]

    schema5 = make_grader_schema(5)
    assert len(schema5["properties"]) == 5
    assert schema5["required"] == ["c1", "c2", "c3", "c4", "c5"]


def test_determine_route():
    """Verify routing decision logic."""
    assert determine_route(["correct", "incorrect", "incorrect"]) == "generate"
    assert determine_route(["incorrect", "correct", "ambiguous"]) == "generate"
    assert determine_route(["ambiguous", "incorrect", "incorrect"]) == "fallback"
    assert determine_route(["ambiguous", "ambiguous"]) == "fallback"
    assert determine_route(["incorrect", "incorrect", "incorrect"]) == "rewrite"
    assert determine_route([]) == "rewrite"


def test_grade_chunks_mocked():
    """Verify batched grading with mocked Ollama response."""
    mock_payload = {
        "message": {
            "content": json.dumps({
                "c1": "correct",
                "c2": "ambiguous",
                "c3": "incorrect",
            })
        }
    }
    chunks = [
        {"id": "doc_1", "text": "Passage 1 content", "metadata": {"role": "Facts"}, "distance": 0.25},
        {"id": "doc_2", "text": "Passage 2 content", "metadata": {"role": "Reasoning"}, "distance": 0.45},
        {"id": "doc_3", "text": "Passage 3 content", "metadata": {"role": "Decision"}, "distance": 0.85},
    ]

    with patch("src.rag.grader.call_grader_ollama", return_value=(mock_payload, 150.0)):
        graded, route, duration = grade_chunks("What was the outcome?", chunks)
        assert route == "generate"
        assert len(graded) == 3
        assert graded[0]["grade"] == "correct"
        assert graded[0]["score"] == 0.9
        assert graded[1]["grade"] == "ambiguous"
        assert graded[1]["score"] == 0.5
        assert graded[2]["grade"] == "incorrect"
        assert graded[2]["score"] == 0.1
        assert duration == 150.0


def test_grade_node_mocked():
    """Verify grade_node correctly updates JurisGraphState."""
    mock_payload = {
        "message": {
            "content": json.dumps({
                "c1": "correct",
                "c2": "incorrect",
                "c3": "incorrect",
            })
        }
    }
    state: JurisGraphState = {
        "question": "What relief was granted?",
        "document_id": "sample2",
        "mode": "role",
        "mapped_roles": ["Decision"],
        "role_kind": "role",
        "role_confidence": 0.95,
        "retrieved_chunks": [
            {"id": "c1", "text": "The petitions are allowed.", "metadata": {"role": "Decision"}, "distance": 0.2},
            {"id": "c2", "text": "Notice under section 131.", "metadata": {"role": "Facts"}, "distance": 0.6},
            {"id": "c3", "text": "Background facts.", "metadata": {"role": "Facts"}, "distance": 0.8},
        ],
        "graded_chunks": [],
        "verified_context": "",
        "fallback_context": "",
        "retry_count": 0,
        "relax_level": 0,
        "generation": "",
        "faithfulness_score": 0.0,
        "eval_status": "",
        "route": "",
        "trace": [],
    }

    with patch("src.rag.grader.call_grader_ollama", return_value=(mock_payload, 120.0)):
        result = grade_node(state)
        assert result["route"] == "generate"
        assert len(result["graded_chunks"]) == 3
        assert result["verified_context"] == "The petitions are allowed."
        assert len(result["trace"]) == 1
        assert result["trace"][0]["node"] == "grade_node"
        assert result["trace"][0]["details"]["route"] == "generate"


def test_grade_node_all_incorrect_routes_to_rewrite():
    """Verify all incorrect chunks route to rewrite."""
    mock_payload = {
        "message": {
            "content": json.dumps({
                "c1": "incorrect",
                "c2": "incorrect",
            })
        }
    }
    state: JurisGraphState = {
        "question": "Irrelevant query",
        "document_id": "sample2",
        "mode": "role",
        "mapped_roles": [],
        "role_kind": "whole_document",
        "role_confidence": 0.5,
        "retrieved_chunks": [
            {"id": "c1", "text": "Random text 1", "metadata": {}, "distance": 0.7},
            {"id": "c2", "text": "Random text 2", "metadata": {}, "distance": 0.8},
        ],
        "graded_chunks": [],
        "verified_context": "",
        "fallback_context": "",
        "retry_count": 0,
        "relax_level": 0,
        "generation": "",
        "faithfulness_score": 0.0,
        "eval_status": "",
        "route": "",
        "trace": [],
    }

    with patch("src.rag.grader.call_grader_ollama", return_value=(mock_payload, 90.0)):
        result = grade_node(state)
        assert result["route"] == "rewrite"
        assert result["verified_context"] == ""


def test_cosine_distance_baseline():
    """Verify no-LLM baseline grading by cosine distance."""
    chunks = [
        {"id": "c1", "distance": 0.35, "text": "Very close chunk", "metadata": {}},
        {"id": "c2", "distance": 0.55, "text": "Far chunk", "metadata": {}},
    ]
    graded, route = grade_chunks_by_distance(chunks, distance_threshold=0.50)
    assert graded[0]["grade"] == "correct"
    assert graded[1]["grade"] == "incorrect"
    assert route == "generate"

    # Higher threshold classifies both as correct
    graded2, route2 = grade_chunks_by_distance(chunks, distance_threshold=0.60)
    assert graded2[0]["grade"] == "correct"
    assert graded2[1]["grade"] == "correct"
    assert route2 == "generate"


def test_real_ollama_grader_smoke():
    """Live smoke test executing live against Ollama (skips if server is offline)."""
    try:
        check_ollama()
    except Exception as e:
        pytest.skip(f"Ollama server is unavailable: {e}")

    question = "What relief did the High Court grant?"
    chunks = [
        {"id": "live_1", "text": "The writ petition is allowed and the warrant of authorization is quashed.", "metadata": {"role": "Decision"}, "distance": 0.22},
        {"id": "live_2", "text": "The assessee sold 11,11,486 shares on 01.07.2021.", "metadata": {"role": "Facts"}, "distance": 0.65},
    ]

    graded, route, wall_ms = grade_chunks(question, chunks, max_chars=800)
    assert len(graded) == 2
    assert route in {"generate", "fallback", "rewrite"}
    assert all(c["grade"] in {"correct", "ambiguous", "incorrect"} for c in graded)
    assert wall_ms > 0.0
