"""
Unit tests for Module 9: Grounded Generation with Abstention (src/rag/generator.py).

Tests (all LLM calls mocked):
1. Invalid citations: dropped/filtered to only existing context chunks.
2. Ungrounded numbers: flags ungrounded_numbers=True when answer mentions dates/numbers/sections absent from cited text.
3. Grounded numbers: passes when numbers/dates/sections appear in cited text.
4. Abstention: returns answerable=False, ABSTENTION_TEXT, and status='abstained' when chunks do not answer.
5. Immediate abstention: abstains without LLM call when no chunks are provided.
6. External-context labeling: prefixes external web block and uses [W1], [W2] identifiers with low_confidence status.
7. Source mapping: resolves citations to chunk id, role, first paragraph marker, and sentence range.
"""

from unittest.mock import patch, MagicMock
import json

from src.rag.generator import (
    generate_grounded_answer,
    format_context_for_prompt,
    resolve_citations_to_sources,
    ABSTENTION_TEXT,
)


def _mock_ollama_response(answerable: bool, answer: str, citations: list):
    """Helper to mock urllib.request.urlopen returning structured JSON."""
    content = json.dumps({
        "answerable": answerable,
        "answer": answer,
        "citations": citations,
    })
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "message": {"content": content}
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_invalid_citations_are_filtered():
    """Verify citations referencing non-existent chunks (e.g. 99, -1, 'two') are stripped."""
    verified_chunks = [
        {"chunk_id": "c1", "text": "The High Court dismissed the writ petition on 12 March 2021.", "role": "Ruling"},
        {"chunk_id": "c2", "text": "The respondent filed an appeal under Section 34.", "role": "Facts"},
    ]

    mock_resp = _mock_ollama_response(
        answerable=True,
        answer="The High Court dismissed the writ petition [1].",
        citations=[1, 99, -1, "invalid", 2],
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer("What happened to the petition?", verified_chunks)

        assert res["answerable"] is True
        # Only valid 1-indexed citations in range [1, 2] are retained
        assert res["citations"] == [1, 2]
        assert len(res["sources"]) == 2
        assert res["sources"][0]["chunk_id"] == "c1"
        assert res["sources"][1]["chunk_id"] == "c2"


def test_ungrounded_numbers_flagged():
    """Verify that dates/numbers/sections invented by LLM trigger ungrounded_numbers=True and abstention."""
    verified_chunks = [
        {"chunk_id": "c1", "text": "The appellant argued that the penalty was unjust and arbitrary.", "role": "Argument"},
    ]

    # LLM invents section 420 and year 2022 not in chunk
    mock_resp = _mock_ollama_response(
        answerable=True,
        answer="The penalty of 50000 was imposed under Section 420 in 2022 [1].",
        citations=[1],
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer("What was the penalty?", verified_chunks)

        assert res["ungrounded_numbers"] is True
        assert res["status"] == "abstained"
        assert res["answer"] == ABSTENTION_TEXT


def test_grounded_numbers_pass():
    """Verify that numbers/dates/sections present in cited chunk pass grounding check."""
    verified_chunks = [
        {"chunk_id": "c1", "text": "Penalty under Section 271(1)(c) was initiated for assessment year 1998-99.", "role": "Facts"},
    ]

    mock_resp = _mock_ollama_response(
        answerable=True,
        answer="Penalty was initiated under Section 271 for assessment year 1998 [1].",
        citations=[1],
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer("Which section was invoked?", verified_chunks)

        assert res["ungrounded_numbers"] is False
        assert res["status"] == "ready"
        assert res["answerable"] is True
        assert "Section 271" in res["answer"]


def test_abstention_when_unanswerable():
    """Verify generator abstains with standard message when LLM declares answerable=False."""
    verified_chunks = [
        {"chunk_id": "c1", "text": "The court adjourned the hearing until next Tuesday.", "role": "Facts"},
    ]

    mock_resp = _mock_ollama_response(
        answerable=False,
        answer=ABSTENTION_TEXT,
        citations=[],
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer("Who was the arbitrator appointed?", verified_chunks)

        assert res["answerable"] is False
        assert res["status"] == "abstained"
        assert res["answer"] == ABSTENTION_TEXT
        assert res["citations"] == []
        assert res["sources"] == []


def test_immediate_abstention_on_empty_context():
    """Verify generator returns abstention immediately without network call if context is empty."""
    with patch("urllib.request.urlopen") as mock_url:
        res = generate_grounded_answer("Any question?", verified_chunks=[], fallback_chunks=[])

        assert res["answerable"] is False
        assert res["status"] == "abstained"
        assert res["answer"] == ABSTENTION_TEXT
        mock_url.assert_not_called()


def test_external_context_labeling_and_low_confidence():
    """Verify external web snippets are labeled with distinct header, [W1] citations, and low_confidence status."""
    fb_chunks = [
        {"snippet": "The arbitrator was Justice K. Ramaswamy appointed in 2019.", "url": "https://example.com/arbitration"},
    ]

    # Format context test
    fmt, chunk_map = format_context_for_prompt([], fallback_chunks=fb_chunks, is_web=True)
    assert "External web sources, not part of the judgment" in fmt
    assert "[W1]" in fmt

    # LLM execution test
    mock_resp = _mock_ollama_response(
        answerable=True,
        answer="According to external web sources, Justice K. Ramaswamy was appointed in 2019 [1].",
        citations=[1],
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer(
            question="Who was the arbitrator?",
            verified_chunks=[],
            fallback_chunks=fb_chunks,
            is_web_fallback=True,
        )

        assert res["status"] == "low_confidence"
        assert res["answerable"] is True
        assert len(res["citations"]) == 1


def test_source_mapping_structure():
    """Verify sources metadata mapping correctly extracts paragraph markers and sentence ranges."""
    chunk = {
        "chunk_id": "doc1_para4",
        "text": "14. It is well settled that under Section 9, an interim measure can be granted.",
        "role": "Precedent",
        "metadata": {"start_sentence_idx": 45, "end_sentence_idx": 50},
    }
    chunk_map = {1: chunk}
    sources = resolve_citations_to_sources([1], chunk_map)

    assert len(sources) == 1
    src = sources[0]
    assert src["citation_num"] == 1
    assert src["chunk_id"] == "doc1_para4"
    assert src["role"] == "Precedent"
    assert src["first_para_marker"] == "14."
    assert src["sentence_range"] == (45, 50)


def test_abstention_phrasing_detection():
    """Verify natural-language abstention phrases force status='abstained' even if answerable=True."""
    chunks = [
        {"chunk_id": "c1", "text": "General background facts about property.", "role": "Facts"}
    ]
    phrases = [
        "Based on the provided context, it is unclear whether the applicant filed an advance ruling.",
        "The retrieved parts of this judgment do not contain sufficient information to answer.",
        "There is no information in the provided context to determine this claim.",
    ]
    for phr in phrases:
        mock_resp = _mock_ollama_response(
            answerable=True,
            answer=phr,
            citations=[],
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = generate_grounded_answer(
                question="Was an advance ruling filed?",
                verified_chunks=chunks,
            )
            assert res["status"] == "abstained", f"Expected abstained for phrasing: {phr}"
            assert res["answerable"] is False
