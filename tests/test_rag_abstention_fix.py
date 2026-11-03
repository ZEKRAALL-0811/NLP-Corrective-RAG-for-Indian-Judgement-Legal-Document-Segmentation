"""
Regression test suite for JurisGraph Legal RAG abstention fix.
Tests the 6 regression requirements:
1. Relevant evidence is retrieved even if initial role prediction is ARGUMENTS OF RESPONDENT.
2. Correct relevance grading accepting the Section 12 passage with score/confidence > 0.5.
3. Correct grounded answer generating Section 12 with citation [1].
4. Trace consistency ensuring candidate ID, status, and non-zero score agree (no 0.00 score for CORRECT candidate).
5. Genuine abstention when query is unsupported by judgment context.
6. Document isolation ensuring retrieval never returns chunks from another document.
"""

import pytest
import math
from unittest.mock import MagicMock, patch

from src.rag.graph import (
    JurisGraphState,
    retrieve_node,
    grade_node,
)
from src.rag.grader import (
    grade_chunks,
    grade_chunks_by_cross_encoder,
    GRADE_SCORES,
)
from src.rag.reranker import score_chunks
from src.rag.generator import generate_grounded_answer, ABSTENTION_TEXT
from src.rag.vectorstore import LegalVectorStore


SAMPLE3_SECTION12_TEXT = (
    "26. Now coming to the quantum of punishment imposed by the learned single Judge is concerned, "
    "though Section 12 of the Contempt of Courts Act provides the punishment for contempt with simple "
    "imprisonment for a term which may extend to six months, or with fine which may extend to two thousand "
    "rupees, or with both, in view of the finding recorded by the learned single Judge exercising his "
    "discretionary power, which is extracted as follows..."
)

SAMPLE3_SECTION12_CHUNK = {
    "id": "sample3_role_117",
    "chunk_id": "sample3_role_117",
    "text": SAMPLE3_SECTION12_TEXT,
    "metadata": {
        "document_id": "sample3",
        "document_name": "sample3.pdf",
        "role": "Reasoning",
        "chunk_idx": 117,
        "rel_pos": 0.9358,
    },
    "distance": 0.3553,
}

SAMPLE3_RESPONDENT_CHUNK = {
    "id": "sample3_role_81",
    "chunk_id": "sample3_role_81",
    "text": "The respondent contended that the Contempt of Courts Act, 1971 could not be invoked without sanction.",
    "metadata": {
        "document_id": "sample3",
        "document_name": "sample3.pdf",
        "role": "Arguments of Respondent",
        "chunk_idx": 81,
        "rel_pos": 0.65,
    },
    "distance": 0.3543,
}


def test_regression_1_relevant_evidence_retrieved_despite_wrong_role():
    """
    Test 1: Relevant evidence is retrieved.
    Verify that the query can retrieve the chunk containing the reference to Section 12,
    even if the initial role prediction is ARGUMENTS OF RESPONDENT.
    """
    state: JurisGraphState = {
        "question": "Under what section was the Contempt of Courts Act invoked?",
        "document_id": "sample3",
        "mode": "role",
        "mapped_roles": ["Arguments of Respondent"],
        "relax_level": 0,
        "trace": [],
    }

    mock_store = MagicMock()
    # First call: role-constrained retrieval (returns respondent chunk)
    # Second call: bounded cross-role fallback retrieval (returns Section 12 chunk)
    mock_store.retrieve.side_effect = [
        [SAMPLE3_RESPONDENT_CHUNK],
        [SAMPLE3_SECTION12_CHUNK],
    ]

    out = retrieve_node(state, store=mock_store)
    retrieved = out["retrieved_chunks"]

    # Verify candidate pool contains the Section 12 chunk
    retrieved_texts = [c["text"] for c in retrieved]
    assert any("Section 12" in t for t in retrieved_texts), "Section 12 passage must be retrieved via bounded cross-role fallback"
    assert any(c.get("id") == "sample3_role_117" for c in retrieved)


def test_regression_2_correct_relevance_grading():
    """
    Test 2: Correct relevance grading.
    Verify that the genuinely relevant passage is accepted by the grader using the real
    score contract and configured threshold (confidence > 0.5 and graded CORRECT).
    """
    question = "Under what section was the Contempt of Courts Act invoked?"
    chunks = [SAMPLE3_SECTION12_CHUNK]

    # Test cross-encoder scoring locally
    scored, ms = score_chunks(question, chunks)
    assert len(scored) == 1
    raw_score = scored[0]["rerank_score"]
    prob = 1.0 / (1.0 + math.exp(-raw_score))

    # Cross-encoder confidence must be greater than 0.5
    assert prob > 0.5 or raw_score > 0.5, f"Expected cross-encoder score > 0.5, got raw={raw_score}, prob={prob}"

    # Verify cross-encoder grader assigns CORRECT with confidence > 0.5
    graded, route = grade_chunks_by_cross_encoder(question, chunks, threshold=0.50)
    assert len(graded) == 1
    assert graded[0]["grade"] == "correct"
    assert graded[0]["confidence"] > 0.5
    assert route == "generate"


def test_regression_3_correct_grounded_answer():
    """
    Test 3: Correct grounded answer.
    Verify that the pipeline answers that the Act was invoked under Section 12,
    with citation [1] linked to the supporting passage.
    """
    question = "Under what section was the Contempt of Courts Act invoked?"
    verified = [SAMPLE3_SECTION12_CHUNK]

    res = generate_grounded_answer(
        question=question,
        verified_chunks=verified,
    )

    assert res["status"] == "ready", f"Expected ready status, got {res['status']}"
    assert "12" in res["answer"], f"Expected Section 12 in answer, got: {res['answer']}"
    assert "[1]" in res["answer"], f"Expected bracketed citation [1] in answer, got: {res['answer']}"
    assert any(s.get("chunk_id") == "sample3_role_117" for s in res["sources"])


def test_regression_4_trace_consistency():
    """
    Test 4: Trace consistency.
    Verify that candidate ID, relevance status, score, and generation input agree.
    A candidate must not be labeled CORRECT while its displayed score incorrectly defaults to 0.00.
    """
    question = "Under what section was the Contempt of Courts Act invoked?"
    chunks = [SAMPLE3_SECTION12_CHUNK]

    # Mock grade_chunks to return CORRECT
    mock_graded = [{
        "chunk_id": "sample3_role_117",
        "grade": "correct",
        "score": 0.9,
        "confidence": 0.9,
        "rerank_score": 2.22,
        "role": "Reasoning",
        "distance": 0.3553,
        "text": SAMPLE3_SECTION12_TEXT,
        "metadata": SAMPLE3_SECTION12_CHUNK["metadata"],
    }]

    with patch("src.rag.graph.grade_chunks", return_value=(mock_graded, "generate", 15.0)):
        state: JurisGraphState = {
            "question": question,
            "document_id": "sample3",
            "mode": "role",
            "mapped_roles": ["Reasoning"],
            "retrieved_chunks": chunks,
            "trace": [],
        }
        out = grade_node(state)

        graded = out["graded_chunks"]
        assert len(graded) == 1
        g = graded[0]

        # Verify candidate ID
        assert g["chunk_id"] == "sample3_role_117"
        # Verify status
        assert g["grade"] == "correct"
        # Verify score is not 0.00
        conf = g.get("confidence") or g.get("score") or g.get("rerank_score", 0.0)
        assert conf > 0.0, f"Displayed score must not be 0.00 for CORRECT candidate, got: {conf}"
        assert conf >= 0.9

        # Verify verified_context passed to generation contains this chunk
        verified = out["verified_context"]
        assert len(verified) == 1
        cid = verified[0].get("id") or verified[0].get("chunk_id")
        assert cid == "sample3_role_117"


def test_regression_5_genuine_abstention():
    """
    Test 5: Genuine abstention.
    Use a question whose answer is absent from the document and verify that the system
    still abstains rather than inventing an answer.
    """
    question = "What were the primary arguments of the petitioner regarding the appointment?"
    # Chunks are completely irrelevant to appointments
    irrelevant_chunks = [SAMPLE3_RESPONDENT_CHUNK]

    res = generate_grounded_answer(
        question=question,
        verified_chunks=irrelevant_chunks,
    )

    assert res["status"] == "abstained"
    assert any(phr in res["answer"].lower() for phr in ["do not answer", "do not contain sufficient", "no reliable answer", "insufficient"])
    assert res["citations"] == []


def test_regression_6_document_isolation():
    """
    Test 6: Document isolation.
    Verify that retrieval for an uploaded document does not return chunks from a different
    document or stale prior upload.
    """
    state: JurisGraphState = {
        "question": "Under what section was the Contempt of Courts Act invoked?",
        "document_id": "sample3",
        "mode": "role",
        "mapped_roles": ["Arguments of Respondent"],
        "relax_level": 0,
        "trace": [],
    }

    sample2_chunk = {
        "id": "sample2_role_10",
        "text": "Irrelevant text from sample2",
        "metadata": {"document_id": "sample2", "role": "Arguments of Respondent"},
        "distance": 0.20,
    }

    mock_store = MagicMock()
    # Mock returns only sample3 chunks for sample3 queries
    def mock_retrieve(**kwargs):
        doc_id = kwargs.get("document_id")
        if doc_id == "sample3":
            return [SAMPLE3_RESPONDENT_CHUNK]
        return [sample2_chunk]

    mock_store.retrieve.side_effect = mock_retrieve

    out = retrieve_node(state, store=mock_store)
    retrieved = out["retrieved_chunks"]

    for c in retrieved:
        doc = c.get("metadata", {}).get("document_id") or c.get("document_id")
        assert doc == "sample3", f"Expected document_id 'sample3', found '{doc}'"
        assert "sample2" not in c.get("id", "")


def test_regression_7_acceptance_test1_source_card_excerpt_not_empty_or_div():
    """
    Acceptance Test 1: Verify cited source card for sample3_role_117 contains non-empty
    text and valid paragraph marker (26.), eliminating empty tags or malformed </div> rendering.
    """
    from src.rag.generator import resolve_citations_to_sources, extract_first_para_marker

    text = SAMPLE3_SECTION12_TEXT
    chunk_map = {
        1: {
            "chunk_id": "sample3_role_117",
            "role": "Reasoning",
            "text": text,
            "metadata": {"start_sentence_idx": 337, "end_sentence_idx": 337},
        }
    }

    # Verify paragraph marker extraction finds embedded paragraph 26.
    marker = extract_first_para_marker(text)
    assert marker == "26.", f"Expected para marker '26.', got '{marker}'"

    # Verify source resolution populates text, snippet, citation_num, citation_index
    sources = resolve_citations_to_sources([1], chunk_map)
    assert len(sources) == 1
    src = sources[0]
    assert src["chunk_id"] == "sample3_role_117"
    assert src["citation_num"] == 1
    assert src["citation_index"] == 1
    assert src["role"] == "Reasoning"
    assert src["first_para_marker"] == "26."
    assert "Section 12" in src["snippet"]
    assert "Section 12" in src["text"]
    assert src["snippet"] != ""
    assert src["text"] != ""


def test_regression_8_acceptance_test2_first_point_framed_for_determination():
    """
    Acceptance Test 2: Verify first point framed for determination retrieves
    sample3_role_98 (Paragraph 19) and correctly generates the point with citation [1].
    """
    from src.rag.role_mapper import map_query_to_roles

    q = "What was the first point framed for determination by the High Court?"
    mapping = map_query_to_roles(q)
    assert "Issue" in mapping.roles

    sample3_point1_chunk = {
        "id": "sample3_role_98",
        "chunk_id": "sample3_role_98",
        "text": (
            "19. Having heard the submissions of the respective learned Senior Counsel appearing for the parties, "
            "the first point that arises for consideration is whether the application filed by the appellant under "
            "Article 215 of the Constitution of India read with Section 2(b) of the Contempt of Courts Act is "
            "maintainable for holding the respondents including the judgment debtors guilty of committing contempt?"
        ),
        "metadata": {
            "document_id": "sample3",
            "role": "Issue",
            "chunk_idx": 98,
            "rel_pos": 0.7514,
        },
        "role": "Issue",
    }

    res = generate_grounded_answer(question=q, verified_chunks=[sample3_point1_chunk])
    assert res["status"] == "ready"
    assert "[1]" in res["answer"]
    assert any(term in res["answer"].lower() for term in ["maintainable", "article 215", "section 2(b)"])
    assert res["citations"] == [1]


def test_regression_9_acceptance_test3_nclt_order_binding_effect():
    """
    Acceptance Test 3: Verify NCLT order binding effect maps to Reasoning, evaluates
    sample3_role_116, and generates answer grounded in moratorium / IBC Section 14 with citation [1].
    """
    from src.rag.role_mapper import map_query_to_roles

    q = "Is the order of NCLT binding on the parties?"
    mapping = map_query_to_roles(q)
    assert "Reasoning" in mapping.roles

    sample3_nclt_chunk = {
        "id": "sample3_role_116",
        "chunk_id": "sample3_role_116",
        "text": (
            "25. Moreover, the main object of passing the orders of attachment and prohibition were to restrain "
            "the respondents and other third parties from creating any further encumbrance over the property and "
            "whatever encumbrance created subsequent to the orders of attachment and prohibition will not bind "
            "the appellant/decree holder in the execution proceedings, as the fact remains that by suppressing "
            "the material facts, parallel proceedings have been initiated before the statutory authority/NCLT... "
            "Since the matter is pending before the National Company Law Tribunal and a moratorium has been imposed "
            "under Section 14 of the Insolvency & Bankruptcy Code, civil proceedings against the company cannot proceed."
        ),
        "metadata": {
            "document_id": "sample3",
            "role": "Reasoning",
            "chunk_idx": 116,
            "rel_pos": 0.9246,
        },
        "role": "Reasoning",
    }

    res = generate_grounded_answer(question=q, verified_chunks=[sample3_nclt_chunk])
    assert res["status"] == "ready"
    assert "[1]" in res["answer"]
    assert any(term in res["answer"].lower() for term in ["moratorium", "insolvency", "bankruptcy", "bind", "section 14"])
    assert res["citations"] == [1]

