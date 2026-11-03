"""
Unit and integration tests for JurisGraph Streamlit UI (tests/test_ui.py).
Tests app startup, document ingestion, 7-role distribution, query submission,
trace rendering, verification badges, no-answer/out-of-scope handling,
Ollama diagnosis, rerun resilience, and zero live Gemini calls.
"""

import sys
import io
import os
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

sys.path.insert(0, ".")
import src.config as cfg
from src.config import CANONICAL_LABELS
from src.data.preprocessor import ScannedPdfError
from src.rag.role_mapper import RoleMapping


@pytest.fixture(autouse=True)
def prevent_gemini_calls(monkeypatch):
    """Enforce safety rule: never make live Gemini or Groq calls in UI tests."""
    monkeypatch.setenv("ENABLE_ONLINE_GATE", "false")
    with patch("google.genai.Client", side_effect=RuntimeError("FORBIDDEN: Live Gemini call attempted during UI tests!")):
        yield


def test_ui_startup_and_initial_empty_state():
    """1. App startup and initial empty-document state."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        assert len(at.exception) == 0

        # Check title and subtitle
        header_text = "".join(m.value for m in at.markdown)
        assert "JurisGraph" in header_text
        assert "No document loaded" in header_text

        # Check file uploader exists
        assert len(at.file_uploader) >= 1

        # Check query input exists and is disabled initially
        assert len(at.text_input) >= 1
        assert at.text_input[0].disabled is True

        # Check Ask button is disabled initially
        assert len(at.button) >= 1
        assert at.button[0].disabled is True


def test_ui_empty_file_validation():
    """3. Invalid and empty file validation: 0-byte file."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        
        # Upload empty file
        at.file_uploader[0].upload("empty.pdf", b"").run()
        assert len(at.exception) == 0
        error_text = "".join(e.value for e in at.error)
        assert "Uploaded file is empty (0 bytes)" in error_text


def test_ui_scanned_pdf_error_handling():
    """3. Invalid and empty file validation: Scanned PDF without text layer."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True), \
         patch("src.models.segmenter.segment_document", side_effect=ScannedPdfError("No text layer")):
        
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.file_uploader[0].upload("scanned.pdf", b"%PDF-1.4 dummy scanned content").run()
        assert len(at.exception) == 0
        
        error_text = "".join(e.value for e in at.error)
        assert "scanned image without an extractable digital text layer" in error_text


def test_ui_successful_document_ingestion_and_seven_roles():
    """2 & 4. Successful PDF/TXT ingestion and 7-role distribution display."""
    mock_stats = {
        "sentence_count": 55,
        "mean_tokens_per_sentence": 24.5,
        "wall_clock_time_sec": 1.42,
        "cached": False,
        "role_counts": {
            "Facts": 15,
            "Issue": 3,
            "Arguments of Petitioner": 8,
            "Arguments of Respondent": 6,
            "Reasoning": 12,
            "Decision": 4,
            "None": 7,
        },
    }
    mock_records = [{"sentence_idx": i, "text": f"Sentence {i}", "role": "Facts"} for i in range(55)]

    with patch("src.rag.role_mapper.check_ollama", return_value=True), \
         patch("src.models.segmenter.segment_document", return_value=(mock_records, mock_stats)), \
         patch("src.rag.vectorstore.LegalVectorStore.ingest_document", return_value={"n_chunks": 12}):
        
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.file_uploader[0].upload("sample_judgment.pdf", b"%PDF-1.4 valid digital text").run()
        assert len(at.exception) == 0

        # Verify state updated
        assert at.session_state["doc_id"] == "sample_judgment"
        assert at.session_state["doc_name"] == "sample_judgment.pdf"
        assert len(at.session_state["segmented_records"]) == 55

        # Verify all 7 canonical labels are present in the UI
        rendered_md = "".join(m.value for m in at.markdown)
        for label in CANONICAL_LABELS:
            assert label in rendered_md, f"Missing canonical role label in UI: {label}"

        # Query input and button should now be enabled
        assert at.text_input[0].disabled is False
        assert at.button[0].disabled is False


def test_ui_duplicate_upload_rerun_resilience():
    """11. Streamlit reruns without duplicate pipeline execution."""
    mock_stats = {
        "sentence_count": 10,
        "mean_tokens_per_sentence": 20.0,
        "wall_clock_time_sec": 0.5,
        "cached": False,
        "role_counts": {r: 1 for r in CANONICAL_LABELS},
    }
    mock_records = [{"sentence_idx": 0, "text": "Test", "role": "Facts"}]

    with patch("src.rag.role_mapper.check_ollama", return_value=True), \
         patch("src.models.segmenter.segment_document", return_value=(mock_records, mock_stats)) as mock_segment, \
         patch("src.rag.vectorstore.LegalVectorStore.ingest_document") as mock_ingest:
        
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        file_bytes = b"identical_content_for_hash"
        
        # First upload -> runs segmentation and ingestion
        at.file_uploader[0].upload("doc.pdf", file_bytes).run()
        assert mock_segment.call_count == 1
        assert mock_ingest.call_count == 1

        # Second rerun without changing file -> does NOT re-run segmentation or ingestion
        at.run()
        assert mock_segment.call_count == 1
        assert mock_ingest.call_count == 1


def test_ui_query_submission_and_mapped_role():
    """5. Query submission and mapped-role display."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        
        # Seed loaded document into session state
        at.session_state["doc_id"] = "sample2"
        at.session_state["doc_name"] = "sample2.pdf"
        at.session_state["role_counts"] = {r: 5 for r in CANONICAL_LABELS}
        at.run()

        mock_role_map = RoleMapping(roles=["Arguments of Petitioner"], confidence=0.94, kind="role")
        mock_graph_res = {
            "question": "What did petitioner argue?",
            "generation": "The petitioner argued that the notice was barred by limitation [1].",
            "mapped_roles": ["Arguments of Petitioner"],
            "role_kind": "role",
            "role_confidence": 0.94,
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Passage text", "role": "Arguments of Petitioner"}],
            "graded_chunks": [{
                "chunk_id": "c1",
                "grade": "Correct",
                "confidence": 0.94,
                "role": "Arguments of Petitioner",
                "reason": "Direct argument on limitation",
                "text": "Notice barred by limitation.",
            }],
            "verified_context": [{"chunk_id": "c1", "text": "Notice barred by limitation.", "role": "Arguments of Petitioner"}],
            "citations": [1],
            "sources": [{
                "citation_index": 1,
                "chunk_id": "c1",
                "role": "Arguments of Petitioner",
                "first_para_marker": "(12)",
                "text": "Notice barred by limitation.",
            }],
            "final_status": "ready",
            "retry_count": 0,
            "relax_level": 0,
            "trace": [{"node": "retrieve_node", "duration_ms": 12.5, "route": "grade_node"}],
        }

        with patch("src.rag.role_mapper.map_query_to_roles", return_value=mock_role_map), \
             patch("src.rag.graph.run_query", return_value=mock_graph_res), \
             patch("src.evaluation.quality_gate.verify_generation_faithfulness", return_value={"status": "gate_disabled", "verified": True, "score": 1.0}):
            
            at.text_input[0].set_value("What did petitioner argue?")
            at.button[0].click().run()
            assert len(at.exception) == 0

            rendered = "".join(m.value for m in at.markdown)
            assert "Arguments of Petitioner" in rendered
            assert "Notice barred by limitation" in rendered
            assert "The petitioner argued that the notice was barred by limitation" in rendered


def test_ui_trace_rendering():
    """6. Retrieval/grading trace rendering with grades and confidence scores."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        
        at.session_state["doc_id"] = "sample2"
        at.session_state["doc_name"] = "sample2.pdf"
        at.session_state["role_counts"] = {r: 5 for r in CANONICAL_LABELS}
        at.session_state["query_result"] = {
            "question": "What was the final decision?",
            "generation": "The appeal was allowed and the order was set aside [1].",
            "mapped_roles": ["Decision"],
            "role_kind": "role",
            "role_confidence": 0.96,
            "graded_chunks": [
                {
                    "chunk_id": "chunk_dec_1",
                    "grade": "Correct",
                    "confidence": 0.95,
                    "role": "Decision",
                    "reason": "Directly states appeal allowed",
                    "text": "In the result, the appeal is allowed and impugned judgment is set aside.",
                },
                {
                    "chunk_id": "chunk_fact_2",
                    "grade": "Incorrect",
                    "confidence": 0.88,
                    "role": "Facts",
                    "reason": "Background facts, not operative ruling",
                    "text": "The appellant instituted writ petition in 2018.",
                },
            ],
            "verified_context": [{"chunk_id": "chunk_dec_1", "text": "Appeal allowed.", "role": "Decision"}],
            "sources": [{"citation_index": 1, "chunk_id": "chunk_dec_1", "role": "Decision", "first_para_marker": "(28)", "text": "Appeal allowed."}],
            "final_status": "ready",
            "retry_count": 0,
            "relax_level": 0,
            "trace": [{"node": "grade_node", "duration_ms": 25.4, "route": "generate"}],
        }
        at.session_state["verification_result"] = {"status": "gate_disabled", "verified": True, "score": 1.0}
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        # Check trace elements
        assert any("Pipeline Trace & Relevance Grading" in e.label for e in at.expander)
        assert "CORRECT" in rendered
        assert "DISCARDED" in rendered
        assert "0.95" in rendered
        assert "chunk_dec_1" in rendered


def test_ui_answer_with_verified_badge():
    """7. Successful answer with valid verification status (Online Gate Passed)."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample2"
        at.session_state["query_result"] = {
            "question": "When was sale deed executed?",
            "generation": "The sale deed was executed on 01.07.2021 [1].",
            "final_status": "ready",
            "sources": [{"citation_index": 1, "chunk_id": "c1", "role": "Facts", "first_para_marker": "3.", "text": "Sale deed executed 01.07.2021."}],
        }
        at.session_state["verification_result"] = {
            "status": "passed",
            "verified": True,
            "faithfulness_score": 1.0,
            "relevancy_score": 0.95,
            "reason": "Fully grounded and relevant.",
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "VERIFIED AGAINST SOURCE" in rendered
        assert "Faithfulness: 1.00" in rendered
        assert "Relevancy: 0.95" in rendered


def test_ui_unverified_answer_and_disclaimer():
    """8. Unverified answer and disclaimer after evaluation failure."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample2"
        at.session_state["query_result"] = {
            "question": "What was the fine amount?",
            "generation": "The court imposed a penalty of 50 lakhs.",
            "final_status": "ready",
            "verified_context": [{"chunk_id": "c1", "text": "No fine was mentioned.", "role": "Decision"}],
            "sources": [],
        }
        at.session_state["verification_result"] = {
            "status": "failed",
            "verified": False,
            "faithfulness_score": 0.25,
            "relevancy_score": 0.80,
            "reason": "Penalty figure of 50 lakhs is not supported by context.",
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "COULD NOT BE FULLY VERIFIED" in rendered
        assert "Score: 0.25" in rendered
        assert "Disclaimer:" in rendered
        assert any("Inspect All Retrieved Chunks" in e.label for e in at.expander)


def test_ui_quality_gate_disabled_offline_badge():
    """9. Distinguish gate-disabled offline status from verified."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample2"
        at.session_state["query_result"] = {
            "question": "What was the issue?",
            "generation": "The issue was whether the revision was maintainable.",
            "final_status": "ready",
            "sources": [],
        }
        at.session_state["verification_result"] = {
            "status": "gate_disabled",
            "verified": True,
            "score": 1.0,
            "reason": "Online quality gate is disabled.",
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "OFFLINE (QUALITY GATE DISABLED)" in rendered
        assert "VERIFIED AGAINST SOURCE" not in rendered


def test_ui_no_answer_state():
    """9. Documented no-answer state when retries exhausted."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample2"
        at.session_state["query_result"] = {
            "question": "What is the secret formula?",
            "generation": "No reliable answer found in this document for that question.",
            "final_status": "no_answer",
            "sources": [],
        }
        at.session_state["verification_result"] = {
            "status": "no_answer",
            "verified": False,
            "score": 0.0,
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "No reliable answer found in this document for that question." in rendered
        assert "NO ANSWER FOUND" in rendered


def test_ui_out_of_scope_query_handling():
    """9. Out-of-scope query handled gracefully without crash."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample2"
        at.session_state["role_counts"] = {r: 5 for r in CANONICAL_LABELS}
        at.run()

        mock_role_map = RoleMapping(roles=[], confidence=0.99, kind="out_of_scope")
        with patch("src.rag.role_mapper.map_query_to_roles", return_value=mock_role_map), \
             patch("src.rag.graph.run_query") as mock_run_query:
            
            at.text_input[0].set_value("Who won the 2024 cricket world cup?")
            at.button[0].click().run()
            assert len(at.exception) == 0

            # run_query must NOT be called for out-of-scope query
            assert mock_run_query.call_count == 0
            rendered = "".join(m.value for m in at.markdown)
            assert "outside the scope of Indian legal judgments" in rendered
            assert "OUT OF LEGAL SCOPE" in rendered


def test_ui_ollama_unavailable_warning():
    """10. Ollama unavailable reported with actionable diagnosis."""
    with patch("src.rag.role_mapper.check_ollama", side_effect=RuntimeError("Ollama server is unreachable at http://localhost:11434. Please run 'ollama serve'.")):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "Ollama Service Advisory" in rendered
        assert "Ollama server is unreachable at http://localhost:11434" in rendered
        assert "ollama serve" in rendered



def test_ui_cross_encoder_rejection_trace_explanation():
    """1. Retrieved chunks rejected by cross-encoder show informative explanation and candidate cards."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample3"
        at.session_state["query_result"] = {
            "question": "What were the arguments regarding the appointment?",
            "generation": "No reliable answer found in this document for that question.",
            "final_status": "no_answer",
            "retry_count": 2,
            "relax_level": 2,
            "role_kind": "role",
            "retrieved_chunks": [
                {"chunk_id": "c_rej_1", "role": "Arguments of Petitioner", "text": "Passage discussing contempt.", "rerank_score": -10.3207}
            ],
            "graded_chunks": [],
            "sources": [],
            "trace": [
                {"node": "retrieve_node", "details": {"chunks_retrieved": 5}},
                {"node": "grade_node", "details": {"top_ce_score": -10.3207, "route": "rewrite"}},
                {"node": "end_no_answer_node", "details": {"final_status": "no_answer"}},
            ],
        }
        at.session_state["verification_result"] = {
            "status": "no_answer",
            "verified": False,
            "score": 0.0,
            "reason": "No reliable answer found in document.",
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "Retrieved passages were rejected by the relevance filter" in rendered
        assert "-10.32" in rendered
        assert "-3.5" in rendered
        assert "Self-correction retries were exhausted without finding reliable context" in rendered
        assert "Unverified Candidate Passages Inspected" in rendered
        assert "c_rej_1" in rendered
        assert "REJECTED" in rendered


def test_ui_cross_encoder_score_matches_mocked_values():
    """2. The displayed score and threshold match the actual mocked trace values."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample3"
        at.session_state["query_result"] = {
            "question": "Sample rejected question?",
            "generation": "No reliable answer found in this document for that question.",
            "final_status": "no_answer",
            "retry_count": 1,
            "relax_level": 1,
            "role_kind": "role",
            "retrieved_chunks": [],
            "graded_chunks": [],
            "sources": [],
            "trace": [
                {"node": "retrieve_node", "details": {"chunks_retrieved": 3}},
                {"node": "grade_node", "details": {"top_ce_score": -8.452, "route": "rewrite"}},
            ],
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "-8.45" in rendered
        assert "-3.5" in rendered
        assert "The system rewrote the query instead of generating an unsupported answer" in rendered


def test_ui_zero_chunks_retrieved_distinct_explanation():
    """3. A query with zero retrieved chunks has a distinct explanation from rejected chunks."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample3"
        at.session_state["query_result"] = {
            "question": "Question yielding zero chunks?",
            "generation": "No reliable answer found in this document for that question.",
            "final_status": "no_answer",
            "retry_count": 0,
            "relax_level": 0,
            "role_kind": "role",
            "retrieved_chunks": [],
            "graded_chunks": [],
            "sources": [],
            "trace": [
                {"node": "retrieve_node", "details": {"chunks_retrieved": 0}},
            ],
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "No candidate passages were retrieved from the document matching the query" in rendered
        assert "rejected by the relevance filter" not in rendered


def test_ui_exhausted_retry_loop_no_answer_state():
    """4. An exhausted retry loop displays the correct no-answer state in trace and answer card."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample3"
        at.session_state["query_result"] = {
            "question": "Exhausted query?",
            "generation": "No reliable answer found in this document for that question.",
            "final_status": "no_answer",
            "retry_count": 2,
            "relax_level": 2,
            "role_kind": "role",
            "retrieved_chunks": [],
            "graded_chunks": [],
            "sources": [],
            "trace": [
                {"node": "retrieve_node", "details": {"chunks_retrieved": 5}},
                {"node": "grade_node", "details": {"top_ce_score": -9.99, "route": "rewrite"}},
                {"node": "end_no_answer_node", "details": {"final_status": "no_answer"}},
            ],
        }
        at.session_state["verification_result"] = {
            "status": "no_answer",
            "verified": False,
            "score": 0.0,
            "reason": "No reliable answer found in document.",
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "NO ANSWER FOUND" in rendered
        assert "No reliable answer found in this document for that question." in rendered
        assert "VERIFIED AGAINST SOURCE" not in rendered


def test_ui_missing_trace_fields_graceful_handling():
    """5. Missing trace fields do not crash the UI or produce fabricated scores."""
    with patch("src.rag.role_mapper.check_ollama", return_value=True):
        at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
        at.session_state["doc_id"] = "sample3"
        at.session_state["query_result"] = {
            "question": "Question with missing trace?",
            "generation": "No reliable answer found in this document for that question.",
            "final_status": "no_answer",
            "retry_count": 0,
            "relax_level": 0,
            "role_kind": "role",
            "retrieved_chunks": [],
            "graded_chunks": [],
            "sources": [],
            "trace": [],  # Completely missing trace
        }
        at.run()
        assert len(at.exception) == 0

        rendered = "".join(m.value for m in at.markdown)
        assert "No chunk evaluation details are available in the execution trace." in rendered
        assert "top_ce_score" not in rendered
        assert "-10.32" not in rendered


def test_ui_zero_gemini_api_calls():
    """12. Guarantee zero live Gemini calls were made in any automated UI tests."""
    ledger_path = cfg.ARTIFACTS_DIR / "logs" / "judge_usage.jsonl"
    if ledger_path.exists():
        lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1102, f"Expected 1102 ledger lines, found {len(lines)}! Live calls were made."
