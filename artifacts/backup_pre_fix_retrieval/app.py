"""
JurisGraph — Agentic Corrective RAG for Indian Legal Documents
Clinical Dark-Mode Streamlit Dashboard (Module 11)

Adheres strictly to Docs/UIUX_Brief_JurisGraph.md:
- Background: #0B0D12
- Card Surface: #161922
- Borders: 1px solid #1E212B
- Primary Accent: #4C6FFF
- Primary Text: #E7E9EE, Muted Text: #9AA1AE
- Status: Correct #3DD68C, Ambiguous #F5B841, Incorrect #63687A, Warning #F0654B
- Single scrolling column, no modals, monospace metadata/scores.
"""

import os
import sys
import time
import signal
import threading
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Defensive monkey-patch for DeepEval signal registration in Streamlit worker threads
_orig_signal = signal.signal
def _safe_signal(sig, handler):
    try:
        return _orig_signal(sig, handler)
    except ValueError:
        return None
signal.signal = _safe_signal

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.config import CANONICAL_LABELS
from src.data.preprocessor import ScannedPdfError
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore
from src.rag.graph import run_query, get_default_store, TAU_LOW, TAU_HIGH
from src.rag.role_mapper import map_query_to_roles, check_ollama, RoleMapping
from src.evaluation.quality_gate import verify_generation_faithfulness


# -----------------------------------------------------------------------------
# 1. Custom CSS Theme (Clinical Dark Mode)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Main application background and typography */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B0D12 !important;
    color: #E7E9EE !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Card Surface container */
.jg-card {
    background-color: #161922;
    border: 1px solid #1E212B;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.jg-header {
    border-bottom: 1px solid #1E212B;
    padding-bottom: 16px;
    margin-bottom: 24px;
}

.jg-title {
    font-size: 24px;
    font-weight: 600;
    color: #E7E9EE;
    margin: 0;
    letter-spacing: -0.01em;
}

.jg-subtitle {
    font-size: 14px;
    font-weight: 400;
    color: #9AA1AE;
    margin-top: 4px;
    margin-bottom: 0;
}

/* Document status banner */
.jg-status-banner {
    background-color: #161922;
    border: 1px solid #1E212B;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.jg-status-banner.loaded {
    border-left: 3px solid #4C6FFF;
}

.jg-status-banner.empty {
    border-left: 3px solid #63687A;
    color: #9AA1AE;
}

/* Role distribution pill cards */
.jg-role-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.jg-role-pill {
    background-color: #12141C;
    border: 1px solid #1E212B;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.jg-role-pill-name {
    color: #9AA1AE;
    font-weight: 500;
}

.jg-role-pill-count {
    color: #E7E9EE;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* Role Tags / Chips */
.jg-tag {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.02em;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}

.jg-tag-role { background-color: rgba(76, 111, 255, 0.15); color: #7C98FF; border: 1px solid rgba(76, 111, 255, 0.3); }
.jg-tag-facts { background-color: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
.jg-tag-issue { background-color: rgba(167, 139, 250, 0.15); color: #A78BFA; border: 1px solid rgba(167, 139, 250, 0.3); }
.jg-tag-petitioner { background-color: rgba(96, 165, 250, 0.15); color: #60A5FA; border: 1px solid rgba(96, 165, 250, 0.3); }
.jg-tag-respondent { background-color: rgba(251, 146, 60, 0.15); color: #FB923C; border: 1px solid rgba(251, 146, 60, 0.3); }
.jg-tag-reasoning { background-color: rgba(192, 132, 252, 0.15); color: #C084FC; border: 1px solid rgba(192, 132, 252, 0.3); }
.jg-tag-decision { background-color: rgba(61, 214, 140, 0.15); color: #3DD68C; border: 1px solid rgba(61, 214, 140, 0.3); }
.jg-tag-none { background-color: rgba(99, 104, 122, 0.15); color: #9AA1AE; border: 1px solid rgba(99, 104, 122, 0.3); }

/* Grading status badges */
.jg-grade-correct {
    background-color: rgba(61, 214, 140, 0.15);
    color: #3DD68C;
    border: 1px solid rgba(61, 214, 140, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.jg-grade-ambiguous {
    background-color: rgba(245, 184, 65, 0.15);
    color: #F5B841;
    border: 1px solid rgba(245, 184, 65, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.jg-grade-incorrect {
    background-color: rgba(99, 104, 122, 0.15);
    color: #9AA1AE;
    border: 1px solid rgba(99, 104, 122, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Verification Badges */
.jg-badge-verified {
    background-color: rgba(61, 214, 140, 0.12);
    color: #3DD68C;
    border: 1px solid #3DD68C;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.jg-badge-unverified {
    background-color: rgba(240, 101, 75, 0.12);
    color: #F0654B;
    border: 1px solid #F0654B;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.jg-badge-offline {
    background-color: rgba(99, 104, 122, 0.12);
    color: #9AA1AE;
    border: 1px solid #63687A;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Monospace trace styles */
.jg-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #9AA1AE;
}

.jg-mono-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #E7E9EE;
}

/* Source Citation Box */
.jg-source-item {
    background-color: #12141C;
    border: 1px solid #1E212B;
    border-radius: 6px;
    padding: 12px 14px;
    margin-top: 10px;
    font-size: 13px;
    line-height: 1.5;
}

/* Streamlit Button override */
button[kind="primary"] {
    background-color: #4C6FFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

button[kind="primary"]:hover {
    background-color: #3B5BDB !important;
}

/* Streamlit input fields */
div[data-baseweb="input"] {
    background-color: #161922 !important;
    border-color: #1E212B !important;
    border-radius: 6px !important;
    color: #E7E9EE !important;
}

/* Expander customization */
div[data-testid="stExpander"] {
    background-color: #161922 !important;
    border: 1px solid #1E212B !important;
    border-radius: 6px !important;
    margin-top: 12px !important;
}
</style>
"""


# -----------------------------------------------------------------------------
# 2. Session State Management
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    """Initializes persistent Streamlit session state fields."""
    defaults = {
        "doc_id": None,
        "doc_name": None,
        "file_hash": None,
        "segmented_records": [],
        "segmentation_stats": {},
        "role_counts": {r: 0 for r in CANONICAL_LABELS},
        "current_query": "",
        "query_result": None,
        "verification_result": None,
        "ollama_warning": None,
        "error_message": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_role_tag_html(role: str) -> str:
    """Returns styled HTML inline tag for canonical rhetorical roles."""
    clean = str(role).strip()
    tag_class = "jg-tag-role"
    if clean == "Facts":
        tag_class = "jg-tag-facts"
    elif clean == "Issue":
        tag_class = "jg-tag-issue"
    elif clean == "Arguments of Petitioner":
        tag_class = "jg-tag-petitioner"
    elif clean == "Arguments of Respondent":
        tag_class = "jg-tag-respondent"
    elif clean == "Reasoning":
        tag_class = "jg-tag-reasoning"
    elif clean == "Decision":
        tag_class = "jg-tag-decision"
    elif clean == "None":
        tag_class = "jg-tag-none"
    return f'<span class="jg-tag {tag_class}">{clean}</span>'


def get_grade_badge_html(grade: str) -> str:
    """Returns styled HTML badge for Correct / Ambiguous / Incorrect grades."""
    g = str(grade).strip().capitalize()
    if g == "Correct":
        return '<span class="jg-grade-correct">✓ CORRECT</span>'
    elif g == "Ambiguous":
        return '<span class="jg-grade-ambiguous">? AMBIGUOUS</span>'
    else:
        return '<span class="jg-grade-incorrect">✗ DISCARDED</span>'


def get_trace_explanation(res: Dict[str, Any]) -> str:
    """
    Produces a grounded, informative explanation when no LLM-graded chunks are shown.
    Distinguishes:
    1. Out-of-scope query.
    2. No chunks retrieved from document.
    3. Chunks retrieved but rejected by Cross-Encoder relevance filtering (with real scores/thresholds).
    4. Chunks retrieved but rejected by LLM relevance grader.
    5. Retries exhausted without verified context.
    6. Genuine missing trace data.
    """
    role_kind = res.get("role_kind", "")
    if role_kind == "out_of_scope":
        return "No chunks retrieved for out-of-scope question."

    trace = res.get("trace", [])
    retrieved_chunks = res.get("retrieved_chunks", [])
    final_status = res.get("final_status", "")
    retry_count = res.get("retry_count", 0)

    # 1. Zero chunks retrieved
    retrieve_events = [ev for ev in trace if ev.get("node") == "retrieve_node"]
    if retrieve_events:
        total_retrieved = sum(ev.get("details", {}).get("chunks_retrieved", 0) for ev in retrieve_events)
        if total_retrieved == 0:
            return "No candidate passages were retrieved from the document matching the query."

    # 2. Cross-Encoder rejection in grade_node events
    grade_events = [ev for ev in trace if ev.get("node") == "grade_node"]
    ce_scores = [
        ev.get("details", {}).get("top_ce_score")
        for ev in grade_events
        if ev.get("details", {}).get("top_ce_score") is not None
    ]

    if ce_scores:
        top_score = ce_scores[0]
        if top_score < TAU_LOW:
            if final_status == "no_answer" and retry_count >= 2:
                return (
                    f"Retrieved passages were rejected by the relevance filter. "
                    f"The highest cross-encoder score was {top_score:.2f}, below the configured lower threshold of {TAU_LOW}. "
                    f"Self-correction retries were exhausted without finding reliable context; "
                    f"the system abstained instead of generating an unsupported answer."
                )
            else:
                return (
                    f"Retrieved passages were rejected by the relevance filter. "
                    f"The highest cross-encoder score was {top_score:.2f}, below the configured lower threshold of {TAU_LOW}. "
                    f"The system rewrote the query instead of generating an unsupported answer."
                )

    # 3. LLM grader rejection
    for ev in grade_events:
        details = ev.get("details", {})
        grades = details.get("grades", [])
        if grades and all(g == "incorrect" for g in grades):
            return "Retrieved passages were evaluated by the relevance grader, but all candidates were deemed irrelevant or off-topic."

    # 4. Missing trace data or unrecorded evaluation details
    if not trace or (grade_events and not ce_scores):
        return "No chunk evaluation details are available in the execution trace."

    # 5. Retries exhausted without verified context
    if final_status == "no_answer":
        return "Retrieval and self-correction retries were exhausted without finding verified supporting context in this document."

    return "No verified context passages were available for generation."


# -----------------------------------------------------------------------------
# 3. Core App Entry Point
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="JurisGraph — Agentic Corrective RAG",
        page_icon="⚖",
        layout="centered",
    )
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        """
        <div class="jg-header">
            <h1 class="jg-title">JurisGraph</h1>
            <p class="jg-subtitle">Agentic Corrective RAG for Indian Legal Documents · Rhetorical Role Segmentation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # System Preflight / Diagnostics
    # -------------------------------------------------------------------------
    ollama_ok = True
    try:
        check_ollama(timeout=1.5)
    except Exception as e:
        ollama_ok = False
        st.session_state.ollama_warning = str(e)

    if not ollama_ok and st.session_state.ollama_warning:
        st.markdown(
            f"""
            <div class="jg-card" style="border-left: 3px solid #F0654B;">
                <div style="font-size: 13px; font-weight: 600; color: #F0654B; margin-bottom: 4px;">
                    ⚠ Ollama Service Advisory
                </div>
                <div style="font-size: 13px; color: #E7E9EE; line-height: 1.4;">
                    {st.session_state.ollama_warning}
                </div>
                <div style="font-size: 12px; color: #9AA1AE; margin-top: 6px;">
                    Role mapping, grading, and generation require <code>llama3.2:3b</code> on <code>http://localhost:11434</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Section A: Document Ingestion
    # -------------------------------------------------------------------------
    # Landing status indicator
    if st.session_state.doc_id:
        n_sents = st.session_state.segmentation_stats.get("sentence_count", len(st.session_state.segmented_records))
        st.markdown(
            f"""
            <div class="jg-status-banner loaded">
                <div>
                    <span style="color: #9AA1AE; font-size: 12px; text-transform: uppercase; letter-spacing: 0.02em;">Document Loaded</span><br/>
                    <strong style="color: #E7E9EE; font-size: 14px;">{st.session_state.doc_name}</strong>
                    <span class="jg-mono" style="margin-left: 8px;">({n_sents} sentences segmented)</span>
                </div>
                <div>
                    <span class="jg-grade-correct">READY FOR QUERY</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="jg-status-banner empty">
                <div>No document loaded. Upload an Indian court judgment (PDF or TXT) to begin.</div>
                <div><span class="jg-grade-incorrect">IDLE</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Ingestion Card
    with st.container():
        st.markdown('<div class="jg-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-bottom: 8px;">Document Ingestion</div>',
            unsafe_allow_html=True,
        )
        
        uploaded_file = st.file_uploader(
            "Upload judgment file",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            key="judgment_uploader",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) == 0:
                st.error("Uploaded file is empty (0 bytes). Please upload a valid judgment document.")
            else:
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                
                # Ingest only if new file or file changed
                if file_hash != st.session_state.file_hash:
                    with st.spinner("Preprocessing document and classifying rhetorical roles (InLegalBERT + Viterbi)..."):
                        try:
                            # Save temporary file for preprocessor
                            temp_dir = cfg.CACHE_DIR / "uploads"
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            temp_path = temp_dir / uploaded_file.name
                            temp_path.write_bytes(file_bytes)

                            # 1. Segment sentences and smooth roles
                            records, stats = segment_document(temp_path, use_cache=True, write_preview=False)

                            if len(records) == 0:
                                st.warning("No sentences could be extracted from this document. Please check the file.")
                            else:
                                # 2. Idempotent ChromaDB vector ingestion
                                doc_id = Path(uploaded_file.name).stem
                                vstore = get_default_store()
                                ingest_res = vstore.ingest_document(
                                    segmented=records,
                                    document_id=doc_id,
                                    mode="role",
                                    document_name=uploaded_file.name,
                                )

                                # Update session state
                                st.session_state.file_hash = file_hash
                                st.session_state.doc_id = doc_id
                                st.session_state.doc_name = uploaded_file.name
                                st.session_state.segmented_records = records
                                st.session_state.segmentation_stats = stats
                                st.session_state.role_counts = stats.get("role_counts", {})
                                st.session_state.query_result = None
                                st.session_state.verification_result = None

                        except ScannedPdfError:
                            st.error(
                                "This PDF appears to be a scanned image without an extractable digital text layer. "
                                "JurisGraph supports digital legal judgments; OCR is not supported."
                            )
                        except Exception as exc:
                            st.error(f"Ingestion failed: {exc}")

        # Display 7-Role Distribution if document is loaded
        if st.session_state.doc_id and st.session_state.role_counts:
            st.markdown(
                """
                <div style="margin-top: 14px; font-size: 12px; font-weight: 500; color: #9AA1AE; text-transform: uppercase; letter-spacing: 0.02em;">
                    Rhetorical Role Distribution (Exact 7 Canonical Labels)
                </div>
                """,
                unsafe_allow_html=True,
            )

            pill_htmls = []
            for role in CANONICAL_LABELS:
                count = st.session_state.role_counts.get(role, 0)
                pill_htmls.append(
                    f'<div class="jg-role-pill">'
                    f'<span class="jg-role-pill-name">{role}</span>'
                    f'<span class="jg-role-pill-count">{count}</span>'
                    f'</div>'
                )

            st.markdown(f'<div class="jg-role-grid">{"".join(pill_htmls)}</div>', unsafe_allow_html=True)
            
            # Throughput metrics
            stats = st.session_state.segmentation_stats
            w_sec = stats.get("wall_clock_time_sec", 0.0)
            cached_flag = " (cached)" if stats.get("cached") else ""
            st.markdown(
                f"""
                <div class="jg-mono" style="margin-top: 10px; font-size: 11px; color: #63687A;">
                    Throughput: {len(st.session_state.segmented_records)} sentences in {w_sec:.2f}s{cached_flag} · Token mean: {stats.get('mean_tokens_per_sentence', 'N/A')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Section B: Query Input & Role Mapping
    # -------------------------------------------------------------------------
    st.markdown('<div class="jg-card">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-bottom: 8px;">Query & Retrieval</div>',
        unsafe_allow_html=True,
    )

    query_input = st.text_input(
        "Ask a question about this judgment",
        value=st.session_state.current_query,
        placeholder="e.g. What did the petitioner argue regarding the limitation period?",
        disabled=not bool(st.session_state.doc_id),
        key="user_query_input",
    )

    col_btn, col_tag = st.columns([1, 3])
    with col_btn:
        ask_clicked = st.button(
            "Ask Question",
            type="primary",
            disabled=not bool(st.session_state.doc_id),
        )

    # Handle Query Execution
    if ask_clicked:
        if not query_input.strip():
            st.warning("Please enter a question.")
        else:
            q_clean = query_input.strip()
            st.session_state.current_query = q_clean

        # Role Mapping pre-check
        with st.spinner("Mapping query to rhetorical roles..."):
            try:
                role_mapping = map_query_to_roles(q_clean, mode="hybrid")
            except Exception as e:
                role_mapping = RoleMapping(roles=[], confidence=0.0, kind="whole_document", reason="Fallback")

        if role_mapping.kind == "out_of_scope":
            st.session_state.query_result = {
                "question": q_clean,
                "generation": "This query appears to be outside the scope of Indian legal judgments. Please ask a case-related question.",
                "mapped_roles": [],
                "role_kind": "out_of_scope",
                "role_confidence": role_mapping.confidence,
                "retrieved_chunks": [],
                "graded_chunks": [],
                "verified_context": [],
                "citations": [],
                "sources": [],
                "final_status": "out_of_scope",
                "trace": [],
            }
            st.session_state.verification_result = {
                "verified": False,
                "status": "out_of_scope",
                "score": 0.0,
                "reason": "Query out of legal scope.",
            }
        else:
            with st.spinner("Retrieving, grading relevance, and generating grounded answer..."):
                try:
                    result = run_query(
                        question=q_clean,
                        document_id=st.session_state.doc_id,
                        mode="role",
                    )
                    st.session_state.query_result = result

                    # Module 10: Online Quality Gate Verification
                    if result.get("final_status") == "no_answer":
                        ver_res = {
                            "verified": False,
                            "status": "no_answer",
                            "score": 0.0,
                            "reason": "No reliable answer found in document.",
                        }
                    elif result.get("final_status") == "abstained":
                        ver_res = {
                            "verified": True,
                            "status": "abstained",
                            "score": 1.0,
                            "reason": "Abstention response; verification skipped.",
                        }
                    else:
                        ver_res = verify_generation_faithfulness(
                            question=q_clean,
                            context_chunks=result.get("verified_context", []),
                            answer=result.get("generation", ""),
                        )
                    st.session_state.verification_result = ver_res

                except Exception as exc:
                    st.error(f"Pipeline execution failed: {exc}")
                    st.session_state.query_result = None
                    st.session_state.verification_result = None

    # Render Mapped Role Tag if a query result exists
    if st.session_state.query_result:
        res = st.session_state.query_result
        roles = res.get("mapped_roles", [])
        kind = res.get("role_kind", "")
        conf = res.get("role_confidence", 0.0)

        with col_tag:
            if kind == "out_of_scope":
                st.markdown(
                    '<span class="jg-tag jg-tag-none">OUT OF LEGAL SCOPE</span>',
                    unsafe_allow_html=True,
                )
            elif roles:
                tags = " ".join([get_role_tag_html(r) for r in roles])
                st.markdown(
                    f'<div style="padding-top: 6px;">'
                    f'<span class="jg-mono" style="margin-right: 6px;">Searching:</span>'
                    f'{tags}'
                    f'<span class="jg-mono" style="margin-left: 8px;">(conf: {conf:.2f})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="padding-top: 6px;">'
                    '<span class="jg-tag jg-tag-role">WHOLE DOCUMENT</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Section C & D: Retrieval and Grading Trace (Collapsible)
    # -------------------------------------------------------------------------
    if st.session_state.query_result:
        res = st.session_state.query_result
        graded = res.get("graded_chunks", [])
        retries = res.get("retry_count", 0)
        is_web = res.get("external_context", False)

        with st.expander("🔍 Pipeline Trace & Relevance Grading", expanded=False):
            st.markdown(
                f"""
                <div class="jg-mono" style="margin-bottom: 12px;">
                    <strong>Active Query:</strong> {res.get('question', '')}<br/>
                    <strong>Retries:</strong> {retries} / 2 &nbsp;|&nbsp; 
                    <strong>Relaxation Level:</strong> {res.get('relax_level', 0)} &nbsp;|&nbsp; 
                    <strong>Web Fallback:</strong> {'Active (Tavily)' if is_web else 'Inactive'}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not graded:
                explanation = get_trace_explanation(res)
                st.markdown(
                    f'<div class="jg-mono" style="color: #9AA1AE; font-size: 13px; line-height: 1.5; margin-bottom: 12px;">{explanation}</div>',
                    unsafe_allow_html=True,
                )
                
                # If unverified candidate chunks exist in state, display them as inspected & rejected
                candidate_chunks = res.get("retrieved_chunks", [])
                if candidate_chunks and res.get("role_kind") != "out_of_scope":
                    st.markdown(
                        '<div style="font-size: 12px; font-weight: 600; color: #63687A; text-transform: uppercase; letter-spacing: 0.02em; margin-top: 10px; margin-bottom: 6px;">Unverified Candidate Passages Inspected</div>',
                        unsafe_allow_html=True,
                    )
                    for idx, chunk in enumerate(candidate_chunks[:3], start=1):
                        role = chunk.get("role", chunk.get("metadata", {}).get("role", "None"))
                        cid = chunk.get("chunk_id", chunk.get("id", f"c_{idx}"))
                        text = chunk.get("text", "")
                        excerpt = text[:240].strip() + ("..." if len(text) > 240 else "")
                        ce_score = chunk.get("rerank_score")
                        score_str = f"CE score: {ce_score:.2f}" if ce_score is not None else ""
                        st.markdown(
                            f"""
                            <div class="jg-source-item" style="opacity: 0.85;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <div>
                                        <span class="jg-mono-score">[{idx}] {cid}</span> &nbsp;
                                        {get_role_tag_html(role)}
                                    </div>
                                    <div>
                                        <span class="jg-grade-incorrect">REJECTED</span>
                                        <span class="jg-mono-score" style="margin-left: 8px;">{score_str}</span>
                                    </div>
                                </div>
                                <div style="color: #9AA1AE; font-size: 11px; font-family: 'JetBrains Mono', monospace;">
                                    {excerpt}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    '<div style="font-size: 13px; font-weight: 600; color: #E7E9EE; margin-bottom: 8px;">Retrieved Passage Evaluation</div>',
                    unsafe_allow_html=True,
                )
                for idx, chunk in enumerate(graded, start=1):
                    grade = chunk.get("grade", "Incorrect")
                    conf = chunk.get("confidence") or chunk.get("score") or chunk.get("rerank_score", 0.0)
                    role = chunk.get("role", "None")
                    reason = chunk.get("reason", "")
                    cid = chunk.get("chunk_id", f"c_{idx}")
                    excerpt = chunk.get("text", "")[:280].strip() + ("..." if len(chunk.get("text", "")) > 280 else "")

                    st.markdown(
                        f"""
                        <div class="jg-source-item">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <div>
                                    <span class="jg-mono-score">[{idx}] {cid}</span> &nbsp;
                                    {get_role_tag_html(role)}
                                </div>
                                <div>
                                    {get_grade_badge_html(grade)}
                                    <span class="jg-mono-score" style="margin-left: 8px;">{conf:.2f}</span>
                                </div>
                            </div>
                            <div class="jg-mono" style="color: #E7E9EE; margin-bottom: 4px;">
                                <em>"{reason}"</em>
                            </div>
                            <div style="color: #9AA1AE; font-size: 12px; font-family: 'JetBrains Mono', monospace;">
                                {excerpt}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Node Timeline from Trace
            trace_events = res.get("trace", [])
            if trace_events:
                st.markdown(
                    '<div style="font-size: 13px; font-weight: 600; color: #E7E9EE; margin-top: 16px; margin-bottom: 6px;">Execution Timeline</div>',
                    unsafe_allow_html=True,
                )
                timeline_rows = []
                for ev in trace_events:
                    node = ev.get("node", "")
                    dur = ev.get("duration_ms", 0.0)
                    route = ev.get("route", "")
                    timeline_rows.append(
                        f"<tr>"
                        f"<td class='jg-mono' style='padding: 4px 8px; color: #4C6FFF;'>{node}</td>"
                        f"<td class='jg-mono' style='padding: 4px 8px;'>{dur:.1f}ms</td>"
                        f"<td class='jg-mono' style='padding: 4px 8px; color: #9AA1AE;'>{route}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"""
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="border-bottom: 1px solid #1E212B; color: #9AA1AE; text-align: left;">
                                <th style="padding: 4px 8px;">Node</th>
                                <th style="padding: 4px 8px;">Duration</th>
                                <th style="padding: 4px 8px;">Route</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(timeline_rows)}</tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # Section E: Answer and Verification Status
    # -------------------------------------------------------------------------
    if st.session_state.query_result:
        res = st.session_state.query_result
        ver = st.session_state.verification_result or {}
        generation = res.get("generation", "")
        sources = res.get("sources", [])
        final_status = res.get("final_status", "ready")

        st.markdown('<div class="jg-card">', unsafe_allow_html=True)
        
        # Header with Verification Badge
        ver_status = ver.get("status", "gate_disabled")
        badge_html = ""
        
        if ver_status == "passed":
            f_score = ver.get("faithfulness_score", ver.get("score", 1.0))
            r_score = ver.get("relevancy_score", 1.0)
            badge_html = (
                f'<span class="jg-badge-verified">✓ VERIFIED AGAINST SOURCE</span> '
                f'<span class="jg-mono" style="margin-left: 8px;">(Faithfulness: {f_score:.2f} · Relevancy: {r_score:.2f})</span>'
            )
        elif ver_status == "failed" or ver_status == "could not be fully verified":
            f_score = ver.get("faithfulness_score", ver.get("score", 0.0))
            badge_html = (
                f'<span class="jg-badge-unverified">⚠ COULD NOT BE FULLY VERIFIED</span> '
                f'<span class="jg-mono" style="margin-left: 8px;">(Score: {f_score:.2f})</span>'
            )
        elif ver_status == "not_verified":
            badge_html = (
                '<span class="jg-badge-unverified">⚠ UNVERIFIED (EVALUATION UNAVAILABLE)</span>'
            )
        elif ver_status == "gate_disabled":
            badge_html = (
                '<span class="jg-badge-offline">○ OFFLINE (QUALITY GATE DISABLED)</span>'
            )
        elif ver_status == "abstained":
            badge_html = (
                '<span class="jg-badge-offline">○ ABSTAINED</span>'
            )
        elif ver_status == "no_answer":
            badge_html = (
                '<span class="jg-badge-offline">○ NO ANSWER FOUND</span>'
            )

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 14px; font-weight: 600; color: #E7E9EE;">Answer</div>
                <div>{badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Answer Text
        st.markdown(
            f"""
            <div style="font-size: 15px; line-height: 1.6; color: #E7E9EE; margin-bottom: 16px;">
                {generation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Warnings or failure disclaimers
        if ver_status in ("failed", "could not be fully verified", "not_verified"):
            st.markdown(
                f"""
                <div style="background-color: rgba(240, 101, 75, 0.08); border-left: 3px solid #F0654B; padding: 10px 14px; border-radius: 4px; font-size: 12px; color: #F0654B; margin-bottom: 16px;">
                    <strong>Disclaimer:</strong> {ver.get('reason', 'This response could not be verified against the source context by the automated quality gate. Please review the retrieved source passages below.')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Source Citations
        if sources:
            st.markdown(
                '<div style="font-size: 13px; font-weight: 600; color: #9AA1AE; text-transform: uppercase; letter-spacing: 0.02em; margin-bottom: 8px;">Cited Source Passages</div>',
                unsafe_allow_html=True,
            )
            for src in sources:
                cit_num = src.get("citation_num") or src.get("citation_index", "")
                role = src.get("role", "Legal Passage")
                marker = src.get("first_para_marker", "N/A")
                cid = src.get("chunk_id", "")
                p_text = src.get("snippet") or src.get("text", "")
                if not p_text and cid and hasattr(get_default_store, "__call__"):
                    try:
                        vdoc = get_default_store().role_collection.get(ids=[cid])
                        if vdoc and vdoc.get("documents"):
                            p_text = vdoc["documents"][0][:280] + "..."
                    except Exception:
                        pass
                if not p_text:
                    p_text = "Source passage cited by answer."

                st.markdown(
                    f"""
                    <div class="jg-source-item">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <div>
                                <span class="jg-mono-score">[{cit_num}] {cid}</span> &nbsp;
                                {get_role_tag_html(role)}
                            </div>
                            <div class="jg-mono">
                                Para Marker: {marker}
                            </div>
                        </div>
                        <div class="jg-mono" style="font-size: 12px; color: #E7E9EE; white-space: pre-wrap; line-height: 1.5; margin-top: 4px;">{p_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif final_status == "ready" and not sources:
            st.markdown(
                '<div class="jg-mono" style="color: #63687A;">No explicit citation markers in answer.</div>',
                unsafe_allow_html=True,
            )

        # If verification failed or unverified, offer expandable raw retrieved chunks for manual inspection
        if ver_status in ("failed", "could not be fully verified"):
            ver_ctx = res.get("verified_context", [])
            with st.expander("🔎 Inspect All Retrieved Chunks (Manual Review)", expanded=True):
                for i, c in enumerate(ver_ctx, start=1):
                    st.markdown(
                        f"""
                        <div class="jg-source-item" style="border-left: 2px solid #F0654B;">
                            <div class="jg-mono-score">Passage {i} — {c.get('chunk_id', '')} ({c.get('role', '')})</div>
                            <div style="font-size: 12px; color: #E7E9EE; margin-top: 4px;">{c.get('text', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
