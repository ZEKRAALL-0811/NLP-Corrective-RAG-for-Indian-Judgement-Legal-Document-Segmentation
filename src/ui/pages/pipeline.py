"""
Page C — RAG Pipeline
Interactive architectural flow diagram, technical stage breakdowns,
and real-time execution trace for JurisGraph.
"""

import streamlit as st
from typing import Dict, Any, List, Tuple

import src.config as cfg
from src.rag.graph import TAU_LOW, TAU_HIGH


def make_clean_table_html(headers: List[str], rows: List[List[Tuple[str, str]]]) -> str:
    """Builds a single-line, unindented HTML table to guarantee markdown parser never emits <pre><code>."""
    th_html = "".join(f"<th>{h}</th>" for h in headers)
    tr_html = "".join(
        f"<tr>{''.join(f'<td{td[1]}>{td[0]}</td>' for td in r)}</tr>"
        for r in rows
    )
    return f'<table class="jg-table"><thead><tr>{th_html}</tr></thead><tbody>{tr_html}</tbody></table>'


def render_pipeline_page():
    """Renders the dedicated RAG Pipeline visualization page."""
    st.markdown(
        '<div class="jg-page-header">'
        '<h1 class="jg-page-title">RAG Pipeline Architecture</h1>'
        '<p class="jg-page-subtitle">End-to-end flow from document ingestion to grounded answer generation.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 1. Five Main Stage Cards
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            '<div class="jg-pipe-stage-card">'
            '<div class="jg-pipe-stage-num">1</div>'
            '<div class="jg-pipe-stage-title">Document Ingestion</div>'
            '<div class="jg-pipe-stage-item">PDF/TXT extraction</div>'
            '<div class="jg-pipe-stage-item">Header/footer removal</div>'
            '<div class="jg-pipe-stage-item">Preservation validation</div>'
            '<div class="jg-pipe-stage-item">Sentence segmentation</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="jg-pipe-stage-card">'
            '<div class="jg-pipe-stage-num">2</div>'
            '<div class="jg-pipe-stage-title">Role Classification</div>'
            '<div class="jg-pipe-stage-item">InLegalBERT model</div>'
            '<div class="jg-pipe-stage-item">7 canonical roles</div>'
            '<div class="jg-pipe-stage-item">Viterbi smoothing</div>'
            '<div class="jg-pipe-stage-item">Chunk creation</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="jg-pipe-stage-card">'
            '<div class="jg-pipe-stage-num">3</div>'
            '<div class="jg-pipe-stage-title">Vector Store</div>'
            '<div class="jg-pipe-stage-item">ChromaDB persistence</div>'
            '<div class="jg-pipe-stage-item">Role-aware metadata</div>'
            '<div class="jg-pipe-stage-item">BGE-small embeddings</div>'
            '<div class="jg-pipe-stage-item">Position-based indexing</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            '<div class="jg-pipe-stage-card">'
            '<div class="jg-pipe-stage-num">4</div>'
            '<div class="jg-pipe-stage-title">Retrieval & Grading</div>'
            '<div class="jg-pipe-stage-item">Query role mapping</div>'
            '<div class="jg-pipe-stage-item">Role-filtered search</div>'
            '<div class="jg-pipe-stage-item">Cross-Encoder reranking</div>'
            '<div class="jg-pipe-stage-item">Self-correction loop</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            '<div class="jg-pipe-stage-card">'
            '<div class="jg-pipe-stage-num">5</div>'
            '<div class="jg-pipe-stage-title">Generation & Audit</div>'
            '<div class="jg-pipe-stage-item">Context assembly</div>'
            '<div class="jg-pipe-stage-item">Grounded generation</div>'
            '<div class="jg-pipe-stage-item">Strict source citation</div>'
            '<div class="jg-pipe-stage-item">Quality gate audit</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Visual Architecture Flow Diagram (Dual Branch)
    # -------------------------------------------------------------------------
    st.markdown('<div class="jg-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="jg-card-header">'
        '<div class="jg-card-title">End-to-End Pipeline Data Flow</div>'
        '<div class="jg-card-caption">Dual-branch architecture: Document Processing & Corrective Query Execution</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Branch A: Document Processing & Vector Indexing
    branch_a_boxes = [
        ("Court Judgment", "PDF / TXT Ingestion", False),
        ("Text & Structure", "Preservation Check", False),
        ("Sentence Splitting", "Spacy Legal Tokenizer", False),
        ("Role Classification", "InLegalBERT + Viterbi", False),
        ("Vector Store", "ChromaDB (BGE-small)", True),
    ]

    branch_a_items = []
    for idx, (title, sub, is_active) in enumerate(branch_a_boxes):
        active_cls = " active" if is_active else ""
        color_style = "color: #4C6FFF;" if is_active else "color: #E7E9EE;"
        sub_color = "color: #7C98FF;" if is_active else "color: #8E97A8;"
        branch_a_items.append(
            f'<div class="jg-flow-box{active_cls}" style="flex: 1 1 140px; min-width: 120px;">'
            f'<strong style="{color_style} font-size: 11px;">{title}</strong><br/>'
            f'<span class="jg-mono" style="font-size: 10px; {sub_color}">{sub}</span>'
            f'</div>'
        )
        if idx < len(branch_a_boxes) - 1:
            branch_a_items.append('<div class="jg-flow-arrow" style="padding: 0 4px;">→</div>')

    # Branch B: Corrective Query Retrieval & Generation
    branch_b_boxes = [
        ("Legal Query", "User Question", False),
        ("Query Routing", "Hybrid Role Router", False),
        ("Vector Retrieval", "ChromaDB Filtered", False),
        ("Rerank & Grading", "Cross-Encoder &tau;", False),
        ("Self-Correction", "CRAG Rewrite Loop", False),
        ("Grounded Answer", "llama3.2:3b Citations", True),
    ]

    branch_b_items = []
    for idx, (title, sub, is_active) in enumerate(branch_b_boxes):
        active_cls = " active" if is_active else ""
        color_style = "color: #3DD68C;" if is_active else "color: #E7E9EE;"
        sub_color = "color: #3DD68C;" if is_active else "color: #8E97A8;"
        branch_b_items.append(
            f'<div class="jg-flow-box{active_cls}" style="flex: 1 1 130px; min-width: 115px;">'
            f'<strong style="{color_style} font-size: 11px;">{title}</strong><br/>'
            f'<span class="jg-mono" style="font-size: 10px; {sub_color}">{sub}</span>'
            f'</div>'
        )
        if idx < len(branch_b_boxes) - 1:
            branch_b_items.append('<div class="jg-flow-arrow" style="padding: 0 4px;">→</div>')

    diagram_html = (
        '<div style="padding: 8px 0;">'
        '<div style="font-size: 11px; font-weight: 600; color: #8E97A8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">'
        'Branch A: Document Processing & Vector Indexing'
        '</div>'
        f'<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 20px;">'
        f'{"".join(branch_a_items)}'
        '</div>'
        '<div style="font-size: 11px; font-weight: 600; color: #8E97A8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">'
        'Branch B: Corrective Query Retrieval & Generation'
        '</div>'
        f'<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 6px;">'
        f'{"".join(branch_b_items)}'
        '</div>'
        '</div>'
    )

    st.markdown(diagram_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Live Execution Trace for Current Session (Real Data Only)
    # -------------------------------------------------------------------------
    st.markdown('<div class="jg-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="jg-card-header">'
        '<div class="jg-card-title">Live Pipeline Execution Trace</div>'
        '<div class="jg-card-caption">Actual execution metrics from the most recent query</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    query_res = st.session_state.get("query_result")
    if query_res and query_res.get("trace"):
        trace_events = query_res.get("trace", [])
        total_time_ms = sum(ev.get("duration_ms", 0.0) for ev in trace_events)

        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            st.markdown(
                f'<div class="jg-kpi-card">'
                f'<div class="jg-kpi-label">Active Question</div>'
                f'<div style="font-size: 13px; font-weight: 600; color: #E7E9EE; word-break: break-word;">'
                f'{query_res.get("question", "N/A")[:48]}...'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_t2:
            st.markdown(
                f'<div class="jg-kpi-card">'
                f'<div class="jg-kpi-label">Nodes Executed</div>'
                f'<div class="jg-kpi-value">{len(trace_events)}</div>'
                f'<div class="jg-kpi-subtext">Total pipeline steps</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_t3:
            st.markdown(
                f'<div class="jg-kpi-card">'
                f'<div class="jg-kpi-label">Trace Duration</div>'
                f'<div class="jg-kpi-value">{total_time_ms:.1f}ms</div>'
                f'<div class="jg-kpi-subtext">Wall-clock trace sum</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_t4:
            st.markdown(
                f'<div class="jg-kpi-card">'
                f'<div class="jg-kpi-label">Final Status</div>'
                f'<div class="jg-kpi-value" style="font-size: 18px; color: #3DD68C;">'
                f'{query_res.get("final_status", "ready").upper()}'
                f'</div>'
                f'<div class="jg-kpi-subtext">Retries: {query_res.get("retry_count", 0)} / 2</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        trace_headers = ["Node", "Status", "Duration", "Details & Decisions"]
        trace_rows = []
        for ev in trace_events:
            node_name = ev.get("node", "unknown")
            status = ev.get("status", "ok")
            dur = ev.get("duration_ms", 0.0)
            details = ev.get("details", {})

            det_parts = []
            for k, v in details.items():
                if k not in ["grades"]:
                    det_parts.append(f"{k}: {v}")
            det_summary = ", ".join(det_parts[:3]) if det_parts else "completed"

            trace_rows.append([
                (node_name, ' class="jg-mono" style="font-weight: 600; color: #4C6FFF;"'),
                (f'<span class="jg-grade-correct">{status.upper()}</span>', ''),
                (f"{dur:.1f}ms", ' class="jg-mono"'),
                (det_summary, ' style="font-size: 11px; color: #8E97A8;"'),
            ])

        st.markdown(make_clean_table_html(trace_headers, trace_rows), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background-color: #0E1118; border: 1px dashed #262C38; border-radius: 6px; padding: 24px; text-align: center;">'
            '<div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-bottom: 4px;">No live query trace in current session</div>'
            '<div style="font-size: 12px; color: #8E97A8; max-width: 500px; margin: 0 auto;">'
            'Execute a query on the <strong>Upload & Query</strong> page to view real-time node durations, '
            'routing decisions, and cross-encoder scores in this timeline.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Technical Deep Dive / Stage Details (Expandable)
    # -------------------------------------------------------------------------
    st.markdown(
        '<div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-top: 20px; margin-bottom: 10px;">'
        'Stage Specifications & Model Configurations'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Technical Details: InLegalBERT & Viterbi HMM Smoothing", expanded=False):
        st.markdown(
            '<div class="jg-mono" style="font-size: 12px; line-height: 1.6; color: #E7E9EE;">'
            '<strong>Base Model:</strong> <code>law-ai/InLegalBERT</code> fine-tuned on Indian Supreme Court / High Court rhetorical roles.<br/>'
            '<strong>Canonical Labels (7):</strong> Facts, Issue, Arguments of Petitioner, Arguments of Respondent, Reasoning, Decision, None.<br/>'
            '<strong>Viterbi HMM:</strong> Enforces legal structural priors. E.g., Facts preceding Issue, Arguments preceding Reasoning, Reasoning preceding Decision.<br/>'
            '<strong>Tau Threshold:</strong> 0.0 – 0.15 thresholding for probability smoothing.<br/>'
            '<strong>Implementation:</strong> <code>src/models/segmenter.py</code>'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Technical Details: ChromaDB Vector Store & Embeddings", expanded=False):
        st.markdown(
            f'<div class="jg-mono" style="font-size: 12px; line-height: 1.6; color: #E7E9EE;">'
            f'<strong>Vector Store:</strong> ChromaDB persistent storage at <code>{cfg.CHROMA_DIR}</code>.<br/>'
            f'<strong>Embedding Model:</strong> <code>BAAI/bge-small-en-v1.5</code> (384-dimensional dense semantic vectors).<br/>'
            f'<strong>Indexing Metadata:</strong> document_id, role, sentence_idx, start_char, end_char, paragraph_marker, page_num.<br/>'
            f'<strong>Chunking Strategy:</strong> Role-boundary-aware chunking preserving rhetorical transitions.<br/>'
            f'<strong>Implementation:</strong> <code>src/rag/vectorstore.py</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Technical Details: Cross-Encoder Reranking & Self-Correction Guardrails", expanded=False):
        st.markdown(
            f'<div class="jg-mono" style="font-size: 12px; line-height: 1.6; color: #E7E9EE;">'
            f'<strong>Reranker:</strong> <code>cross-encoder/ms-marco-MiniLM-L-6-v2</code>.<br/>'
            f'<strong>Thresholds:</strong> &tau;<sub>low</sub> = {TAU_LOW} (below this, chunks are discarded), &tau;<sub>high</sub> = {TAU_HIGH} (verified context).<br/>'
            f'<strong>Self-Correction Loop:</strong> When top score &lt; &tau;<sub>low</sub>, query is rewritten with legal synonyms and retry count increments.<br/>'
            f'<strong>Budget Guardrails:</strong> MAX_RETRIES = 2, MAX_LLM_CALLS = 6, MAX_WALL_CLOCK = 150.0s.<br/>'
            f'<strong>Abstention:</strong> When retries exhaust without reliable context, returns documented no-answer state.<br/>'
            f'<strong>Implementation:</strong> <code>src/rag/graph.py</code> and <code>src/rag/grader.py</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
