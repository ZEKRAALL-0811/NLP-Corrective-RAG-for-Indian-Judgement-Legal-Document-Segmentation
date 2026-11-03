"""
Page A — Home Dashboard
Landing view for JurisGraph with:
- Wordmark & descriptor
- Quick actions for document upload and querying
- 4 Core capability overview cards
- Example questions
- Current document status panel
- System dependency health indicators (real runtime checks)
- Real evaluation summary preview
"""

import streamlit as st
from typing import Dict, Any, Callable
from pathlib import Path

from src.ui.data_loader import check_system_health
from src.ui.components import get_role_tag_html


def render_home_page(navigate_fn: Callable[[str], None]):
    """Renders the Home Dashboard page."""
    # Hero Title & Subtitle
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 class="jg-page-title" style="font-size: 32px; font-weight: 700; margin-bottom: 4px;">JurisGraph</h1>
            <div style="font-size: 16px; font-weight: 500; color: #E7E9EE; margin-bottom: 6px;">
                Agentic Corrective RAG for Indian Legal Documents
            </div>
            <div style="font-size: 13px; color: #4C6FFF; font-weight: 500; letter-spacing: 0.02em; margin-bottom: 12px;">
                Rhetorical Role Segmentation &nbsp;·&nbsp; Grounded Answers &nbsp;·&nbsp; Source Attribution
            </div>
            <p style="font-size: 14px; color: #8E97A8; line-height: 1.6; max-width: 820px; margin-bottom: 0;">
                Upload Indian court judgments (PDF or TXT), explore rhetorical structure, and get grounded, 
                citation-backed answers using an agentic RAG pipeline with InLegalBERT role classification, 
                Cross-Encoder reranking, and self-corrective retrieval.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent Action Buttons
    col_act1, col_act2, col_act_spacer = st.columns([1.2, 1.4, 2.0])
    with col_act1:
        if st.button("Upload Document", type="primary", key="home_btn_upload"):
            navigate_fn("Upload & Query")
    with col_act2:
        if st.button("Ask a Question", type="secondary", key="home_btn_ask"):
            navigate_fn("Upload & Query")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 4 Overview Capability Cards
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.markdown(
            """
            <div class="jg-kpi-card">
                <div style="font-size: 11px; font-weight: 600; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                    Rhetorical Roles
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-bottom: 6px;">
                    Role Segmentation
                </div>
                <div style="font-size: 12px; color: #8E97A8; line-height: 1.4;">
                    Identify Facts, Issues, Arguments, Reasoning, and Decisions with InLegalBERT.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c2:
        st.markdown(
            """
            <div class="jg-kpi-card">
                <div style="font-size: 11px; font-weight: 600; color: #3DD68C; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                    Targeted Retrieval
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-bottom: 6px;">
                    Role-Aware Retrieval
                </div>
                <div style="font-size: 12px; color: #8E97A8; line-height: 1.4;">
                    Map legal questions to specific rhetorical roles to filter candidate context.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c3:
        st.markdown(
            """
            <div class="jg-kpi-card">
                <div style="font-size: 11px; font-weight: 600; color: #A78BFA; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                    Grounded Output
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-bottom: 6px;">
                    Corrective Synthesis
                </div>
                <div style="font-size: 12px; color: #8E97A8; line-height: 1.4;">
                    Cross-encoder grading with self-correction retry loop and source attribution.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c4:
        st.markdown(
            """
            <div class="jg-kpi-card">
                <div style="font-size: 11px; font-weight: 600; color: #F5B841; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                    Verification
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-bottom: 6px;">
                    Evaluation & Insights
                </div>
                <div style="font-size: 12px; color: #8E97A8; line-height: 1.4;">
                    Faithfulness verification, abstention testing, and per-class performance metrics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Example Questions Row
    st.markdown(
        """
        <div style="font-size: 12px; font-weight: 600; color: #8E97A8; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 8px;">
            Example Legal Questions
        </div>
        """,
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(4)
    examples = [
        "Who is the petitioner?",
        "What were the key arguments?",
        "What is the final decision?",
        "Explain the legal issue.",
    ]
    for idx, ex in enumerate(examples):
        with ex_cols[idx]:
            if st.button(ex, key=f"home_ex_{idx}", type="secondary"):
                st.session_state["current_query"] = ex
                navigate_fn("Upload & Query")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Two column layout: Active Document status on Left, System Health on Right
    col_doc, col_sys = st.columns([1.5, 1.2])

    with col_doc:
        st.markdown('<div class="jg-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="jg-card-title" style="margin-bottom: 12px;">Active Document Status</div>',
            unsafe_allow_html=True,
        )

        doc_id = st.session_state.get("doc_id")
        doc_name = st.session_state.get("doc_name")
        records = st.session_state.get("segmented_records", [])
        stats = st.session_state.get("segmentation_stats", {})
        n_sents = stats.get("sentence_count", len(records))

        if doc_id:
            st.markdown(
                f"""
                <div class="jg-status-banner loaded" style="margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 11px; color: #8E97A8; text-transform: uppercase;">Current Document</div>
                        <strong style="color: #E7E9EE; font-size: 14px;">{doc_name}</strong>
                        <div class="jg-mono" style="font-size: 11px; margin-top: 2px;">
                            ID: {doc_id} &nbsp;·&nbsp; {n_sents} sentences segmented
                        </div>
                    </div>
                    <div>
                        <span class="jg-grade-correct">READY FOR QUERY</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("Query Active Document", key="home_query_doc", type="primary"):
                    navigate_fn("Upload & Query")
            with col_b2:
                if st.button("Inspect Structure", key="home_inspect_doc", type="secondary"):
                    navigate_fn("Document Analysis")
        else:
            st.markdown(
                """
                <div class="jg-status-banner empty" style="margin-bottom: 12px;">
                    <div>
                        <div style="font-weight: 600; color: #E7E9EE; margin-bottom: 2px;">No document loaded</div>
                        <div style="font-size: 12px; color: #8E97A8;">
                            Upload an Indian court judgment (PDF/TXT) or select a pre-indexed sample.
                        </div>
                    </div>
                    <div>
                        <span class="jg-grade-incorrect">IDLE</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Go to Upload & Ingestion", key="home_goto_upload", type="primary"):
                navigate_fn("Upload & Query")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_sys:
        st.markdown('<div class="jg-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="jg-card-title" style="margin-bottom: 12px;">System & Service Health</div>',
            unsafe_allow_html=True,
        )

        health = check_system_health()

        # Ollama row
        ol = health.get("ollama", {})
        ol_badge = '<span class="jg-grade-correct">ONLINE</span>' if ol.get("is_ok") else '<span class="jg-badge-unverified">OFFLINE</span>'
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #1E232F;">
                <div>
                    <span style="font-size: 12px; font-weight: 600; color: #E7E9EE;">Ollama LLM</span>
                    <div class="jg-mono" style="font-size: 10px; color: #8E97A8;">{ol.get('model', 'llama3.2:3b')}</div>
                </div>
                <div>{ol_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ChromaDB row
        ch = health.get("chromadb", {})
        ch_badge = '<span class="jg-grade-correct">ONLINE</span>' if ch.get("is_ok") else '<span class="jg-badge-offline">IDLE</span>'
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #1E232F;">
                <div>
                    <span style="font-size: 12px; font-weight: 600; color: #E7E9EE;">ChromaDB Vector Store</span>
                    <div class="jg-mono" style="font-size: 10px; color: #8E97A8;">{ch.get('message', 'Ready')}</div>
                </div>
                <div>{ch_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # InLegalBERT row
        bert = health.get("inlegalbert", {})
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0;">
                <div>
                    <span style="font-size: 12px; font-weight: 600; color: #E7E9EE;">InLegalBERT + Viterbi</span>
                    <div class="jg-mono" style="font-size: 10px; color: #8E97A8;">{bert.get('message', 'Loaded')}</div>
                </div>
                <div><span class="jg-grade-correct">READY</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)


