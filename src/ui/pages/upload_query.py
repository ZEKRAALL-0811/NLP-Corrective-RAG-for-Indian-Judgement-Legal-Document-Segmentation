"""
Page B — Upload & Query
Ingestion of PDF/TXT judgments, role classification,
grounded querying with confidence gauge and pipeline trace.
Preserves all backend functions and test assertions using module lookups.
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

import src.config as cfg
from src.config import CANONICAL_LABELS
from src.data.preprocessor import (
    ScannedPdfError,
    EmptyFileError,
    EncryptedPdfError,
    CorruptPdfError,
    TextPreservationError,
)
import src.models.segmenter as segmenter_mod
import src.rag.vectorstore as vectorstore_mod
import src.rag.graph as graph_mod
import src.rag.role_mapper as role_mapper_mod
import src.evaluation.quality_gate as quality_gate_mod

from src.ui.components import (
    get_role_tag_html,
    get_grade_badge_html,
    render_confidence_gauge,
    get_trace_explanation,
)
from src.ui.data_loader import get_available_samples


def render_upload_query_page():
    """Renders the Upload & Query page and Query Result view."""
    # -------------------------------------------------------------------------
    # 1. Document Ingestion Card
    # -------------------------------------------------------------------------
    st.markdown('<div class="jg-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="jg-card-title" style="margin-bottom: 12px;">Document Ingestion</div>',
        unsafe_allow_html=True,
    )

    col_up, col_doc_status = st.columns([1.6, 1.2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload judgment file",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            key="judgment_uploader",
        )

        # Pre-indexed benchmark sample selector
        avail_samples = get_available_samples()
        sample_keys = ["-- Select pre-indexed benchmark sample --"] + list(avail_samples.keys())

        # Check URL query param for sample
        default_idx = 0
        qp_sample = st.query_params.get("sample", "")
        if qp_sample and not st.session_state.get("doc_id"):
            for idx, k in enumerate(sample_keys):
                if qp_sample.lower() in k.lower():
                    default_idx = idx
                    break

        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            selected_sample = st.selectbox(
                "Or choose benchmark judgment:",
                options=sample_keys,
                index=default_idx,
                label_visibility="collapsed",
                key="sample_selector_box",
            )
        with col_s2:
            load_sample_btn = st.button("Load Sample", key="btn_load_sample")

    with col_doc_status:
        # Document loaded card or idle banner
        if st.session_state.get("doc_id"):
            stats = st.session_state.get("segmentation_stats", {})
            n_sents = stats.get("sentence_count", len(st.session_state.get("segmented_records", [])))
            doc_name = st.session_state.get("doc_name", "document.pdf")
            st.markdown(
                f"""
                <div style="background-color: #0E1118; border: 1px solid #262C38; border-radius: 6px; padding: 14px 16px;">
                    <div style="font-size: 11px; color: #8E97A8; text-transform: uppercase;">Active Document</div>
                    <div style="font-size: 14px; font-weight: 600; color: #E7E9EE; margin-top: 2px;">{doc_name}</div>
                    <div class="jg-mono" style="font-size: 11px; margin-top: 4px; color: #8E97A8;">
                        ID: {st.session_state.get('doc_id')} &nbsp;·&nbsp; {n_sents} sentences
                    </div>
                    <div style="margin-top: 8px;">
                        <span class="jg-grade-correct">Document ready for analysis</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background-color: #0E1118; border: 1px solid #262C38; border-radius: 6px; padding: 14px 16px;">
                    <div style="font-size: 11px; color: #8E97A8; text-transform: uppercase;">Active Document</div>
                    <div style="font-size: 13px; font-weight: 500; color: #8E97A8; margin-top: 4px;">No document loaded</div>
                    <div style="font-size: 11px; color: #63687A; margin-top: 4px;">
                        Upload a file or choose a benchmark judgment on the left.
                    </div>
                    <div style="margin-top: 8px;">
                        <span class="jg-badge-offline">IDLE</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Ingestion Actions
    if (load_sample_btn or default_idx > 0) and selected_sample in avail_samples:
        if st.session_state.get("doc_name") != selected_sample:
            s_path = avail_samples[selected_sample]
            file_bytes = s_path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            doc_id = s_path.stem
            with st.spinner(f"Loading {selected_sample}..."):
                records, stats = segmenter_mod.segment_document(s_path, use_cache=True, write_preview=False)
                st.session_state.file_hash = file_hash
                st.session_state.doc_id = doc_id
                st.session_state.doc_name = selected_sample
                st.session_state.segmented_records = records
                st.session_state.segmentation_stats = stats
                st.session_state.role_counts = stats.get("role_counts", {})
                st.session_state.query_result = None
                st.session_state.verification_result = None
                st.rerun()

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) == 0:
            st.error("Uploaded file is empty (0 bytes). Please upload a valid judgment document.")
        else:
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            if file_hash != st.session_state.get("file_hash"):
                with st.spinner("Preprocessing document and classifying rhetorical roles (InLegalBERT + Viterbi)..."):
                    try:
                        temp_dir = cfg.CACHE_DIR / "uploads"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        temp_path = temp_dir / uploaded_file.name
                        temp_path.write_bytes(file_bytes)

                        # 1. Segment sentences and smooth roles
                        records, stats = segmenter_mod.segment_document(temp_path, use_cache=True, write_preview=False)

                        if len(records) == 0:
                            st.warning("No sentences could be extracted from this document. Please check the file.")
                        else:
                            # 2. Idempotent ChromaDB vector ingestion
                            doc_id = Path(uploaded_file.name).stem
                            vstore = graph_mod.get_default_store()
                            vstore.ingest_document(
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
                    except EmptyFileError:
                        st.error("Uploaded file is empty (0 bytes). Please upload a valid judgment document.")
                    except EncryptedPdfError:
                        st.error("This PDF is password-protected or encrypted. Please upload an unencrypted judgment PDF.")
                    except CorruptPdfError as exc:
                        st.error(f"PDF structure is corrupted or unreadable: {exc}")
                    except TextPreservationError as exc:
                        st.error(f"Ingestion failed: {exc}")
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")

    # Display 7-Role Distribution if document is loaded
    if st.session_state.get("doc_id") and st.session_state.get("role_counts"):
        st.markdown(
            """
            <div style="margin-top: 14px; font-size: 11px; font-weight: 600; color: #8E97A8; text-transform: uppercase; letter-spacing: 0.03em;">
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

        stats = st.session_state.get("segmentation_stats", {})
        w_sec = stats.get("wall_clock_time_sec", 0.0)
        cached_flag = " (cached)" if stats.get("cached") else ""
        st.markdown(
            f"""
            <div class="jg-mono" style="margin-top: 8px; font-size: 11px; color: #63687A;">
                Throughput: {len(st.session_state.get('segmented_records', []))} sentences in {w_sec:.2f}s{cached_flag} · Token mean: {stats.get('mean_tokens_per_sentence', 'N/A')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Query Input & Action
    # -------------------------------------------------------------------------
    st.markdown('<div class="jg-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="jg-card-title" style="margin-bottom: 8px;">Query & Retrieval</div>',
        unsafe_allow_html=True,
    )

    query_input = st.text_input(
        "Ask a question about this judgment",
        value=st.session_state.get("current_query", ""),
        placeholder="Ask a question about this judgment...",
        disabled=not bool(st.session_state.get("doc_id")),
        key="user_query_input",
    )

    col_btn, col_tag = st.columns([1, 3])
    with col_btn:
        ask_clicked = st.button(
            "Ask Question",
            type="primary",
            disabled=not bool(st.session_state.get("doc_id")),
            key="btn_ask",
        )

    # Handle Query Execution
    if ask_clicked:
        if not query_input.strip():
            st.warning("Please enter a question.")
        else:
            q_clean = query_input.strip()
            st.session_state["current_query"] = q_clean

            # Role Mapping pre-check
            with st.spinner("Mapping query to rhetorical roles..."):
                try:
                    role_mapping = role_mapper_mod.map_query_to_roles(q_clean, mode="hybrid")
                except Exception:
                    role_mapping = role_mapper_mod.RoleMapping(roles=[], confidence=0.0, kind="whole_document", reason="Fallback")

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
                        result = graph_mod.run_query(
                            question=q_clean,
                            document_id=st.session_state.doc_id,
                            mode="role",
                        )
                        st.session_state.query_result = result

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
                            ver_res = quality_gate_mod.verify_generation_faithfulness(
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
    if st.session_state.get("query_result"):
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
    # 3. Query Results View
    # -------------------------------------------------------------------------
    if st.session_state.get("query_result"):
        res = st.session_state.query_result
        ver = st.session_state.get("verification_result") or {}
        generation = res.get("generation", "")
        sources = res.get("sources", [])
        final_status = res.get("final_status", "ready")
        ver_status = ver.get("status", "gate_disabled")

        st.markdown('<div class="jg-card">', unsafe_allow_html=True)

        # Header with Verification Badge & New Query button
        badge_html = ""
        if ver_status == "passed":
            f_score = ver.get("faithfulness_score", ver.get("score", 1.0))
            r_score = ver.get("relevancy_score", 1.0)
            badge_html = (
                f'<span class="jg-badge-verified">✓ VERIFIED AGAINST SOURCE</span> '
                f'<span class="jg-mono" style="margin-left: 8px;">(Faithfulness: {f_score:.2f} · Relevancy: {r_score:.2f})</span>'
            )
        elif ver_status in ("failed", "could not be fully verified"):
            f_score = ver.get("faithfulness_score", ver.get("score", 0.0))
            badge_html = (
                f'<span class="jg-badge-unverified">⚠ COULD NOT BE FULLY VERIFIED</span> '
                f'<span class="jg-mono" style="margin-left: 8px;">(Score: {f_score:.2f})</span>'
            )
        elif ver_status == "not_verified":
            badge_html = '<span class="jg-badge-unverified">⚠ UNVERIFIED (EVALUATION UNAVAILABLE)</span>'
        elif ver_status == "gate_disabled":
            badge_html = '<span class="jg-badge-offline">○ OFFLINE (QUALITY GATE DISABLED)</span>'
        elif ver_status == "abstained":
            badge_html = '<span class="jg-badge-offline">○ ABSTAINED</span>'
        elif ver_status == "no_answer":
            badge_html = '<span class="jg-badge-offline">○ NO ANSWER FOUND</span>'

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <div class="jg-card-title">Answer</div>
                <div>{badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Answer content & Confidence Gauge Layout
        col_ans, col_gauge = st.columns([2.2, 1.0])

        with col_ans:
            st.markdown(
                f"""
                <div style="font-size: 15px; line-height: 1.6; color: #E7E9EE; margin-bottom: 16px;">
                    {generation}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Failure / Unverified Disclaimer
            if ver_status in ("failed", "could not be fully verified", "not_verified"):
                st.markdown(
                    f"""
                    <div style="background-color: rgba(240, 101, 75, 0.08); border-left: 3px solid #F0654B; padding: 10px 14px; border-radius: 4px; font-size: 12px; color: #F0654B; margin-bottom: 16px;">
                        <strong>Disclaimer:</strong> {ver.get('reason', 'This response could not be verified against the source context by the automated quality gate. Please review the retrieved source passages below.')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_gauge:
            # Determine real confidence score from backend signals
            conf_val: Optional[float] = None
            caption_str = "Relevance & verification confidence score"

            if final_status in ("no_answer", "out_of_scope"):
                conf_val = 0.0
                caption_str = "No verified context found for query"
            elif ver_status == "passed":
                conf_val = ver.get("faithfulness_score", ver.get("score", 1.0))
                caption_str = "Verified by LLM faithfulness quality gate"
            elif ver_status in ("failed", "could not be fully verified"):
                conf_val = ver.get("faithfulness_score", ver.get("score", 0.25))
                caption_str = "Verification failed by quality gate"
            else:
                # Offline / gate disabled: extract top graded chunk confidence or role confidence
                graded_chunks = res.get("graded_chunks", [])
                correct_chunks = [c for c in graded_chunks if c.get("grade") == "Correct"]
                if correct_chunks and correct_chunks[0].get("confidence") is not None:
                    conf_val = float(correct_chunks[0]["confidence"])
                    caption_str = "Cross-encoder relevance confidence"
                elif res.get("role_confidence") is not None:
                    conf_val = float(res["role_confidence"])
                    caption_str = "Query role mapping confidence"

            st.markdown(render_confidence_gauge(conf_val, label="Confidence Score", caption=caption_str), unsafe_allow_html=True)

        # Cited Source Passages
        if sources:
            st.markdown(
                '<div style="font-size: 13px; font-weight: 600; color: #8E97A8; text-transform: uppercase; letter-spacing: 0.02em; margin-top: 14px; margin-bottom: 8px;">Cited Source Passages</div>',
                unsafe_allow_html=True,
            )
            for src in sources:
                cit_num = src.get("citation_num") or src.get("citation_index", "")
                role = src.get("role", "Legal Passage")
                marker = src.get("first_para_marker", "N/A")
                cid = src.get("chunk_id", "")
                p_text = src.get("snippet") or src.get("text", "")
                if not p_text and cid and hasattr(graph_mod.get_default_store, "__call__"):
                    try:
                        vdoc = graph_mod.get_default_store().role_collection.get(ids=[cid])
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
                '<div class="jg-mono" style="color: #63687A; margin-top: 8px;">No explicit citation markers in answer.</div>',
                unsafe_allow_html=True,
            )

        # Expandable manual inspection if verification failed
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

        # ---------------------------------------------------------------------
        # 4. Pipeline Trace & Relevance Grading (Collapsible)
        # ---------------------------------------------------------------------
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
                        f"<td class='jg-mono' style='padding: 4px 8px; color: #8E97A8;'>{route}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"""
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="border-bottom: 1px solid #262C38; color: #8E97A8; text-align: left;">
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
