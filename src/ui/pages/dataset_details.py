"""
Page E — Dataset Details
Structured inspection of actual benchmark datasets, gold question distributions,
and rhetorical role taxonomic definitions.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.config import CANONICAL_LABELS
from src.ui.data_loader import get_dataset_summary
from src.ui.components import get_role_tag_html


def make_clean_table_html(headers: List[str], rows: List[List[Tuple[str, str]]]) -> str:
    """Builds a single-line, unindented HTML table to guarantee markdown parser never emits <pre><code>."""
    th_html = "".join(f"<th>{h}</th>" for h in headers)
    tr_html = "".join(
        f"<tr>{''.join(f'<td{td[1]}>{td[0]}</td>' for td in r)}</tr>"
        for r in rows
    )
    return f'<table class="jg-table"><thead><tr>{th_html}</tr></thead><tbody>{tr_html}</tbody></table>'


def render_dataset_page():
    """Renders the Dataset Details page."""
    st.markdown(
        '<div class="jg-page-header">'
        '<h1 class="jg-page-title">Benchmark Dataset & Evaluation Suite</h1>'
        '<p class="jg-page-subtitle">Curated Indian legal judgments, grounded gold questions, and rhetorical role annotations.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    data_summary = get_dataset_summary()
    total_docs = data_summary["total_documents"]
    total_q = data_summary["total_questions"]
    total_unans = data_summary["total_unanswerable"]
    roles_count = data_summary["canonical_roles_count"]
    doc_comp = data_summary["doc_composition"]
    grounded_questions = data_summary["grounded_questions"]

    # -------------------------------------------------------------------------
    # 1. KPI Cards Row
    # -------------------------------------------------------------------------
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        st.markdown(
            f'<div class="jg-kpi-card">'
            f'<div class="jg-kpi-label">Benchmark Judgments</div>'
            f'<div class="jg-kpi-value">{total_docs}</div>'
            f'<div class="jg-kpi-subtext">Indian High Court judgments</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_k2:
        st.markdown(
            f'<div class="jg-kpi-card">'
            f'<div class="jg-kpi-label">Gold Questions</div>'
            f'<div class="jg-kpi-value">{total_q}</div>'
            f'<div class="jg-kpi-subtext">Grounded in judgment text</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_k3:
        st.markdown(
            f'<div class="jg-kpi-card">'
            f'<div class="jg-kpi-label">Unanswerable Tests</div>'
            f'<div class="jg-kpi-value">{total_unans}</div>'
            f'<div class="jg-kpi-subtext">For abstention audit</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_k4:
        st.markdown(
            f'<div class="jg-kpi-card">'
            f'<div class="jg-kpi-label">Rhetorical Roles</div>'
            f'<div class="jg-kpi-value">{roles_count}</div>'
            f'<div class="jg-kpi-subtext">Canonical legal labels</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Dataset Composition Table & Question Types Breakdown
    # -------------------------------------------------------------------------
    col_comp, col_types = st.columns([1.75, 1.25])

    with col_comp:
        st.markdown('<div class="jg-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="jg-card-header">'
            '<div class="jg-card-title">Dataset Composition</div>'
            '<div class="jg-card-caption">Gold benchmark judgment portfolio</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        doc_headers = ["Document", "Type / Court", "Pages", "Gold Qs", "Unans", "Description"]
        doc_rows = []
        for d in doc_comp:
            doc_rows.append([
                (d['doc_name'], ' class="jg-mono" style="font-weight: 600; color: #4C6FFF;"'),
                (d['doc_type'], ' style="font-size: 11px; color: #E7E9EE;"'),
                (str(d['pages']), ' class="jg-mono"'),
                (str(d['questions']), ' class="jg-mono" style="color: #3DD68C;"'),
                (str(d['unanswerable']), ' class="jg-mono" style="color: #F5B841;"'),
                (d['description'], ' style="font-size: 11px; color: #8E97A8;"'),
            ])

        st.markdown(make_clean_table_html(doc_headers, doc_rows), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_types:
        st.markdown('<div class="jg-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="jg-card-header">'
            '<div class="jg-card-title">Question Taxonomy</div>'
            '<div class="jg-card-caption">Rhetorical focus of benchmark questions</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        role_dist = data_summary.get("role_distribution", {})
        type_items = [
            ("Facts & Procedural History", role_dist.get("Facts", 4), "#38BDF8"),
            ("Legal Issues Framed", role_dist.get("Issue", 2), "#A78BFA"),
            ("Arguments of Petitioner", role_dist.get("Arguments of Petitioner", 4), "#60A5FA"),
            ("Arguments of Respondent", role_dist.get("Arguments of Respondent", 6), "#FB923C"),
            ("Judicial Reasoning", role_dist.get("Reasoning", 4), "#C084FC"),
            ("Final Decision & Relief", role_dist.get("Decision", 2), "#3DD68C"),
            ("Case Metadata / Bench", role_dist.get("None", 2), "#8E97A8"),
            ("Out of Scope / Adversarial", role_dist.get("OUT_OF_SCOPE", 5), "#F0654B"),
        ]

        list_items_html = "".join([
            f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #1E232F; font-size: 11px;">'
            f'<div style="display: flex; align-items: center; gap: 6px;">'
            f'<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {color};"></span>'
            f'<span style="color: #E7E9EE;">{name}</span>'
            f'</div>'
            f'<div class="jg-mono" style="font-weight: 600; color: #E7E9EE;">{count}</div>'
            f'</div>'
            for name, count, color in type_items
        ])
        st.markdown(f'<div>{list_items_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Canonical Rhetorical Roles Specification Table
    # -------------------------------------------------------------------------
    st.markdown('<div class="jg-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="jg-card-header">'
        '<div class="jg-card-title">Canonical Rhetorical Roles (InLegalBERT Annotation Schema)</div>'
        '<div class="jg-card-caption">7 distinct structural functions in Indian court judgments</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    schema_rows = [
        ("Facts", "Chronological background, procedural history, transactions, FIRs, notices, and evidentiary facts.", "Chronological grounding, locus standi, factual dispute identification."),
        ("Issue", "Substantive or procedural legal questions, points for determination, or constitutional issues framed by the court.", "Pinpointing exact questions of law or statutory interpretation."),
        ("Arguments of Petitioner", "Legal submissions, precedents, interpretations, and pleas advanced by appellant / petitioner counsel.", "Retrieving specific applicant arguments and counter-assertions."),
        ("Arguments of Respondent", "Rebuttals, defenses, counter-arguments, and justifications advanced by respondent / State / Revenue counsel.", "Analyzing State or opposing party defenses and statutory reliance."),
        ("Reasoning", "Analysis, statutory interpretation, jurisprudential doctrine, and evidentiary appraisal by the Bench.", "Understanding the ratio decidendi and judicial thought process."),
        ("Decision", "Operative orders, directions, writ disposal, appeal outcome (allowed/dismissed), cost orders.", "Extracting the final enforceable decree or outcome of the petition."),
        ("None", "Case title, bench names, advocate appearances, citation headers, footnotes, administrative text.", "Case metadata, judicial officer verification, procedural dates."),
    ]

    schema_headers = ["Canonical Role", "Structural Definition", "RAG Retrieval Value"]
    schema_table_rows = []
    for role, defn, obj in schema_rows:
        schema_table_rows.append([
            (get_role_tag_html(role), ' style="width: 220px; white-space: nowrap;"'),
            (defn, ' style="font-size: 12px; color: #E7E9EE; line-height: 1.4;"'),
            (obj, ' style="font-size: 11px; color: #8E97A8; line-height: 1.4;"'),
        ])

    st.markdown(make_clean_table_html(schema_headers, schema_table_rows), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Benchmark Question Explorer
    # -------------------------------------------------------------------------
    with st.expander("Inspect Benchmark Questions (Gold & Unanswerable)", expanded=False):
        st.markdown(
            '<div style="font-size: 12px; color: #8E97A8; margin-bottom: 10px;">Select a question below to copy or explore its ground truth citation.</div>',
            unsafe_allow_html=True,
        )

        q_headers = ["#", "Sample", "Question", "Target Role", "Gold Answer Hint"]
        q_table_rows = []
        for idx, q in enumerate(grounded_questions[:20], start=1):
            q_text = q.get("question", "")
            s = q.get("sample", "")
            role = q.get("role", "")
            hint = q.get("answer_hint", "")[:120] + "..." if len(q.get("answer_hint", "")) > 120 else q.get("answer_hint", "")
            q_table_rows.append([
                (str(idx), ' class="jg-mono"'),
                (s, ' class="jg-mono" style="color: #4C6FFF;"'),
                (q_text, ' style="font-size: 12px; color: #E7E9EE;"'),
                (get_role_tag_html(role), ''),
                (hint, ' class="jg-mono" style="font-size: 11px; color: #8E97A8;"'),
            ])

        st.markdown(make_clean_table_html(q_headers, q_table_rows), unsafe_allow_html=True)
