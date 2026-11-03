"""
Reusable UI Components for JurisGraph Clinical Dark-Mode Workspace
Includes:
- Semicircular SVG Confidence Gauge
- Canonical Role Tags & Badges
- Execution Trace & Timeline Renderer
- Pipeline Architecture Visual Cards
"""

from typing import Dict, Any, List, Optional
import math
import textwrap
from src.config import CANONICAL_LABELS
from src.rag.graph import TAU_LOW, TAU_HIGH


def get_role_tag_html(role: str) -> str:
    """Returns styled HTML inline tag for canonical rhetorical roles."""
    clean = str(role).strip()
    tag_class = "jg-tag-none"
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


def render_confidence_gauge(
    score: Optional[float],
    label: str = "Confidence Score",
    caption: str = "Relevance & verification confidence score",
    width: int = 200,
    height: int = 120,
) -> str:
    """
    Renders an elegant, restrained SVG semicircular gauge.
    Requirements:
    - Clearly labeled 'Confidence Score'.
    - Displays actual numeric score.
    - Subtle solid-color gauge with muted neutral base.
    - If no valid score exists, displays 'Unavailable'.
    """
    if score is None or not isinstance(score, (int, float)) or math.isnan(score):
        display_score = "Unavailable"
        stroke_color = "#63687A"
        pct = 0.0
    else:
        score = max(0.0, min(1.0, float(score)))
        display_score = f"{score:.2f}"
        pct = score
        if score >= 0.80:
            stroke_color = "#3DD68C"  # Muted green
        elif score >= 0.60:
            stroke_color = "#38BDF8"  # Muted sky
        elif score >= 0.40:
            stroke_color = "#F5B841"  # Muted amber
        else:
            stroke_color = "#F0654B"  # Muted coral/red

    # Arc calculations: radius 70, center (100, 95)
    # Semicircle from angle -180 deg (x=30, y=95) to 0 deg (x=170, y=95)
    arc_length = math.pi * 70.0
    dash_offset = arc_length * (1.0 - pct)

    svg_html = textwrap.dedent(f"""
    <div class="jg-gauge-container">
        <div class="jg-gauge-label">{label}</div>
        <svg viewBox="0 0 200 115" width="{width}" height="{height}" style="overflow: visible;">
            <!-- Background Arc -->
            <path d="M 30 95 A 70 70 0 0 1 170 95" 
                  fill="none" 
                  stroke="#1E232F" 
                  stroke-width="12" 
                  stroke-linecap="round"/>
            <!-- Active Score Arc -->
            <path d="M 30 95 A 70 70 0 0 1 170 95" 
                  fill="none" 
                  stroke="{stroke_color}" 
                  stroke-width="12" 
                  stroke-linecap="round"
                  stroke-dasharray="{arc_length:.2f}"
                  stroke-dashoffset="{dash_offset:.2f}"
                  style="transition: stroke-dashoffset 400ms ease, stroke 300ms ease;"/>
            <!-- Value text -->
            <text x="100" y="85" 
                  text-anchor="middle" 
                  fill="#E7E9EE" 
                  font-family="'JetBrains Mono', monospace" 
                  font-size="24" 
                  font-weight="700">
                {display_score}
            </text>
            <text x="100" y="105" 
                  text-anchor="middle" 
                  fill="#8E97A8" 
                  font-family="'Inter', sans-serif" 
                  font-size="11">
                Scale: 0.0 – 1.0
            </text>
        </svg>
        <div class="jg-gauge-caption">{caption}</div>
    </div>
    """).strip()
    return svg_html


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
