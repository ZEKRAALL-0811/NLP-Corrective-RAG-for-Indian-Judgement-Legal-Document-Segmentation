"""
Cross-Encoder Reranker & Relevance Scorer (Module 8 / Part A).

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers CrossEncoder, CPU)
to score (question, chunk text up to 512 tokens) for candidate chunks.
Provides:
- Singleton CPU CrossEncoder instance.
- score_chunks(question, chunks): attaches cross-encoder score to chunks.
- rerank_chunks(question, chunks, top_k): re-orders chunks by cross-encoder score.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import os
import re
import psutil

_CROSS_ENCODER_MODEL: Optional[Any] = None
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder():
    """Lazily loads and returns the singleton CPU CrossEncoder model."""
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER_MODEL = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME,
            device="cpu",
            max_length=512,
        )
    return _CROSS_ENCODER_MODEL


def get_cross_encoder_memory_footprint_mb() -> float:
    """Returns the RSS memory in MB of the current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def compute_evidence_bonus(
    question: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Computes an evidence-first score bonus for legal factual queries
    (case identity, challenged instruments, specific statutory rules/sections, dates).
    Ensures direct explicit legal evidence outranks generic argument discussions.
    """
    if not question or not text:
        return 0.0

    meta = metadata or {}
    bonus = 0.0
    q_lower = question.lower()
    t_lower = text.lower()

    # 1. Government Order / G.O. entity match
    if ("g.o" in q_lower or "government order" in q_lower) and ("g.o" in t_lower or "government order" in t_lower):
        if re.search(r"g\.?\s*o\.?(?:\s*\([a-z.]+\))?\s*(?:ms\.?\s*)?(?:no\.?\s*)?\d+", text, re.I):
            bonus += 4.0
        else:
            bonus += 1.5

    # 2. Specific Rules match
    if "rule" in q_lower and "rule" in t_lower:
        q_rules = re.findall(r"\b(?:rules?|rule)\s+(\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?)\b", q_lower)
        if q_rules:
            if any(r in t_lower for r in q_rules):
                bonus += 5.0
            elif re.search(r"\brules?\s+\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?\b", text, re.I):
                bonus += 2.0
        else:
            if re.search(r"\brules?\s+\d+[a-z]*(?:\s*(?:to|-)\s*\d+[a-z]*)?\b", text, re.I):
                bonus += 3.0
            else:
                bonus += 1.0

    # 3. Party identification match
    if re.search(r"\b(?:who\s+(?:is|was|are|were)\s+(?:the\s+)?(?:petitioner|appellant|respondent)|who\s+filed|which\s+(?:party|association|person|company|entity)\s+filed|name\s+(?:of\s+)?(?:the\s+)?(?:petitioner|appellant))\b", q_lower):
        if re.search(r"\b(?:petition(?:er)?\s+has\s+been\s+filed\s+by|filed\s+by\s+(?:the\s+)?|writ\s+petition\s+has\s+been\s+filed|petitioner\(s\)|appellant\(s\))\b", text, re.I):
            bonus += 4.0
        elif "petitioner" in t_lower or "appellant" in t_lower:
            bonus += 1.0

    # 4. Specific subject matter keyword match (e.g. green fund)
    if "green fund" in q_lower and "green fund" in t_lower:
        bonus += 2.0

    # 5. Introductory zone boost for case-identity evidence
    rel_pos = meta.get("rel_pos", 0.5)
    if rel_pos <= 0.08 and bonus > 0:
        bonus += 2.0

    return bonus


def score_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Scores a list of chunks against a question using the cross-encoder and evidence-first weighting.

    Args:
        question: User query string.
        chunks: List of retrieved chunk dictionaries.

    Returns:
        scored_chunks: Copy of chunks with 'rerank_score', 'ce_score', and 'evidence_bonus' attached.
        duration_ms: Wall-clock scoring latency in ms.
    """
    if not chunks:
        return [], 0.0

    model = get_cross_encoder()
    pairs = [(question, c.get("text", "")) for c in chunks]

    t0 = time.time()
    raw_scores = model.predict(pairs)
    duration_ms = (time.time() - t0) * 1000.0

    scored = []
    for c, score in zip(chunks, raw_scores):
        c_copy = dict(c)
        meta = c_copy.get("metadata", {})
        bon = compute_evidence_bonus(question, c_copy.get("text", ""), meta)
        c_copy["ce_score"] = float(score)
        c_copy["evidence_bonus"] = bon
        c_copy["rerank_score"] = float(score) + bon
        scored.append(c_copy)

    return scored, duration_ms


def rerank_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Scores and sorts chunks in descending order of cross-encoder score.

    Args:
        question: User query string.
        chunks: List of retrieved chunk dictionaries.
        top_k: Optional limit on returned chunks.

    Returns:
        reranked_chunks: Chunks sorted by 'rerank_score' descending.
        duration_ms: Total scoring duration in ms.
    """
    scored, duration_ms = score_chunks(question, chunks)
    reranked = sorted(scored, key=lambda x: x.get("rerank_score", -999.0), reverse=True)
    if top_k is not None:
        reranked = reranked[:top_k]
    return reranked, duration_ms
