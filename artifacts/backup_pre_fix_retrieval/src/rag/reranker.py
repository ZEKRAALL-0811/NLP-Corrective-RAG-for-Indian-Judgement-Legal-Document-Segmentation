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


def score_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Scores a list of chunks against a question using the cross-encoder.

    Args:
        question: User query string.
        chunks: List of retrieved chunk dictionaries.

    Returns:
        scored_chunks: Copy of chunks with 'rerank_score' attached.
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
        c_copy["rerank_score"] = float(score)
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
