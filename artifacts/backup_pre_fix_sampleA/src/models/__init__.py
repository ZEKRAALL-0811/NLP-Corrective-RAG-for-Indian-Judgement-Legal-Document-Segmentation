"""
Model modules: InLegalBERT classifier, training loop, and segmentation pipeline.
"""

from src.models.viterbi import smooth_roles
from src.models.segmenter import (
    LegalSegmenter,
    get_segmenter,
    segment_document,
    segment_sentences,
)

__all__ = [
    "smooth_roles",
    "LegalSegmenter",
    "get_segmenter",
    "segment_document",
    "segment_sentences",
]
