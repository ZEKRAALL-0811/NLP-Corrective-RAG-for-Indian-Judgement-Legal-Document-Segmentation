"""
Module 2: InLegalBERT Classifier
Wraps law-ai/InLegalBERT with a 7-class sequence classification head.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import BASE_MODEL_NAME, NUM_LABELS, LABEL2ID, ID2LABEL


def load_tokenizer(model_name: str = BASE_MODEL_NAME) -> AutoTokenizer:
    """Load the InLegalBERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_model(model_name: str = BASE_MODEL_NAME, num_labels: int = NUM_LABELS):
    """
    Load InLegalBERT with a sequence classification head.
    Configures label2id/id2label so the checkpoint is self-documenting.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,  # classifier head is new
    )
    return model
