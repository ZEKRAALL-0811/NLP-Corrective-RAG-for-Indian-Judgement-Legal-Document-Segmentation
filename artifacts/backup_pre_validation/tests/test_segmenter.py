"""
Unit and Integration Tests for Module 3: Segmentation Pipeline

Tests:
1. Preprocessor: conservative text cleaning, page-number stripping, hyphenation fix.
2. Repeating header/footer removal on synthetic multi-page document.
3. Paragraph line-joining before sentence segmentation.
4. Early ScannedPdfError exception guard.
5. spaCy sentencizer: sentence boundary splitting on legal texts.
6. Segmenter: security assertion guarding against smoke-test weights.
7. Output record validation: required JSON keys, probability sum to 1.0, valid canonical labels.
8. Caching & preview generation.
"""

from pathlib import Path
import pytest
import numpy as np

from src.config import CANONICAL_LABELS
from src.data.preprocessor import (
    clean_legal_text,
    remove_headers_and_footers,
    join_paragraph_lines,
    split_sentences,
    preprocess_document,
    ScannedPdfError,
)
from src.models.segmenter import (
    LegalSegmenter,
    segment_sentences,
    save_human_readable_preview,
)


def test_clean_legal_text_hyphens_and_page_numbers():
    raw_text = """
    IN THE SUPREME COURT OF INDIA
    Page 1 of 12
    The appellant filed an appli-
    cation challenging the juris-
    diction of the tribunal.
    12
    - 14 -
    [Page 3]
    The High Court dismissed the writ petition.
    """
    cleaned, _ = clean_legal_text(raw_text)

    # Obvious page numbers stripped
    assert "Page 1 of 12" not in cleaned
    assert "\n12\n" not in cleaned
    assert "- 14 -" not in cleaned
    assert "[Page 3]" not in cleaned

    # Hyphenated line breaks resolved
    assert "application" in cleaned
    assert "jurisdiction" in cleaned


def test_remove_headers_and_footers_synthetic_multipage():
    # Construct a 5-page document with repeated headers and footers
    pages = []
    for p in range(1, 6):
        pages.append(f"""https://indiankanoon.org/doc/1234567/ {p}/5
IN THE SUPREME COURT OF INDIA
[Cites 12, Cited by 2]
This is genuine judgment paragraph on page {p}.
The High Court held that the prosecution proved its case.
Indian Kanoon - http://indiankanoon.org/doc/1234567/
CASE RECAST AI
Page {p} of 5""")

    cleaned_pages, stats = remove_headers_and_footers(pages, freq_threshold=0.40)

    assert stats["num_pages"] == 5
    assert stats["lines_removed"] > 0
    assert len(stats["repeating_headers"]) > 0

    # Ensure repeating headers were stripped from all pages
    for cp in cleaned_pages:
        assert "https://indiankanoon.org/doc/1234567/" not in cp
        assert "[Cites 12, Cited by 2]" not in cp
        assert "CASE RECAST AI" not in cp
        # Middle content must be preserved!
        assert "This is genuine judgment paragraph" in cp
        assert "The High Court held that the prosecution proved its case." in cp


def test_paragraph_line_joining():
    text = """First line of paragraph one.
Second line of paragraph one.
Third line of paragraph one.

First line of paragraph two.
Second line of paragraph two."""

    joined = join_paragraph_lines(text)
    paragraphs = joined.split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0] == "First line of paragraph one. Second line of paragraph one. Third line of paragraph one."
    assert paragraphs[1] == "First line of paragraph two. Second line of paragraph two."


def test_scanned_pdf_error():
    # Verify exception is a subclass of Exception
    assert issubclass(ScannedPdfError, Exception)


def test_spacy_sentencizer_split():
    text = (
        "The appellant and the deceased were married on 12th May 1998. "
        "The question arises whether the conviction is sustainable. "
        "We find no merit in the appeal."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The appellant and the deceased were married on 12th May 1998."
    assert sentences[1] == "The question arises whether the conviction is sustainable."
    assert sentences[2] == "We find no merit in the appeal."


def test_preprocess_document_stats():
    text = "The appeal is allowed.\n\nThe judgment of the High Court is set aside."
    sentences, stats = preprocess_document(text)
    assert len(sentences) == 2
    assert stats["sentence_count"] == 2
    assert stats["mean_tokens_per_sentence"] > 0
    assert "header_removal" in stats
    assert "legalseg_avg_sentences" not in stats


def test_party_details_override():
    from src.models.segmenter import is_party_details
    party_line = "C.K.Babu, S/o C.Kothandapani, No.12, Second Street, Chennai - 600 079. ... 11 th Respondent"
    assert is_party_details(party_line) is True

    substantive_line = "Learned counsel for the Respondent vehemently argued that no cause of action arose."
    assert is_party_details(substantive_line) is False


def test_list_marker_backward_merge_exemption():
    from src.data.preprocessor import is_backward_merge_candidate
    list_item = "e. The appointment of the respondent was contrary to rules."
    assert is_backward_merge_candidate(list_item) is False

    short_frag = "and dismissed."
    assert is_backward_merge_candidate(short_frag) is True


def test_segmenter_smoke_test_guard():
    with pytest.raises(ValueError, match="Smoke-test weights"):
        LegalSegmenter(model_path="artifacts/model_checkpoint/best_model_smoke_test")



def test_segment_sentences_output_format():
    test_sentences = [
        "The appellant and the deceased were married on 12th May 1998.",
        "Learned senior counsel for the petitioner contended that the arrest was illegal.",
        "In the result, the appeal is dismissed with costs.",
    ]
    records = segment_sentences(test_sentences, batch_size=2)
    assert len(records) == 3

    for i, rec in enumerate(records):
        assert rec["sentence_idx"] == i
        assert rec["text"] == test_sentences[i]
        assert rec["raw_role"] in CANONICAL_LABELS
        assert rec["role"] in CANONICAL_LABELS
        assert 0.0 <= rec["raw_confidence"] <= 1.0
        assert len(rec["role_probs"]) == 7
        assert np.isclose(sum(rec["role_probs"]), 1.0, atol=1e-4)


def test_preview_writer(tmp_path: Path):
    sample_records = [
        {
            "sentence_idx": 0,
            "text": "The appellant filed an appeal.",
            "role": "Facts",
            "role_probs": [0.85, 0.02, 0.03, 0.02, 0.03, 0.02, 0.03],
        }
    ]
    out_file = tmp_path / "test_preview.txt"
    save_human_readable_preview(sample_records, out_file)
    content = out_file.read_text(encoding="utf-8")
    assert "[0] Facts (85.0%) The appellant filed an appeal." in content


if __name__ == "__main__":
    pytest.main(["-v", __file__])
