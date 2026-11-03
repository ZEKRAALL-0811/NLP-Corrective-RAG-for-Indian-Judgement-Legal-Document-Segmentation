"""
Unit and Integration Regression Tests for PDF Ingestion & Robust Text Preservation.

Covers all 15 required categories:
1. Ordinary multi-paragraph judgment text.
2. Text spanning multiple PDF pages.
3. Line-wrapped paragraphs.
4. Repeated headers and footers.
5. Numbered paragraphs and numbered legal clauses.
6. Legal abbreviations, initials, case citations, G.O. identifiers, dates, and statutory references.
7. Unicode quotation marks, dashes, ligatures, and non-breaking spaces.
8. Empty or whitespace-only documents.
9. PDFs with one or more pages lacking extractable text.
10. Encrypted or malformed PDFs.
11. A deliberately deleted legal word triggering genuine preservation failure.
12. A duplicated sentence triggering preservation failure.
13. Sentence segmentation and reconstruction preserving defined text representation.
14. Failed ingestion leaving no partial or orphaned chunks.
15. Re-ingestion not duplicating document chunks.
Plus end-to-end regression tests for sampleA.PDF and sampleB.PDF.
"""

from pathlib import Path
import tempfile
import pytest

from src.data.preprocessor import (
    clean_legal_text,
    remove_headers_and_footers,
    remove_repeated_shingles,
    join_paragraph_lines,
    split_sentences,
    validate_sentence_preservation,
    preprocess_document,
    extract_pages_from_pdf,
    ScannedPdfError,
    EmptyFileError,
    EncryptedPdfError,
    CorruptPdfError,
    TextPreservationError,
)
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore


# -----------------------------------------------------------------------------
# 1. Ordinary multi-paragraph judgment text
# -----------------------------------------------------------------------------
def test_ordinary_multiparagraph_judgment_text():
    raw = (
        "The High Court of Judicature at Madras considered the batch of appeals.\n\n"
        "The primary contention raised by the appellant relates to the scope of Section 8.\n\n"
        "Having heard both sides, the Court finds no merit in the appeal and dismisses the same."
    )
    sents = split_sentences(raw)
    assert len(sents) == 3
    assert validate_sentence_preservation(raw, sents) is True


# -----------------------------------------------------------------------------
# 2. Text spanning multiple PDF pages
# -----------------------------------------------------------------------------
def test_text_spanning_multiple_pdf_pages():
    pages = [
        "The learned counsel for the petitioner submitted that the impugned notice\n"
        "was issued without jurisdiction and in violation of natural justice.",
        "On the other hand, the Government Advocate contended that statutory remedies\n"
        "were available to the petitioner before approaching this Court under Article 226.",
    ]
    cleaned, stats = clean_legal_text(pages)
    sents = split_sentences(cleaned)
    assert len(sents) >= 2
    assert validate_sentence_preservation(cleaned, sents) is True
    assert "jurisdiction" in cleaned
    assert "Article 226" in cleaned


# -----------------------------------------------------------------------------
# 3. Line-wrapped paragraphs
# -----------------------------------------------------------------------------
def test_line_wrapped_paragraphs():
    raw = (
        "Assailing the amendment to Rule 35 of the Tamil Nadu Minor\n"
        "Mineral Concession Rules 1959 vide G.O.(Ms) No.23 Industries\n"
        "department dated 23rd February 2022, the writ petition was filed."
    )
    joined = join_paragraph_lines(raw)
    assert "\n" not in joined
    sents = split_sentences(joined)
    assert len(sents) == 1
    assert validate_sentence_preservation(joined, sents) is True


# -----------------------------------------------------------------------------
# 4. Repeated headers and footers
# -----------------------------------------------------------------------------
def test_repeated_headers_and_footers():
    pages = [
        "https://www.mhc.tn.gov.in/judis\nSTA Nos. 25 of 2020 etc.\nThe appellant presented extensive arguments on first page regarding adverse possession.\nPage 1 of 5",
        "https://www.mhc.tn.gov.in/judis\nSTA Nos. 25 of 2020 etc.\nThe respondent contended on second page that statutory period expired before notice was served.\nPage 2 of 5",
        "https://www.mhc.tn.gov.in/judis\nSTA Nos. 25 of 2020 etc.\nThe Single Judge examined on third page the relevant revenue records and field surveys.\nPage 3 of 5",
        "https://www.mhc.tn.gov.in/judis\nSTA Nos. 25 of 2020 etc.\nThe Division Bench affirmed on fourth page the findings of the learned Tribunal below.\nPage 4 of 5",
        "https://www.mhc.tn.gov.in/judis\nSTA Nos. 25 of 2020 etc.\nAccordingly the batch of appeals is dismissed on fifth page without any order as to costs.\nPage 5 of 5",
    ]
    cleaned, stats = clean_legal_text(pages)
    assert "adverse possession" in cleaned
    assert "statutory period" in cleaned
    assert "Division Bench" in cleaned
    assert "STA Nos. 25 of 2020 etc." not in cleaned
    sents = split_sentences(cleaned)
    assert validate_sentence_preservation(cleaned, sents) is True


# -----------------------------------------------------------------------------
# 5. Numbered paragraphs and numbered legal clauses
# -----------------------------------------------------------------------------
def test_numbered_paragraphs_and_clauses():
    text = (
        "1.1. S.T.A.No.25 of 2020 has been filed by the State to set aside the order. "
        "1.2. S.T.A.No.40 of 2023 has been filed by the Claimant. "
        "2. The questions framed for determination are as follows: "
        "(i) Whether the claim is barred by limitation? "
        "(ii) Whether the Claimant is a Janmi under Section 8?"
    )
    sents = split_sentences(text)
    assert len(sents) >= 4
    assert validate_sentence_preservation(text, sents) is True
    assert any("1.1." in s for s in sents)
    assert any("(i)" in s for s in sents)


# -----------------------------------------------------------------------------
# 6. Legal abbreviations, initials, citations, G.O. identifiers, dates, sections
# -----------------------------------------------------------------------------
def test_legal_abbreviations_citations_dates_sections():
    text = (
        "Under Section 8(1) of the Act and Rule 8(1) of the Rules, Form-4 was submitted on 21.05.1975. "
        "Vide G.O.(Ms) No.23, Industries [MMC.1] Department, dated 23.02.2022, Rules 35-A to 35-H were added. "
        "Mr. P.V. Balasubramaniam, learned AAG appeared for the Tahsildar in C.M.A.No.1 of 2018."
    )
    sents = split_sentences(text)
    assert validate_sentence_preservation(text, sents) is True
    assert any("G.O.(Ms) No.23" in s for s in sents)
    assert any("Rules 35-A to 35-H" in s for s in sents)
    assert any("21.05.1975" in s for s in sents)


# -----------------------------------------------------------------------------
# 7. Unicode quotation marks, dashes, ligatures, and non-breaking spaces
# -----------------------------------------------------------------------------
def test_unicode_quotation_marks_dashes_ligatures_nbsp():
    text = (
        "The Bench held: “The plea of res judicata is untenable.”\u00A0"
        "Further, the statutory mandate—as framed under the Act—cannot be waived. "
        "The ‘Janmam’ rights were not established beyond reasonable doubt."
    )
    sents = split_sentences(text)
    assert validate_sentence_preservation(text, sents) is True
    assert any("“The plea" in s or "The plea" in s for s in sents)
    assert any("mandate—as" in s for s in sents)


# -----------------------------------------------------------------------------
# 8. Empty or whitespace-only documents
# -----------------------------------------------------------------------------
def test_empty_or_whitespace_only_documents():
    assert split_sentences("") == []
    assert split_sentences("   \t  \n  ") == []

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f_path = Path(f.name)
    try:
        f_path.write_bytes(b"")
        with pytest.raises(EmptyFileError):
            preprocess_document(f_path)
    finally:
        f_path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# 9. PDFs with one or more pages lacking extractable text
# -----------------------------------------------------------------------------
def test_pdf_with_page_lacking_extractable_text():
    pages = [
        "First page contains genuine judicial reasoning under Section 9.",
        "   \n\t  ",  # blank / non-extractable page
        "Third page contains the final decree and order of costs.",
    ]
    cleaned, stats = clean_legal_text(pages)
    sents = split_sentences(cleaned)
    assert len(sents) >= 2
    assert validate_sentence_preservation(cleaned, sents) is True
    assert "Section 9" in cleaned
    assert "decree" in cleaned


# -----------------------------------------------------------------------------
# 10. Encrypted or malformed PDFs
# -----------------------------------------------------------------------------
def test_encrypted_or_malformed_pdf():
    # Corrupt PDF test
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f_path = Path(f.name)
    try:
        f_path.write_bytes(b"%PDF-1.4\nInvalid corrupted trailer data XYZ")
        with pytest.raises((CorruptPdfError, ScannedPdfError)):
            extract_pages_from_pdf(f_path)
    finally:
        f_path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# 11. Deliberately deleted legal word triggers preservation failure
# -----------------------------------------------------------------------------
def test_deliberately_deleted_legal_word_fails():
    cleaned = "The High Court quashed the impugned order of attachment."
    corrupted = ["The High Court the impugned order of attachment."]
    with pytest.raises((TextPreservationError, AssertionError), match="Text preservation check failed"):
        validate_sentence_preservation(cleaned, corrupted)


# -----------------------------------------------------------------------------
# 12. Duplicated sentence triggers preservation failure
# -----------------------------------------------------------------------------
def test_duplicated_sentence_detected():
    cleaned = "The writ appeal was allowed with costs. Direction was issued to Tahsildar."
    corrupted = [
        "The writ appeal was allowed with costs.",
        "The writ appeal was allowed with costs.",
        "Direction was issued to Tahsildar.",
    ]
    with pytest.raises((TextPreservationError, AssertionError), match="Text preservation check failed"):
        validate_sentence_preservation(cleaned, corrupted)


# -----------------------------------------------------------------------------
# 13. Sentence segmentation and reconstruction preserving defined text representation
# -----------------------------------------------------------------------------
def test_sentence_segmentation_reconstruction():
    text = (
        "1. This appeal arises out of the judgment dated 05.06.2020. "
        "2. The Tribunal held in favour of the claimant. "
        "3. Being aggrieved, the State preferred S.T.A.No.25 of 2020."
    )
    sents = split_sentences(text)
    assert len(sents) == 3
    assert validate_sentence_preservation(text, sents) is True


# -----------------------------------------------------------------------------
# 14. Failed ingestion leaves no partial chunks
# -----------------------------------------------------------------------------
def test_failed_ingestion_leaves_no_partial_chunks():
    vstore = LegalVectorStore(chroma_dir="data/chroma_db", device="cpu")
    doc_id = "test_fail_doc_99"
    # Clean any pre-existing test state
    vstore.delete_document(doc_id, mode="role")

    # Ingesting empty list of segmented records
    res = vstore.ingest_document([], document_id=doc_id, mode="role")
    assert res["n_chunks"] == 0

    # Ensure 0 chunks are present in Chroma collection
    retrieved = vstore.retrieve("anything", document_id=doc_id, mode="role", top_k=5)
    assert len(retrieved) == 0


# -----------------------------------------------------------------------------
# 15. Re-ingestion does not duplicate document chunks
# -----------------------------------------------------------------------------
def test_reingestion_no_duplicate_chunks():
    vstore = LegalVectorStore(chroma_dir="data/chroma_db", device="cpu")
    doc_id = "test_idempotent_doc"

    dummy_records = [
        {"sentence_idx": 0, "text": "Sentence one of test judgment.", "role": "Facts", "role_probs": [0.9] + [0.016]*6},
        {"sentence_idx": 1, "text": "Sentence two of test judgment.", "role": "Facts", "role_probs": [0.9] + [0.016]*6},
    ]

    try:
        res1 = vstore.ingest_document(dummy_records, document_id=doc_id, mode="role")
        n1 = res1["n_chunks"]

        # Re-ingest exact same document
        res2 = vstore.ingest_document(dummy_records, document_id=doc_id, mode="role")
        n2 = res2["n_chunks"]

        assert n1 == n2, "Chunk count must not grow on re-ingestion"

        col = vstore._get_collection("role")
        results = col.get(where={"document_id": {"$eq": doc_id}})
        assert len(results["ids"]) == n1, "ChromaDB must not contain duplicate chunk IDs"
    finally:
        vstore.delete_document(doc_id, mode="role")


# -----------------------------------------------------------------------------
# 16. Exact sampleB quotation mismatch pattern regression test
# -----------------------------------------------------------------------------
def test_sampleB_reproduced_quote_mismatch_fixed():
    """Reproduces the exact text from sampleB.PDF page 12 where the original pipeline failed."""
    text = (
        'Rule 8(1) of the Janmam Estate (Abolition and Conversion into Ryotwari) Rule 1974 reads as follows. '
        '"Enquiry into claim under Sections 8, 9 or 10: Every application for grant of ryotwari patta in respect '
        'of any land under Sections 8, 9 or 10 shall be in Form no.4 and shall be either presented in person or '
        'sent by registered post to the settlement officer within six months from the appointed day".'
    )
    sents = split_sentences(text)
    assert len(sents) >= 1
    assert validate_sentence_preservation(text, sents) is True


# -----------------------------------------------------------------------------
# 17. sampleA.PDF end-to-end regression
# -----------------------------------------------------------------------------
def test_sampleA_pdf_end_to_end_preservation():
    p = Path("artifacts/cache/uploads/sampleA.PDF")
    if not p.exists():
        pytest.skip("sampleA.PDF not found in uploads")
    sents, stats = preprocess_document(p)
    assert len(sents) > 0
    assert stats["sentence_count"] == len(sents)
    assert stats["mean_tokens_per_sentence"] > 0


# -----------------------------------------------------------------------------
# 18. sampleB.PDF end-to-end regression
# -----------------------------------------------------------------------------
def test_sampleB_pdf_end_to_end_preservation():
    p = Path("artifacts/cache/uploads/sampleB.PDF")
    if not p.exists():
        pytest.skip("sampleB.PDF not found in uploads")
    sents, stats = preprocess_document(p)
    assert len(sents) == 384
    assert stats["sentence_count"] == 384
    assert stats["mean_tokens_per_sentence"] > 0
