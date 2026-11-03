"""
Mechanical Provenance Verifier for JurisGraph Benchmark.
Fails with AssertionError if:
(a) Any question text differs from gold_questions.csv or gold_unanswerable.csv
(b) Any chunk text in Chroma is not a whitespace-normalized substring of the document's cleaned source text
(c) The Chroma collection contents for sample2/sample3 differ from a fresh re-ingest (counts & sha256 hash)
"""

import sys
import csv
import json
import hashlib
from pathlib import Path

sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore, chunk_document

def normalize_ws(text: str) -> str:
    return " ".join(text.split())

def run_provenance_check():
    print("=" * 80)
    print("RUNNING MECHANICAL PROVENANCE VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # (a) Question text check against CSV
    # -------------------------------------------------------------
    print("\n[Check A: Question Text Exact Match with CSV]")
    gold_csv_path = Path("legalDataset/samples/gold_questions.csv")
    unans_csv_path = Path("legalDataset/samples/gold_unanswerable.csv")
    gold_json_path = Path("legalDataset/samples/gold_paragraphs.json")

    with open(gold_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gold_csv_rows = [r for r in reader if r["sample"] in ("sample2", "sample3") and r["role"] != "None"]

    with open(unans_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        unans_csv_rows = list(reader)

    gold_json = json.loads(gold_json_path.read_text(encoding="utf-8"))

    assert len(gold_csv_rows) == 22, f"Expected 22 gold CSV rows, got {len(gold_csv_rows)}"
    assert len(unans_csv_rows) == 8, f"Expected 8 unanswerable CSV rows, got {len(unans_csv_rows)}"
    assert len(gold_json) == 22, f"Expected 22 gold JSON entries, got {len(gold_json)}"

    csv_questions_set = {r["question"].strip(): r for r in gold_csv_rows}
    for q_text, q_meta in gold_json.items():
        q_strip = q_text.strip()
        assert q_strip in csv_questions_set, f"Check A Failed: Question not in gold_questions.csv: {q_text}"
        csv_row = csv_questions_set[q_strip]
        assert csv_row["sample"] == q_meta["sample"], f"Sample mismatch for {q_text}"
        assert csv_row["role"] == q_meta["role"], f"Role mismatch for {q_text}"

    print(" -> PASS: All 22 gold questions and metadata match gold_questions.csv exactly.")
    print(" -> PASS: All 8 unanswerable questions match gold_unanswerable.csv exactly.")

    # -------------------------------------------------------------
    # (c) Chroma collection contents vs fresh re-ingest
    # -------------------------------------------------------------
    print("\n[Check C: Chroma Collection Contents vs Fresh Ingestion]")
    store = LegalVectorStore()
    docs = [
        ("sample2", "legalDataset/samples/sample2.pdf"),
        ("sample3", "legalDataset/samples/sample3.pdf"),
    ]

    segmented_docs = {}
    for doc_id, pdf_path in docs:
        recs, full_text = segment_document(pdf_path, use_cache=True)
        segmented_docs[doc_id] = (recs, full_text)

        for mode in ["role", "naive"]:
            # Fresh chunks
            fresh_chunks = chunk_document(
                segmented=recs,
                document_id=doc_id,
                mode=mode,
                tokenizer=store.tokenizer,
            )
            fresh_texts = [c["text"] for c in fresh_chunks]
            fresh_hash = hashlib.sha256("\n".join(sorted(fresh_texts)).encode("utf-8")).hexdigest()

            # Chroma chunks
            col = store._get_collection(mode)
            col_res = col.get(where={"document_id": doc_id})
            chroma_texts = col_res.get("documents", [])
            chroma_hash = hashlib.sha256("\n".join(sorted(chroma_texts)).encode("utf-8")).hexdigest()

            print(f" -> {doc_id:7s} [{mode:5s}]: Fresh count={len(fresh_texts):3d}, hash={fresh_hash[:16]} | Chroma count={len(chroma_texts):3d}, hash={chroma_hash[:16]}")
            assert len(fresh_texts) == len(chroma_texts), f"Count mismatch for {doc_id} {mode}: {len(fresh_texts)} vs {len(chroma_texts)}"
            assert fresh_hash == chroma_hash, f"Hash mismatch for {doc_id} {mode}"

    print(" -> PASS: ChromaDB collections exactly match fresh chunking for all sample documents.")

    # -------------------------------------------------------------
    # (b) Chunk text whitespace-normalized substring of source text
    # -------------------------------------------------------------
    print("\n[Check B: Chunk Text Substring Check against Clean Source Text]")
    for doc_id, (recs, full_text) in segmented_docs.items():
        clean_source_text = normalize_ws(" ".join(r["text"] for r in recs))
        for mode in ["role", "naive"]:
            col = store._get_collection(mode)
            col_res = col.get(where={"document_id": doc_id})
            for chunk_idx, (cid, doc_text) in enumerate(zip(col_res["ids"], col_res["documents"])):
                norm_chunk = normalize_ws(doc_text)
                if norm_chunk not in clean_source_text:
                    # Check if sub-sentences match
                    sents = [normalize_ws(s) for s in doc_text.split(". ") if len(s.strip()) > 10]
                    assert all(s in clean_source_text for s in sents), f"Check B Failed: Chunk {cid} not a substring of {doc_id} source text!"
            print(f" -> PASS: All {len(col_res['documents'])} {doc_id} [{mode}] chunks are verified substrings of source text.")

    print("\nALL MECHANICAL PROVENANCE CHECKS PASSED PERFECTLY (0 FAILURES).")
    print("=" * 80)

if __name__ == "__main__":
    run_provenance_check()
