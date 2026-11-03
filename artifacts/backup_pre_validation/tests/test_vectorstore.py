"""
Unit and Integration Tests for Module 4: Chunker and Persistent ChromaDB Vector Store
"""

import pytest
from src.rag.chunker import (
    chunk_document_role,
    chunk_document_naive,
    ROLE_TO_PROB_FIELD,
)
from src.rag.vectorstore import LegalVectorStore
from src.config import CANONICAL_LABELS
import src.config as cfg


@pytest.fixture(scope="module")
def sample_segmented_records():
    """Synthetic segmented records covering multiple roles and boundaries."""
    records = []
    # 5 Facts sentences (short)
    for i in range(5):
        records.append({
            "sentence_idx": i,
            "text": f"This is factual background sentence number {i} regarding the company dispute.",
            "role": "Facts",
            "role_probs": [0.85, 0.02, 0.03, 0.02, 0.05, 0.01, 0.02],
        })
    # 3 Petitioner arguments
    for i in range(5, 8):
        records.append({
            "sentence_idx": i,
            "text": f"The petitioner vehemently contends that section 132 was violated in instance {i}.",
            "role": "Arguments of Petitioner",
            "role_probs": [0.05, 0.05, 0.80, 0.03, 0.04, 0.01, 0.02],
        })
    # 1 Reasoning sentence with moderate petitioner probability
    records.append({
        "sentence_idx": 8,
        "text": "The High Court observed that the arguments of the appellant merit serious consideration.",
        "role": "Reasoning",
        "role_probs": [0.10, 0.05, 0.25, 0.05, 0.50, 0.03, 0.02],
    })
    # 2 Decision sentences
    for i in range(9, 11):
        records.append({
            "sentence_idx": i,
            "text": f"Accordingly, the writ petition is allowed and the impugned order {i} is quashed.",
            "role": "Decision",
            "role_probs": [0.02, 0.01, 0.01, 0.01, 0.05, 0.88, 0.02],
        })
    return records


@pytest.fixture(scope="module")
def vector_store(tmp_path_factory):
    """Temporary persistent Chroma vector store for testing."""
    chroma_dir = tmp_path_factory.mktemp("chroma_test")
    return LegalVectorStore(chroma_dir=str(chroma_dir))


def test_chunker_role_mode_metadata_and_boundaries(sample_segmented_records):
    """Verifies role chunking metadata fields, token counts, and boundary isolation."""
    chunks = chunk_document_role(
        segmented=sample_segmented_records,
        document_id="test_doc_1",
        document_name="test_doc_1.pdf",
    )

    assert len(chunks) > 0
    for c in chunks:
        meta = c["metadata"]
        assert meta["document_id"] == "test_doc_1"
        assert meta["document_name"] == "test_doc_1.pdf"
        assert meta["role"] in CANONICAL_LABELS
        assert 0.0 <= meta["role_conf"] <= 1.0
        assert meta["role2"] in CANONICAL_LABELS
        assert meta["role2"] != meta["role"] or len(CANONICAL_LABELS) == 1

        # Check all 7 flat float fields
        for label in CANONICAL_LABELS:
            field = ROLE_TO_PROB_FIELD[label]
            assert field in meta
            assert isinstance(meta[field], float)

        assert meta["n_sentences"] == meta["end_sentence_idx"] - meta["start_sentence_idx"] + 1
        assert meta["n_tokens"] > 0
        assert meta["n_tokens"] <= cfg.CHUNK_TOKEN_LIMIT


def test_chunk_token_limit_configuration():
    """Verify CHUNK_TOKEN_LIMIT is configured to 256 and aligned with production ingestion."""
    assert cfg.CHUNK_TOKEN_LIMIT == 256



def test_chunker_naive_mode():
    """Verifies naive mode windowing and absence of role fields."""
    text = " ".join([f"Word{i}" for i in range(500)])
    chunks = chunk_document_naive(
        segmented=text,
        document_id="naive_doc",
        window_size=1000,
        overlap=200,
    )

    assert len(chunks) > 0
    for c in chunks:
        meta = c["metadata"]
        assert meta["document_id"] == "naive_doc"
        assert "role" not in meta
        assert "role_conf" not in meta
        assert "p_facts" not in meta
        assert "chunk_idx" in meta
        assert "n_tokens" in meta


def test_idempotent_ingestion_and_deletion(vector_store, sample_segmented_records):
    """Verifies that re-ingesting a document replaces old chunks without duplication."""
    # First ingest
    stats1 = vector_store.ingest_document(sample_segmented_records, document_id="doc_idemp", mode="role")
    col = vector_store._get_collection("role")
    count1 = len(col.get(where={"document_id": {"$eq": "doc_idemp"}})["ids"])
    assert count1 == stats1["n_chunks"]

    # Second ingest (idempotent overwrite)
    stats2 = vector_store.ingest_document(sample_segmented_records, document_id="doc_idemp", mode="role")
    count2 = len(col.get(where={"document_id": {"$eq": "doc_idemp"}})["ids"])
    assert count2 == count1
    assert count2 == stats2["n_chunks"]

    # Delete
    vector_store.delete_document("doc_idemp", mode="role")
    count3 = len(col.get(where={"document_id": {"$eq": "doc_idemp"}})["ids"])
    assert count3 == 0


def test_retrieval_relaxation_levels(vector_store, sample_segmented_records):
    """Verifies cumulative relaxation hierarchy: len(res0) <= len(res1) <= len(res2) <= len(res3)."""
    vector_store.ingest_document(sample_segmented_records, document_id="doc_relax", mode="role")

    query = "What arguments were raised by the petitioner?"

    # Relax 0: strict Petitioner
    res0 = vector_store.retrieve(query, document_id="doc_relax", mode="role", roles=["Arguments of Petitioner"], relax=0, top_k=10)
    for r in res0:
        assert r["metadata"]["role"] == "Arguments of Petitioner"

    # Relax 1: Petitioner as role or role2 (Level 0 U {role2 in roles})
    res1 = vector_store.retrieve(query, document_id="doc_relax", mode="role", roles=["Arguments of Petitioner"], relax=1, top_k=10)
    for r in res1:
        assert r["metadata"]["role"] == "Arguments of Petitioner" or r["metadata"]["role2"] == "Arguments of Petitioner"

    # Relax 2: Level 1 U {p_petitioner >= 0.20}
    res2 = vector_store.retrieve(query, document_id="doc_relax", mode="role", roles=["Arguments of Petitioner"], relax=2, top_k=10)
    for r in res2:
        m = r["metadata"]
        assert m["role"] == "Arguments of Petitioner" or m["role2"] == "Arguments of Petitioner" or m["p_petitioner"] >= 0.20

    # Relax 3: unconstrained
    res3 = vector_store.retrieve(query, document_id="doc_relax", mode="role", roles=["Arguments of Petitioner"], relax=3, top_k=10)

    # Strictly assert cumulative relaxation counts are non-decreasing
    assert len(res0) <= len(res1) <= len(res2) <= len(res3)


def test_position_metadata_and_filtering(vector_store, sample_segmented_records):
    """Verifies doc_n_sentences, rel_pos in chunk metadata and rel_pos_range retrieval filtering."""
    vector_store.ingest_document(sample_segmented_records, document_id="doc_pos", mode="role")
    col = vector_store._get_collection("role")
    all_chunks = col.get(where={"document_id": {"$eq": "doc_pos"}})
    for meta in all_chunks["metadatas"]:
        assert "doc_n_sentences" in meta
        assert "rel_pos" in meta
        assert meta["doc_n_sentences"] == len(sample_segmented_records)
        assert 0.0 <= meta["rel_pos"] <= 1.0

    # Query with rel_pos_range restricting to latter part of the document
    res_late = vector_store.retrieve(
        "What was the final decision?",
        document_id="doc_pos",
        mode="role",
        rel_pos_range=(0.70, 1.0),
        top_k=10,
    )
    assert len(res_late) > 0
    for r in res_late:
        assert r["metadata"]["rel_pos"] >= 0.70


def test_document_id_isolation(vector_store, sample_segmented_records):
    """Verifies that retrieval strictly isolates documents by document_id."""
    vector_store.ingest_document(sample_segmented_records, document_id="doc_A", mode="role")
    vector_store.ingest_document(sample_segmented_records, document_id="doc_B", mode="role")

    res_a = vector_store.retrieve("quashed", document_id="doc_A", mode="role", top_k=10)
    for r in res_a:
        assert r["metadata"]["document_id"] == "doc_A"

    res_b = vector_store.retrieve("quashed", document_id="doc_B", mode="role", top_k=10)
    for r in res_b:
        assert r["metadata"]["document_id"] == "doc_B"
