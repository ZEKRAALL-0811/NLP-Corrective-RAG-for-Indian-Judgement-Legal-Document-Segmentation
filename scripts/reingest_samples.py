import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore

print("Re-ingesting sample2 and sample3 into ChromaDB...")
recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

store = LegalVectorStore()
for doc_id, recs, name in [("sample2", recs2, "sample2.pdf"), ("sample3", recs3, "sample3.pdf")]:
    r_role = store.ingest_document(recs, document_id=doc_id, mode="role", document_name=name)
    r_naive = store.ingest_document(recs, document_id=doc_id, mode="naive", document_name=name)
    print(f"[{doc_id}] Role chunks: {r_role['n_chunks']} | Naive chunks: {r_naive['n_chunks']}")

print("Re-ingestion complete!")
