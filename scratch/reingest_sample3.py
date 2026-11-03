import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore

print("Segmenting sample3.pdf...")
recs, stats = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)
print(f"Sample3 segmented: {len(recs)} sentences.")

store = LegalVectorStore()
print("Ingesting sample3 role chunks into ChromaDB...")
role_chunks = store.ingest_document(recs, document_id="sample3", mode="role")
print(f"Ingested {len(role_chunks)} role chunks for sample3.")
