"""
JurisGraph RAG (Retrieval-Augmented Generation) Module
Implements role-tagged and naive chunking, persistent ChromaDB vector store,
and rhetorical role-filtered semantic retrieval.
"""

from src.rag.chunker import chunk_document, chunk_document_role, chunk_document_naive
from src.rag.vectorstore import LegalVectorStore, ingest_document, delete_document, retrieve

__all__ = [
    "chunk_document",
    "chunk_document_role",
    "chunk_document_naive",
    "LegalVectorStore",
    "ingest_document",
    "delete_document",
    "retrieve",
]
