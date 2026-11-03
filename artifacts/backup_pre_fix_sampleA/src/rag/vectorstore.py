"""
JurisGraph ChromaDB Vector Store (Module 4)

Provides persistent vector storage and semantic retrieval for legal judgment chunks:
1. Persistent ChromaDB Client at artifacts/chroma/.
2. Embeddings with BAAI/bge-small-en-v1.5 via sentence-transformers on CPU.
3. BGE query instruction prefix applied ONLY for query embeddings.
4. Normalized embeddings with cosine distance.
5. Separate collections for 'role' and 'naive' chunking modes.
6. Idempotent ingestion (replaces old chunks for the same document_id).
7. Hierarchical role-relaxation retrieval:
   - Relax 0: strict role match (role in roles)
   - Relax 1: primary or secondary role match (role or role2 in roles)
   - Relax 2: probability threshold (p_<role> >= 0.20)
   - Relax 3: unconstrained document retrieval (no role filter)
   - Always constrained by document_id.
"""

from typing import List, Dict, Any, Union, Optional, Tuple
import time
import os
import psutil
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    BGE_QUERY_PREFIX,
    ROLE_COLLECTION_NAME,
    NAIVE_COLLECTION_NAME,
    CANONICAL_LABELS,
)
from src.rag.chunker import (
    chunk_document,
    ROLE_TO_PROB_FIELD,
    ROLE_ALIASES,
)


class LegalVectorStore:
    """
    Persistent ChromaDB vector store for legal RAG pipelines.
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        self.chroma_dir = str(chroma_dir or CHROMA_DIR)
        self.model_name = str(embedding_model_name or EMBEDDING_MODEL_NAME)
        self.device = device

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.chroma_dir)

        # Initialize embedding model and tokenizer (timing model load separately)
        t_load_0 = time.time()
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_load_time_sec = round(time.time() - t_load_0, 3)

        # Get or create collections with cosine distance metric
        self.role_collection = self.client.get_or_create_collection(
            name=ROLE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.naive_collection = self.client.get_or_create_collection(
            name=NAIVE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_collection(self, mode: str):
        """Returns the appropriate Chroma collection based on mode."""
        mode_clean = mode.lower().strip()
        if mode_clean == "role":
            return self.role_collection
        elif mode_clean == "naive":
            return self.naive_collection
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'role' or 'naive'.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds document chunks without any prefix.
        Normalizes embeddings for exact cosine similarity.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a search query with BGE's instruction prefix.
        Normalizes embeddings for cosine distance.
        """
        prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
        emb = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )
        return emb.tolist()

    def delete_document(self, document_id: str, mode: Optional[str] = None) -> None:
        """
        Deletes all chunks belonging to document_id.
        If mode is specified, deletes from that mode's collection;
        otherwise deletes from both collections.
        """
        doc_id_str = str(document_id)
        modes = [mode] if mode else ["role", "naive"]
        for m in modes:
            col = self._get_collection(m)
            col.delete(where={"document_id": {"$eq": doc_id_str}})

    def ingest_document(
        self,
        segmented: Any,
        document_id: str,
        mode: str = "role",
        document_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Idempotent document ingestion:
        1. Deletes old chunks for document_id in the target collection.
        2. Chunks the document according to mode.
        3. Computes BGE embeddings.
        4. Inserts chunks with rich metadata into ChromaDB.

        Returns:
            Dictionary of ingestion statistics.
        """
        t0 = time.time()
        process = psutil.Process(os.getpid())
        ram_before_mb = process.memory_info().rss / (1024 * 1024)

        doc_id_str = str(document_id)
        mode_clean = mode.lower().strip()

        # 1. Idempotent deletion of existing document chunks
        self.delete_document(doc_id_str, mode=mode_clean)

        # 2. Chunk document
        chunks = chunk_document(
            segmented=segmented,
            document_id=doc_id_str,
            document_name=document_name,
            mode=mode_clean,
            tokenizer=self.tokenizer,
        )

        if not chunks:
            return {
                "document_id": doc_id_str,
                "mode": mode_clean,
                "n_chunks": 0,
                "ingest_time_sec": round(time.time() - t0, 3),
                "ram_mb": round(ram_before_mb, 2),
            }

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{doc_id_str}_{mode_clean}_{c['chunk_idx']}" for c in chunks]

        # 3. Compute normalized BGE embeddings with timing
        t_embed_0 = time.time()
        embeddings = self.embed_documents(texts)
        t_embed = round(time.time() - t_embed_0, 3)

        # 4. Insert into persistent Chroma collection
        col = self._get_collection(mode_clean)
        col.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        elapsed = round(time.time() - t0, 3)
        ram_after_mb = process.memory_info().rss / (1024 * 1024)

        return {
            "document_id": doc_id_str,
            "mode": mode_clean,
            "n_chunks": len(chunks),
            "model_load_time_sec": getattr(self, "model_load_time_sec", 0.0),
            "embedding_time_sec": t_embed,
            "per_chunk_embed_ms": round((t_embed / max(len(chunks), 1)) * 1000, 2),
            "chunks_per_sec": round(len(chunks) / max(t_embed, 1e-4), 1),
            "total_ingest_time_sec": elapsed,
            "ram_mb": round(ram_after_mb, 2),
        }

    def _normalize_roles(
        self, roles: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """Normalizes role names and handles casing / aliases."""
        if roles is None:
            return []
        if isinstance(roles, str):
            roles = [roles]

        norm_roles = []
        for r in roles:
            clean = r.strip()
            lower = clean.lower()
            if clean in CANONICAL_LABELS:
                norm_roles.append(clean)
            elif lower in ROLE_ALIASES:
                norm_roles.append(ROLE_ALIASES[lower])
            else:
                norm_roles.append(clean)
        return norm_roles

    def retrieve(
        self,
        question: str,
        document_id: str,
        mode: str = "role",
        roles: Optional[Union[str, List[str]]] = None,
        relax: int = 0,
        top_k: int = 5,
        rel_pos_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks from ChromaDB with cumulative relaxation and position filtering.

        Args:
            question: Search query string.
            document_id: Document to filter on (always enforced).
            mode: 'role' or 'naive'.
            roles: Desired rhetorical role(s) or None.
            relax: Cumulative relaxation level:
                0: Level 0 = role in roles
                1: Level 1 = Level 0 U {role2 in roles}
                2: Level 2 = Level 1 U {p_role >= 0.20}
                3: Level 3 = unconstrained by role (all document chunks)
            top_k: Number of chunks to retrieve.
            rel_pos_range: Optional (lo, hi) relative document position filter [0.0, 1.0].

        Returns:
            List of dicts: [{'text': ..., 'metadata': ..., 'distance': ...}]
        """
        doc_id_str = str(document_id)
        mode_clean = mode.lower().strip()
        col = self._get_collection(mode_clean)

        norm_roles = self._normalize_roles(roles)

        # Build ChromaDB 'where' filter
        # Always filter by document_id
        doc_filter = {"document_id": {"$eq": doc_id_str}}
        and_conditions = [doc_filter]

        # Optional relative position range filter: rel_pos in [lo, hi]
        if rel_pos_range is not None:
            lo, hi = rel_pos_range
            and_conditions.append({"rel_pos": {"$gte": float(lo)}})
            and_conditions.append({"rel_pos": {"$lte": float(hi)}})

        # Cumulative role filtering
        if mode_clean != "naive" and norm_roles and relax < 3:
            if relax == 0:
                # Level 0: role in roles
                if len(norm_roles) == 1:
                    and_conditions.append({"role": {"$eq": norm_roles[0]}})
                else:
                    and_conditions.append({"role": {"$in": norm_roles}})

            elif relax == 1:
                # Level 1: Level 0 U {role2 in roles}
                if len(norm_roles) == 1:
                    r = norm_roles[0]
                    and_conditions.append({"$or": [{"role": {"$eq": r}}, {"role2": {"$eq": r}}]})
                else:
                    and_conditions.append({"$or": [{"role": {"$in": norm_roles}}, {"role2": {"$in": norm_roles}}]})

            elif relax == 2:
                # Level 2: Level 1 U {p_role >= 0.20}
                prob_conds = []
                for r in norm_roles:
                    field = ROLE_TO_PROB_FIELD.get(r)
                    if field:
                        prob_conds.append({field: {"$gte": 0.20}})

                if len(norm_roles) == 1:
                    r = norm_roles[0]
                    or_list = [{"role": {"$eq": r}}, {"role2": {"$eq": r}}] + prob_conds
                else:
                    or_list = [{"role": {"$in": norm_roles}}, {"role2": {"$in": norm_roles}}] + prob_conds
                and_conditions.append({"$or": or_list})

        if len(and_conditions) == 1:
            where_clause = and_conditions[0]
        else:
            where_clause = {"$and": and_conditions}

        # Embed query with BGE prefix
        query_vec = self.embed_query(question)

        # Perform semantic search in Chroma
        try:
            results = col.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # If no items match where filter in collection
            if "no documents" in str(e).lower() or "empty" in str(e).lower():
                return []
            raise e

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        output = []
        for cid, d, m, dist in zip(results["ids"][0], docs, metas, dists):
            output.append({
                "id": cid,
                "chunk_id": cid,
                "text": d,
                "metadata": m,
                "distance": float(dist),
            })

        return output


# Singleton instance for high efficiency
_GLOBAL_STORE: Optional[LegalVectorStore] = None


def get_vector_store() -> LegalVectorStore:
    """Returns the persistent singleton LegalVectorStore instance."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = LegalVectorStore()
    return _GLOBAL_STORE


def ingest_document(
    segmented: Any,
    document_id: str,
    mode: str = "role",
    document_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Top-level ingest function matching requested API."""
    store = get_vector_store()
    return store.ingest_document(
        segmented=segmented,
        document_id=document_id,
        mode=mode,
        document_name=document_name,
    )


def delete_document(document_id: str, mode: Optional[str] = None) -> None:
    """Top-level delete function matching requested API."""
    store = get_vector_store()
    store.delete_document(document_id=document_id, mode=mode)


def retrieve(
    question: str,
    document_id: str,
    mode: str = "role",
    roles: Optional[Union[str, List[str]]] = None,
    relax: int = 0,
    top_k: int = 5,
    rel_pos_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Top-level retrieve function matching requested API."""
    store = get_vector_store()
    return store.retrieve(
        question=question,
        document_id=document_id,
        mode=mode,
        roles=roles,
        relax=relax,
        top_k=top_k,
        rel_pos_range=rel_pos_range,
    )

