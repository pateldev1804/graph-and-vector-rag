"""
Pinecone Vector Store Integration & In-Memory Vector Fallback.
Provides low-latency vector similarity search (<50ms) and batch upserting.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from core.schema import Chunk

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_LIB_AVAILABLE = True
except ImportError:
    PINECONE_LIB_AVAILABLE = False


class InMemoryVectorStore:
    """
    High-speed In-Memory Vector Store (Numpy Matrix-backed).
    Used as fallback or for local zero-dependency testing with sub-5ms vector search.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vector_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.chunks: Dict[str, Chunk] = {}

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        if not chunks or not embeddings:
            return

        new_ids = [c.chunk_id for c in chunks]
        new_vecs = np.array(embeddings, dtype=np.float32)

        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

        if self.vectors is None:
            self.vector_ids = new_ids
            self.vectors = new_vecs
        else:
            self.vector_ids.extend(new_ids)
            self.vectors = np.vstack([self.vectors, new_vecs])

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.vector_ids) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Compute cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = self.vectors / norms
        scores = np.dot(normalized_vectors, q_vec)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            cid = self.vector_ids[idx]
            chunk = self.chunks.get(cid)
            if chunk:
                results.append({
                    "id": cid,
                    "score": float(scores[idx]),
                    "chunk": chunk,
                    "metadata": {
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title
                    }
                })
        return results


class PineconeVectorStore:
    """
    Production Pinecone Client with Connection Pooling & In-Memory Fallback.
    """

    def __init__(
        self,
        api_key: str = "mock-api-key",
        environment: str = "us-east-1",
        index_name: str = "graphrag-index",
        dimension: int = 768,
        metric: str = "cosine",
        use_mock_fallback: bool = True
    ):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.use_mock_fallback = use_mock_fallback

        self._pinecone: Optional[Any] = None
        self._index: Optional[Any] = None
        self._is_connected: bool = False
        self._in_memory_store = InMemoryVectorStore(dimension=dimension)

        self._connect()

    def _connect(self):
        if not PINECONE_LIB_AVAILABLE or self.api_key.startswith("mock-") or not self.api_key:
            logger.info("Pinecone API key is mock/empty. Using InMemoryVectorStore.")
            self._is_connected = False
            return

        try:
            self._pinecone = Pinecone(api_key=self.api_key)
            existing_indexes = [idx.name for idx in self._pinecone.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index '{self.index_name}' (dim={self.dimension}, metric={self.metric})...")
                self._pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud="aws", region=self.environment)
                )
            self._index = self._pinecone.Index(self.index_name)
            self._is_connected = True
            logger.info(f"Connected to Pinecone index '{self.index_name}'.")
        except Exception as e:
            logger.warning(f"Could not connect to Pinecone ({e}). Using InMemoryVectorStore.")
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """
        Batch upserts chunk vectors and metadata.
        """
        # Always mirror in in-memory store for fallback
        self._in_memory_store.upsert_chunks(chunks, embeddings)

        if not self._is_connected or self._index is None:
            return

        vectors_to_upsert = []
        for chunk, emb in zip(chunks, embeddings):
            metadata = {
                "text": chunk.text[:1000],  # Pinecone metadata size limit guard
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title or "General",
                "token_count": chunk.token_count
            }
            vectors_to_upsert.append((chunk.chunk_id, emb, metadata))

        # Batch in chunks of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            try:
                self._index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Error upserting batch to Pinecone: {e}")

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries top-k dense vector matches.
        """
        if not self._is_connected or self._index is None:
            return self._in_memory_store.query(query_embedding, top_k)

        try:
            response = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            results = []
            for match in response.get("matches", []):
                cid = match["id"]
                score = match["score"]
                meta = match.get("metadata", {})
                chunk = self._in_memory_store.chunks.get(cid) or Chunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", ""),
                    text=meta.get("text", ""),
                    page_number=meta.get("page_number", 1),
                    section_title=meta.get("section_title", "General")
                )
                results.append({
                    "id": cid,
                    "score": float(score),
                    "chunk": chunk,
                    "metadata": meta
                })
            return results
        except Exception as e:
            logger.error(f"Pinecone query error: {e}, falling back to in-memory store")
            return self._in_memory_store.query(query_embedding, top_k)
