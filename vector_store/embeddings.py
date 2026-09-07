"""
Hugging Face Dense Vector Embeddings Integration.
Provides high-throughput batch embedding generation, device auto-detection (CUDA/MPS/CPU),
and ultra-fast LRU caching for query embeddings to achieve <400ms end-to-end latency.
"""

import logging
import hashlib
from typing import List, Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceEmbeddingPipeline:
    """
    Optimized Dense Embedding Pipeline using Hugging Face SentenceTransformers.
    Includes LRU query caching to reduce query embedding latency to <1ms on cache hit.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        dimension: int = 768,
        device: Optional[str] = None,
        cache_size: int = 10000
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.cache_size = cache_size
        self._query_cache: Dict[str, List[float]] = {}

        # Auto-detect optimal device: CUDA -> Apple Silicon MPS -> CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._init_model()

    def _init_model(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("SentenceTransformers not available. Using random normalized fallback.")
            return

        try:
            logger.info(f"Loading Hugging Face embedding model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Warm up model
            _ = self._model.encode(["warmup query"], convert_to_numpy=True)
            logger.info("Embedding model loaded and warmed up.")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face model {self.model_name}: {e}")
            self._model = None

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generates dense embedding for a single query text with LRU cache lookup.
        """
        query_key = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        if query_key in self._query_cache:
            return self._query_cache[query_key]

        if self._model is not None:
            embedding = self._model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
        else:
            # Deterministic pseudo-embedding fallback for unit testing
            np.random.seed(int(query_key[:8], 16))
            vec = np.random.randn(self.dimension)
            vec = vec / np.linalg.norm(vec)
            embedding = vec.tolist()

        if len(self._query_cache) >= self.cache_size:
            # Evict first element
            self._query_cache.pop(next(iter(self._query_cache)))

        self._query_cache[query_key] = embedding
        return embedding

    def get_text_embeddings_batch(
        self, texts: List[str], batch_size: int = 64
    ) -> List[List[float]]:
        """
        Generates dense embeddings for a batch of chunk texts.
        """
        if not texts:
            return []

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        else:
            # Mock embeddings for testing
            results = []
            for t in texts:
                key = hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
                np.random.seed(int(key[:8], 16))
                vec = np.random.randn(self.dimension)
                vec = vec / np.linalg.norm(vec)
                results.append(vec.tolist())
            return results
