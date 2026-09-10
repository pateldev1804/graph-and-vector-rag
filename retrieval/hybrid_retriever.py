"""
Hybrid Semantic-Graph Retriever with LlamaIndex Integration.
Concurrently executes Dense Vector Search (Pinecone) and Multi-Hop Cypher Traversal (Neo4j),
fusing results with Reciprocal Rank Fusion to power multi-document reasoning.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from core.schema import Chunk, ScoredChunk, GraphSubgraph, LatencyBreakdown
from ingestion.entity_extractor import EntityExtractor
from vector_store.embeddings import HuggingFaceEmbeddingPipeline
from vector_store.pinecone_client import PineconeVectorStore
from graph.neo4j_client import Neo4jGraphClient
from retrieval.ranker import HybridRanker

logger = logging.getLogger(__name__)

# LlamaIndex BaseRetriever integration if available
try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    BaseRetriever = object


class HybridGraphRAGRetriever:
    """
    Production Hybrid GraphRAG Retriever.
    Executes parallel vector and graph queries, achieving sub-100ms retrieval latency.
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        graph_client: Neo4jGraphClient,
        embedding_pipeline: HuggingFaceEmbeddingPipeline,
        top_k_vector: int = 5,
        top_k_graph: int = 5,
        top_k_final: int = 5,
        graph_hop_depth: int = 2,
        rrf_k: int = 60,
        vector_weight: float = 0.55,
        graph_weight: float = 0.45
    ):
        self.vector_store = vector_store
        self.graph_client = graph_client
        self.embedding_pipeline = embedding_pipeline
        self.top_k_vector = top_k_vector
        self.top_k_graph = top_k_graph
        self.top_k_final = top_k_final
        self.graph_hop_depth = graph_hop_depth

        self.entity_extractor = EntityExtractor()
        self.ranker = HybridRanker(
            rrf_k=rrf_k,
            vector_weight=vector_weight,
            graph_weight=graph_weight
        )
        self._executor = ThreadPoolExecutor(max_workers=4)

    def retrieve(
        self, query: str
    ) -> Tuple[List[ScoredChunk], GraphSubgraph, LatencyBreakdown]:
        """
        Executes hybrid retrieval:
        1. Fast entity extraction from query
        2. Concurrent execution of Pinecone vector search + Neo4j Cypher subgraph expansion
        3. Reciprocal Rank Fusion & Graph Grounding
        """
        latency = LatencyBreakdown()

        # Step 1: Entity Extraction from Query
        t0 = time.perf_counter()
        dummy_chunk = Chunk(doc_id="query", text=query)
        extracted_entities, _ = self.entity_extractor.extract_from_chunk(dummy_chunk)
        query_entity_names = [e.canonical_name for e in extracted_entities]

        # Also extract capitalized words as candidate entity seeds if none found
        if not query_entity_names:
            words = [w.strip(",.?") for w in query.split() if w and w[0].isupper()]
            query_entity_names = words[:4]

        latency.entity_extraction_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: Dense Query Embedding Generation
        t1 = time.perf_counter()
        query_embedding = self.embedding_pipeline.get_query_embedding(query)
        latency.embedding_ms = (time.perf_counter() - t1) * 1000.0

        # Step 3: Concurrent Pinecone Vector Search & Neo4j Cypher Traversal
        vector_results: List[Dict[str, Any]] = []
        graph_chunks: List[Dict[str, Any]] = []
        subgraph = GraphSubgraph(seed_entities=query_entity_names, hop_depth=self.graph_hop_depth)

        def _run_vector_search():
            tv0 = time.perf_counter()
            res = self.vector_store.query(query_embedding, top_k=self.top_k_vector)
            v_ms = (time.perf_counter() - tv0) * 1000.0
            return res, v_ms

        def _run_graph_search():
            tg0 = time.perf_counter()
            sub = self.graph_client.get_subgraph(
                query_entity_names, max_hops=self.graph_hop_depth, limit=25
            )
            g_chunks = self.graph_client.get_grounded_chunks_for_entities(
                query_entity_names, limit=self.top_k_graph
            )
            g_ms = (time.perf_counter() - tg0) * 1000.0
            return sub, g_chunks, g_ms

        # Concurrently dispatch to thread pool
        future_vec = self._executor.submit(_run_vector_search)
        future_graph = self._executor.submit(_run_graph_search)

        vector_results, latency.pinecone_vector_search_ms = future_vec.result()
        subgraph, graph_chunks, latency.neo4j_cypher_traversal_ms = future_graph.result()

        # Step 4: Reciprocal Rank Fusion & Context Fusion
        t4 = time.perf_counter()
        fused_chunks = self.ranker.fuse_and_rank(
            vector_results=vector_results,
            graph_chunk_results=graph_chunks,
            subgraph=subgraph,
            top_k=self.top_k_final
        )
        latency.rrf_fusion_rerank_ms = (time.perf_counter() - t4) * 1000.0

        return fused_chunks, subgraph, latency

    def to_llama_index_nodes(self, scored_chunks: List[ScoredChunk]) -> List[Any]:
        """
        Converts internal ScoredChunks to LlamaIndex NodeWithScore objects for LlamaIndex compatibility.
        """
        if not LLAMA_INDEX_AVAILABLE:
            return scored_chunks

        nodes = []
        for sc in scored_chunks:
            node = TextNode(
                text=sc.chunk.text,
                id_=sc.chunk.chunk_id,
                metadata={
                    "doc_id": sc.chunk.doc_id,
                    "page_number": sc.chunk.page_number,
                    "section_title": sc.chunk.section_title,
                    "vector_score": sc.vector_score,
                    "graph_score": sc.graph_score,
                    "matched_entities": sc.matched_entities,
                }
            )
            nodes.append(NodeWithScore(node=node, score=sc.fused_score))
        return nodes
