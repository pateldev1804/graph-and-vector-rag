"""
Benchmark Suite: Query Latency & <400ms SLA Verification.
Tests cold and warm query latency across single-hop and multi-hop questions,
measuring p50, p95, and p99 against the target budget.
"""

import time
import logging
from typing import List

from config.settings import settings
from vector_store.embeddings import HuggingFaceEmbeddingPipeline
from vector_store.pinecone_client import PineconeVectorStore
from graph.neo4j_client import Neo4jGraphClient
from retrieval.hybrid_retriever import HybridGraphRAGRetriever
from engine.graphrag_engine import GraphRAGQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_BENCHMARK_QUERIES = [
    "What technologies does Neo4j integrate with?",
    "How does Pinecone handle vector indexing for LlamaIndex?",
    "What company acquired the database division of TechCorp LLC?",
    "Which subsidiaries are located in North America?",
    "What is the revenue metric reported by Alphabet Inc?",
    "Explain the relationship between PyMuPDF, OCR, and Document ingestion.",
    "What dependencies exist between the RAG pipeline and Transformers?",
    "How do multi-hop Cypher queries eliminate context fragmentation?",
    "What was the financial growth reported for Q4 2024?",
    "Which organizations partnered with Microsoft Corp in 2025?",
]


def run_latency_benchmark(num_iterations: int = 50) -> dict:
    print("\n" + "=" * 70)
    print("🚀 STARTING GRAPHRAG LATENCY BENCHMARK (<400ms SLA VERIFICATION)")
    print("=" * 70)

    # 1. Initialize components
    print("⚡ Initializing Dense Embedding Pipeline (Hugging Face)...")
    emb_pipe = HuggingFaceEmbeddingPipeline(
        model_name=settings.embedding.model_name,
        dimension=settings.embedding.dimension
    )

    print("⚡ Initializing Pinecone Vector Store...")
    vec_store = PineconeVectorStore(
        api_key=settings.pinecone.api_key,
        index_name=settings.pinecone.index_name,
        dimension=settings.embedding.dimension
    )

    print("⚡ Initializing Neo4j Knowledge Graph...")
    graph_client = Neo4jGraphClient(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database
    )

    retriever = HybridGraphRAGRetriever(
        vector_store=vec_store,
        graph_client=graph_client,
        embedding_pipeline=emb_pipe,
        top_k_vector=settings.retrieval.top_k_vector,
        top_k_graph=settings.retrieval.top_k_graph_entities,
        top_k_final=settings.retrieval.top_k_final,
        graph_hop_depth=settings.retrieval.graph_hop_depth
    )

    engine = GraphRAGQueryEngine(
        retriever=retriever,
        latency_budget_ms=settings.retrieval.latency_budget_ms
    )

    # 2. Warm up
    print("🔥 Warming up pipeline...")
    _ = engine.query("Warmup query for embeddings and graph")

    # 3. Benchmark Loop
    print(f"📊 Running {num_iterations} benchmark query iterations...")
    for i in range(num_iterations):
        query = SAMPLE_BENCHMARK_QUERIES[i % len(SAMPLE_BENCHMARK_QUERIES)]
        res = engine.query(query)

    # 4. Print Summary
    summary_table = engine.latency_tracker.format_summary_table()
    print("\n" + summary_table + "\n")

    percentiles = engine.latency_tracker.compute_percentiles()
    sla_pass = percentiles["p95_ms"] <= settings.retrieval.latency_budget_ms
    print(f"🏆 SLA TARGET MET (<400ms at p95): {'✅ YES' if sla_pass else '❌ NO'}")
    print("=" * 70 + "\n")
    return percentiles


if __name__ == "__main__":
    run_latency_benchmark(num_iterations=30)
