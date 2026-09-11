"""
FastAPI REST API Server for Hybrid GraphRAG Pipeline.
Provides high-throughput endpoints for sub-400ms querying, PDF ingestion, and graph inspection.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import tempfile
import os
import logging

from config.settings import settings
from core.schema import GraphRAGQueryResult
from vector_store.embeddings import HuggingFaceEmbeddingPipeline
from vector_store.pinecone_client import PineconeVectorStore
from graph.neo4j_client import Neo4jGraphClient
from retrieval.hybrid_retriever import HybridGraphRAGRetriever
from engine.graphrag_engine import GraphRAGQueryEngine
from ingestion.batch_ingestor import BatchIngestor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid GraphRAG Engine API",
    description="Sub-400ms Hybrid GraphRAG Pipeline integrating Neo4j, Pinecone, and LlamaIndex",
    version="1.0.0"
)

# Global instances
emb_pipeline = HuggingFaceEmbeddingPipeline(
    model_name=settings.embedding.model_name,
    dimension=settings.embedding.dimension
)
vec_store = PineconeVectorStore(
    api_key=settings.pinecone.api_key,
    index_name=settings.pinecone.index_name,
    dimension=settings.embedding.dimension
)
graph_client = Neo4jGraphClient(
    uri=settings.neo4j.uri,
    username=settings.neo4j.username,
    password=settings.neo4j.password,
    database=settings.neo4j.database
)
retriever = HybridGraphRAGRetriever(
    vector_store=vec_store,
    graph_client=graph_client,
    embedding_pipeline=emb_pipeline,
    top_k_vector=settings.retrieval.top_k_vector,
    top_k_graph=settings.retrieval.top_k_graph_entities,
    top_k_final=settings.retrieval.top_k_final,
    graph_hop_depth=settings.retrieval.graph_hop_depth
)
engine = GraphRAGQueryEngine(
    retriever=retriever,
    latency_budget_ms=settings.retrieval.latency_budget_ms
)
ingestor = BatchIngestor(
    num_workers=settings.ingestion.num_workers,
    chunk_size=settings.ingestion.chunk_size,
    chunk_overlap=settings.ingestion.chunk_overlap,
    enable_ocr=settings.ingestion.enable_ocr_fallback
)


class QueryRequest(BaseModel):
    query: str = Field(..., example="What technologies does Neo4j integrate with?")
    top_k: int = Field(default=5)


class QueryResponseModel(BaseModel):
    query: str
    response: str
    total_latency_ms: float
    sla_compliance: bool
    latency_breakdown: Dict[str, float]
    provenance_documents: List[str]
    multi_hop_paths: List[List[str]]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "neo4j_connected": graph_client.is_connected,
        "pinecone_connected": vec_store.is_connected,
        "embedding_device": emb_pipeline.device,
        "embedding_model": emb_pipeline.model_name
    }


@app.post("/query", response_model=QueryResponseModel)
def run_query(req: QueryRequest):
    try:
        result: GraphRAGQueryResult = engine.query(req.query)
        return QueryResponseModel(
            query=result.query,
            response=result.response,
            total_latency_ms=round(result.latency.total_latency_ms, 2),
            sla_compliance=result.latency.total_latency_ms <= settings.retrieval.latency_budget_ms,
            latency_breakdown={
                "entity_extraction_ms": round(result.latency.entity_extraction_ms, 2),
                "embedding_ms": round(result.latency.embedding_ms, 2),
                "pinecone_vector_search_ms": round(result.latency.pinecone_vector_search_ms, 2),
                "neo4j_cypher_traversal_ms": round(result.latency.neo4j_cypher_traversal_ms, 2),
                "rrf_fusion_rerank_ms": round(result.latency.rrf_fusion_rerank_ms, 2),
                "llm_generation_ms": round(result.latency.llm_generation_ms, 2),
            },
            provenance_documents=result.provenance_documents,
            multi_hop_paths=result.multi_hop_paths
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_uploaded_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stats, docs, chunks = ingestor.ingest_files(
            [tmp_path],
            vector_store=vec_store,
            graph_store=graph_client,
            embedding_model=emb_pipeline
        )
        return {
            "status": "success",
            "filename": file.filename,
            "stats": stats.summary()
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/metrics")
def get_latency_metrics():
    return engine.latency_tracker.compute_percentiles()
