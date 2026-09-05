"""
Configuration module for Hybrid GraphRAG Pipeline.
Manages environment variables, Neo4j connection, Pinecone settings,
embedding models, LLM parameters, and ingestion performance tuning.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jConfig(BaseModel):
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    username: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="graphrag_password", description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j target database")
    max_connection_pool_size: int = Field(default=50, description="Connection pool size")
    connection_timeout: float = Field(default=5.0, description="Connection timeout in seconds")
    use_mock_if_unavailable: bool = Field(
        default=True, 
        description="Fallback to in-memory graph if Neo4j instance is not reachable"
    )


class PineconeConfig(BaseModel):
    api_key: str = Field(default="mock-pinecone-api-key", description="Pinecone API Key")
    environment: str = Field(default="us-east-1", description="Pinecone cloud region")
    index_name: str = Field(default="graphrag-index", description="Pinecone index name")
    dimension: int = Field(default=768, description="Embedding dimension (e.g. 768 for all-mpnet-base-v2)")
    metric: str = Field(default="cosine", description="Distance metric")
    use_mock_if_unavailable: bool = Field(
        default=True,
        description="Fallback to in-memory vector store if Pinecone is not reachable"
    )


class EmbeddingConfig(BaseModel):
    model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Hugging Face embedding model name"
    )
    dimension: int = Field(default=768, description="Vector dimension")
    batch_size: int = Field(default=64, description="Embedding inference batch size")
    device: str = Field(default="cpu", description="Torch device (cpu, cuda, mps)")
    cache_size: int = Field(default=10000, description="LRU cache size for query embeddings")


class LLMConfig(BaseModel):
    provider: str = Field(default="huggingface", description="LLM provider (huggingface, local, gemini, openai)")
    model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", description="Model name or path")
    max_new_tokens: int = Field(default=384, description="Max generated tokens")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    context_window: int = Field(default=4096, description="Context window size")
    api_key: Optional[str] = Field(default=None, description="Optional API key for cloud providers")


class IngestionConfig(BaseModel):
    chunk_size: int = Field(default=384, description="Target chunk size in tokens/words")
    chunk_overlap: int = Field(default=64, description="Chunk overlap")
    num_workers: int = Field(default=4, description="Multiprocessing worker processes for PDF parsing")
    enable_ocr_fallback: bool = Field(default=True, description="Run Tesseract OCR on scanned/image pages")
    ocr_language: str = Field(default="eng", description="OCR language")
    extract_entities_during_ingest: bool = Field(default=True, description="Extract entities and relations during ingestion")
    batch_size: int = Field(default=20, description="Batch size of documents per processing task")


class RetrievalConfig(BaseModel):
    top_k_vector: int = Field(default=5, description="Top-k vector matches from Pinecone")
    top_k_graph_entities: int = Field(default=5, description="Top-k entities to seed graph traversal")
    graph_hop_depth: int = Field(default=2, description="Multi-hop traversal depth in Neo4j")
    top_k_final: int = Field(default=5, description="Final fused top-k nodes returned to synthesis")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing parameter")
    vector_weight: float = Field(default=0.55, description="Weight for vector similarity score in fusion")
    graph_weight: float = Field(default=0.45, description="Weight for graph centrality & relationship score")
    latency_budget_ms: float = Field(default=400.0, description="Target SLA latency budget in ms")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    log_level: str = Field(default="INFO", description="Logging level")


settings = Settings()
