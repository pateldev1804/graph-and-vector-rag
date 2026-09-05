"""
Domain Schemas and Data Models for GraphRAG Pipeline.
Defines Documents, Chunks, Entities, Relations, Subgraphs, and Latency Metrics.
"""

from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import time
import uuid


class DocumentMetadata(BaseModel):
    source_path: str
    filename: str
    total_pages: int = 1
    doc_type: str = "pdf"
    file_size_bytes: int = 0
    created_at: float = Field(default_factory=time.time)
    title: Optional[str] = None
    author: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str
    metadata: DocumentMetadata
    page_count: int = 1
    page_texts: Dict[int, str] = Field(default_factory=dict)
    ocr_applied_pages: List[int] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    page_number: int = 1
    section_title: Optional[str] = None
    section_hierarchy: List[str] = Field(default_factory=list)
    token_count: int = 0
    vector_id: Optional[str] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    canonical_name: str
    entity_type: str  # ORGANIZATION, PERSON, LOCATION, CONCEPT, FINANCIAL_METRIC, TECHNOLOGY, POLICY, EVENT
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_doc_ids: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    relation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity: str  # Canonical name or entity_id
    target_entity: str  # Canonical name or entity_id
    relation_type: str  # e.g., OWNS, SUBSIDIARY_OF, IMPACTS, DEPENDS_ON, REPORTED, PART_OF, INCREASES
    description: Optional[str] = None
    weight: float = 1.0
    confidence: float = 1.0
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_doc_ids: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphSubgraph(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    seed_entities: List[str] = Field(default_factory=list)
    hop_depth: int = 1


class ScoredChunk(BaseModel):
    chunk: Chunk
    vector_score: float = 0.0
    graph_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    matched_entities: List[str] = Field(default_factory=list)
    graph_provenance: List[str] = Field(default_factory=list)


class LatencyBreakdown(BaseModel):
    entity_extraction_ms: float = 0.0
    embedding_ms: float = 0.0
    pinecone_vector_search_ms: float = 0.0
    neo4j_cypher_traversal_ms: float = 0.0
    rrf_fusion_rerank_ms: float = 0.0
    llm_generation_ms: float = 0.0
    total_latency_ms: float = 0.0

    def exceeds_budget(self, budget_ms: float = 400.0) -> bool:
        return self.total_latency_ms > budget_ms


class GraphRAGQueryResult(BaseModel):
    query: str
    response: str
    retrieved_chunks: List[ScoredChunk]
    subgraph: GraphSubgraph
    latency: LatencyBreakdown
    provenance_documents: List[str]
    multi_hop_paths: List[List[str]] = Field(default_factory=list)
