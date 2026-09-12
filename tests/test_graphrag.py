"""
Comprehensive Unit & Integration Test Suite for Hybrid GraphRAG.
Tests PDF Parsing, Semantic Chunking, Entity Extraction, Neo4j Schema/Cypher traversal,
Pinecone Vector Indexing, Hybrid RRF Ranking, and <400ms Query Latency.
"""

import pytest
import os
import tempfile
import pymupdf as fitz

from core.schema import Document, DocumentMetadata, Chunk, Entity, Relationship
from ingestion.pdf_parser import PDFParser
from ingestion.chunker import HierarchicalSemanticChunker
from ingestion.entity_extractor import EntityExtractor
from ingestion.batch_ingestor import BatchIngestor
from graph.neo4j_client import InMemoryGraphStore, Neo4jGraphClient
from vector_store.pinecone_client import InMemoryVectorStore, PineconeVectorStore
from vector_store.embeddings import HuggingFaceEmbeddingPipeline
from retrieval.ranker import HybridRanker
from retrieval.hybrid_retriever import HybridGraphRAGRetriever
from engine.graphrag_engine import GraphRAGQueryEngine


@pytest.fixture
def sample_pdf_path():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_textbox(
        fitz.Rect(50, 50, 500, 700),
        "# Enterprise Infrastructure Report\n"
        "Alpha Technologies Inc. acquired Beta Systems Corp for $1.5 billion in 2024.\n"
        "Beta Systems Corp developed the patented GraphDB Engine.",
        fontsize=12
    )
    doc.save(pdf_path)
    doc.close()

    yield pdf_path

    if os.path.exists(pdf_path):
        os.remove(pdf_path)


def test_pdf_parsing_and_chunking(sample_pdf_path):
    parser = PDFParser(enable_ocr=False)
    doc = parser.parse(sample_pdf_path)

    assert doc.page_count == 1
    assert "Enterprise Infrastructure Report" in doc.raw_text
    assert doc.metadata.doc_type == "pdf"

    chunker = HierarchicalSemanticChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    assert chunks[0].doc_id == doc.doc_id
    assert chunks[0].page_number == 1


def test_entity_and_relation_extraction():
    chunk = Chunk(
        doc_id="DOC_001",
        text="Alpha Technologies Inc. acquired Beta Systems Corp. Beta Systems Corp is located in San Francisco."
    )
    extractor = EntityExtractor()
    entities, rels = extractor.extract_from_chunk(chunk)

    entity_names = [e.canonical_name for e in entities]
    assert "Alpha Technologies Inc." in entity_names
    assert "Beta Systems Corp." in entity_names

    rel_types = [r.relation_type for r in rels]
    assert "ACQUIRED" in rel_types


def test_in_memory_graph_and_multi_hop_traversal():
    doc = Document(
        doc_id="DOC_1",
        raw_text="Test",
        metadata=DocumentMetadata(source_path="/test", filename="test.pdf", title="Test Doc")
    )
    chunk = Chunk(doc_id="DOC_1", text="Alpha Corp acquired Beta LLC.")
    ent1 = Entity(name="Alpha Corp", canonical_name="Alpha Corp", entity_type="ORGANIZATION", source_chunk_ids=[chunk.chunk_id])
    ent2 = Entity(name="Beta LLC", canonical_name="Beta LLC", entity_type="ORGANIZATION", source_chunk_ids=[chunk.chunk_id])
    rel = Relationship(source_entity="Alpha Corp", target_entity="Beta LLC", relation_type="ACQUIRED")

    graph = InMemoryGraphStore()
    graph.add_documents_and_chunks([doc], [chunk])
    graph.add_entities_and_relations([ent1, ent2], [rel], [chunk])

    subgraph = graph.get_subgraph_for_entities(["Alpha Corp"], max_hops=2)
    assert len(subgraph.edges) == 1
    assert subgraph.edges[0]["source"] == "Alpha Corp"
    assert subgraph.edges[0]["target"] == "Beta LLC"
    assert subgraph.edges[0]["relation"] == "ACQUIRED"


def test_vector_store_indexing_and_search():
    store = InMemoryVectorStore(dimension=4)
    chunks = [
        Chunk(doc_id="D1", text="Machine learning with PyTorch"),
        Chunk(doc_id="D2", text="Graph databases with Neo4j")
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]
    store.upsert_chunks(chunks, embeddings)

    # Search for Neo4j
    results = store.query([0.0, 0.9, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["chunk"].doc_id == "D2"


def test_rrf_hybrid_ranker():
    ranker = HybridRanker(rrf_k=60)
    c1 = Chunk(doc_id="D1", text="Chunk 1")
    c2 = Chunk(doc_id="D2", text="Chunk 2")

    vec_results = [{"id": c1.chunk_id, "score": 0.95, "chunk": c1}]
    graph_chunks = [{"chunk_id": c2.chunk_id, "doc_id": "D2", "text": "Chunk 2", "graph_relevance_score": 2.0}]

    subgraph = InMemoryGraphStore().get_subgraph_for_entities([])
    fused = ranker.fuse_and_rank(vec_results, graph_chunks, subgraph, top_k=2)

    assert len(fused) == 2
    assert fused[0].fused_score > 0
    assert fused[1].fused_score > 0


def test_end_to_end_query_latency():
    emb = HuggingFaceEmbeddingPipeline()
    vec = PineconeVectorStore(api_key="mock-key", dimension=emb.dimension)
    graph = Neo4jGraphClient(use_mock_fallback=True)

    retriever = HybridGraphRAGRetriever(
        vector_store=vec,
        graph_client=graph,
        embedding_pipeline=emb
    )
    engine = GraphRAGQueryEngine(retriever=retriever, latency_budget_ms=400.0)

    # Ingest a sample doc
    c = Chunk(doc_id="D1", text="Neo4j powers multi-hop GraphRAG pipelines.")
    vec.upsert_chunks([c], [emb.get_query_embedding(c.text)])

    result = engine.query("How does Neo4j power GraphRAG?")
    assert result.latency.total_latency_ms < 400.0
    assert result.response is not None
