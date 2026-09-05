"""
High-Throughput Batch Ingestion Pipeline.
Processes 10k+ unstructured documents using parallel workers (ProcessPool/ThreadPool),
scaling throughput to 1,000+ pages/hour with PyMuPDF + OCR and batch graph/vector upsert.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from core.schema import Document, Chunk, Entity, Relationship
from ingestion.pdf_parser import PDFParser
from ingestion.chunker import HierarchicalSemanticChunker
from ingestion.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


def _parse_single_pdf_worker(
    file_path: str,
    enable_ocr: bool,
    ocr_min_char_threshold: int,
    chunk_size: int,
    chunk_overlap: int
) -> Tuple[Document, List[Chunk]]:
    """
    Worker function executed in parallel processes to parse PDF and generate chunks.
    """
    parser = PDFParser(
        enable_ocr=enable_ocr,
        ocr_min_char_threshold=ocr_min_char_threshold
    )
    doc = parser.parse(file_path)
    chunker = HierarchicalSemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(doc)
    return doc, chunks


class BatchIngestionStats:
    def __init__(self):
        self.total_docs: int = 0
        self.total_pages: int = 0
        self.total_chunks: int = 0
        self.total_entities: int = 0
        self.total_relationships: int = 0
        self.ocr_pages_count: int = 0
        self.start_time: float = time.time()
        self.end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time > 0 else time.time()
        return max(end - self.start_time, 0.001)

    @property
    def pages_per_hour(self) -> float:
        return (self.total_pages / self.elapsed_seconds) * 3600.0

    @property
    def pages_per_second(self) -> float:
        return self.total_pages / self.elapsed_seconds

    def summary(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_docs,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
            "ocr_pages_count": self.ocr_pages_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "pages_per_hour": round(self.pages_per_hour, 1),
            "pages_per_second": round(self.pages_per_second, 2),
        }


class BatchIngestor:
    """
    Production Batch Ingestor scaling to 10k+ documents and 1,000+ pages/hour.
    """

    def __init__(
        self,
        num_workers: int = 4,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        enable_ocr: bool = True,
        ocr_min_char_threshold: int = 50
    ):
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr
        self.ocr_min_char_threshold = ocr_min_char_threshold
        self.entity_extractor = EntityExtractor()

    def ingest_files(
        self,
        file_paths: List[str],
        vector_store=None,
        graph_store=None,
        embedding_model=None
    ) -> Tuple[BatchIngestionStats, List[Document], List[Chunk]]:
        """
        Ingests a list of PDF files across parallel worker processes,
        extracts entities and relationships, and upserts to vector store and graph store.
        """
        stats = BatchIngestionStats()
        all_documents: List[Document] = []
        all_chunks: List[Chunk] = []

        logger.info(f"Starting parallel ingestion for {len(file_paths)} files with {self.num_workers} workers...")

        # 1. Parallel PDF parsing & hierarchical chunking
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {
                executor.submit(
                    _parse_single_pdf_worker,
                    fp,
                    self.enable_ocr,
                    self.ocr_min_char_threshold,
                    self.chunk_size,
                    self.chunk_overlap
                ): fp
                for fp in file_paths
            }

            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                try:
                    doc, chunks = future.result()
                    all_documents.append(doc)
                    all_chunks.extend(chunks)

                    stats.total_docs += 1
                    stats.total_pages += doc.page_count
                    stats.total_chunks += len(chunks)
                    stats.ocr_pages_count += len(doc.ocr_applied_pages)

                except Exception as exc:
                    logger.error(f"Error parsing file {fp}: {exc}")

        # 2. Extract Entities and Relationships across all chunks
        entities_dict, relationships = self.entity_extractor.extract_from_chunks(all_chunks)
        stats.total_entities = len(entities_dict)
        stats.total_relationships = len(relationships)

        # 3. Batch Vector Embedding & Upsert into Pinecone
        if vector_store and embedding_model and all_chunks:
            logger.info(f"Generating dense embeddings for {len(all_chunks)} chunks...")
            texts = [c.text for c in all_chunks]
            embeddings = embedding_model.get_text_embeddings_batch(texts)
            for chunk, emb in zip(all_chunks, embeddings):
                chunk.vector_id = chunk.chunk_id
            vector_store.upsert_chunks(all_chunks, embeddings)

        # 4. Batch Graph Ingestion into Neo4j
        if graph_store:
            logger.info(f"Upserting graph nodes & relationships into Neo4j...")
            graph_store.batch_upsert_documents_and_chunks(all_documents, all_chunks)
            graph_store.batch_upsert_entities_and_relations(
                list(entities_dict.values()),
                relationships,
                all_chunks
            )

        stats.end_time = time.time()
        logger.info(
            f"Ingestion complete: {stats.total_pages} pages processed in {stats.elapsed_seconds:.2f}s "
            f"({stats.pages_per_hour:.1f} pages/hour)"
        )
        return stats, all_documents, all_chunks

    def ingest_directory(
        self,
        dir_path: str,
        vector_store=None,
        graph_store=None,
        embedding_model=None
    ) -> Tuple[BatchIngestionStats, List[Document], List[Chunk]]:
        """
        Recursively discovers all PDFs in a directory and ingests them.
        """
        p = Path(dir_path)
        pdf_files = [str(f) for f in p.glob("**/*.pdf")]
        if not pdf_files:
            logger.warning(f"No PDF files found in directory {dir_path}")
            return BatchIngestionStats(), [], []
        return self.ingest_files(pdf_files, vector_store, graph_store, embedding_model)
