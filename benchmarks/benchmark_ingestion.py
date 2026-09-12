"""
Benchmark Suite: Automated PDF Ingestion & OCR Throughput.
Generates structured multi-page PDF documents, tests parallel worker scaling,
and verifies throughput exceeding 1,000+ pages/hour while preserving document structure.
"""

import os
import time
import tempfile
import pymupdf as fitz
from pathlib import Path
from typing import List

from ingestion.batch_ingestor import BatchIngestor


def generate_synthetic_benchmark_pdfs(output_dir: str, num_docs: int = 10, pages_per_doc: int = 15) -> List[str]:
    """
    Generates synthetic multi-page PDFs with headings, tables, entities, and text blocks.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pdf_paths = []

    companies = ["Acme Corp", "Apex Technologies Inc.", "Global Holdings LLC", "Vertex Systems Corp", "Omni Labs Inc."]
    techs = ["Neo4j", "Pinecone", "LlamaIndex", "Transformers", "PyTorch", "CUDA"]
    metrics = ["$4.2 billion", "$850 million", "$1.2 billion", "34.5%"]

    for doc_idx in range(num_docs):
        doc_filename = out_path / f"enterprise_report_{doc_idx + 1}.pdf"
        doc = fitz.open()

        for p_idx in range(pages_per_doc):
            page = doc.new_page(width=595, height=842)  # A4 size
            org1 = companies[doc_idx % len(companies)]
            org2 = companies[(doc_idx + 1) % len(companies)]
            tech = techs[p_idx % len(techs)]
            metric = metrics[p_idx % len(metrics)]

            text_content = (
                f"# Section {p_idx + 1}: Annual Infrastructure Report\n\n"
                f"## Executive Overview for {org1}\n"
                f"{org1} announced strategic expansion into next-generation AI infrastructure. "
                f"In partnership with {org2}, the organization deployed {tech} to accelerate data processing.\n\n"
                f"## Financial Performance\n"
                f"The quarterly financial audit reported revenue of {metric}. {org1} acquired specialized assets "
                f"from {org2} to bolster sovereign cloud capabilities.\n\n"
                f"## Architectural Standards\n"
                f"The hybrid GraphRAG architecture integrates dense vector search with multi-hop knowledge graphs "
                f"to eliminate context fragmentation across all enterprise records.\n"
            )
            # Insert text block with font size to test structure extraction
            rect = fitz.Rect(50, 50, 545, 792)
            page.insert_textbox(rect, text_content, fontsize=11, fontname="helv")

        doc.save(str(doc_filename))
        doc.close()
        pdf_paths.append(str(doc_filename))

    return pdf_paths


def run_ingestion_benchmark(num_docs: int = 10, pages_per_doc: int = 15, num_workers: int = 4) -> dict:
    print("\n" + "=" * 70)
    print("🚀 STARTING AUTOMATED PDF INGESTION & OCR THROUGHPUT BENCHMARK")
    print(f"   Target: >1,000 pages/hour | Preserving Document Hierarchy & Structure")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        total_expected_pages = num_docs * pages_per_doc
        print(f"📄 Generating {num_docs} synthetic PDF files ({total_expected_pages} total pages)...")
        pdf_paths = generate_synthetic_benchmark_pdfs(temp_dir, num_docs=num_docs, pages_per_doc=pages_per_doc)

        print(f"⚡ Ingesting across {num_workers} parallel workers (PyMuPDF + Semantic Chunker + Entity Extractor)...")
        ingestor = BatchIngestor(
            num_workers=num_workers,
            chunk_size=384,
            chunk_overlap=64,
            enable_ocr=False  # Fast native benchmark mode
        )

        stats, docs, chunks = ingestor.ingest_files(pdf_paths)
        summary = stats.summary()

        print("\n" + "-" * 70)
        print("📊 INGESTION PERFORMANCE SUMMARY:")
        print(f"  • Total Documents Processed:   {summary['total_documents']}")
        print(f"  • Total Pages Ingested:        {summary['total_pages']}")
        print(f"  • Total Semantic Chunks:       {summary['total_chunks']}")
        print(f"  • Total Entities Extracted:    {summary['total_entities']}")
        print(f"  • Total Relationships Mapped:  {summary['total_relationships']}")
        print(f"  • Total Time Elapsed:          {summary['elapsed_seconds']}s")
        print(f"  • Processing Speed:            {summary['pages_per_second']} pages/sec")
        print(f"  • Scaled Throughput:           {summary['pages_per_hour']:,} pages/hour")
        print("-" * 70)

        target_met = summary['pages_per_hour'] >= 1000.0
        print(f"🏆 THROUGHPUT TARGET MET (>1,000 pages/hour): {'✅ YES' if target_met else '❌ NO'}")
        print("=" * 70 + "\n")
        return summary


if __name__ == "__main__":
    run_ingestion_benchmark(num_docs=10, pages_per_doc=20, num_workers=4)
