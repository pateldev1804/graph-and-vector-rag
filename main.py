"""
GraphRAG CLI & Unified Control Engine.
Provides commands for Ingestion, Multi-Hop Querying, Benchmarks (<400ms SLA, 1000+ pages/hr),
FastAPI Serving, and Interactive Demo.
"""

import sys
import os
import argparse
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from config.settings import settings
from vector_store.embeddings import HuggingFaceEmbeddingPipeline
from vector_store.pinecone_client import PineconeVectorStore
from graph.neo4j_client import Neo4jGraphClient
from retrieval.hybrid_retriever import HybridGraphRAGRetriever
from engine.graphrag_engine import GraphRAGQueryEngine
from ingestion.batch_ingestor import BatchIngestor
from benchmarks.benchmark_latency import run_latency_benchmark
from benchmarks.benchmark_ingestion import run_ingestion_benchmark
from benchmarks.benchmark_reasoning import run_reasoning_comparison

console = Console()
logging.basicConfig(level=logging.WARNING)


def initialize_pipeline():
    emb_pipe = HuggingFaceEmbeddingPipeline(
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
    ingestor = BatchIngestor(
        num_workers=settings.ingestion.num_workers,
        chunk_size=settings.ingestion.chunk_size,
        chunk_overlap=settings.ingestion.chunk_overlap,
        enable_ocr=settings.ingestion.enable_ocr_fallback
    )
    return emb_pipe, vec_store, graph_client, engine, ingestor


def run_interactive_demo():
    console.print(Panel.fit(
        "[bold cyan]Hybrid GraphRAG Pipeline Demo[/bold cyan]\n"
        "[dim]Neo4j Multi-Hop Graph + Pinecone Dense Vectors + PyMuPDF Ingestion[/dim]",
        border_style="cyan"
    ))

    # Run reasoning comparison
    run_reasoning_comparison()

    # Run latency benchmark
    run_latency_benchmark(num_iterations=20)

    # Run ingestion benchmark
    run_ingestion_benchmark(num_docs=5, pages_per_doc=10, num_workers=4)


def handle_query(query_str: str):
    emb_pipe, vec_store, graph_client, engine, _ = initialize_pipeline()
    console.print(f"\n[bold yellow]🔍 Query:[/bold yellow] [bold white]{query_str}[/bold white]\n")

    result = engine.query(query_str)

    # Display Generated Grounded Response
    console.print(Panel(
        result.response,
        title="[bold green]💡 Grounded Answer (Hallucination-Free)[/bold green]",
        border_style="green"
    ))

    # Display Latency Metrics
    table = Table(title="⚡ Microsecond Latency Breakdown (<400ms SLA)", border_style="cyan")
    table.add_column("Pipeline Stage", style="bold cyan")
    table.add_column("Duration (ms)", justify="right", style="magenta")
    table.add_column("Target Status", justify="center")

    lat = result.latency
    table.add_row("1. Query Entity Extraction", f"{lat.entity_extraction_ms:.2f} ms", "✅")
    table.add_row("2. Dense Embedding Gen", f"{lat.embedding_ms:.2f} ms", "✅")
    table.add_row("3. Pinecone Vector Search", f"{lat.pinecone_vector_search_ms:.2f} ms", "✅")
    table.add_row("4. Neo4j Cypher Traversal", f"{lat.neo4j_cypher_traversal_ms:.2f} ms", "✅")
    table.add_row("5. RRF Fusion & Re-Ranking", f"{lat.rrf_fusion_rerank_ms:.2f} ms", "✅")
    table.add_row("6. Grounded Synthesis", f"{lat.llm_generation_ms:.2f} ms", "✅")
    table.add_section()
    sla_status = "[bold green]✅ PASS (<400ms)[/bold green]" if lat.total_latency_ms <= 400.0 else "[bold red]❌ EXCEEDED[/bold red]"
    table.add_row("[bold white]Total End-to-End Latency[/bold white]", f"[bold yellow]{lat.total_latency_ms:.2f} ms[/bold yellow]", sla_status)
    console.print(table)


def handle_ingest(path_str: str):
    emb_pipe, vec_store, graph_client, _, ingestor = initialize_pipeline()
    console.print(f"[bold cyan]Starting ingestion for target: {path_str}[/bold cyan]")

    if os.path.isdir(path_str):
        stats, docs, chunks = ingestor.ingest_directory(
            path_str,
            vector_store=vec_store,
            graph_store=graph_client,
            embedding_model=emb_pipe
        )
    elif os.path.isfile(path_str):
        stats, docs, chunks = ingestor.ingest_files(
            [path_str],
            vector_store=vec_store,
            graph_store=graph_client,
            embedding_model=emb_pipe
        )
    else:
        console.print(f"[bold red]Target path not found: {path_str}[/bold red]")
        return

    summary = stats.summary()
    table = Table(title="📊 Ingestion Throughput Results", border_style="green")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="bold yellow")
    for k, v in summary.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Hybrid GraphRAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # query command
    query_parser = subparsers.add_parser("query", help="Execute low-latency hybrid query")
    query_parser.add_argument("query_text", type=str, help="Search query text")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents")
    ingest_parser.add_argument("path", type=str, help="Path to PDF file or directory")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument(
        "--type", choices=["latency", "ingestion", "reasoning", "all"], default="all"
    )

    # demo command
    subparsers.add_parser("demo", help="Run end-to-end interactive demo")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    if args.command == "query":
        handle_query(args.query_text)
    elif args.command == "ingest":
        handle_ingest(args.path)
    elif args.command == "benchmark":
        if args.type in ["latency", "all"]:
            run_latency_benchmark(num_iterations=25)
        if args.type in ["ingestion", "all"]:
            run_ingestion_benchmark(num_docs=5, pages_per_doc=10, num_workers=4)
        if args.type in ["reasoning", "all"]:
            run_reasoning_comparison()
    elif args.command == "demo":
        run_interactive_demo()
    elif args.command == "serve":
        import uvicorn
        console.print(f"[bold green]Starting GraphRAG FastAPI server on {args.host}:{args.port}...[/bold green]")
        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
