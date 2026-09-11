"""
High-Precision Latency Profiler & SLA Budget Monitor.
Tracks microsecond-level latency breakdown for all GraphRAG stages,
calculating p50, p95, and p99 metrics against the <400ms SLA budget.
"""

from typing import List, Dict, Any
import numpy as np
from core.schema import LatencyBreakdown


class LatencyTracker:
    """
    Tracks and summarizes query latencies across GraphRAG pipeline stages.
    """

    def __init__(self, latency_budget_ms: float = 400.0):
        self.latency_budget_ms = latency_budget_ms
        self.records: List[LatencyBreakdown] = []

    def record(self, latency: LatencyBreakdown):
        self.records.append(latency)

    def compute_percentiles(self) -> Dict[str, Any]:
        if not self.records:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "sla_compliance_pct": 100.0}

        totals = np.array([r.total_latency_ms for r in self.records])
        p50 = float(np.percentile(totals, 50))
        p95 = float(np.percentile(totals, 95))
        p99 = float(np.percentile(totals, 99))
        mean = float(np.mean(totals))
        compliant = float(np.sum(totals <= self.latency_budget_ms) / len(totals) * 100.0)

        # Stage averages
        stage_avgs = {
            "entity_extraction_ms": float(np.mean([r.entity_extraction_ms for r in self.records])),
            "embedding_ms": float(np.mean([r.embedding_ms for r in self.records])),
            "vector_search_ms": float(np.mean([r.pinecone_vector_search_ms for r in self.records])),
            "graph_traversal_ms": float(np.mean([r.neo4j_cypher_traversal_ms for r in self.records])),
            "rrf_fusion_ms": float(np.mean([r.rrf_fusion_rerank_ms for r in self.records])),
            "llm_generation_ms": float(np.mean([r.llm_generation_ms for r in self.records])),
        }

        return {
            "count": len(self.records),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "mean_ms": round(mean, 2),
            "sla_budget_ms": self.latency_budget_ms,
            "sla_compliance_pct": round(compliant, 2),
            "stage_breakdown_avg_ms": {k: round(v, 2) for k, v in stage_avgs.items()}
        }

    def format_summary_table(self) -> str:
        stats = self.compute_percentiles()
        if stats["count"] == 0:
            return "No queries tracked yet."

        breakdown = stats["stage_breakdown_avg_ms"]
        lines = [
            "=" * 60,
            f"  GRAPHRAG LATENCY PROFILE & SLA BENCHMARK ({stats['count']} Queries)",
            "=" * 60,
            f"  • SLA Budget:            {stats['sla_budget_ms']} ms",
            f"  • Mean Latency:          {stats['mean_ms']} ms",
            f"  • P50 Latency:           {stats['p50_ms']} ms",
            f"  • P95 Latency:           {stats['p95_ms']} ms",
            f"  • P99 Latency:           {stats['p99_ms']} ms",
            f"  • SLA Compliance Rate:   {stats['sla_compliance_pct']}% (<{stats['sla_budget_ms']}ms)",
            "-" * 60,
            "  Stage Breakdown (Average):",
            f"    - Query Entity Extraction:  {breakdown['entity_extraction_ms']:>6.2f} ms",
            f"    - Dense Vector Embedding:   {breakdown['embedding_ms']:>6.2f} ms",
            f"    - Pinecone Vector Search:   {breakdown['vector_search_ms']:>6.2f} ms",
            f"    - Neo4j Cypher Traversal:   {breakdown['graph_traversal_ms']:>6.2f} ms",
            f"    - RRF Fusion & Re-Ranking:  {breakdown['rrf_fusion_ms']:>6.2f} ms",
            f"    - LLM Inference Synthesis:  {breakdown['llm_generation_ms']:>6.2f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)
