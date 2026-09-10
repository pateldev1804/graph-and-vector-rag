"""
Reciprocal Rank Fusion (RRF) & Graph-Grounded Context Fusion Ranker.
Combines dense vector similarity scores and knowledge graph structural relevance,
grounding retrieved chunks with verified graph relationships to eliminate hallucinations.
"""

from typing import List, Dict, Any, Set, Tuple
from core.schema import Chunk, ScoredChunk, GraphSubgraph


class HybridRanker:
    """
    Combines Vector Search and Graph Traversal results using Reciprocal Rank Fusion (RRF)
    and Graph Centrality / Multi-hop weight adjustments.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        vector_weight: float = 0.55,
        graph_weight: float = 0.45
    ):
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def fuse_and_rank(
        self,
        vector_results: List[Dict[str, Any]],
        graph_chunk_results: List[Dict[str, Any]],
        subgraph: GraphSubgraph,
        top_k: int = 5
    ) -> List[ScoredChunk]:
        """
        Fuses vector search results and graph traversal matches using weighted RRF.
        """
        chunk_map: Dict[str, ScoredChunk] = {}

        # 1. Process Vector Search Results (Dense semantic matching)
        for rank, item in enumerate(vector_results):
            cid = item["id"]
            chunk = item["chunk"]
            vector_score = item["score"]
            rrf_score = self.vector_weight * (1.0 / (self.rrf_k + rank + 1))

            if cid not in chunk_map:
                chunk_map[cid] = ScoredChunk(
                    chunk=chunk,
                    vector_score=vector_score,
                    graph_score=0.0,
                    fused_score=rrf_score,
                    matched_entities=[],
                    graph_provenance=[]
                )
            else:
                chunk_map[cid].vector_score = vector_score
                chunk_map[cid].fused_score += rrf_score

        # 2. Process Graph Chunk Results (Multi-hop structural matching)
        for rank, item in enumerate(graph_chunk_results):
            cid = item["chunk_id"]
            graph_score = item.get("graph_relevance_score", 1.0)
            entities = item.get("mentioned_entities", [])
            rrf_score = self.graph_weight * (1.0 / (self.rrf_k + rank + 1))

            if cid not in chunk_map:
                # If chunk came solely from graph, reconstruct Chunk object
                chunk = Chunk(
                    chunk_id=cid,
                    doc_id=item.get("doc_id", ""),
                    text=item.get("text", ""),
                    page_number=item.get("page_number", 1),
                    section_title=item.get("section_title", "General")
                )
                chunk_map[cid] = ScoredChunk(
                    chunk=chunk,
                    vector_score=0.0,
                    graph_score=graph_score,
                    fused_score=rrf_score,
                    matched_entities=entities,
                    graph_provenance=[f"EntityMatch: {', '.join(entities)}"]
                )
            else:
                chunk_map[cid].graph_score = graph_score
                chunk_map[cid].fused_score += rrf_score
                chunk_map[cid].matched_entities.extend(entities)
                chunk_map[cid].graph_provenance.append(f"EntityMatch: {', '.join(entities)}")

        # 3. Apply Multi-Hop Subgraph Centrality Bonus
        subgraph_entity_names = {node.get("canonical_name", "") for node in subgraph.nodes}
        for sc in chunk_map.values():
            overlap = set(sc.matched_entities).intersection(subgraph_entity_names)
            if overlap:
                # Centrality bonus for chunks anchoring active multi-hop entities
                sc.fused_score *= (1.0 + 0.15 * len(overlap))

        # Sort descending by fused score
        sorted_scored_chunks = sorted(
            chunk_map.values(),
            key=lambda sc: sc.fused_score,
            reverse=True
        )

        return sorted_scored_chunks[:top_k]

    @staticmethod
    def format_graph_grounding_context(subgraph: GraphSubgraph) -> str:
        """
        Formats verified knowledge graph triples for LLM prompt context to eliminate hallucination.
        """
        if not subgraph.edges:
            return ""

        triples = []
        for edge in subgraph.edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "RELATION")
            desc = edge.get("description", "")
            if desc:
                triples.append(f"• ({src}) -[{rel}]-> ({tgt}): {desc}")
            else:
                triples.append(f"• ({src}) -[{rel}]-> ({tgt})")

        return "=== KNOWLEDGE GRAPH VERIFIED TRIPLETS (Ground Truth) ===\n" + "\n".join(triples)
