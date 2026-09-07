"""
Neo4j Graph Store Client with Connection Pooling & In-Memory Graph Fallback.
Provides low-latency graph execution, connection pooling, and multi-hop Cypher queries.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx

from core.schema import Document, Chunk, Entity, Relationship, GraphSubgraph
from graph.cypher_schema import SCHEMA_CONSTRAINTS, SCHEMA_INDEXES
from graph.cypher_queries import (
    CYPHER_1_HOP_NEIGHBORS,
    CYPHER_2_HOP_SUBGRAPH,
    CYPHER_MULTI_DOC_BRIDGES,
    CYPHER_SHORTEST_PATH,
    CYPHER_GET_CHUNKS_FOR_ENTITIES,
)

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_LIB_AVAILABLE = True
except ImportError:
    NEO4J_LIB_AVAILABLE = False


class InMemoryGraphStore:
    """
    High-performance In-Memory Knowledge Graph Store (NetworkX-backed).
    Provides Cypher-equivalent graph traversal semantics when Neo4j is not connected.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.entities: Dict[str, Entity] = {}
        self.entity_to_chunks: Dict[str, Set[str]] = {}

    def add_documents_and_chunks(self, documents: List[Document], chunks: List[Chunk]):
        for doc in documents:
            self.documents[doc.doc_id] = doc
            self.graph.add_node(
                f"doc:{doc.doc_id}",
                node_type="Document",
                doc_id=doc.doc_id,
                title=doc.metadata.title or doc.metadata.filename
            )

        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            chunk_node = f"chunk:{chunk.chunk_id}"
            self.graph.add_node(
                chunk_node,
                node_type="Chunk",
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                page_number=chunk.page_number,
                section_title=chunk.section_title
            )
            # Connect Document to Chunk
            self.graph.add_edge(f"doc:{chunk.doc_id}", chunk_node, rel_type="CONTAINS_CHUNK")
            # Connect linear chunks
            if chunk.next_chunk_id:
                self.graph.add_edge(chunk_node, f"chunk:{chunk.next_chunk_id}", rel_type="NEXT_CHUNK")

    def add_entities_and_relations(
        self, entities: List[Entity], relations: List[Relationship], chunks: List[Chunk]
    ):
        for ent in entities:
            self.entities[ent.canonical_name] = ent
            ent_node = f"entity:{ent.canonical_name}"
            self.graph.add_node(
                ent_node,
                node_type="Entity",
                name=ent.name,
                canonical_name=ent.canonical_name,
                entity_type=ent.entity_type,
                confidence=ent.confidence
            )

            # Link chunks to entities
            for chunk_id in ent.source_chunk_ids:
                if chunk_id in self.chunks:
                    self.graph.add_edge(f"chunk:{chunk_id}", ent_node, rel_type="MENTIONS")
                    self.entity_to_chunks.setdefault(ent.canonical_name, set()).add(chunk_id)

        for rel in relations:
            src_node = f"entity:{rel.source_entity}"
            tgt_node = f"entity:{rel.target_entity}"
            if not self.graph.has_node(src_node):
                self.graph.add_node(src_node, node_type="Entity", canonical_name=rel.source_entity, entity_type="CONCEPT")
            if not self.graph.has_node(tgt_node):
                self.graph.add_node(tgt_node, node_type="Entity", canonical_name=rel.target_entity, entity_type="CONCEPT")

            self.graph.add_edge(
                src_node,
                tgt_node,
                rel_type="RELATION",
                explicit_rel=rel.relation_type,
                description=rel.description,
                weight=rel.weight
            )

    def get_subgraph_for_entities(self, seed_entities: List[str], max_hops: int = 2, limit: int = 25) -> GraphSubgraph:
        nodes = []
        edges = []
        visited_nodes = set()
        visited_edges = set()

        seeds = [f"entity:{s}" for s in seed_entities if f"entity:{s}" in self.graph]

        for seed in seeds:
            # Multi-hop breadth-first traversal
            lengths = nx.single_source_shortest_path_length(self.graph.to_undirected(), seed, cutoff=max_hops)
            for node, dist in lengths.items():
                if dist <= max_hops:
                    node_data = self.graph.nodes.get(node, {})
                    if node not in visited_nodes and node_data.get("node_type") == "Entity":
                        visited_nodes.add(node)
                        nodes.append({
                            "id": node,
                            "canonical_name": node_data.get("canonical_name", node),
                            "entity_type": node_data.get("entity_type", "CONCEPT"),
                            "hop_distance": dist
                        })

        for node_id in visited_nodes:
            for neighbor in self.graph.successors(node_id):
                if neighbor in visited_nodes:
                    edge_dict = self.graph.get_edge_data(node_id, neighbor)
                    for key, data in edge_dict.items():
                        edge_sig = (node_id, neighbor, data.get("explicit_rel", "RELATION"))
                        if edge_sig not in visited_edges and len(edges) < limit:
                            visited_edges.add(edge_sig)
                            edges.append({
                                "source": self.graph.nodes[node_id].get("canonical_name", node_id),
                                "target": self.graph.nodes[neighbor].get("canonical_name", neighbor),
                                "relation": data.get("explicit_rel", "RELATION"),
                                "description": data.get("description", ""),
                                "weight": data.get("weight", 1.0)
                            })

        return GraphSubgraph(
            nodes=nodes,
            edges=edges,
            seed_entities=seed_entities,
            hop_depth=max_hops
        )

    def get_chunks_for_entities(self, seed_entities: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        scored_chunks: Dict[str, Dict[str, Any]] = {}
        for ent in seed_entities:
            chunk_ids = self.entity_to_chunks.get(ent, set())
            for cid in chunk_ids:
                if cid in self.chunks:
                    chunk = self.chunks[cid]
                    if cid not in scored_chunks:
                        scored_chunks[cid] = {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "text": chunk.text,
                            "page_number": chunk.page_number,
                            "section_title": chunk.section_title,
                            "mentioned_entities": [ent],
                            "graph_relevance_score": 1.0
                        }
                    else:
                        scored_chunks[cid]["mentioned_entities"].append(ent)
                        scored_chunks[cid]["graph_relevance_score"] += 1.0

        sorted_chunks = sorted(scored_chunks.values(), key=lambda x: x["graph_relevance_score"], reverse=True)
        return sorted_chunks[:limit]

    def get_multi_doc_bridges(self, entity_names: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        bridges = []
        for ent in entity_names:
            chunk_ids = list(self.entity_to_chunks.get(ent, set()))
            if len(chunk_ids) >= 2:
                for i in range(len(chunk_ids)):
                    for j in range(i + 1, len(chunk_ids)):
                        c1 = self.chunks[chunk_ids[i]]
                        c2 = self.chunks[chunk_ids[j]]
                        if c1.doc_id != c2.doc_id:
                            d1 = self.documents.get(c1.doc_id)
                            d2 = self.documents.get(c2.doc_id)
                            bridges.append({
                                "shared_entity": ent,
                                "doc1_title": d1.metadata.title if d1 else c1.doc_id,
                                "chunk1_id": c1.chunk_id,
                                "chunk1_text": c1.text[:200],
                                "doc2_title": d2.metadata.title if d2 else c2.doc_id,
                                "chunk2_id": c2.chunk_id,
                                "chunk2_text": c2.text[:200]
                            })
                            if len(bridges) >= limit:
                                return bridges
        return bridges


class Neo4jGraphClient:
    """
    Neo4j Production Client with Pooled Connections, Fast Cypher Execution,
    and In-Memory Graph Fallback.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "graphrag_password",
        database: str = "neo4j",
        max_pool_size: int = 50,
        use_mock_fallback: bool = True
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_pool_size = max_pool_size
        self.use_mock_fallback = use_mock_fallback

        self._driver: Optional[Any] = None
        self._is_connected: bool = False
        self._in_memory_store = InMemoryGraphStore()

        self._connect()

    def _connect(self):
        if not NEO4J_LIB_AVAILABLE:
            logger.info("Neo4j Python driver not available. Using In-Memory GraphStore.")
            self._is_connected = False
            return

        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=self.max_pool_size
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            self._is_connected = True
            logger.info(f"Successfully connected to Neo4j instance at {self.uri}")
            self.init_schema()
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j ({e}). Active Fallback: InMemoryGraphStore.")
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def init_schema(self):
        """Initializes unique constraints and fulltext indexes on Neo4j."""
        if not self._is_connected:
            return

        with self._driver.session(database=self.database) as session:
            for constraint in SCHEMA_CONSTRAINTS:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint notice: {e}")

            for index in SCHEMA_INDEXES:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index notice: {e}")
            logger.info("Neo4j graph schema constraints and indexes validated.")

    def batch_upsert_documents_and_chunks(self, documents: List[Document], chunks: List[Chunk]):
        # Always mirror in in-memory store for fallback/tests
        self._in_memory_store.add_documents_and_chunks(documents, chunks)

        if not self._is_connected:
            return

        doc_params = [
            {
                "doc_id": d.doc_id,
                "title": d.metadata.title or d.metadata.filename,
                "filename": d.metadata.filename,
                "total_pages": d.metadata.total_pages,
                "created_at": d.metadata.created_at
            }
            for d in documents
        ]

        chunk_params = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "page_number": c.page_number,
                "section_title": c.section_title or "General",
                "token_count": c.token_count,
                "vector_id": c.vector_id or c.chunk_id,
                "next_chunk_id": c.next_chunk_id
            }
            for c in chunks
        ]

        with self._driver.session(database=self.database) as session:
            # Batch upsert Documents
            session.run(
                """
                UNWIND $docs AS d
                MERGE (doc:Document {doc_id: d.doc_id})
                SET doc.title = d.title,
                    doc.filename = d.filename,
                    doc.total_pages = d.total_pages,
                    doc.created_at = d.created_at
                """,
                docs=doc_params
            )

            # Batch upsert Chunks and connect to Document
            session.run(
                """
                UNWIND $chunks AS c
                MERGE (chunk:Chunk {chunk_id: c.chunk_id})
                SET chunk.doc_id = c.doc_id,
                    chunk.text = c.text,
                    chunk.page_number = c.page_number,
                    chunk.section_title = c.section_title,
                    chunk.token_count = c.token_count,
                    chunk.vector_id = c.vector_id
                WITH chunk, c
                MATCH (doc:Document {doc_id: c.doc_id})
                MERGE (doc)-[:CONTAINS_CHUNK {page: c.page_number}]->(chunk)
                """,
                chunks=chunk_params
            )

            # Batch upsert NEXT_CHUNK relationships
            session.run(
                """
                UNWIND $chunks AS c
                WITH c WHERE c.next_chunk_id IS NOT NULL
                MATCH (c1:Chunk {chunk_id: c.chunk_id})
                MATCH (c2:Chunk {chunk_id: c.next_chunk_id})
                MERGE (c1)-[:NEXT_CHUNK]->(c2)
                """,
                chunks=chunk_params
            )

    def batch_upsert_entities_and_relations(
        self, entities: List[Entity], relations: List[Relationship], chunks: List[Chunk]
    ):
        self._in_memory_store.add_entities_and_relations(entities, relations, chunks)

        if not self._is_connected:
            return

        ent_params = [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "canonical_name": e.canonical_name,
                "entity_type": e.entity_type,
                "confidence": e.confidence,
                "chunk_ids": e.source_chunk_ids
            }
            for e in entities
        ]

        rel_params = [
            {
                "source": r.source_entity,
                "target": r.target_entity,
                "relation_type": r.relation_type,
                "description": r.description or "",
                "weight": r.weight,
                "confidence": r.confidence
            }
            for r in relations
        ]

        with self._driver.session(database=self.database) as session:
            # Batch upsert Entities
            session.run(
                """
                UNWIND $entities AS e
                MERGE (ent:Entity {canonical_name: e.canonical_name})
                SET ent.name = e.name,
                    ent.entity_type = e.entity_type,
                    ent.confidence = e.confidence
                WITH ent, e
                UNWIND e.chunk_ids AS cid
                MATCH (c:Chunk {chunk_id: cid})
                MERGE (c)-[:MENTIONS {confidence: e.confidence}]->(ent)
                """,
                entities=ent_params
            )

            # Batch upsert Relationships
            session.run(
                """
                UNWIND $rels AS r
                MERGE (s:Entity {canonical_name: r.source})
                MERGE (t:Entity {canonical_name: r.target})
                MERGE (s)-[rel:RELATION {relation_type: r.relation_type}]->(t)
                SET rel.description = r.description,
                    rel.weight = r.weight,
                    rel.confidence = r.confidence
                """,
                rels=rel_params
            )

    def get_subgraph(self, seed_entities: List[str], max_hops: int = 2, limit: int = 25) -> GraphSubgraph:
        """
        Executes multi-hop traversal in Neo4j (or In-Memory store if Neo4j is offline).
        """
        if not self._is_connected:
            return self._in_memory_store.get_subgraph_for_entities(seed_entities, max_hops, limit)

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    CYPHER_2_HOP_SUBGRAPH,
                    seed_entities=seed_entities,
                    limit=limit
                )
                nodes = []
                edges = []
                seen_nodes = set()

                for record in result:
                    src = record["source"]
                    tgt = record["target"]
                    tgt_type = record["target_type"]
                    rel = record["relation"]
                    desc = record["description"]
                    weight = record["weight"]

                    if src not in seen_nodes:
                        seen_nodes.add(src)
                        nodes.append({"canonical_name": src, "entity_type": "CONCEPT"})
                    if tgt not in seen_nodes:
                        seen_nodes.add(tgt)
                        nodes.append({"canonical_name": tgt, "entity_type": tgt_type})

                    edges.append({
                        "source": src,
                        "target": tgt,
                        "relation": rel,
                        "description": desc,
                        "weight": weight
                    })

                return GraphSubgraph(
                    nodes=nodes,
                    edges=edges,
                    seed_entities=seed_entities,
                    hop_depth=max_hops
                )
        except Exception as e:
            logger.error(f"Neo4j query error: {e}, falling back to in-memory store")
            return self._in_memory_store.get_subgraph_for_entities(seed_entities, max_hops, limit)

    def get_grounded_chunks_for_entities(self, seed_entities: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top grounded chunks referencing seed entities.
        """
        if not self._is_connected:
            return self._in_memory_store.get_chunks_for_entities(seed_entities, limit)

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    CYPHER_GET_CHUNKS_FOR_ENTITIES,
                    entity_names=seed_entities,
                    limit=limit
                )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Neo4j chunk retrieval error: {e}")
            return self._in_memory_store.get_chunks_for_entities(seed_entities, limit)

    def get_multi_doc_bridges(self, entity_names: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Discovers cross-document entity bridges.
        """
        if not self._is_connected:
            return self._in_memory_store.get_multi_doc_bridges(entity_names, limit)

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    CYPHER_MULTI_DOC_BRIDGES,
                    entity_names=entity_names,
                    limit=limit
                )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Neo4j bridge query error: {e}")
            return self._in_memory_store.get_multi_doc_bridges(entity_names, limit)

    def close(self):
        if self._driver:
            self._driver.close()
