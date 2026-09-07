"""Knowledge Graph package for Neo4j, Cypher schemas, and multi-hop queries."""
from graph.neo4j_client import Neo4jGraphClient, InMemoryGraphStore
from graph.cypher_schema import SCHEMA_CONSTRAINTS, SCHEMA_INDEXES, GRAPH_SCHEMA_DOC
from graph.cypher_queries import (
    CYPHER_1_HOP_NEIGHBORS,
    CYPHER_2_HOP_SUBGRAPH,
    CYPHER_MULTI_DOC_BRIDGES,
    CYPHER_SHORTEST_PATH,
    CYPHER_GET_CHUNKS_FOR_ENTITIES,
)

__all__ = [
    "Neo4jGraphClient",
    "InMemoryGraphStore",
    "SCHEMA_CONSTRAINTS",
    "SCHEMA_INDEXES",
    "GRAPH_SCHEMA_DOC",
    "CYPHER_1_HOP_NEIGHBORS",
    "CYPHER_2_HOP_SUBGRAPH",
    "CYPHER_MULTI_DOC_BRIDGES",
    "CYPHER_SHORTEST_PATH",
    "CYPHER_GET_CHUNKS_FOR_ENTITIES",
]
