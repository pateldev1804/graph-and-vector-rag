"""
Custom Cypher Graph Schemas, Constraints, and Index Definitions for GraphRAG.
Defines explicit node schemas, relational constraints, and fulltext search indexes.
"""

from typing import List

# Schema Constraints: Ensure uniqueness and fast primary key lookups
SCHEMA_CONSTRAINTS: List[str] = [
    # Document Constraints
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    
    # Chunk Constraints
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    
    # Entity Constraints
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT entity_canonical_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE",
    
    # Section Constraints
    "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
]

# Schema Indexes: Accelerate graph traversals, entity lookups, and multi-hop hops
SCHEMA_INDEXES: List[str] = [
    # Lookup Indexes
    "CREATE INDEX chunk_doc_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    
    # Fulltext Search Indexes for fast hybrid entity & text matching
    "CREATE FULLTEXT INDEX entity_fulltext_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.canonical_name, e.description]",
    "CREATE FULLTEXT INDEX chunk_fulltext_idx IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text, c.section_title]",
]

# Graph Data Model Schema Description
GRAPH_SCHEMA_DOC = """
=== GRAPH SCHEMA SPECIFICATION ===
NODES:
  (:Document {doc_id: STRING, title: STRING, filename: STRING, total_pages: INT, created_at: FLOAT})
  (:Section  {section_id: STRING, title: STRING, doc_id: STRING, level: INT})
  (:Chunk    {chunk_id: STRING, doc_id: STRING, text: STRING, page_number: INT, token_count: INT, vector_id: STRING})
  (:Entity   {entity_id: STRING, name: STRING, canonical_name: STRING, entity_type: STRING, confidence: FLOAT})
    Specific Labels: (:Organization), (:Person), (:Technology), (:Metric), (:Concept), (:Policy)

RELATIONSHIPS:
  (:Document)-[:CONTAINS_CHUNK {page: INT}]->(:Chunk)
  (:Chunk)-[:NEXT_CHUNK]->(:Chunk)                       [Preserves linear context]
  (:Chunk)-[:MENTIONS {confidence: FLOAT}]->(:Entity)    [Chunk to Entity Grounding]
  (:Entity)-[:RELATION {
      relation_type: STRING, 
      description: STRING, 
      weight: FLOAT, 
      confidence: FLOAT, 
      source_chunk_id: STRING, 
      doc_id: STRING
  }]->(:Entity)                                          [Multi-Hop Knowledge Triples]
  (:Document)-[:BRIDGES_TO {shared_entities: INT}]->(:Document) [Cross-doc coherence]
"""
