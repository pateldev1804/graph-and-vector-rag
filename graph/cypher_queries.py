"""
Production Cypher Query Templates for Hybrid GraphRAG.
Optimized for multi-hop graph traversal, cross-document entity bridging,
and grounded context retrieval with sub-40ms execution time.
"""

# 1-Hop Neighborhood Expansion
CYPHER_1_HOP_NEIGHBORS = """
MATCH (e:Entity)
WHERE e.canonical_name IN $entity_names OR e.name IN $entity_names
OPTIONAL MATCH (e)-[r:RELATION]-(target:Entity)
RETURN e.canonical_name AS source,
       type(r) AS relation_type,
       r.relation_type AS explicit_rel,
       r.description AS description,
       r.weight AS weight,
       target.canonical_name AS target,
       target.entity_type AS target_type
LIMIT $limit
"""

# 2-Hop Multi-Hop Dependency Expansion
CYPHER_2_HOP_SUBGRAPH = """
MATCH (seed:Entity)
WHERE seed.canonical_name IN $seed_entities OR seed.name IN $seed_entities
MATCH path = (seed)-[r1:RELATION]-(hop1:Entity)-[r2:RELATION*0..1]-(hop2:Entity)
WITH seed, hop1, hop2, relationships(path) AS rels
UNWIND rels AS rel
RETURN DISTINCT 
    startNode(rel).canonical_name AS source,
    rel.relation_type AS relation,
    rel.description AS description,
    rel.weight AS weight,
    endNode(rel).canonical_name AS target,
    endNode(rel).entity_type AS target_type
LIMIT $limit
"""

# Multi-Document Entity Bridge: Solves Context Fragmentation across disparate documents
CYPHER_MULTI_DOC_BRIDGES = """
MATCH (e:Entity)
WHERE e.canonical_name IN $entity_names
MATCH (c1:Chunk)-[:MENTIONS]->(e)<-[:MENTIONS]-(c2:Chunk)
WHERE c1.doc_id <> c2.doc_id
MATCH (d1:Document {doc_id: c1.doc_id})
MATCH (d2:Document {doc_id: c2.doc_id})
RETURN e.canonical_name AS shared_entity,
       d1.title AS doc1_title,
       c1.chunk_id AS chunk1_id,
       c1.text AS chunk1_text,
       d2.title AS doc2_title,
       c2.chunk_id AS chunk2_id,
       c2.text AS chunk2_text
LIMIT $limit
"""

# Shortest-Path Query for Multi-Entity Reasoning
CYPHER_SHORTEST_PATH = """
MATCH (start:Entity {canonical_name: $start_entity}), (end_node:Entity {canonical_name: $end_entity})
MATCH path = shortestPath((start)-[:RELATION*..4]-(end_node))
RETURN [n in nodes(path) | n.canonical_name] AS path_nodes,
       [r in relationships(path) | r.relation_type] AS path_relations,
       length(path) AS hop_length
"""

# Grounded Chunk Retrieval for Subgraph Entities
CYPHER_GET_CHUNKS_FOR_ENTITIES = """
MATCH (e:Entity)
WHERE e.canonical_name IN $entity_names
MATCH (c:Chunk)-[m:MENTIONS]->(e)
RETURN c.chunk_id AS chunk_id,
       c.doc_id AS doc_id,
       c.text AS text,
       c.page_number AS page_number,
       c.section_title AS section_title,
       collect(e.canonical_name) AS mentioned_entities,
       sum(coalesce(m.confidence, 1.0)) AS graph_relevance_score
ORDER BY graph_relevance_score DESC
LIMIT $limit
"""

# Linear Sequence Context Expansion (Retrieve predecessor and successor chunks)
CYPHER_GET_CHUNK_SURROUNDING_CONTEXT = """
MATCH (c:Chunk {chunk_id: $chunk_id})
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
RETURN c.chunk_id AS target_id,
       c.text AS target_text,
       prev.text AS prev_text,
       next.text AS next_text
"""
