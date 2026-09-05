"""
Entity & Relationship Extraction Engine.
Extracts multi-type entities (ORGANIZATION, PERSON, TECHNOLOGY, FINANCIAL_METRIC, CONCEPT)
and semantic relational edges (OWNS, ACQUIRED, DEPENDS_ON, INCREASED_BY, PARTNERS_WITH).
Supports fast pattern/NER heuristics for high-throughput scaling and transformer integration.
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from core.schema import Chunk, Entity, Relationship


class EntityExtractor:
    """
    High-throughput Entity and Relationship Extraction Engine.
    Maps complex entity networks, multi-hop dependencies, and chunk provenance.
    """

    # Precompiled regex patterns for fast extraction
    ORG_PATTERN = re.compile(
        r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Group|Technologies|Holdings|Bank|Labs|Systems|AI))\b',
        re.MULTILINE
    )
    METRIC_PATTERN = re.compile(
        r'(\$[0-9]+(?:\.[0-9]+)?\s*(?:billion|million|trillion|B|M|k)?|[0-9]+(?:\.[0-9]+)?\s*%)',
        re.IGNORECASE
    )
    TECH_PATTERN = re.compile(
        r'\b(Neo4j|Pinecone|LlamaIndex|PyMuPDF|Transformers|PyTorch|CUDA|Kubernetes|Docker|PostgreSQL|Redis|FastAPI|LangChain|GraphRAG|Milvus|Qdrant|BERT|RoBERTa|GPT-4|Gemini|Claude)\b',
        re.IGNORECASE
    )

    # Relational trigger patterns: (Source) -> [TRIGGER] -> (Target)
    RELATION_TRIGGERS = [
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:formally|subsequently|officially|recently|successfully)?\s*(?:acquired|purchased|bought)\s+([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)', re.IGNORECASE), "ACQUIRED", 1.0),
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:is a subsidiary of|owned by)\s+([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)', re.IGNORECASE), "SUBSIDIARY_OF", 0.95),
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:partnered with|collaborated with|signed agreement with)\s+([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)', re.IGNORECASE), "PARTNERS_WITH", 0.9),
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:developed|built|launched|created|released|owns|controls|powers)\s+(?:and owns\s+)?(?:the\s+)?([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Engine|Database|Model|System|Platform|Framework))?)', re.IGNORECASE), "DEVELOPS_AND_OWNS", 0.95),
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:depends on|relies on|integrates with)\s+([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)', re.IGNORECASE), "DEPENDS_ON", 0.9),
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:increased by|grew by|surged)\s+(\$[0-9]+[\w\s]*|[0-9]+%)', re.IGNORECASE), "INCREASED_BY", 0.9),
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:reported revenue of|generated)\s+(\$[0-9]+[\w\s]*)', re.IGNORECASE), "REPORTED_METRIC", 0.9),
        (re.compile(r'([A-Z][A-Za-z0-9\s]+?(?:\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?))?)\s+(?:is located in|headquartered in|based in)\s+([A-Z][A-Za-z0-9\s]+)', re.IGNORECASE), "LOCATED_IN", 0.9),
    ]

    def __init__(self, canonical_map: Optional[Dict[str, str]] = None):
        self.canonical_map = canonical_map or {
            "apple": "Apple Inc.",
            "apple inc": "Apple Inc.",
            "google": "Google LLC",
            "alphabet": "Alphabet Inc.",
            "microsoft": "Microsoft Corp.",
            "amazon": "Amazon.com Inc.",
            "meta": "Meta Platforms Inc.",
            "nvidia": "NVIDIA Corp.",
        }

    def canonicalize(self, name: str) -> str:
        cleaned = re.sub(r'\s+', ' ', name.strip().strip(",;:()\"'!?"))
        cleaned = re.sub(r'\bInc\.?$', 'Inc.', cleaned)
        cleaned = re.sub(r'\bCorp\.?$', 'Corp.', cleaned)
        cleaned = re.sub(r'\bLtd\.?$', 'Ltd.', cleaned)
        cleaned = re.sub(r'\bLLC\.?$', 'LLC', cleaned)
        if cleaned.endswith(".") and not any(cleaned.endswith(suffix) for suffix in ["Inc.", "Corp.", "Ltd.", "Co."]):
            cleaned = cleaned.rstrip(".")
        key = cleaned.lower()
        return self.canonical_map.get(key, cleaned)

    def extract_from_chunk(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extracts entities and relational edges from a single text chunk.
        """
        text = chunk.text
        entities_dict: Dict[str, Entity] = {}
        relationships: List[Relationship] = []

        # 1. Extract Organizations
        for match in self.ORG_PATTERN.finditer(text):
            raw_name = match.group(1).strip()
            canonical = self.canonicalize(raw_name)
            if canonical not in entities_dict:
                entities_dict[canonical] = Entity(
                    name=raw_name,
                    canonical_name=canonical,
                    entity_type="ORGANIZATION",
                    confidence=0.92,
                    source_chunk_ids=[chunk.chunk_id],
                    source_doc_ids=[chunk.doc_id]
                )

        # 2. Extract Technologies
        for match in self.TECH_PATTERN.finditer(text):
            raw_name = match.group(1).strip()
            canonical = self.canonicalize(raw_name)
            if canonical not in entities_dict:
                entities_dict[canonical] = Entity(
                    name=raw_name,
                    canonical_name=canonical,
                    entity_type="TECHNOLOGY",
                    confidence=0.95,
                    source_chunk_ids=[chunk.chunk_id],
                    source_doc_ids=[chunk.doc_id]
                )

        # 3. Extract Metrics
        for match in self.METRIC_PATTERN.finditer(text):
            raw_val = match.group(1).strip()
            if raw_val not in entities_dict:
                entities_dict[raw_val] = Entity(
                    name=raw_val,
                    canonical_name=raw_val,
                    entity_type="METRIC",
                    confidence=0.88,
                    source_chunk_ids=[chunk.chunk_id],
                    source_doc_ids=[chunk.doc_id]
                )

        # 4. Extract Relationships & Triples
        known_entity_names = list(entities_dict.keys())

        for pattern, rel_type, base_conf in self.RELATION_TRIGGERS:
            for match in pattern.finditer(text):
                src_raw = match.group(1).strip()
                tgt_raw = match.group(2).strip()

                src_can = self.canonicalize(src_raw)
                tgt_can = self.canonicalize(tgt_raw)

                # Disambiguate against known organizations/technologies
                for known in known_entity_names:
                    if known in src_raw or src_raw.endswith(known):
                        src_can = known
                        break
                for known in known_entity_names:
                    if known in tgt_raw or tgt_raw.startswith(known):
                        tgt_can = known
                        break

                # Ensure source and target entities exist
                if src_can not in entities_dict and len(src_can) > 2:
                    entities_dict[src_can] = Entity(
                        name=src_raw,
                        canonical_name=src_can,
                        entity_type="CONCEPT",
                        confidence=0.80,
                        source_chunk_ids=[chunk.chunk_id],
                        source_doc_ids=[chunk.doc_id]
                    )

                if tgt_can not in entities_dict and len(tgt_can) > 1:
                    entities_dict[tgt_can] = Entity(
                        name=tgt_raw,
                        canonical_name=tgt_can,
                        entity_type="CONCEPT" if not tgt_can.startswith("$") else "METRIC",
                        confidence=0.80,
                        source_chunk_ids=[chunk.chunk_id],
                        source_doc_ids=[chunk.doc_id]
                    )

                rel = Relationship(
                    source_entity=src_can,
                    target_entity=tgt_can,
                    relation_type=rel_type,
                    description=f"{src_can} {rel_type} {tgt_can}",
                    confidence=base_conf,
                    source_chunk_ids=[chunk.chunk_id],
                    source_doc_ids=[chunk.doc_id]
                )
                relationships.append(rel)

        return list(entities_dict.values()), relationships

    def extract_from_chunks(
        self, chunks: List[Chunk]
    ) -> Tuple[Dict[str, Entity], List[Relationship]]:
        """
        Extracts and merges entities and relationships across a batch of chunks.
        """
        all_entities: Dict[str, Entity] = {}
        all_relationships: List[Relationship] = []

        for chunk in chunks:
            ents, rels = self.extract_from_chunk(chunk)
            for ent in ents:
                if ent.canonical_name in all_entities:
                    existing = all_entities[ent.canonical_name]
                    if chunk.chunk_id not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chunk.chunk_id)
                    if chunk.doc_id not in existing.source_doc_ids:
                        existing.source_doc_ids.append(chunk.doc_id)
                else:
                    all_entities[ent.canonical_name] = ent

            all_relationships.extend(rels)

        return all_entities, all_relationships
