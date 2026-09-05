"""
Hierarchical Semantic Chunker.
Splits parsed documents while preserving section headers, hierarchy breadcrumbs,
and linear sequence links (prev_chunk_id, next_chunk_id).
"""

import re
import uuid
from typing import List, Optional
from core.schema import Document, Chunk


class HierarchicalSemanticChunker:
    """
    Hierarchical Semantic Chunker that preserves document structural hierarchy,
    heading paths, page bounds, and adjacent node pointers.
    """

    def __init__(self, chunk_size: int = 384, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Splits a Document into structured Chunks with section hierarchy and sequence pointers.
        """
        chunks: List[Chunk] = []
        current_section_h1: Optional[str] = None
        current_section_h2: Optional[str] = None

        for page_num, page_text in document.page_texts.items():
            if not page_text.strip():
                continue

            lines = page_text.split("\n")
            current_buffer: List[str] = []
            current_tokens = 0

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Detect Markdown/Structure headings
                if stripped.startswith("# "):
                    current_section_h1 = stripped[2:].strip()
                    current_section_h2 = None
                elif stripped.startswith("## "):
                    current_section_h2 = stripped[3:].strip()

                line_words = stripped.split()
                line_tokens = len(line_words)

                if current_tokens + line_tokens > self.chunk_size and current_buffer:
                    chunk_text = " ".join(current_buffer)
                    section_hierarchy = [h for h in [current_section_h1, current_section_h2] if h]
                    section_title = current_section_h2 or current_section_h1 or "General"

                    chunk = Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=document.doc_id,
                        text=chunk_text,
                        page_number=page_num,
                        section_title=section_title,
                        section_hierarchy=section_hierarchy,
                        token_count=current_tokens,
                        metadata={
                            "filename": document.metadata.filename,
                            "doc_title": document.metadata.title,
                            "page": page_num,
                            "section": section_title
                        }
                    )
                    chunks.append(chunk)

                    # Retain overlap words for semantic continuity
                    overlap_words = " ".join(current_buffer).split()[-self.chunk_overlap:]
                    current_buffer = [" ".join(overlap_words), stripped]
                    current_tokens = len(overlap_words) + line_tokens
                else:
                    current_buffer.append(stripped)
                    current_tokens += line_tokens

            # Flush remaining buffer on page
            if current_buffer:
                chunk_text = " ".join(current_buffer)
                section_hierarchy = [h for h in [current_section_h1, current_section_h2] if h]
                section_title = current_section_h2 or current_section_h1 or "General"

                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=document.doc_id,
                    text=chunk_text,
                    page_number=page_num,
                    section_title=section_title,
                    section_hierarchy=section_hierarchy,
                    token_count=current_tokens,
                    metadata={
                        "filename": document.metadata.filename,
                        "doc_title": document.metadata.title,
                        "page": page_num,
                        "section": section_title
                    }
                )
                chunks.append(chunk)

        # Link adjacent chunks sequentially (Linear sequence preservation)
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunks[i].next_chunk_id = chunks[i + 1].chunk_id

        return chunks
