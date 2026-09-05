"""
High-Throughput PDF Parser with Structural Preservation & OCR Fallback.
Uses PyMuPDF (fitz) for ultra-fast native extraction and layout analysis.
Falls back to pytesseract OCR for scanned or image-based pages.
"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pymupdf as fitz
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from core.schema import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class LayoutBlock(dict):
    """Represents a structured text or table block extracted from a page."""
    pass


class PDFParser:
    """
    Production-grade PDF Ingestion Parser.
    Preserves document structure (headings, hierarchy, paragraphs, tables)
    and falls back to OCR when page text density is low.
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_min_char_threshold: int = 50,
        ocr_dpi: int = 200,
        ocr_lang: str = "eng"
    ):
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_min_char_threshold = ocr_min_char_threshold
        self.ocr_dpi = ocr_dpi
        self.ocr_lang = ocr_lang

    def parse(self, file_path: str) -> Document:
        """
        Parses a PDF file from disk, preserving layout hierarchy and running OCR if needed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_size = path.stat().st_size
        doc = fitz.open(str(path))
        total_pages = len(doc)

        metadata_dict = doc.metadata or {}
        doc_metadata = DocumentMetadata(
            source_path=str(path.absolute()),
            filename=path.name,
            total_pages=total_pages,
            doc_type="pdf",
            file_size_bytes=file_size,
            title=metadata_dict.get("title") or path.stem,
            author=metadata_dict.get("author") or "Unknown",
            custom_metadata={
                "creation_date": metadata_dict.get("creationDate", ""),
                "producer": metadata_dict.get("producer", ""),
            }
        )

        # 0. Check for password/encryption
        if doc.is_encrypted:
            if not doc.authenticate(""):
                logger.warning(f"Encrypted PDF detected ({file_path}), skipping password-protected document.")
                doc.close()
                return Document(
                    raw_text="",
                    metadata=doc_metadata,
                    page_count=total_pages
                )

        page_texts: Dict[int, str] = {}
        ocr_applied_pages: List[int] = []
        full_doc_lines: List[str] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Try structured text extraction via PyMuPDF dict layout + Table extraction
            page_text, blocks = self._extract_page_structure(page)

            # 2. Check if text is sparse (scanned page / raster PDF)
            if len(page_text.strip()) < self.ocr_min_char_threshold and self.enable_ocr:
                ocr_text = self._run_ocr_on_page(page)
                if len(ocr_text.strip()) > len(page_text.strip()):
                    page_text = ocr_text
                    ocr_applied_pages.append(page_num)

            page_texts[page_num] = page_text
            if page_text:
                full_doc_lines.append(f"--- Page {page_num} ---\n{page_text}")

            # Free page resource from memory
            page = None

        doc.close()

        raw_text = "\n\n".join(full_doc_lines)
        return Document(
            raw_text=raw_text,
            metadata=doc_metadata,
            page_count=total_pages,
            page_texts=page_texts,
            ocr_applied_pages=ocr_applied_pages
        )

    def _extract_page_structure(self, page: fitz.Page) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts text and markdown tables from PyMuPDF page dictionary while detecting headings.
        """
        page_lines: List[str] = []
        structured_blocks: List[Dict[str, Any]] = []

        # 1. Extract Native Tables (PyMuPDF find_tables)
        table_markdowns: List[str] = []
        try:
            tabs = page.find_tables()
            if tabs.tables:
                for tab in tabs:
                    df_markdown = tab.to_markdown()
                    if df_markdown.strip():
                        table_markdowns.append(df_markdown.strip())
        except Exception as e:
            logger.debug(f"Table detection notice: {e}")

        # 2. Extract Text Blocks with Font Hierarchy
        text_page = page.get_text("dict")
        blocks = text_page.get("blocks", [])

        # Sort blocks by vertical position then horizontal (reading order)
        blocks = sorted(blocks, key=lambda b: (round(b.get("bbox", [0, 0, 0, 0])[1] / 10) * 10, b.get("bbox", [0, 0, 0, 0])[0]))

        font_sizes: List[float] = []
        for b in blocks:
            if b.get("type") == 0:  # Text block
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_sizes.append(span.get("size", 10.0))

        avg_font_size = sum(font_sizes) / max(len(font_sizes), 1) if font_sizes else 11.0

        for b in blocks:
            if b.get("type") == 0:  # Text block
                block_text_lines = []
                is_heading = False
                heading_level = 0

                for line in b.get("lines", []):
                    line_spans = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        span_size = span.get("size", 10.0)
                        if span_size > avg_font_size * 1.3:
                            is_heading = True
                            heading_level = 1 if span_size > avg_font_size * 1.6 else 2
                        line_spans.append(span_text)
                    
                    line_content = "".join(line_spans).strip()
                    if line_content:
                        block_text_lines.append(line_content)

                if block_text_lines:
                    full_block_text = "\n".join(block_text_lines)
                    prefix = "## " if is_heading and heading_level == 2 else ("# " if is_heading else "")
                    page_lines.append(f"{prefix}{full_block_text}")
                    structured_blocks.append({
                        "text": full_block_text,
                        "bbox": b.get("bbox"),
                        "is_heading": is_heading,
                        "heading_level": heading_level,
                    })

        # Append structured tables if detected
        if table_markdowns:
            page_lines.append("\n### Structured Tables:\n" + "\n\n".join(table_markdowns))

        return "\n\n".join(page_lines), structured_blocks

    def _run_ocr_on_page(self, page: fitz.Page) -> str:
        """
        Renders the page to high-res pixmap and runs Tesseract OCR.
        """
        try:
            zoom = self.ocr_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image, lang=self.ocr_lang)
            # Explicit cleanup of pixmap
            pix = None
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed on page: {e}")
            return ""
