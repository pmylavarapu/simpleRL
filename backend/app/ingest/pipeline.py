from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Tuple

from ..models import Document, Span
from .ocr import ocr_pages
from .pdf_parser import extract_text_layer_spans


def ingest_pdf(pdf_path: Path, original_name: str) -> Tuple[Document, List[Span]]:
    """Run text-layer extraction, then OCR any pages that came back text-sparse."""
    doc_id = uuid.uuid4().hex[:12]
    text_spans, has_text, page_count = extract_text_layer_spans(pdf_path, doc_id)
    pages_to_ocr = [i + 1 for i, ok in enumerate(has_text) if not ok]

    # Drop text-layer spans on pages we're going to OCR (they were too sparse to trust).
    text_spans = [s for s in text_spans if s.page not in pages_to_ocr]
    ocr_spans = ocr_pages(pdf_path, doc_id, pages_to_ocr)

    all_spans = text_spans + ocr_spans
    all_spans.sort(key=lambda s: (s.page, s.bbox[1], s.bbox[0]))

    if pages_to_ocr and any(has_text):
        source = "mixed"
    elif pages_to_ocr:
        source = "ocr"
    else:
        source = "text_layer"

    doc = Document(
        doc_id=doc_id,
        name=original_name,
        page_count=page_count,
        source=source,
    )
    return doc, all_spans
