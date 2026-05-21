from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Tuple

import pdfplumber

from ..models import Span


MIN_TEXT_CHARS_PER_PAGE = 50  # below this, treat the page as scanned and OCR it


def extract_text_layer_spans(
    pdf_path: Path, doc_id: str
) -> Tuple[List[Span], List[bool], int]:
    """Extract line-level spans from the PDF text layer.

    Returns (spans, per_page_has_text, page_count). Bboxes are normalized to
    0..1 of page dimensions with origin top-left.
    """
    spans: List[Span] = []
    has_text: List[bool] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_w = float(page.width or 1.0)
            page_h = float(page.height or 1.0)
            try:
                lines = page.extract_text_lines(
                    layout=False, return_chars=False, strip=True
                )
            except Exception:
                lines = []

            page_chars = sum(len(line.get("text", "")) for line in lines)
            has_text.append(page_chars >= MIN_TEXT_CHARS_PER_PAGE)

            for line in lines:
                text = (line.get("text") or "").strip()
                if not text:
                    continue
                x0 = float(line["x0"])
                x1 = float(line["x1"])
                y0 = float(line["top"])
                y1 = float(line["bottom"])
                spans.append(
                    Span(
                        span_id=uuid.uuid4().hex[:12],
                        doc_id=doc_id,
                        page=page_idx + 1,
                        bbox=(
                            max(0.0, x0 / page_w),
                            max(0.0, y0 / page_h),
                            min(1.0, x1 / page_w),
                            min(1.0, y1 / page_h),
                        ),
                        text=text,
                        source="text_layer",
                    )
                )
        page_count = len(pdf.pages)
    return spans, has_text, page_count
