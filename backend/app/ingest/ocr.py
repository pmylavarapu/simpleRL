from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import pytesseract
from pdf2image import convert_from_path

from ..models import Span


OCR_DPI = 200


def ocr_pages(pdf_path: Path, doc_id: str, page_numbers: List[int]) -> List[Span]:
    """OCR the given 1-based page numbers; return line-level spans with normalized bboxes."""
    if not page_numbers:
        return []

    spans: List[Span] = []
    for page_num in page_numbers:
        images = convert_from_path(
            str(pdf_path),
            dpi=OCR_DPI,
            first_page=page_num,
            last_page=page_num,
        )
        if not images:
            continue
        img = images[0]
        img_w, img_h = img.size
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n = len(data["text"])

        # Group words into lines using (block_num, par_num, line_num).
        groups: Dict[Tuple[int, int, int], List[int]] = {}
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)

        for idxs in groups.values():
            idxs.sort(key=lambda i: data["left"][i])
            x0 = min(data["left"][i] for i in idxs)
            y0 = min(data["top"][i] for i in idxs)
            x1 = max(data["left"][i] + data["width"][i] for i in idxs)
            y1 = max(data["top"][i] + data["height"][i] for i in idxs)
            text = " ".join(
                data["text"][i].strip() for i in idxs if data["text"][i].strip()
            )
            if not text:
                continue
            spans.append(
                Span(
                    span_id=uuid.uuid4().hex[:12],
                    doc_id=doc_id,
                    page=page_num,
                    bbox=(
                        max(0.0, x0 / img_w),
                        max(0.0, y0 / img_h),
                        min(1.0, x1 / img_w),
                        min(1.0, y1 / img_h),
                    ),
                    text=text,
                    source="ocr",
                )
            )
    return spans
