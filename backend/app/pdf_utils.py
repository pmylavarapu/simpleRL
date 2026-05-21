"""Rasterize PDF pages to PNG bytes for vision models."""
from __future__ import annotations

import base64
from pathlib import Path

import pymupdf


def page_to_png_b64(pdf_file: Path, page_index: int, dpi: int = 150) -> str:
    with pymupdf.open(pdf_file) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        return base64.standard_b64encode(pix.tobytes("png")).decode("ascii")


def page_count(pdf_file: Path) -> int:
    with pymupdf.open(pdf_file) as doc:
        return doc.page_count
