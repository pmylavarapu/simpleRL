"""Simple JSON-on-disk storage for cases. No DB for v1."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from .config import DATA_DIR
from .schemas import CaseManifest, OnePageSummary, PageExtraction, PdfFile


def new_case_id() -> str:
    return uuid.uuid4().hex[:12]


def case_dir(case_id: str) -> Path:
    return DATA_DIR / case_id


def pdf_path(case_id: str, pdf_id: str) -> Path:
    return case_dir(case_id) / "pdfs" / f"{pdf_id}.pdf"


def manifest_path(case_id: str) -> Path:
    return case_dir(case_id) / "manifest.json"


def extractions_path(case_id: str) -> Path:
    return case_dir(case_id) / "extractions.json"


def summary_path(case_id: str) -> Path:
    return case_dir(case_id) / "summary.json"


def init_case(case_id: str) -> None:
    (case_dir(case_id) / "pdfs").mkdir(parents=True, exist_ok=True)


def save_pdf(case_id: str, filename: str, source_path: Path) -> PdfFile:
    """Copy a PDF into the case directory and return its descriptor."""
    import pymupdf  # lazy import

    pdf_id = uuid.uuid4().hex[:10]
    dest = pdf_path(case_id, pdf_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, dest)
    with pymupdf.open(dest) as doc:
        pages = doc.page_count
    return PdfFile(pdf_id=pdf_id, filename=filename, pages=pages)


def write_manifest(manifest: CaseManifest) -> None:
    manifest_path(manifest.case_id).write_text(manifest.model_dump_json(indent=2))


def read_manifest(case_id: str) -> Optional[CaseManifest]:
    p = manifest_path(case_id)
    if not p.exists():
        return None
    return CaseManifest.model_validate_json(p.read_text())


def write_extractions(case_id: str, extractions: list[PageExtraction]) -> None:
    extractions_path(case_id).write_text(
        json.dumps([e.model_dump() for e in extractions], indent=2)
    )


def read_extractions(case_id: str) -> Optional[list[PageExtraction]]:
    p = extractions_path(case_id)
    if not p.exists():
        return None
    return [PageExtraction.model_validate(x) for x in json.loads(p.read_text())]


def write_summary(case_id: str, summary: OnePageSummary) -> None:
    summary_path(case_id).write_text(summary.model_dump_json(indent=2))


def read_summary(case_id: str) -> Optional[OnePageSummary]:
    p = summary_path(case_id)
    if not p.exists():
        return None
    return OnePageSummary.model_validate_json(p.read_text())
