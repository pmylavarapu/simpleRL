from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import storage
from .extraction import extract_page
from .planner import generate_plan
from .schemas import CaseManifest, PdfFile
from .summarization import summarize


router = APIRouter()


@router.post("/api/cases")
async def create_case(files: list[UploadFile]) -> dict:
    case_id = storage.new_case_id()
    storage.init_case(case_id)
    pdfs: list[PdfFile] = []

    for upload in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await upload.read())
            tmp_path = Path(tmp.name)
        try:
            pdfs.append(storage.save_pdf(case_id, upload.filename or "file.pdf", tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    manifest = CaseManifest(case_id=case_id, pdfs=pdfs)
    storage.write_manifest(manifest)
    return manifest.model_dump()


@router.post("/api/cases/{case_id}/extract")
def run_extraction(case_id: str) -> dict:
    manifest = storage.read_manifest(case_id)
    if not manifest:
        raise HTTPException(404, "case not found")

    pages = []
    for pdf in manifest.pdfs:
        pdf_file = storage.pdf_path(case_id, pdf.pdf_id)
        for page_idx in range(pdf.pages):
            pages.append(extract_page(pdf.pdf_id, pdf_file, page_idx))

    storage.write_extractions(case_id, pages)
    return {
        "case_id": case_id,
        "page_count": len(pages),
        "fact_count": sum(len(p.facts) for p in pages),
    }


@router.post("/api/cases/{case_id}/summarize")
def run_summarize(case_id: str) -> dict:
    pages = storage.read_extractions(case_id)
    if pages is None:
        raise HTTPException(400, "run /extract first")
    summary = summarize(pages)
    summary.plan = generate_plan(pages, summary)
    storage.write_summary(case_id, summary)
    return summary.model_dump()


@router.get("/api/cases/{case_id}/summary")
def get_summary(case_id: str) -> dict:
    summary = storage.read_summary(case_id)
    if not summary:
        raise HTTPException(404, "summary not generated")
    return summary.model_dump()


@router.get("/api/cases/{case_id}/manifest")
def get_manifest(case_id: str) -> dict:
    manifest = storage.read_manifest(case_id)
    if not manifest:
        raise HTTPException(404, "case not found")
    return manifest.model_dump()


@router.get("/api/cases/{case_id}/pdfs/{pdf_id}")
def get_pdf(case_id: str, pdf_id: str) -> FileResponse:
    p = storage.pdf_path(case_id, pdf_id)
    if not p.exists():
        raise HTTPException(404, "pdf not found")
    return FileResponse(p, media_type="application/pdf")
