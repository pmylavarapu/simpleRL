from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..ingest.pipeline import ingest_pdf
from ..models import Document, IngestResponse, Span
from ..storage import store


router = APIRouter()


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    sid = store.new_session()
    documents: List[Document] = []
    all_spans: List[Span] = []

    for f in files:
        name = f.filename or "document.pdf"
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"not a PDF: {name}")

        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty upload: {name}")

        # Write to a temp path inside the session, then run the pipeline (which
        # mints the real doc_id), then rename to that doc_id.pdf.
        import uuid as _uuid

        tmp_did = _uuid.uuid4().hex[:12]
        tmp_path = store.pdf_path(sid, tmp_did)
        tmp_path.write_bytes(data)
        try:
            doc, spans = ingest_pdf(tmp_path, name)
        except Exception as e:  # noqa: BLE001
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"ingest failed for {name}: {e}")

        final_path = store.pdf_path(sid, doc.doc_id)
        if final_path != tmp_path:
            tmp_path.rename(final_path)

        store.add(sid, doc, spans)
        documents.append(doc)
        all_spans.extend(spans)

    return IngestResponse(session_id=sid, documents=documents, spans=all_spans)


@router.get("/api/sessions/{sid}/documents/{did}/pdf")
def get_pdf(sid: str, did: str) -> FileResponse:
    path = store.pdf_path(sid, did)
    if not path.exists():
        raise HTTPException(status_code=404, detail="pdf not found")
    return FileResponse(str(path), media_type="application/pdf")


@router.get("/api/sessions/{sid}/spans", response_model=List[Span])
def get_spans(sid: str) -> List[Span]:
    return store.spans(sid)


@router.get("/api/sessions/{sid}/documents", response_model=List[Document])
def get_documents(sid: str) -> List[Document]:
    return store.documents(sid)
