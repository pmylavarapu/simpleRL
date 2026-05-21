from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models import OnePager
from ..storage import store
from ..summarize.onepager import build_one_pager


router = APIRouter()


@router.post("/api/sessions/{sid}/summary", response_model=OnePager)
def create_summary(sid: str) -> OnePager:
    spans = store.spans(sid)
    if not spans:
        raise HTTPException(status_code=404, detail="no spans for this session")
    return build_one_pager(spans)
