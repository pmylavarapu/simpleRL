from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


SpanSource = Literal["text_layer", "ocr"]
DocSource = Literal["text_layer", "ocr", "mixed"]


class Span(BaseModel):
    span_id: str
    doc_id: str
    page: int  # 1-based page number
    bbox: Tuple[float, float, float, float]  # normalized 0..1, (x0, y0, x1, y1), top-left origin
    text: str
    source: SpanSource


class Document(BaseModel):
    doc_id: str
    name: str
    page_count: int
    source: DocSource


class IngestResponse(BaseModel):
    session_id: str
    documents: List[Document]
    spans: List[Span]


# ---------- Extraction (deterministic) ----------


class Extraction(BaseModel):
    """A structured fact pulled deterministically from spans (e.g. a med, lab, EF)."""

    kind: Literal["medication", "lab", "echo", "cath"]
    name: str
    value: Optional[str] = None
    units: Optional[str] = None
    qualifier: Optional[str] = None  # e.g. "abnormal", "reduced", "severe"
    source_span_ids: List[str] = Field(default_factory=list)


# ---------- One-pager ----------


SectionKey = Literal[
    "hpi",
    "family_history",
    "social_history",
    "past_medical_history",
    "past_surgical_history",
    "medications",
    "objective",
    "labs",
    "cardiology_imaging_procedures",
]


SECTION_TITLES: dict[str, str] = {
    "hpi": "History of Present Illness",
    "family_history": "Family History",
    "social_history": "Social History",
    "past_medical_history": "Past Medical History",
    "past_surgical_history": "Past Surgical History",
    "medications": "Medications",
    "objective": "Objective",
    "labs": "Labs",
    "cardiology_imaging_procedures": "Cardiology Imaging & Procedures",
}


class SectionItem(BaseModel):
    text: str
    importance: int = Field(ge=1, le=5)
    evidence_span_ids: List[str] = Field(default_factory=list)


class Section(BaseModel):
    key: SectionKey
    title: str
    items: List[SectionItem] = Field(default_factory=list)


class GuidelineCitation(BaseModel):
    chunk_id: str
    document: str  # the guideline PDF filename
    page: int
    snippet: str
    score: float


class Problem(BaseModel):
    name: str
    summary: str
    importance: int = Field(ge=1, le=5)
    evidence_span_ids: List[str] = Field(default_factory=list)
    plan: Optional[str] = None  # None ⇒ "no guideline-backed recommendation"
    plan_citations: List[GuidelineCitation] = Field(default_factory=list)


class OnePager(BaseModel):
    sections: List[Section]
    problems: List[Problem]
    warnings: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "Informational only. Generated from uploaded records; not a medical device, "
        "not for clinical decision-making."
    )
