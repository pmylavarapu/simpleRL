"""Section synthesis with evidence-gated validation."""

from __future__ import annotations

from typing import Iterable, List

from ..llm.client import LLMClient
from ..models import SECTION_TITLES, Extraction, Section, SectionItem, Span
from .prompts import SECTION_SYSTEM, patient_context_text, section_user_prompt
from .schemas import LLMSectionOutput


SECTION_ORDER: list[str] = [
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


def _validate(items: list, valid_span_ids: set[str]) -> List[SectionItem]:
    out: List[SectionItem] = []
    for it in items:
        evidence = [sid for sid in it.evidence_span_ids if sid in valid_span_ids]
        if not evidence:
            continue
        out.append(
            SectionItem(
                text=it.text.strip(),
                importance=max(1, min(5, int(it.importance))),
                evidence_span_ids=evidence,
            )
        )
    return out


def synthesize_sections(
    llm: LLMClient,
    spans: List[Span],
    extractions: List[Extraction],
) -> List[Section]:
    valid_span_ids = {s.span_id for s in spans}
    context = patient_context_text(spans, extractions)
    sections: List[Section] = []
    for key in SECTION_ORDER:
        title = SECTION_TITLES[key]
        result = llm.parse_structured(
            system=SECTION_SYSTEM,
            context_text=context,
            user_text=section_user_prompt(key, title),
            response_model=LLMSectionOutput,
        )
        items = _validate(result.items, valid_span_ids)
        sections.append(Section(key=key, title=title, items=items))  # type: ignore[arg-type]
    return sections


# ---------- Deterministic fallback when no LLM is available ----------


def synthesize_sections_deterministic(
    spans: List[Span], extractions: List[Extraction]
) -> List[Section]:
    """Render extractions directly into the structured sections they belong to. No
    narrative synthesis — just facts with provenance."""
    meds = [e for e in extractions if e.kind == "medication"]
    labs = [e for e in extractions if e.kind == "lab"]
    echos = [e for e in extractions if e.kind == "echo"]
    caths = [e for e in extractions if e.kind == "cath"]

    sections: List[Section] = []
    for key in SECTION_ORDER:
        title = SECTION_TITLES[key]
        items: List[SectionItem] = []
        if key == "medications":
            for m in meds:
                items.append(
                    SectionItem(
                        text=f"{m.name} {m.value or ''}".strip(),
                        importance=3,
                        evidence_span_ids=m.source_span_ids,
                    )
                )
        elif key == "labs":
            for lab in labs:
                qual = f" ({lab.qualifier})" if lab.qualifier else ""
                items.append(
                    SectionItem(
                        text=f"{lab.name} {lab.value} {lab.units or ''}{qual}".strip(),
                        importance=4 if lab.qualifier else 2,
                        evidence_span_ids=lab.source_span_ids,
                    )
                )
        elif key == "cardiology_imaging_procedures":
            for e in echos + caths:
                qual = f" ({e.qualifier})" if e.qualifier else ""
                val = f": {e.value} {e.units or ''}".strip() if e.value else ""
                items.append(
                    SectionItem(
                        text=f"{e.name}{val}{qual}".strip(),
                        importance=4 if e.qualifier in ("reduced", "severe", "abnormal") else 3,
                        evidence_span_ids=e.source_span_ids,
                    )
                )
        sections.append(Section(key=key, title=title, items=items))  # type: ignore[arg-type]
    return sections
