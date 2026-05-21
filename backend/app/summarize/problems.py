"""Problem-list extraction + guideline-gated plan generation."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..config import config
from ..guidelines.store import GuidelineChunk, Retriever
from ..llm.client import LLMClient
from ..models import Extraction, GuidelineCitation, Problem, Span
from .prompts import (
    PLAN_SYSTEM,
    PROBLEM_SYSTEM,
    patient_context_text,
    plan_user_prompt,
    problem_user_prompt,
)
from .schemas import LLMPlanOutput, LLMProblem, LLMProblemList


MAX_PROBLEMS = 7


def extract_problems(
    llm: LLMClient,
    spans: List[Span],
    extractions: List[Extraction],
) -> List[LLMProblem]:
    """LLM emits the prioritized problem list (with guideline queries) — validated below."""
    valid_span_ids = {s.span_id for s in spans}
    result = llm.parse_structured(
        system=PROBLEM_SYSTEM,
        context_text=patient_context_text(spans, extractions),
        user_text=problem_user_prompt(),
        response_model=LLMProblemList,
    )
    out: List[LLMProblem] = []
    for p in result.problems:
        evidence = [sid for sid in p.evidence_span_ids if sid in valid_span_ids]
        if not evidence:
            continue
        out.append(
            LLMProblem(
                name=p.name.strip(),
                summary=p.summary.strip(),
                importance=max(1, min(5, int(p.importance))),
                evidence_span_ids=evidence,
                guideline_query=p.guideline_query.strip() or p.name,
            )
        )
    out.sort(key=lambda p: -p.importance)
    return out[:MAX_PROBLEMS]


def _retrieve_guideline_candidates(
    retriever: Optional[Retriever], query: str
) -> List[Tuple[GuidelineChunk, float]]:
    if retriever is None:
        return []
    hits = retriever.query(query, k=4)
    return [(c, s) for c, s in hits if s >= config.guideline_score_threshold]


def generate_plan_for_problem(
    llm: LLMClient,
    problem: LLMProblem,
    retriever: Optional[Retriever],
) -> Tuple[Optional[str], List[GuidelineCitation]]:
    """Return (plan_text, citations). plan_text=None ⇒ no guideline-backed recommendation."""
    candidates = _retrieve_guideline_candidates(retriever, problem.guideline_query)
    if not candidates:
        return None, []

    by_id = {c.chunk_id: (c, s) for c, s in candidates}
    candidate_payload = [
        {"chunk_id": c.chunk_id, "document": c.document, "page": c.page, "snippet": c.text[:600]}
        for c, _ in candidates
    ]
    plan_out = llm.parse_structured(
        system=PLAN_SYSTEM,
        context_text="",  # no need to recache patient context for the plan call
        user_text=plan_user_prompt(problem.name, problem.summary, candidate_payload),
        response_model=LLMPlanOutput,
    )

    if not plan_out.plan.strip() or not plan_out.cited_chunk_ids:
        return None, []

    cited: List[GuidelineCitation] = []
    for cid in plan_out.cited_chunk_ids:
        if cid not in by_id:
            continue
        chunk, score = by_id[cid]
        cited.append(
            GuidelineCitation(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                page=chunk.page,
                snippet=chunk.text[:400],
                score=score,
            )
        )
    if not cited:
        return None, []
    return plan_out.plan.strip(), cited


def build_problem_list(
    llm: Optional[LLMClient],
    spans: List[Span],
    extractions: List[Extraction],
    retriever: Optional[Retriever],
) -> List[Problem]:
    """Top-level: extract problems via LLM, then generate guideline-gated plans for each.
    Without an LLM, fall back to a degenerate problem list derived from abnormal extractions."""
    if llm is None:
        return _fallback_problems(extractions)

    problems = extract_problems(llm, spans, extractions)
    out: List[Problem] = []
    for p in problems:
        plan_text, citations = generate_plan_for_problem(llm, p, retriever)
        out.append(
            Problem(
                name=p.name,
                summary=p.summary,
                importance=p.importance,
                evidence_span_ids=p.evidence_span_ids,
                plan=plan_text,
                plan_citations=citations,
            )
        )
    return out


def _fallback_problems(extractions: List[Extraction]) -> List[Problem]:
    """No-LLM degenerate path: surface abnormal extractions as 'problems' with no plan.
    Lets the demo render something coherent even without API access."""
    out: List[Problem] = []
    for e in extractions:
        if e.qualifier in ("elevated", "reduced", "abnormal", "severe", "low", "mildly reduced"):
            out.append(
                Problem(
                    name=f"{e.name} {e.qualifier}",
                    summary=f"{e.name} {e.value or ''} {e.units or ''} ({e.qualifier}).".strip(),
                    importance=3,
                    evidence_span_ids=e.source_span_ids,
                    plan=None,
                    plan_citations=[],
                )
            )
    out.sort(key=lambda p: -p.importance)
    return out[:MAX_PROBLEMS]
