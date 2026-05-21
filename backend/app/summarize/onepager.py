"""Top-level orchestrator: build the OnePager for a session."""

from __future__ import annotations

from typing import List

from ..config import config
from ..extract.extractors import run_all_extractors
from ..guidelines.store import Retriever, load_index
from ..llm.client import LLMClient, make_llm_client
from ..models import OnePager, Span
from .problems import build_problem_list
from .sections import synthesize_sections, synthesize_sections_deterministic


def _load_retriever() -> tuple[Retriever | None, str | None]:
    try:
        chunks = load_index(config.guidelines_dir)
    except Exception as e:  # noqa: BLE001
        return None, f"Guideline corpus unavailable: {e}"
    if chunks is None:
        return None, (
            f"No guideline index at {config.guidelines_dir}. Run "
            "`python -m app.guidelines build` after dropping ACC/AHA PDFs into that folder."
        )
    if not chunks:
        return None, f"Guideline index empty at {config.guidelines_dir}."
    return Retriever(chunks), None


def build_one_pager(spans: List[Span]) -> OnePager:
    warnings: list[str] = []

    extractions = run_all_extractors(spans)

    llm: LLMClient | None = make_llm_client()
    if llm is None:
        warnings.append(
            f"LLM backend '{config.llm_backend}' unavailable — falling back to "
            "deterministic extraction only. No narrative synthesis, no guideline-backed plans."
        )

    retriever, retriever_warning = _load_retriever()
    if retriever_warning:
        warnings.append(retriever_warning)

    if llm is not None:
        sections = synthesize_sections(llm, spans, extractions)
    else:
        sections = synthesize_sections_deterministic(spans, extractions)

    problems = build_problem_list(llm, spans, extractions, retriever)

    return OnePager(sections=sections, problems=problems, warnings=warnings)
