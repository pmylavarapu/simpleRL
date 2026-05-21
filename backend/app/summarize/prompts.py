"""Prompt strings. Section synthesis and problem extraction share the same patient
context (cached); only the system prompt and the per-section question vary."""

from __future__ import annotations

import json
from typing import Iterable

from ..models import Extraction, Span


SECTION_SYSTEM = """You synthesize a medical record summary from extracted document spans.

HARD RULES — violations cause the item to be dropped:
1. Every item you emit MUST include >=1 span_id from the candidate list as evidence.
2. NEVER invent facts. If the spans do not support an item, omit it.
3. Importance is your judgment of clinical relevance to a cardiologist (1=trivial, 5=critical).
4. Keep each item to one short sentence. No preamble, no narrative paragraphs.
5. Output JSON matching the provided schema. No prose outside the JSON.
"""


PROBLEM_SYSTEM = """You build a prioritized problem list from a patient's record.

HARD RULES:
1. Only include problems supported by >=1 span_id from the candidate list.
2. List at most 7 problems, ordered by clinical importance (severity x recency x persistence).
3. For each problem, write a `guideline_query` suitable for ACC/AHA cardiology guideline search.
4. Importance: 5=imminent risk, 4=major active disease, 3=significant comorbidity, 2=stable chronic, 1=incidental.
5. Output JSON matching the schema.
"""


PLAN_SYSTEM = """You write a plan for one clinical problem, citing ACC/AHA guideline passages.

HARD RULES:
1. Cite ONLY the chunk_ids provided as candidates. NEVER cite anything else.
2. Every clinical assertion in the plan must be directly supported by a cited chunk.
3. If the candidates do not directly address this problem, return plan="" and cited_chunk_ids=[].
4. 1-3 sentences. No hedging, no generalities. Specific guideline-backed recommendations only.
5. Output JSON matching the schema.
"""


def patient_context_text(spans: list[Span], extractions: list[Extraction]) -> str:
    """Single shared block used as the cached prefix for every section/problem call."""
    span_lines = [
        {"id": s.span_id, "doc": s.doc_id, "page": s.page, "text": s.text}
        for s in spans
    ]
    ext_lines = [
        {
            "kind": e.kind,
            "name": e.name,
            "value": e.value,
            "units": e.units,
            "qualifier": e.qualifier,
            "source_span_ids": e.source_span_ids,
        }
        for e in extractions
    ]
    return (
        "PATIENT CONTEXT — extracted spans and deterministic structured facts.\n"
        "SPANS (id, doc, page, text):\n"
        f"{json.dumps(span_lines, ensure_ascii=False)}\n\n"
        "STRUCTURED EXTRACTIONS:\n"
        f"{json.dumps(ext_lines, ensure_ascii=False)}\n"
    )


def section_user_prompt(section_key: str, section_title: str) -> str:
    return (
        f"Synthesize the **{section_title}** section.\n"
        f"Section key: {section_key}\n\n"
        "Return the most important items for this section, supported by the spans above. "
        "If nothing in the record relates to this section, return an empty items list."
    )


def problem_user_prompt() -> str:
    return (
        "Build the prioritized problem list. Use the spans and structured extractions above. "
        "Top 7 max, ordered by importance descending. Each problem needs a guideline_query "
        "we will use to retrieve ACC/AHA cardiology guideline passages."
    )


def plan_user_prompt(problem_name: str, problem_summary: str, candidates: Iterable[dict]) -> str:
    candidate_json = json.dumps(list(candidates), ensure_ascii=False)
    return (
        f"PROBLEM: {problem_name}\n"
        f"SUMMARY: {problem_summary}\n\n"
        "GUIDELINE CANDIDATES (chunk_id, document, page, snippet):\n"
        f"{candidate_json}\n\n"
        "Write the plan citing only these chunk_ids. If the candidates don't support a plan, "
        "return plan='' and cited_chunk_ids=[]."
    )
