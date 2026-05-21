"""Pydantic schemas for LLM-structured output. Each field is chosen so the validator
can fail fast on unsupported items (missing evidence, out-of-range importance)."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class LLMSectionItem(BaseModel):
    text: str = Field(description="One-sentence summary of a single fact for this section.")
    importance: int = Field(ge=1, le=5, description="1=low, 5=critical")
    evidence_span_ids: List[str] = Field(
        description="span_ids from the provided candidates that directly support this item. Must be non-empty.",
    )


class LLMSectionOutput(BaseModel):
    items: List[LLMSectionItem] = Field(default_factory=list)


class LLMProblem(BaseModel):
    name: str = Field(description="Short problem name, e.g. 'Heart failure with reduced EF'")
    summary: str = Field(description="2-3 sentence summary of the problem from the patient's records.")
    importance: int = Field(ge=1, le=5)
    evidence_span_ids: List[str] = Field(
        description="span_ids supporting this problem. Must be non-empty.",
    )
    guideline_query: str = Field(
        description="Search query for ACC/AHA guideline retrieval. Use clinical terminology.",
    )


class LLMProblemList(BaseModel):
    problems: List[LLMProblem] = Field(default_factory=list)


class LLMPlanOutput(BaseModel):
    plan: str = Field(
        description=(
            "1-3 sentence plan, citing ONLY the provided guideline chunks. "
            "If the chunks do not directly support a plan, return an empty string."
        ),
    )
    cited_chunk_ids: List[str] = Field(
        description="chunk_ids from the provided guideline candidates that the plan relies on.",
        default_factory=list,
    )
