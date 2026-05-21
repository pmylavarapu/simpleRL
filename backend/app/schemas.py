from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


FactCategory = Literal[
    "demographic",
    "hpi",
    "fam_hx",
    "soc_hx",
    "pmh",
    "psh",
    "med",
    "allergy",
    "vitals",
    "lab",
    "imaging",
    "procedure",
]


class SourceRef(BaseModel):
    pdf_id: str
    page: int
    snippet: str = Field(
        description="Verbatim text from the source page, used by the frontend "
        "to locate and highlight the supporting evidence."
    )


class ExtractedFact(BaseModel):
    category: FactCategory
    content: str
    date: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    source: SourceRef


class PageExtraction(BaseModel):
    pdf_id: str
    page: int
    raw_text: str
    facts: list[ExtractedFact]


class PatientHeader(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    mrn: Optional[str] = None
    dob: Optional[str] = None


class Claim(BaseModel):
    """A short factual phrase with the source(s) that support it."""

    text: str
    sources: list[SourceRef]


class AssessmentProblem(BaseModel):
    problem: str
    paragraph: list[Claim]


class LabValue(BaseModel):
    value: str
    unit: str
    date: str
    source: SourceRef


class LabPanel(BaseModel):
    name: str
    latest: LabValue
    trend: list[LabValue] = Field(
        default_factory=list,
        description="Prior values, most recent first. Excludes latest.",
    )


class CardiologyStudy(BaseModel):
    study_type: str
    date: str
    key_findings: list[Claim]


class ObjectiveSection(BaseModel):
    labs: list[LabPanel]
    cardiology: list[CardiologyStudy]


class GuidelineCitation(BaseModel):
    guideline: str
    year: int
    section: str
    cor: str
    loe: str
    url: Optional[str] = None
    doi: Optional[str] = None


class PlanItem(BaseModel):
    problem: str
    recommendation: str
    rationale_for_patient: str
    priority: int
    citation: GuidelineCitation
    precondition_evidence: list[SourceRef]


class OnePageSummary(BaseModel):
    patient: PatientHeader
    hpi: list[Claim]
    fam_hx: list[Claim]
    soc_hx: list[Claim]
    pmh: list[Claim]
    psh: list[Claim]
    meds: list[Claim]
    objective: ObjectiveSection
    assessment: list[AssessmentProblem]
    plan: list[PlanItem]


class CaseManifest(BaseModel):
    case_id: str
    pdfs: list[PdfFile]


class PdfFile(BaseModel):
    pdf_id: str
    filename: str
    pages: int


CaseManifest.model_rebuild()
