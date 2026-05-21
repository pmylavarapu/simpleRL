"""Rule-based plan generator.

We never let the LLM invent Plan items. Instead, we hand-curate ACC/AHA
recommendations as structured rules. For each rule:
  1. Check if the patient has a matching problem (by string).
  2. Evaluate preconditions against the extracted facts (EF, eGFR, K, LDL,
     BP, current meds, etc.).
  3. If all preconditions pass, emit the rule's `recommendation_text`
     verbatim, with the original guideline citation and the source refs
     of the facts that satisfied the preconditions.

If a problem on the assessment has no matching rule (or no rule's
preconditions are satisfied), we surface it as a "no rule" plan item that
explicitly says so, rather than guessing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .config import GUIDELINES_PATH
from .schemas import (
    AssessmentProblem,
    GuidelineCitation,
    OnePageSummary,
    PageExtraction,
    PlanItem,
    SourceRef,
)


@dataclass
class PatientFacts:
    """Normalized facts pulled from extractions for precondition checks."""

    meds: list[tuple[str, SourceRef]]
    ef_percent: Optional[tuple[float, SourceRef]]
    egfr: Optional[tuple[float, SourceRef]]
    potassium: Optional[tuple[float, SourceRef]]
    ldl: Optional[tuple[float, SourceRef]]
    bp: Optional[tuple[tuple[int, int], SourceRef]]
    cha2ds2vasc: Optional[tuple[int, SourceRef]]
    problems: list[str]  # canonical problem names from assessment


_EF_RE = re.compile(r"(?:LVEF|EF)\s*(?:of\s*)?(\d{1,3})\s*%", re.IGNORECASE)
_BP_RE = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")
_CHADS_RE = re.compile(r"CHA[₂2]DS[₂2]-?VASc\s*(?:of|=|score)?\s*(\d)", re.IGNORECASE)


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(re.sub(r"[^0-9.\-]", "", s))
    except ValueError:
        return None


def collect_patient_facts(
    pages: list[PageExtraction], assessment: list[AssessmentProblem]
) -> PatientFacts:
    meds: list[tuple[str, SourceRef]] = []
    ef: Optional[tuple[float, SourceRef]] = None
    egfr: Optional[tuple[float, SourceRef]] = None
    k: Optional[tuple[float, SourceRef]] = None
    ldl: Optional[tuple[float, SourceRef]] = None
    bp: Optional[tuple[tuple[int, int], SourceRef]] = None
    chads: Optional[tuple[int, SourceRef]] = None

    for p in pages:
        for f in p.facts:
            src = f.source
            text_for_search = f"{f.content} {f.value or ''} {src.snippet}"

            if f.category == "med":
                meds.append((f.content.lower(), src))

            if f.category == "imaging" and ef is None:
                m = _EF_RE.search(text_for_search)
                if m:
                    ef = (float(m.group(1)), src)

            if f.category == "lab":
                name = f.content.lower()
                val = _to_float(f.value)
                if val is None:
                    continue
                if "egfr" in name or "gfr" in name:
                    if egfr is None or f.date and (egfr[0] is None):
                        egfr = (val, src)
                if "potassium" in name or name.strip() == "k" or name.strip() == "k+":
                    k = (val, src)
                if "ldl" in name:
                    ldl = (val, src)

            if f.category == "vitals":
                m = _BP_RE.search(text_for_search)
                if m and bp is None:
                    bp = ((int(m.group(1)), int(m.group(2))), src)

            m = _CHADS_RE.search(text_for_search)
            if m and chads is None:
                chads = (int(m.group(1)), src)

    return PatientFacts(
        meds=meds,
        ef_percent=ef,
        egfr=egfr,
        potassium=k,
        ldl=ldl,
        bp=bp,
        cha2ds2vasc=chads,
        problems=[ap.problem for ap in assessment],
    )


def _problem_matches(problem: str, patterns: list[str]) -> bool:
    p = problem.lower()
    return any(pat.lower() in p for pat in patterns)


def _eval_precondition(
    pre: dict, facts: PatientFacts
) -> tuple[bool, list[SourceRef]]:
    kind = pre["kind"]
    evidence: list[SourceRef] = []

    if kind == "ef_at_most":
        if facts.ef_percent is None:
            return False, []
        val, src = facts.ef_percent
        return val <= pre["value"], [src]

    if kind == "egfr_at_least":
        if facts.egfr is None:
            return False, []
        val, src = facts.egfr
        return val >= pre["value"], [src]

    if kind == "k_at_most":
        if facts.potassium is None:
            return False, []
        val, src = facts.potassium
        return val <= pre["value"], [src]

    if kind == "ldl_at_least":
        if facts.ldl is None:
            return False, []
        val, src = facts.ldl
        return val >= pre["value"], [src]

    if kind == "bp_above":
        if facts.bp is None:
            return False, []
        (s, d), src = facts.bp
        return (s > pre["systolic"] or d > pre["diastolic"]), [src]

    if kind == "cha2ds2vasc_at_least":
        if facts.cha2ds2vasc is None:
            return False, []
        val, src = facts.cha2ds2vasc
        return val >= pre["value"], [src]

    if kind == "no_med_substring_any":
        values = [v.lower() for v in pre["values"]]
        for med, _src in facts.meds:
            if any(v in med for v in values):
                return False, []
        return True, []

    if kind == "any_med_substring_any":
        values = [v.lower() for v in pre["values"]]
        for med, src in facts.meds:
            if any(v in med for v in values):
                return True, [src]
        return False, []

    if kind == "no_high_intensity_statin":
        # High-intensity = atorvastatin 40-80 mg or rosuvastatin 20-40 mg.
        for med, _src in facts.meds:
            m = re.search(r"atorvastatin\s*(\d{1,3})", med)
            if m and int(m.group(1)) >= 40:
                return False, []
            m = re.search(r"rosuvastatin\s*(\d{1,3})", med)
            if m and int(m.group(1)) >= 20:
                return False, []
        return True, []

    if kind == "very_high_risk_ascvd":
        # ACC/AHA 2018: multiple major ASCVD events OR 1 event + multiple
        # high-risk conditions. We approximate: any ASCVD problem PLUS at
        # least 2 of {HTN, DM, CKD, HF, age>=65, current/prior smoker}.
        ascvd_problem = any(
            re.search(r"\b(CAD|ASCVD|PCI|MI|coronary)\b", p, re.IGNORECASE)
            for p in facts.problems
        )
        if not ascvd_problem:
            return False, []
        hr_conditions = 0
        for p in facts.problems:
            if re.search(r"hypertension|HTN", p, re.IGNORECASE):
                hr_conditions += 1
            if re.search(r"diabetes|DM\b|T2DM|T1DM", p, re.IGNORECASE):
                hr_conditions += 1
            if re.search(r"CKD|kidney", p, re.IGNORECASE):
                hr_conditions += 1
            if re.search(r"heart failure|HFrEF|HFpEF", p, re.IGNORECASE):
                hr_conditions += 1
        return hr_conditions >= 2, []

    # Unknown precondition kind -> fail closed.
    return False, []


def generate_plan(
    pages: list[PageExtraction], summary: OnePageSummary, max_items: int = 6
) -> list[PlanItem]:
    rules = json.loads(GUIDELINES_PATH.read_text())["rules"]
    facts = collect_patient_facts(pages, summary.assessment)

    plan: list[PlanItem] = []
    seen_rules: set[str] = set()

    for problem in summary.assessment:
        matched_any = False
        for rule in rules:
            if rule["id"] in seen_rules:
                continue
            if not _problem_matches(problem.problem, rule["problem_match"]):
                continue

            evidence: list[SourceRef] = []
            ok = True
            for pre in rule["preconditions"]:
                pre_ok, pre_ev = _eval_precondition(pre, facts)
                if not pre_ok:
                    ok = False
                    break
                evidence.extend(pre_ev)

            if not ok:
                continue

            matched_any = True
            seen_rules.add(rule["id"])
            plan.append(
                PlanItem(
                    problem=problem.problem,
                    recommendation=rule["recommendation_text"],
                    rationale_for_patient=rule["rationale_for_patient"],
                    priority=rule["priority"],
                    citation=GuidelineCitation(**rule["citation"]),
                    precondition_evidence=evidence,
                )
            )

        if not matched_any:
            # No ACC/AHA rule fits this problem under current evidence.
            # We do NOT invent advice; we say so.
            plan.append(
                PlanItem(
                    problem=problem.problem,
                    recommendation=(
                        "No matching ACC/AHA recommendation in the curated "
                        "rule set for this problem under the current evidence."
                    ),
                    rationale_for_patient=(
                        "Defer to specialty-specific guidelines."
                    ),
                    priority=3,
                    citation=GuidelineCitation(
                        guideline="N/A",
                        year=0,
                        section="No matching rule",
                        cor="-",
                        loe="-",
                    ),
                    precondition_evidence=[],
                )
            )

    plan.sort(key=lambda x: x.priority)
    return plan[:max_items]
