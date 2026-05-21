"""Deterministic extractors. Each function takes patient spans and emits Extractions,
each tagged with the source span_ids that support it. Regex-driven and conservative —
the LLM step does narrative synthesis on top.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from ..models import Extraction, Span
from .lexicons import CATH_PATTERNS, ECHO_PATTERNS, LAB_PATTERNS, MEDICATIONS


_MED_DOSE_RE = re.compile(
    r"(?i)(?P<name>[a-z][a-z\-/]+(?:\s+[a-z][a-z\-/]+){0,2})\s+"
    r"(?P<dose>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg|mcg|µg|g|units?|iu|ml)\b"
    r"(?P<rest>[^,\n]{0,80})"
)


def extract_medications(spans: Iterable[Span]) -> List[Extraction]:
    out: List[Extraction] = []
    seen: set[tuple[str, str, str]] = set()
    for s in spans:
        for m in _MED_DOSE_RE.finditer(s.text):
            name_raw = m.group("name").strip().lower()
            # take the last 1-2 tokens of the name match (drug names are short)
            tokens = name_raw.split()
            candidate = None
            for k in (1, 2):
                if len(tokens) >= k:
                    cand = " ".join(tokens[-k:])
                    if cand in MEDICATIONS:
                        candidate = cand
                        break
            if candidate is None:
                continue
            dose = m.group("dose")
            unit = m.group("unit").lower()
            key = (candidate, dose, unit)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Extraction(
                    kind="medication",
                    name=candidate,
                    value=f"{dose} {unit}",
                    source_span_ids=[s.span_id],
                )
            )
    return out


def extract_labs(spans: Iterable[Span]) -> List[Extraction]:
    out: List[Extraction] = []
    seen: set[tuple[str, str]] = set()
    for s in spans:
        text = s.text
        for pat, name, units, _kind in LAB_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                value = m.group(1).replace(",", "")
                key = (name, value)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Extraction(
                        kind="lab",
                        name=name,
                        value=value,
                        units=units,
                        qualifier=_lab_qualifier(name, value),
                        source_span_ids=[s.span_id],
                    )
                )
    return out


def _lab_qualifier(name: str, raw_value: str) -> str | None:
    """Light-touch flagging for common cardiology-relevant abnormal thresholds.
    Only flags directional severity; full interpretation belongs to the clinician.
    """
    try:
        v = float(raw_value)
    except ValueError:
        return None
    n = name.lower()
    if n == "hba1c" and v >= 6.5:
        return "elevated"
    if n == "ldl" and v >= 100:
        return "elevated"
    if n in ("bnp",) and v >= 100:
        return "elevated"
    if n == "nt-probnp" and v >= 300:
        return "elevated"
    if n == "troponin" and v >= 0.04:
        return "elevated"
    if n == "egfr" and v < 60:
        return "reduced"
    if n == "creatinine" and v >= 1.3:
        return "elevated"
    if n == "potassium" and (v < 3.5 or v > 5.0):
        return "abnormal"
    if n == "hemoglobin" and v < 12:
        return "low"
    return None


def extract_echo(spans: Iterable[Span]) -> List[Extraction]:
    out: List[Extraction] = []
    seen: set[tuple[str, str]] = set()
    for s in spans:
        text = s.text
        for pat, name in ECHO_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                value: str | None
                qualifier: str | None = None
                if m.groups():
                    value = m.group(1)
                    if name == "LVEF":
                        try:
                            ef = float(value)
                            if ef < 40:
                                qualifier = "reduced"
                            elif ef < 50:
                                qualifier = "mildly reduced"
                            else:
                                qualifier = "preserved"
                        except ValueError:
                            pass
                else:
                    value = m.group(0)
                key = (name, value or "")
                if key in seen:
                    continue
                seen.add(key)
                units = "%" if name == "LVEF" else ("mmHg" if name == "PASP" else None)
                out.append(
                    Extraction(
                        kind="echo",
                        name=name,
                        value=value,
                        units=units,
                        qualifier=qualifier,
                        source_span_ids=[s.span_id],
                    )
                )
    return out


def extract_cath(spans: Iterable[Span]) -> List[Extraction]:
    out: List[Extraction] = []
    seen: set[tuple[str, str]] = set()
    for s in spans:
        text = s.text
        for pat, vessel in CATH_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                pct = m.group(2)
                key = (vessel, pct)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    pct_v = int(pct)
                    qualifier = (
                        "severe" if pct_v >= 70
                        else "moderate" if pct_v >= 50
                        else "mild"
                    )
                except ValueError:
                    qualifier = None
                out.append(
                    Extraction(
                        kind="cath",
                        name=vessel,
                        value=f"{pct}%",
                        qualifier=qualifier,
                        source_span_ids=[s.span_id],
                    )
                )
    return out


def run_all_extractors(spans: List[Span]) -> List[Extraction]:
    """Run every deterministic extractor and concatenate."""
    return (
        extract_medications(spans)
        + extract_labs(spans)
        + extract_echo(spans)
        + extract_cath(spans)
    )
