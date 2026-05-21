"""Synthesize a one-page summary from per-page extracted facts.

The LLM produces structured JSON. Every Claim has a `sources` list referencing
the (pdf_id, page, snippet) of the supporting fact. The Plan section is NOT
produced here; it comes from the rule-based planner so that nothing is
asserted without a matching ACC/AHA rule.
"""
from __future__ import annotations

import json

from anthropic import Anthropic

from .config import ANTHROPIC_API_KEY, SUMMARY_MODEL
from .schemas import OnePageSummary, PageExtraction


SUMMARY_SYSTEM = """You are a medical scribe producing a one-page chart summary
from extracted facts pulled from outside records. Every claim you write must
be supported by one or more facts from the input. Every claim must carry the
`sources` list copied verbatim from the supporting fact(s).

OUTPUT STRICT JSON ONLY, matching this schema:

{
  "patient": {"name": str|null, "age": int|null, "sex": str|null,
              "mrn": str|null, "dob": str|null},
  "hpi":     [{"text": str, "sources":[{pdf_id,page,snippet}, ...]}, ...],
  "fam_hx":  [Claim, ...],
  "soc_hx":  [Claim, ...],
  "pmh":     [Claim, ...],
  "psh":     [Claim, ...],
  "meds":    [Claim, ...],
  "objective": {
    "labs": [
      {"name": str,
       "latest": {"value": str, "unit": str, "date": str, "source": {...}},
       "trend": [LabValue, ...]}
    ],
    "cardiology": [
      {"study_type": str, "date": str,
       "key_findings": [Claim, ...]}
    ]
  },
  "assessment": [
    {"problem": str,
     "paragraph": [Claim, Claim, ...]}
  ]
}

Rules:
- DO NOT output a "plan" key. Plan is produced separately.
- Every Claim's `sources` must come from the input facts; never invent a
  pdf_id, page, or snippet. Copy them verbatim.
- HPI: brief narrative (3-6 claims max).
- PMH: one claim per chronic problem, in roughly decreasing severity.
- Meds: each med is one claim.
- Objective.labs: focus on BMP, CBC, lipid panel (LDL-C, HDL, TG, total chol),
  HbA1c, BNP/NT-proBNP, troponin, INR if present. Skip anything irrelevant.
- Objective.cardiology: every echo / stress / cath / EP study found.
- Assessment.problem: short canonical problem name
  (e.g., "Heart Failure with Reduced Ejection Fraction (HFrEF)",
   "Atrial Fibrillation", "Hypertension", "Hyperlipidemia",
   "Coronary Artery Disease s/p PCI").
- Assessment.paragraph: ordered list of short claims that together form the
  problem paragraph; each must be source-anchored.
- If a section has no facts, output an empty list.
- No prose outside the JSON. No markdown.
"""


def _facts_payload(pages: list[PageExtraction]) -> str:
    flat = []
    for p in pages:
        for f in p.facts:
            flat.append(
                {
                    "category": f.category,
                    "content": f.content,
                    "date": f.date,
                    "value": f.value,
                    "unit": f.unit,
                    "source": f.source.model_dump(),
                }
            )
    return json.dumps(flat, indent=2)


def summarize(pages: list[PageExtraction]) -> OnePageSummary:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    user_payload = (
        "Extracted facts from outside records (JSON array):\n\n"
        + _facts_payload(pages)
        + "\n\nProduce the one-page summary JSON now."
    )

    message = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=8192,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": user_payload}],
    )
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    payload = json.loads(text)
    payload["plan"] = []  # filled in by planner
    return OnePageSummary.model_validate(payload)
