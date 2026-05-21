"""Per-page extraction via Claude vision.

For each PDF page (rasterized to PNG) we ask Claude to:
  1. Transcribe the page text verbatim (raw_text).
  2. Emit a list of structured medical facts, each anchored to a verbatim
     snippet from the page so the frontend can find and highlight it.
"""
from __future__ import annotations

import json
from pathlib import Path

from anthropic import Anthropic

from .config import ANTHROPIC_API_KEY, EXTRACTION_MODEL
from .pdf_utils import page_to_png_b64
from .schemas import ExtractedFact, PageExtraction


EXTRACTION_SYSTEM = """You are a careful medical records information extractor.

You will be shown one page of an outside medical record. Your job is to:

1. TRANSCRIBE the page text verbatim. Preserve dates, numbers, and units.
2. Extract STRUCTURED MEDICAL FACTS in the schema given below.

Rules:
- Every fact MUST include a `snippet` field that is an EXACT, verbatim
  substring from the page (1 phrase to 1 sentence). The frontend uses this
  string to highlight the supporting evidence. Do not paraphrase the snippet.
- Do not invent facts. If the page does not state something, do not extract it.
- Dates should be ISO (YYYY-MM-DD) when possible. Use the date as written
  if you cannot parse it.
- Labs: include `value` and `unit`. `content` should be the lab name
  (e.g., "LDL-C", "Creatinine").
- Medications: include dose/route/frequency in `content`
  (e.g., "Atorvastatin 40 mg PO daily").
- Imaging/procedure: `content` is a short description
  (e.g., "TTE", "PCI to LAD"). Include the date.
- Categories:
  demographic | hpi | fam_hx | soc_hx | pmh | psh | med | allergy |
  vitals | lab | imaging | procedure

Output STRICT JSON only, matching:
{
  "raw_text": "<verbatim page text>",
  "facts": [
    {
      "category": "<one of the categories above>",
      "content": "<short canonical phrase>",
      "date": "<ISO date or null>",
      "value": "<value or null>",
      "unit": "<unit or null>",
      "snippet": "<verbatim substring from the page>"
    }
  ]
}
No prose, no markdown, JSON only.
"""


def extract_page(pdf_id: str, pdf_file: Path, page_index: int) -> PageExtraction:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it before running extraction."
        )

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    image_b64 = page_to_png_b64(pdf_file, page_index)

    message = client.messages.create(
        model=EXTRACTION_MODEL,
        max_tokens=4096,
        system=EXTRACTION_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This is page {page_index + 1} of an outside "
                            "medical record. Extract per the schema."
                        ),
                    },
                ],
            }
        ],
    )

    text = "".join(b.text for b in message.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    payload = json.loads(text)

    facts: list[ExtractedFact] = []
    for f in payload.get("facts", []):
        facts.append(
            ExtractedFact(
                category=f["category"],
                content=f["content"],
                date=f.get("date") or None,
                value=f.get("value") or None,
                unit=f.get("unit") or None,
                source={
                    "pdf_id": pdf_id,
                    "page": page_index,
                    "snippet": f["snippet"],
                },
            )
        )

    return PageExtraction(
        pdf_id=pdf_id,
        page=page_index,
        raw_text=payload.get("raw_text", ""),
        facts=facts,
    )
