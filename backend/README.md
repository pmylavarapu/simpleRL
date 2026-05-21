# Medical Records Summarizer — Backend

One-page synthesis of outside medical records (scanned PDFs) with
ACC/AHA-grounded plan recommendations.

## Pipeline

```
PDFs ──► [vision extract per page] ──► page facts (each w/ verbatim snippet + page#)
                                          │
                                          ▼
                            [LLM summarize] ──► one-page JSON
                            (HPI/PMH/PSH/Meds/Objective/Assessment)
                                          │
                                          ▼
                            [rule-based ACC/AHA planner] ──► Plan items
                            (only emits verbatim guideline text + citation)
```

Nothing in the Plan section is invented by the LLM. Each Plan item comes
from `guidelines/acc_aha.json` (curated subset of ACC/AHA guidelines).
If a problem has no matching rule, the planner says so explicitly.

## Setup

```bash
pip install -r backend/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## End-to-end test on the synthetic patient

```bash
python samples/generate_synthetic_patient.py
python backend/run_pipeline.py samples/outside_records/*.pdf
```

This drops a `data/<case_id>/summary.json` you can inspect.

## Running the API

```bash
uvicorn app.main:app --reload --app-dir backend
```

Endpoints:
- `POST /api/cases` (multipart `files[]`) — upload PDFs → `{case_id, pdfs[]}`
- `POST /api/cases/{id}/extract` — vision extraction, persists `extractions.json`
- `POST /api/cases/{id}/summarize` — produces `summary.json`
- `GET  /api/cases/{id}/summary` — fetch summary JSON
- `GET  /api/cases/{id}/manifest` — list PDFs for the case
- `GET  /api/cases/{id}/pdfs/{pdf_id}` — stream a PDF for the viewer

## Curated guidelines

`backend/guidelines/acc_aha.json` is the source of truth for Plan
recommendations. v1 covers:

- 2022 ACC/AHA/HFSA Heart Failure — ARNI, beta blocker, MRA, SGLT2i in HFrEF
- 2023 ACC/AHA AF — anticoagulation by CHA₂DS₂-VASc
- 2018 ACC/AHA Cholesterol — secondary-prevention statin & very-high-risk add-on
- 2017 ACC/AHA Hypertension — <130/80 target in high CV risk
- 2023 AHA/ACC Chronic Coronary Disease — antiplatelet for secondary prevention

To add a rule, append to `rules[]` with the documented precondition kinds:
`ef_at_most`, `egfr_at_least`, `k_at_most`, `ldl_at_least`, `bp_above`,
`cha2ds2vasc_at_least`, `no_med_substring_any`, `any_med_substring_any`,
`no_high_intensity_statin`, `very_high_risk_ascvd`.
