# Medical Records Summarizer

Local web app that ingests medical-record PDFs (including scanned reports), OCRs them,
and synthesizes a one-page clinical summary with click-to-source citations and an
ACC/AHA-grounded plan.

The demo runs against **synthetic** patient data — no HIPAA concerns. Swap in real
PHI only behind a BAA-covered LLM endpoint.

> Informational only. Not a medical device, not for clinical decision-making.

## What it does

1. **Upload** one or more PDFs for a patient.
2. **OCR** scanned pages (Tesseract); use the text layer when present (pdfplumber).
   Every extracted line carries `(doc, page, bounding-box)` provenance.
3. **Deterministic extractors** pull meds, labs, echo findings, and cath findings.
   Each fact retains its source span IDs.
4. **Section synthesis** (Claude Opus 4.7 by default) writes each section of the
   one-pager. Every claim must cite ≥1 span — items without evidence are dropped.
5. **Problem list** is prioritized; for each problem we retrieve top ACC/AHA passages
   (TF-IDF over the corpus you provide) and the LLM writes a plan citing **only**
   those passages. Below-threshold matches render as
   "No guideline-backed recommendation found".
6. **One-pager** renders left; click any item or problem to jump to the exact page
   and bounding-box on the source PDF (PDF.js viewer, right pane).

## Architecture

```
backend/
  app/
    ingest/      pdfplumber + Tesseract → spans with normalized bboxes
    extract/     deterministic extractors for meds/labs/echo/cath
    llm/         Claude / Ollama clients behind one interface
    guidelines/  TF-IDF index over an ACC/AHA PDF corpus
    summarize/   section synthesis, problem extraction, plan generation
    routers/     FastAPI endpoints
  scripts/
    generate_synthetic_patient.py    # demo data
frontend/
  src/components/  Upload, OnePager, PdfViewer, SpanList
guidelines_corpus/   <- you populate this with ACC/AHA PDFs
```

## Prerequisites

- Python 3.11+
- Node 20+
- System packages: `tesseract-ocr`, `poppler-utils`

Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr poppler-utils`
macOS:         `brew install tesseract poppler`

## Quickstart (demo)

1. **Backend** — install deps, generate a synthetic patient, build the guideline index, launch:

   ```sh
   cd backend
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-demo.txt        # main deps + reportlab for synthetic data

   # Generate a synthetic patient bundle
   PYTHONPATH=. python scripts/generate_synthetic_patient.py --out /tmp/demo_patient

   # Drop ACC/AHA guideline PDFs into ../guidelines_corpus/ then build the index:
   mkdir -p ../guidelines_corpus
   #  ... copy your ACC/AHA PDFs into that folder ...
   python -m app.guidelines build

   # Required for LLM-backed synthesis (Claude is the default backend):
   export ANTHROPIC_API_KEY=sk-ant-...

   uvicorn app.main:app --reload --port 8000
   ```

2. **Frontend** — second terminal:

   ```sh
   cd frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:5173, drop the PDFs from `/tmp/demo_patient/` into the upload box.

## Backend configuration

Environment variables, all optional:

| Var                          | Default                             | Notes |
|------------------------------|-------------------------------------|-------|
| `LLM_BACKEND`                | `claude`                            | `claude` / `ollama` / `none` |
| `ANTHROPIC_API_KEY`          | —                                   | Required when `LLM_BACKEND=claude` |
| `CLAUDE_MODEL`               | `claude-opus-4-7`                   | |
| `OLLAMA_URL`                 | `http://localhost:11434`            | |
| `OLLAMA_MODEL`               | `qwen2.5:7b-instruct-q4_K_M`        | Pull it first: `ollama pull qwen2.5:7b-instruct-q4_K_M` |
| `GUIDELINES_DIR`             | `<repo>/guidelines_corpus`          | |
| `GUIDELINE_SCORE_THRESHOLD`  | `0.15`                              | Below this TF-IDF cosine, no plan |

If the LLM backend or the guideline corpus is missing, the summarizer degrades gracefully:
the deterministic extractors still produce structured sections, and each problem renders
without a plan and with a warning at the top of the one-pager.

## Why these choices

- **Claude Opus 4.7** by default — the demo runs on synthetic data, so we can use the
  strongest model. The LLM is wired behind an `LLMClient` interface; swap to a local
  Ollama model (or a BAA-covered Bedrock endpoint) by setting `LLM_BACKEND` — no code
  changes elsewhere.
- **TF-IDF over ACC/AHA chunks** (not embeddings) — zero embedding-model dependencies,
  runs in-process, and recall is good on jargon-heavy guideline prose.
- **Evidence gating in the prompts** — every LLM-emitted claim must carry a `span_id`
  that exists in the input. The Python validator drops items that don't, so the
  one-pager never asserts anything that isn't anchored to a quoted span.
- **Prompt caching** — each per-section call shares the patient-context block, marked
  with `cache_control: ephemeral`. After the first section, every subsequent section
  pays only for the few hundred tokens of section-specific suffix.

## Status

- [x] Phase 1 — upload → OCR → click span → highlight on PDF
- [x] Phase 2 — deterministic extractors (meds, labs, echo, cath)
- [x] Phase 3 — LLM section synthesis with evidence gating (Claude / Ollama)
- [x] Phase 4 — ACC/AHA TF-IDF corpus index + retriever
- [x] Phase 5 — guideline-gated plan generation
- [x] Phase 6 — one-pager layout, importance ranking, click-to-source UX

Future: importance-based PDF export, persistent patient store, richer extractors.
