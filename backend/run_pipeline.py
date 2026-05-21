"""End-to-end pipeline runner.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python backend/run_pipeline.py samples/outside_records/*.pdf

Outputs:
    data/<case_id>/manifest.json
    data/<case_id>/extractions.json
    data/<case_id>/summary.json   <- the structured one-page summary

Prints the summary path on success.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow running as a script: add repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import storage  # noqa: E402
from app.extraction import extract_page  # noqa: E402
from app.planner import generate_plan  # noqa: E402
from app.schemas import CaseManifest, PdfFile  # noqa: E402
from app.summarization import summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", type=Path)
    args = ap.parse_args()

    case_id = storage.new_case_id()
    storage.init_case(case_id)
    pdfs: list[PdfFile] = []
    for path in args.pdfs:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            sys.exit(2)
        pdfs.append(storage.save_pdf(case_id, path.name, path))

    manifest = CaseManifest(case_id=case_id, pdfs=pdfs)
    storage.write_manifest(manifest)
    print(f"[case] {case_id}: {len(pdfs)} PDFs, "
          f"{sum(p.pages for p in pdfs)} total pages")

    print("[1/3] extracting facts via Claude vision...")
    pages = []
    for pdf in pdfs:
        pdf_file = storage.pdf_path(case_id, pdf.pdf_id)
        for i in range(pdf.pages):
            print(f"   - {pdf.filename} page {i + 1}/{pdf.pages}")
            pages.append(extract_page(pdf.pdf_id, pdf_file, i))
    storage.write_extractions(case_id, pages)
    print(f"      -> {sum(len(p.facts) for p in pages)} facts extracted")

    print("[2/3] generating one-page summary...")
    summary = summarize(pages)

    print("[3/3] running rule-based ACC/AHA planner...")
    summary.plan = generate_plan(pages, summary)
    storage.write_summary(case_id, summary)

    print(f"\nDONE. Summary: data/{case_id}/summary.json")
    print(f"     Extractions: data/{case_id}/extractions.json")
    print(f"     Manifest: data/{case_id}/manifest.json")
    print(f"\nPlan items ({len(summary.plan)}):")
    for item in summary.plan:
        print(f"  [P{item.priority}] {item.problem}")
        print(f"        {item.citation.guideline} ({item.citation.year}) "
              f"COR {item.citation.cor} LOE {item.citation.loe}")


if __name__ == "__main__":
    main()
