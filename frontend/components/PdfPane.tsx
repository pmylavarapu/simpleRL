"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";

import { pdfUrl } from "@/lib/api";
import type { ActiveSource, CaseManifest } from "@/lib/types";

// pdfjs worker — bundled via webpack URL import. Next.js (turbopack/webpack)
// rewrites this to a static asset URL at build time.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

interface Props {
  caseId: string;
  manifest: CaseManifest;
  active: ActiveSource;
}

function normalize(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

/**
 * After react-pdf renders the page text layer, walk its spans and
 * highlight the one(s) that match the active snippet.
 * Strategy:
 *   1) Look for any single span containing the full normalized snippet.
 *   2) Otherwise, take the first significant word (≥5 chars) of the snippet
 *      and highlight the first span whose normalized text contains it.
 * Returns the DOM element that was highlighted (for scrollIntoView).
 */
function highlightInTextLayer(
  pageRoot: HTMLElement,
  snippet: string
): HTMLElement | null {
  const norm = normalize(snippet);
  if (!norm) return null;

  const spans = pageRoot.querySelectorAll<HTMLElement>(
    ".react-pdf__Page__textContent span"
  );
  spans.forEach((s) => s.classList.remove("evidence-highlight"));

  for (const span of Array.from(spans)) {
    if (normalize(span.textContent || "").includes(norm)) {
      span.classList.add("evidence-highlight");
      return span;
    }
  }

  // Fallback: highlight any span containing the first long word.
  const firstWord = norm.split(" ").find((w) => w.length >= 5);
  if (firstWord) {
    for (const span of Array.from(spans)) {
      if (normalize(span.textContent || "").includes(firstWord)) {
        span.classList.add("evidence-highlight");
        return span;
      }
    }
  }
  return null;
}

export function PdfPane({ caseId, manifest, active }: Props) {
  const [pdfIdx, setPdfIdx] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const pageContainerRef = useRef<HTMLDivElement>(null);

  // When the active source changes, jump to the right PDF + page.
  useEffect(() => {
    if (!active) return;
    const idx = manifest.pdfs.findIndex((p) => p.pdf_id === active.pdf_id);
    if (idx >= 0) {
      setPdfIdx(idx);
      setPageNum(active.page + 1);
    }
  }, [active, manifest]);

  // After page render, attempt snippet highlight.
  const onPageRender = () => {
    if (!active || !pageContainerRef.current) return;
    // pdfjs renders the text layer async; small defer.
    setTimeout(() => {
      if (!pageContainerRef.current) return;
      const el = highlightInTextLayer(pageContainerRef.current, active.snippet);
      el?.scrollIntoView({ block: "center", behavior: "smooth" });
    }, 50);
  };

  const current = manifest.pdfs[pdfIdx];
  const file = useMemo(
    () => (current ? pdfUrl(caseId, current.pdf_id) : null),
    [caseId, current]
  );

  if (!current) return <div className="p-4 text-sm text-slate-600">No PDFs.</div>;

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-2 p-2 border-b border-clinical-200 bg-clinical-50 text-xs">
        <select
          value={pdfIdx}
          onChange={(e) => {
            setPdfIdx(Number(e.target.value));
            setPageNum(1);
          }}
          className="bg-white border border-clinical-200 rounded px-1.5 py-0.5"
        >
          {manifest.pdfs.map((p, i) => (
            <option key={p.pdf_id} value={i}>
              {p.filename}
            </option>
          ))}
        </select>
        <button
          className="px-1.5 py-0.5 bg-white border border-clinical-200 rounded disabled:opacity-40"
          onClick={() => setPageNum((n) => Math.max(1, n - 1))}
          disabled={pageNum <= 1}
        >
          ‹
        </button>
        <span>
          page {pageNum} / {numPages || current.pages}
        </span>
        <button
          className="px-1.5 py-0.5 bg-white border border-clinical-200 rounded disabled:opacity-40"
          onClick={() => setPageNum((n) => Math.min(numPages, n + 1))}
          disabled={pageNum >= numPages}
        >
          ›
        </button>
        {active && (
          <span className="ml-auto text-clinical-700 truncate max-w-[60%]">
            Source: “{active.snippet.slice(0, 70)}
            {active.snippet.length > 70 ? "…" : ""}”
          </span>
        )}
      </div>
      <div
        ref={pageContainerRef}
        className="flex-1 overflow-auto bg-clinical-100 p-3"
      >
        {file && (
          <Document
            file={file}
            onLoadSuccess={(doc) => setNumPages(doc.numPages)}
            loading={<div className="p-4 text-sm">Loading PDF…</div>}
          >
            <Page
              pageNumber={pageNum}
              width={680}
              renderTextLayer
              renderAnnotationLayer={false}
              onRenderTextLayerSuccess={onPageRender}
            />
          </Document>
        )}
      </div>
    </div>
  );
}
