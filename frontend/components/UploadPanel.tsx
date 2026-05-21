"use client";

import { useState } from "react";
import { createCase, runExtract, runSummarize } from "@/lib/api";
import type { CaseManifest, OnePageSummary } from "@/lib/types";

interface Props {
  onReady: (manifest: CaseManifest, summary: OnePageSummary) => void;
}

export function UploadPanel({ onReady }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const handleRun = async () => {
    if (files.length === 0) return;
    setErr(null);
    try {
      setBusy("Uploading…");
      const manifest = await createCase(files);

      setBusy("Extracting facts (vision)…");
      await runExtract(manifest.case_id);

      setBusy("Synthesizing summary + plan…");
      const summary = await runSummarize(manifest.case_id);

      onReady(manifest, summary);
    } catch (e: any) {
      setErr(e?.message ?? "error");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-12 p-6 bg-white border border-clinical-200 rounded-md shadow-sm">
      <h1 className="text-lg font-semibold text-clinical-900">
        Outside Records Summarizer
      </h1>
      <p className="text-sm text-slate-600 mt-1">
        Upload one or more outside-records PDFs. The pipeline extracts
        facts per page, synthesizes a one-page summary, and emits an
        ACC/AHA-grounded plan.
      </p>
      <input
        type="file"
        accept="application/pdf"
        multiple
        onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        className="mt-4 block w-full text-sm"
      />
      {files.length > 0 && (
        <ul className="text-xs text-slate-600 mt-2 list-disc list-inside">
          {files.map((f) => (
            <li key={f.name}>
              {f.name}{" "}
              <span className="text-slate-400">
                ({Math.round(f.size / 1024)} KB)
              </span>
            </li>
          ))}
        </ul>
      )}
      <button
        disabled={files.length === 0 || busy !== null}
        onClick={handleRun}
        className="mt-4 px-3 py-1.5 bg-clinical-700 text-white text-sm rounded disabled:opacity-40"
      >
        {busy ?? "Run pipeline"}
      </button>
      {err && <div className="mt-3 text-sm text-red-700">{err}</div>}
    </div>
  );
}
