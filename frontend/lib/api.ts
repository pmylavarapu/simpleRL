import type { CaseManifest, OnePageSummary } from "./types";

export async function createCase(files: File[]): Promise<CaseManifest> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  const r = await fetch("/api/cases", { method: "POST", body: form });
  if (!r.ok) throw new Error(`createCase ${r.status}`);
  return r.json();
}

export async function runExtract(caseId: string): Promise<void> {
  const r = await fetch(`/api/cases/${caseId}/extract`, { method: "POST" });
  if (!r.ok) throw new Error(`extract ${r.status}`);
}

export async function runSummarize(caseId: string): Promise<OnePageSummary> {
  const r = await fetch(`/api/cases/${caseId}/summarize`, { method: "POST" });
  if (!r.ok) throw new Error(`summarize ${r.status}`);
  return r.json();
}

export async function fetchSummary(caseId: string): Promise<OnePageSummary> {
  const r = await fetch(`/api/cases/${caseId}/summary`);
  if (!r.ok) throw new Error(`fetchSummary ${r.status}`);
  return r.json();
}

export function pdfUrl(caseId: string, pdfId: string): string {
  return `/api/cases/${caseId}/pdfs/${pdfId}`;
}
