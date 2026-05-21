import type { IngestResponse, OnePager } from './types'

export async function ingest(files: File[]): Promise<IngestResponse> {
  const fd = new FormData()
  for (const f of files) fd.append('files', f, f.name)
  const res = await fetch('/api/ingest', { method: 'POST', body: fd })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`ingest failed (${res.status}): ${body || res.statusText}`)
  }
  return res.json()
}

export async function summarize(sessionId: string): Promise<OnePager> {
  const res = await fetch(`/api/sessions/${sessionId}/summary`, { method: 'POST' })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`summarize failed (${res.status}): ${body || res.statusText}`)
  }
  return res.json()
}

export function pdfUrl(sessionId: string, docId: string): string {
  return `/api/sessions/${sessionId}/documents/${docId}/pdf`
}
