import { useState } from 'react'
import { ingest } from '../api'
import type { IngestResponse } from '../types'

export function Upload({ onDone }: { onDone: (r: IngestResponse) => void }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function handle(files: FileList | null) {
    if (!files || files.length === 0) return
    setBusy(true)
    setErr(null)
    try {
      const result = await ingest(Array.from(files))
      onDone(result)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="p-6">
      <label className="block border-2 border-dashed border-slate-400 rounded-lg p-10 text-center cursor-pointer hover:bg-slate-50">
        <input
          type="file"
          accept="application/pdf"
          multiple
          className="hidden"
          disabled={busy}
          onChange={(e) => handle(e.target.files)}
        />
        <div className="text-slate-700">
          {busy ? 'Processing… (OCR can take a minute per scanned page)' : 'Drop PDFs here or click to select'}
        </div>
        <div className="text-xs text-slate-500 mt-1">
          Text-layer extraction first, OCR fallback for scans. Files stay on this machine.
        </div>
      </label>
      {err && <div className="mt-3 text-red-600 text-sm whitespace-pre-wrap">{err}</div>}
    </div>
  )
}
