import { useMemo } from 'react'
import type { Document, Span } from '../types'

export function SpanList({
  documents,
  spans,
  selectedId,
  onSelect,
}: {
  documents: Document[]
  spans: Span[]
  selectedId: string | null
  onSelect: (s: Span) => void
}) {
  const byDoc = useMemo(() => {
    const m = new Map<string, Span[]>()
    for (const s of spans) {
      const arr = m.get(s.doc_id) ?? []
      arr.push(s)
      m.set(s.doc_id, arr)
    }
    return m
  }, [spans])

  return (
    <div className="overflow-y-auto h-full">
      {documents.map((d) => (
        <div key={d.doc_id} className="border-b border-slate-200">
          <div className="px-3 py-2 bg-slate-100 text-sm font-medium sticky top-0 z-10">
            {d.name}
            <span className="ml-2 text-xs text-slate-500 font-normal">
              {d.page_count}p · {d.source}
            </span>
          </div>
          <ul>
            {(byDoc.get(d.doc_id) ?? []).map((s) => (
              <li
                key={s.span_id}
                onClick={() => onSelect(s)}
                className={
                  'px-3 py-1.5 text-sm cursor-pointer border-l-2 ' +
                  (selectedId === s.span_id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-transparent hover:bg-slate-50')
                }
              >
                <span className="text-slate-400 text-xs mr-2">p.{s.page}</span>
                {s.text}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
