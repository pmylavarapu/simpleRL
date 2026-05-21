import { useEffect, useMemo, useState } from 'react'
import { summarize } from './api'
import { OnePager } from './components/OnePager'
import { PdfViewer } from './components/PdfViewer'
import { SpanList } from './components/SpanList'
import { Upload } from './components/Upload'
import type { IngestResponse, OnePager as OnePagerData, Span } from './types'

type View = 'summary' | 'spans'

export default function App() {
  const [data, setData] = useState<IngestResponse | null>(null)
  const [summary, setSummary] = useState<OnePagerData | null>(null)
  const [summarizing, setSummarizing] = useState(false)
  const [summaryErr, setSummaryErr] = useState<string | null>(null)
  const [view, setView] = useState<View>('summary')
  const [selected, setSelected] = useState<Span | null>(null)

  const spansById = useMemo(() => {
    const m = new Map<string, Span>()
    if (data) for (const s of data.spans) m.set(s.span_id, s)
    return m
  }, [data])

  useEffect(() => {
    if (!data) return
    setSummary(null)
    setSummaryErr(null)
    setSummarizing(true)
    summarize(data.session_id)
      .then(setSummary)
      .catch((e) => setSummaryErr(e instanceof Error ? e.message : String(e)))
      .finally(() => setSummarizing(false))
  }, [data?.session_id])

  if (!data) {
    return (
      <div className="h-full flex flex-col">
        <Header />
        <div className="flex-1 flex items-center justify-center">
          <div className="w-full max-w-xl">
            <Upload onDone={setData} />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      <Header
        right={
          <div className="flex items-center gap-3">
            <div className="flex text-xs border border-slate-300 rounded overflow-hidden">
              <button
                className={
                  'px-2 py-1 ' +
                  (view === 'summary'
                    ? 'bg-slate-800 text-white'
                    : 'bg-white text-slate-700 hover:bg-slate-100')
                }
                onClick={() => setView('summary')}
              >
                Summary
              </button>
              <button
                className={
                  'px-2 py-1 border-l border-slate-300 ' +
                  (view === 'spans'
                    ? 'bg-slate-800 text-white'
                    : 'bg-white text-slate-700 hover:bg-slate-100')
                }
                onClick={() => setView('spans')}
              >
                All spans
              </button>
            </div>
            <button
              className="text-sm text-blue-600 hover:underline"
              onClick={() => {
                setData(null)
                setSummary(null)
                setSelected(null)
              }}
            >
              New session
            </button>
          </div>
        }
      />
      <div className="flex-1 grid grid-cols-2 min-h-0">
        <div className="border-r min-h-0">
          {view === 'summary' ? (
            summarizing ? (
              <div className="h-full flex items-center justify-center text-slate-500 text-sm p-6 text-center">
                Building the one-pager…<br />
                <span className="text-xs">
                  (LLM section synthesis + guideline-gated plan generation)
                </span>
              </div>
            ) : summaryErr ? (
              <div className="p-6 text-red-700 text-sm">
                <div className="font-medium mb-2">Couldn't build the summary.</div>
                <div className="whitespace-pre-wrap text-xs">{summaryErr}</div>
                <button
                  className="mt-3 text-blue-600 hover:underline"
                  onClick={() => {
                    setSummaryErr(null)
                    setSummarizing(true)
                    summarize(data.session_id)
                      .then(setSummary)
                      .catch((e) => setSummaryErr(String(e)))
                      .finally(() => setSummarizing(false))
                  }}
                >
                  Retry
                </button>
              </div>
            ) : summary ? (
              <OnePager
                data={summary}
                spansById={spansById}
                onSelectSpan={setSelected}
                selectedSpanId={selected?.span_id ?? null}
              />
            ) : null
          ) : (
            <SpanList
              documents={data.documents}
              spans={data.spans}
              selectedId={selected?.span_id ?? null}
              onSelect={setSelected}
            />
          )}
        </div>
        <div className="min-h-0">
          <PdfViewer sessionId={data.session_id} span={selected} />
        </div>
      </div>
    </div>
  )
}

function Header({ right }: { right?: React.ReactNode }) {
  return (
    <header className="px-6 py-3 border-b flex items-center justify-between">
      <div>
        <h1 className="text-lg font-semibold">Medical Records Summarizer</h1>
        <div className="text-xs text-slate-500">
          Demo mode — synthetic data. Informational only; not for clinical decision-making.
        </div>
      </div>
      {right}
    </header>
  )
}
