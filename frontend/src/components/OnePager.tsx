import { useMemo } from 'react'
import type {
  OnePager as OnePagerData,
  Problem,
  Section,
  SectionItem,
  Span,
} from '../types'

interface Props {
  data: OnePagerData
  spansById: Map<string, Span>
  onSelectSpan: (s: Span) => void
  selectedSpanId: string | null
}

export function OnePager({ data, spansById, onSelectSpan, selectedSpanId }: Props) {
  const sectionsByKey = useMemo(() => {
    const m = new Map<string, Section>()
    for (const s of data.sections) m.set(s.key, s)
    return m
  }, [data.sections])

  return (
    <div className="overflow-y-auto h-full bg-white">
      {data.warnings.length > 0 && (
        <div className="m-4 p-3 rounded bg-amber-50 border border-amber-200 text-amber-900 text-xs">
          {data.warnings.map((w, i) => (
            <div key={i}>· {w}</div>
          ))}
        </div>
      )}

      <div className="px-6 py-4 grid grid-cols-2 gap-x-8 gap-y-4">
        <SectionBlock
          section={sectionsByKey.get('hpi')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <SectionBlock
          section={sectionsByKey.get('past_medical_history')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <SectionBlock
          section={sectionsByKey.get('past_surgical_history')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <SectionBlock
          section={sectionsByKey.get('family_history')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <SectionBlock
          section={sectionsByKey.get('social_history')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <SectionBlock
          section={sectionsByKey.get('medications')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <SectionBlock
          section={sectionsByKey.get('objective')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <SectionBlock
          section={sectionsByKey.get('labs')}
          onSelect={onSelectSpan}
          spansById={spansById}
          selectedSpanId={selectedSpanId}
        />
        <div className="col-span-2">
          <SectionBlock
            section={sectionsByKey.get('cardiology_imaging_procedures')}
            onSelect={onSelectSpan}
            spansById={spansById}
            selectedSpanId={selectedSpanId}
          />
        </div>
      </div>

      <div className="px-6 pb-6">
        <h2 className="text-base font-semibold border-b border-slate-300 pb-1 mb-2">
          Assessment &amp; Plan
        </h2>
        {data.problems.length === 0 ? (
          <div className="text-sm text-slate-500">No problems identified.</div>
        ) : (
          <ol className="space-y-3">
            {data.problems.map((p, i) => (
              <ProblemBlock
                key={i}
                problem={p}
                onSelect={onSelectSpan}
                spansById={spansById}
                selectedSpanId={selectedSpanId}
              />
            ))}
          </ol>
        )}
      </div>

      <div className="px-6 pb-6 text-xs text-slate-500 italic">{data.disclaimer}</div>
    </div>
  )
}

function SectionBlock({
  section,
  spansById,
  onSelect,
  selectedSpanId,
}: {
  section: Section | undefined
  spansById: Map<string, Span>
  onSelect: (s: Span) => void
  selectedSpanId: string | null
}) {
  if (!section) return null
  const items = [...section.items].sort((a, b) => b.importance - a.importance)
  return (
    <div>
      <h2 className="text-sm font-semibold text-slate-700 border-b border-slate-200 pb-0.5 mb-1.5">
        {section.title}
      </h2>
      {items.length === 0 ? (
        <div className="text-xs text-slate-400 italic">No relevant items.</div>
      ) : (
        <ul className="space-y-0.5">
          {items.map((it, i) => (
            <ItemLine
              key={i}
              item={it}
              spansById={spansById}
              onSelect={onSelect}
              selectedSpanId={selectedSpanId}
            />
          ))}
        </ul>
      )}
    </div>
  )
}

function ItemLine({
  item,
  spansById,
  onSelect,
  selectedSpanId,
}: {
  item: SectionItem
  spansById: Map<string, Span>
  onSelect: (s: Span) => void
  selectedSpanId: string | null
}) {
  const firstEvidence = item.evidence_span_ids
    .map((id) => spansById.get(id))
    .find((s): s is Span => !!s)
  const isSelected = !!firstEvidence && firstEvidence.span_id === selectedSpanId
  return (
    <li
      className={
        'text-sm cursor-pointer rounded px-1 py-0.5 border-l-2 ' +
        (isSelected
          ? 'border-blue-500 bg-blue-50'
          : 'border-transparent hover:bg-slate-50')
      }
      onClick={() => firstEvidence && onSelect(firstEvidence)}
      title={
        item.evidence_span_ids.length > 1
          ? `${item.evidence_span_ids.length} supporting spans — click to view first`
          : 'Click to view source'
      }
    >
      <span className="inline-block w-1.5 h-1.5 mr-1.5 rounded-full bg-slate-400 align-middle" />
      {item.text}
    </li>
  )
}

function ProblemBlock({
  problem,
  spansById,
  onSelect,
  selectedSpanId,
}: {
  problem: Problem
  spansById: Map<string, Span>
  onSelect: (s: Span) => void
  selectedSpanId: string | null
}) {
  const firstEvidence = problem.evidence_span_ids
    .map((id) => spansById.get(id))
    .find((s): s is Span => !!s)
  const isSelected = !!firstEvidence && firstEvidence.span_id === selectedSpanId
  return (
    <li className="text-sm">
      <div
        className={
          'cursor-pointer rounded px-2 py-1 border-l-2 ' +
          (isSelected
            ? 'border-blue-500 bg-blue-50'
            : 'border-transparent hover:bg-slate-50')
        }
        onClick={() => firstEvidence && onSelect(firstEvidence)}
      >
        <div className="font-semibold">{problem.name}</div>
        <div className="text-slate-700">{problem.summary}</div>
      </div>
      <div className="ml-3 mt-1 text-sm">
        {problem.plan ? (
          <div>
            <span className="font-medium text-slate-700">Plan: </span>
            <span>{problem.plan}</span>
            <div className="text-xs text-slate-500 mt-0.5">
              {problem.plan_citations.map((c, i) => (
                <span key={c.chunk_id} className="mr-2">
                  [{i + 1}] {c.document} p.{c.page}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-xs italic text-slate-500">
            No guideline-backed recommendation found in indexed ACC/AHA corpus.
          </div>
        )}
      </div>
    </li>
  )
}
