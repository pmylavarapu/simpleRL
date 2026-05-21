"use client";

import type { ActiveSource, OnePageSummary } from "@/lib/types";
import { ClaimSpan } from "./Claim";
import { PlanItemView } from "./PlanItem";

interface Props {
  summary: OnePageSummary;
  active: ActiveSource;
  onPick: (src: ActiveSource) => void;
}

function Section({
  title,
  children,
  empty,
}: {
  title: string;
  children: React.ReactNode;
  empty?: boolean;
}) {
  if (empty) return null;
  return (
    <section className="mb-3">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-clinical-700 border-b border-clinical-200 mb-1 pb-0.5">
        {title}
      </h3>
      <div className="text-sm leading-snug space-y-0.5">{children}</div>
    </section>
  );
}

export function SummaryPane({ summary, active, onPick }: Props) {
  const p = summary.patient;
  return (
    <div className="p-5 max-w-[760px] mx-auto bg-white border border-clinical-200 rounded-md shadow-sm">
      <header className="flex items-baseline justify-between border-b border-clinical-300 pb-2 mb-3">
        <div>
          <div className="text-lg font-semibold">
            {p.name ?? "—"}{" "}
            {p.age != null && <span className="text-clinical-700">· {p.age}</span>}
            {p.sex && <span className="text-clinical-700">{p.sex}</span>}
          </div>
          <div className="text-xs text-slate-600">
            DOB {p.dob ?? "—"} · MRN {p.mrn ?? "—"}
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-wider text-clinical-500">
          Outside Records Synthesis
        </div>
      </header>

      <Section title="HPI" empty={summary.hpi.length === 0}>
        <p>
          {summary.hpi.map((c, i) => (
            <span key={i}>
              <ClaimSpan claim={c} active={active} onPick={onPick} inline />
              {i < summary.hpi.length - 1 ? " " : ""}
            </span>
          ))}
        </p>
      </Section>

      <Section title="Family History" empty={summary.fam_hx.length === 0}>
        {summary.fam_hx.map((c, i) => (
          <ClaimSpan key={i} claim={c} active={active} onPick={onPick} />
        ))}
      </Section>

      <Section title="Social History" empty={summary.soc_hx.length === 0}>
        {summary.soc_hx.map((c, i) => (
          <ClaimSpan key={i} claim={c} active={active} onPick={onPick} />
        ))}
      </Section>

      <Section title="Past Medical History" empty={summary.pmh.length === 0}>
        <ul className="list-disc list-inside">
          {summary.pmh.map((c, i) => (
            <li key={i}>
              <ClaimSpan claim={c} active={active} onPick={onPick} inline />
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Past Surgical History" empty={summary.psh.length === 0}>
        <ul className="list-disc list-inside">
          {summary.psh.map((c, i) => (
            <li key={i}>
              <ClaimSpan claim={c} active={active} onPick={onPick} inline />
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Medications" empty={summary.meds.length === 0}>
        <ul className="list-disc list-inside">
          {summary.meds.map((c, i) => (
            <li key={i}>
              <ClaimSpan claim={c} active={active} onPick={onPick} inline />
            </li>
          ))}
        </ul>
      </Section>

      <Section
        title="Objective"
        empty={
          summary.objective.labs.length === 0 &&
          summary.objective.cardiology.length === 0
        }
      >
        {summary.objective.labs.length > 0 && (
          <div className="mb-2">
            <div className="text-xs font-medium text-clinical-700">Labs</div>
            <table className="text-xs w-full">
              <thead className="text-[10px] uppercase text-slate-500">
                <tr>
                  <th className="text-left font-medium pr-2">Test</th>
                  <th className="text-left font-medium pr-2">Latest</th>
                  <th className="text-left font-medium pr-2">Date</th>
                  <th className="text-left font-medium">Trend (most-recent prior)</th>
                </tr>
              </thead>
              <tbody>
                {summary.objective.labs.map((lab, i) => (
                  <tr key={i} className="border-t border-clinical-100">
                    <td className="pr-2 py-0.5 font-medium">{lab.name}</td>
                    <td className="pr-2 py-0.5">
                      <span
                        className="claim"
                        onClick={() => onPick(lab.latest.source)}
                      >
                        {lab.latest.value} {lab.latest.unit}
                      </span>
                    </td>
                    <td className="pr-2 py-0.5 text-slate-500">{lab.latest.date}</td>
                    <td className="py-0.5 text-slate-500">
                      {lab.trend.slice(0, 2).map((v, j) => (
                        <span
                          key={j}
                          className="claim mr-2"
                          onClick={() => onPick(v.source)}
                        >
                          {v.value} ({v.date})
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {summary.objective.cardiology.length > 0 && (
          <div>
            <div className="text-xs font-medium text-clinical-700">
              Cardiology imaging / procedures
            </div>
            <ul className="text-sm space-y-0.5">
              {summary.objective.cardiology.map((st, i) => (
                <li key={i}>
                  <span className="text-xs text-slate-500">
                    {st.date} · {st.study_type}:
                  </span>{" "}
                  {st.key_findings.map((c, j) => (
                    <span key={j}>
                      <ClaimSpan claim={c} active={active} onPick={onPick} inline />
                      {j < st.key_findings.length - 1 ? "; " : ""}
                    </span>
                  ))}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Section>

      <Section title="Assessment" empty={summary.assessment.length === 0}>
        <div className="space-y-2">
          {summary.assessment.map((ap, i) => (
            <div key={i}>
              <div className="text-sm font-semibold text-clinical-900">
                {ap.problem}
              </div>
              <p className="text-sm leading-snug">
                {ap.paragraph.map((c, j) => (
                  <span key={j}>
                    <ClaimSpan claim={c} active={active} onPick={onPick} inline />
                    {j < ap.paragraph.length - 1 ? " " : ""}
                  </span>
                ))}
              </p>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Plan (ACC/AHA grounded)" empty={summary.plan.length === 0}>
        <div className="space-y-1.5">
          {summary.plan.map((it, i) => (
            <PlanItemView key={i} item={it} onPick={onPick} />
          ))}
        </div>
      </Section>
    </div>
  );
}
