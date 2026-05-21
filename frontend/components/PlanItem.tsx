"use client";

import type { PlanItem, ActiveSource } from "@/lib/types";

interface Props {
  item: PlanItem;
  onPick: (src: ActiveSource) => void;
}

const PRIORITY_LABEL: Record<number, string> = {
  1: "P1 — life-saving / GDMT",
  2: "P2 — secondary prevention",
  3: "P3 — risk-factor modification",
};

const PRIORITY_BG: Record<number, string> = {
  1: "bg-priority-1",
  2: "bg-priority-2",
  3: "bg-priority-3",
};

export function PlanItemView({ item, onPick }: Props) {
  const hasRule = item.citation.guideline !== "N/A";
  return (
    <div className="border border-clinical-200 rounded-md p-3 bg-white">
      <div className="flex items-center gap-2 mb-1">
        <span
          className={`text-[10px] font-semibold uppercase tracking-wide text-white px-1.5 py-0.5 rounded ${
            PRIORITY_BG[item.priority] ?? "bg-priority-3"
          }`}
        >
          {PRIORITY_LABEL[item.priority] ?? `P${item.priority}`}
        </span>
        <span className="text-xs font-medium text-clinical-700">{item.problem}</span>
      </div>
      <p className="text-sm leading-snug text-ink">{item.recommendation}</p>
      {item.rationale_for_patient && hasRule && (
        <p className="text-xs italic text-slate-600 mt-1">
          {item.rationale_for_patient}
        </p>
      )}
      <div className="flex flex-wrap gap-1.5 mt-2 items-center">
        {hasRule ? (
          <a
            href={item.citation.url ?? "#"}
            target="_blank"
            rel="noreferrer"
            className="text-[11px] bg-clinical-100 text-clinical-700 border border-clinical-200 rounded px-1.5 py-0.5 hover:bg-clinical-200"
          >
            {item.citation.guideline} ({item.citation.year}) · COR{" "}
            {item.citation.cor} · LOE {item.citation.loe}
          </a>
        ) : (
          <span className="text-[11px] bg-amber-50 text-amber-800 border border-amber-200 rounded px-1.5 py-0.5">
            No matching ACC/AHA rule
          </span>
        )}
        {item.precondition_evidence.map((src, i) => (
          <button
            key={i}
            className="text-[11px] bg-evidence/40 border border-amber-300 text-ink rounded px-1.5 py-0.5 hover:bg-evidence"
            onClick={() => onPick(src)}
            title={src.snippet}
          >
            evidence p.{src.page + 1}
          </button>
        ))}
      </div>
    </div>
  );
}
