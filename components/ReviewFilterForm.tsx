"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";

type Status = "due" | "unseen" | "struggling" | "known" | "all";
type SectionCode = "all" | "I" | "II" | "III" | "IV" | "V" | "VI";

const STATUS_LABELS: Record<Status, string> = {
  due: "Due today",
  unseen: "New / unseen",
  struggling: "Struggling (recent Again / Hard)",
  known: "Known (stable)",
  all: "All cards",
};

const SECTION_LABELS: Record<SectionCode, string> = {
  all: "All (I–VI)",
  I: "I · Physics",
  II: "II · Valvular",
  III: "III · Chambers",
  IV: "IV · Congenital",
  V: "V · Masses/Peri",
  VI: "VI · Misc",
};

const LIMIT_OPTIONS = [10, 20, 30, 50, 100] as const;
const UNLIMITED = 500;

export type SectionCounts = Record<SectionCode, Record<Status, number>>;

export function ReviewFilterForm({
  initial,
  countsBySection,
}: {
  initial: { section: SectionCode; status: Status; limit: number };
  countsBySection: SectionCounts;
}) {
  const router = useRouter();
  const [section, setSection] = useState<SectionCode>(initial.section);
  const [status, setStatus] = useState<Status>(initial.status);
  const [limit, setLimit] = useState<number>(initial.limit);
  const [submitting, setSubmitting] = useState(false);

  const counts = countsBySection[section];
  const matching = counts[status];
  const sessionSize = Math.min(matching, limit);

  const cardStr = useMemo(
    () => `${sessionSize} card${sessionSize === 1 ? "" : "s"}`,
    [sessionSize],
  );

  function start() {
    if (matching === 0) return;
    setSubmitting(true);
    const params = new URLSearchParams({
      start: "1",
      section,
      status,
      limit: String(limit),
    });
    router.push(`/review?${params.toString()}`);
  }

  return (
    <div className="space-y-6 border border-border rounded-lg p-6">
      <div>
        <div className="text-sm font-medium mb-2">Section</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {(Object.keys(SECTION_LABELS) as SectionCode[]).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSection(s)}
              className={`text-left border rounded px-3 py-2 text-sm transition-colors ${
                section === s
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-border hover:border-accent"
              }`}
            >
              {SECTION_LABELS[s]}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="text-sm font-medium mb-2">Filter</div>
        <div className="space-y-1">
          {(Object.entries(STATUS_LABELS) as [Status, string][]).map(([s, label]) => (
            <button
              key={s}
              type="button"
              onClick={() => setStatus(s)}
              className={`w-full flex items-center justify-between border rounded px-3 py-2 text-sm transition-colors ${
                status === s
                  ? "border-accent bg-accent/10"
                  : "border-border hover:border-accent"
              }`}
            >
              <span className="flex items-center">
                <span
                  className={`inline-block w-3 h-3 rounded-full border mr-3 ${
                    status === s ? "bg-accent border-accent" : "border-muted"
                  }`}
                />
                {label}
              </span>
              <span className="text-xs text-muted">{counts[s]} cards</span>
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="text-sm font-medium mb-2">Session size</div>
        <div className="flex flex-wrap gap-2">
          {LIMIT_OPTIONS.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setLimit(n)}
              className={`border rounded px-3 py-2 text-sm transition-colors ${
                limit === n
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-border hover:border-accent"
              }`}
            >
              {n}
            </button>
          ))}
          <button
            type="button"
            onClick={() => setLimit(UNLIMITED)}
            className={`border rounded px-3 py-2 text-sm transition-colors ${
              limit >= UNLIMITED
                ? "border-accent bg-accent/10 text-accent"
                : "border-border hover:border-accent"
            }`}
          >
            Unlimited
          </button>
        </div>
      </div>

      <button
        type="button"
        onClick={start}
        disabled={matching === 0 || submitting}
        className="w-full rounded-md bg-accent text-white py-3 font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {matching === 0 ? "No cards match this filter" : `Start review · ${cardStr}`}
      </button>
    </div>
  );
}
