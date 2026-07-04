"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";

type Status = "smart" | "new" | "incorrect" | "all";
type SectionCode = "all" | "I" | "II" | "III" | "IV" | "V" | "VI";

const STATUS_LABELS: Record<Status, string> = {
  smart: "Spaced repetition",
  new: "New cards",
  incorrect: "Incorrect only",
  all: "All cards",
};

const STATUS_HINT: Record<Status, string> = {
  smart: "Teach me what I don't know — due cards first, new cards mixed in.",
  new: "Cards you've never reviewed.",
  incorrect: "Cards you've missed or that have lapsed.",
  all: "Every card in the selected section.",
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
    <div className="sheet p-6 sm:p-8 space-y-8">
      <div>
        <p className="eyebrow mb-3">Section</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {(Object.keys(SECTION_LABELS) as SectionCode[]).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSection(s)}
              aria-pressed={section === s}
              className={`text-left rounded-md border px-3 py-2.5 text-[13px] transition-all ${
                section === s
                  ? "border-fg bg-fg text-accent-fg"
                  : "border-border hover:border-fg"
              }`}
            >
              {SECTION_LABELS[s]}
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="eyebrow mb-3">Mode</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {(Object.entries(STATUS_LABELS) as [Status, string][]).map(([s, label]) => (
            <button
              key={s}
              type="button"
              onClick={() => setStatus(s)}
              aria-pressed={status === s}
              className={`text-left rounded-md border px-4 py-3 transition-all ${
                status === s
                  ? "border-fg bg-fg text-accent-fg"
                  : "border-border hover:border-fg"
              }`}
            >
              <div className="flex items-baseline justify-between gap-3">
                <div className="text-[14px] font-medium">{label}</div>
                <div className={`text-[12px] tabular ${status === s ? "opacity-70" : "text-muted"}`}>
                  {counts[s]}
                </div>
              </div>
            </button>
          ))}
        </div>
        <p className="text-[12px] text-muted mt-2 leading-relaxed">
          {STATUS_HINT[status]}
        </p>
      </div>

      <div>
        <p className="eyebrow mb-3">Session size</p>
        <div className="flex flex-wrap gap-2">
          {LIMIT_OPTIONS.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setLimit(n)}
              aria-pressed={limit === n}
              className={`rounded-md border px-3.5 py-2 text-[13px] tabular transition-all ${
                limit === n
                  ? "border-fg bg-fg text-accent-fg"
                  : "border-border hover:border-fg"
              }`}
            >
              {n}
            </button>
          ))}
          <button
            type="button"
            onClick={() => setLimit(UNLIMITED)}
            aria-pressed={limit >= UNLIMITED}
            className={`rounded-md border px-3.5 py-2 text-[13px] transition-all ${
              limit >= UNLIMITED
                ? "border-fg bg-fg text-accent-fg"
                : "border-border hover:border-fg"
            }`}
          >
            Unlimited
          </button>
        </div>
      </div>

      <div className="pt-2">
        <button
          type="button"
          onClick={start}
          disabled={matching === 0 || submitting}
          className="w-full rounded-md bg-fg text-accent-fg py-3.5 text-[14px] font-medium hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
        >
          {matching === 0 ? "No cards match" : (
            <>Start review <span className="opacity-60 mx-1">·</span> <span className="tabular">{cardStr}</span></>
          )}
        </button>
      </div>
    </div>
  );
}
