import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";

const SECTION_LABELS: Record<string, string> = {
  I: "Physics · Instrumentation",
  II: "Valvular Heart Disease",
  III: "Chamber Size & Function",
  IV: "Congenital Heart Disease",
  V: "Masses · Pericardial · Contrast",
  VI: "Miscellaneous (Role of Echo)",
};

export default async function Home() {
  const session = await auth();
  const cards = loadAllCards();
  const total = cards.length;

  const bySection = new Map<string, number>();
  for (const c of cards) {
    const sec = c.topic.split(".")[0];
    bySection.set(sec, (bySection.get(sec) ?? 0) + 1);
  }

  // Compute stats when the user is signed in
  let stats: {
    seen: number;
    unseen: number;
    mastered: number;
    struggling: number;
    dueNow: number;
    coveragePct: number;
    masteryPct: number;
    perSection: Map<string, { seen: number; mastered: number; total: number }>;
  } | null = null;

  if (session?.user) {
    const userId = (session.user as { id?: string }).id!;
    const states = await prisma.reviewState.findMany({ where: { userId } });
    const stateByCardId = new Map(states.map((s) => [s.cardId, s]));
    const now = new Date();

    const seen = states.length;
    const unseen = total - seen;
    const mastered = states.filter((s) => s.state === 2 && s.lapses === 0).length;
    const struggling = states.filter((s) => s.state === 1 || s.state === 3 || s.lapses >= 1).length;
    const dueNow = states.filter((s) => s.due.getTime() <= now.getTime()).length;

    const perSection = new Map<string, { seen: number; mastered: number; total: number }>();
    for (const sec of BLUEPRINT) {
      let sSeen = 0, sMastered = 0, sTotal = 0;
      for (const c of cards) {
        if (!c.topic.startsWith(sec.code + ".")) continue;
        sTotal++;
        const s = stateByCardId.get(c.id);
        if (s) {
          sSeen++;
          if (s.state === 2 && s.lapses === 0) sMastered++;
        }
      }
      perSection.set(sec.code, { seen: sSeen, mastered: sMastered, total: sTotal });
    }

    stats = {
      seen,
      unseen,
      mastered,
      struggling,
      dueNow,
      coveragePct: total > 0 ? Math.round((seen / total) * 100) : 0,
      masteryPct: total > 0 ? Math.round((mastered / total) * 100) : 0,
      perSection,
    };
  }

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-2xl sm:text-3xl font-medium tracking-tightest max-w-2xl leading-[1.15]">
          {stats
            ? "Pick up where you left off."
            : "A knowledge base for the echo boards."}
        </h1>
        <p className="text-[14px] text-muted max-w-xl leading-relaxed">
          {stats ? (
            <>
              <span className="tabular">{stats.dueNow}</span> card{stats.dueNow === 1 ? "" : "s"} due ·{" "}
              <span className="tabular">{stats.unseen}</span> unseen ·{" "}
              <span className="tabular">{stats.struggling}</span> to shore up
            </>
          ) : (
            <>
              {total} cards curated from guideline literature, indexed to the official ASE blueprint. FSRS scheduling underneath.
            </>
          )}
        </p>
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Link
            href="/review"
            className="inline-flex items-center gap-2 rounded-md bg-fg text-accent-fg px-4 py-2 text-[13px] font-medium hover:opacity-90 transition-opacity"
          >
            {stats && stats.dueNow > 0 ? `Review ${stats.dueNow} due` : "Start reviewing"} →
          </Link>
          <Link
            href="/decks"
            className="inline-flex items-center gap-2 rounded-md border border-border px-4 py-2 text-[13px] font-medium hover:border-fg transition-colors"
          >
            Browse
          </Link>
        </div>
      </section>

      {stats && (
        <>
          {/* KPI row */}
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <StatTile label="Seen" value={stats.seen} total={total} />
            <StatTile label="Mastered" value={stats.mastered} total={total} />
            <StatTile label="Due" value={stats.dueNow} />
            <StatTile label="Struggling" value={stats.struggling} />
          </section>

          {/* Progress bar */}
          <section className="sheet p-4 space-y-3">
            <div className="flex items-baseline justify-between">
              <p className="eyebrow">Progress</p>
              <p className="text-[11px] text-muted tabular">
                {stats.seen} / {total} seen
              </p>
            </div>
            <ProgressBar seenPct={stats.coveragePct} masteredPct={stats.masteryPct} />
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
              <LegendDot tone="success">
                <span className="tabular">{stats.masteryPct}%</span> mastered
              </LegendDot>
              <LegendDot tone="warning">
                <span className="tabular">{Math.max(0, stats.coveragePct - stats.masteryPct)}%</span> needs review
              </LegendDot>
              <LegendDot tone="border">
                <span className="tabular">{Math.max(0, 100 - stats.coveragePct)}%</span> new
              </LegendDot>
            </div>
          </section>

          {/* Per-section breakdown */}
          <section>
            <div className="flex flex-wrap items-baseline justify-between gap-3 mb-2">
              <p className="eyebrow">By section</p>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
                <LegendDot tone="success">mastered</LegendDot>
                <LegendDot tone="warning">needs review</LegendDot>
                <LegendDot tone="border">new</LegendDot>
              </div>
            </div>
            <ul className="divide-y divide-border border-y border-border">
              {BLUEPRINT.map((sec) => {
                const s = stats!.perSection.get(sec.code)!;
                const seenPct = s.total > 0 ? (s.seen / s.total) * 100 : 0;
                const masteredPct = s.total > 0 ? (s.mastered / s.total) * 100 : 0;
                return (
                  <li key={sec.code}>
                    <Link
                      href={`/review?section=${sec.code}&status=smart&limit=30`}
                      className="group flex items-center py-2.5 px-2 -mx-2 rounded hover:bg-bg-soft/60 transition-colors"
                    >
                      <span className="w-8 shrink-0 text-[12px] font-mono text-muted tabular">
                        {sec.code}.
                      </span>
                      <div className="flex-1 min-w-0 pr-3">
                        <div className="text-[13px] font-medium tracking-tight truncate group-hover:underline underline-offset-4 decoration-1">
                          {sec.title}
                        </div>
                        <div className="mt-1.5 max-w-md">
                          <ProgressBar seenPct={seenPct} masteredPct={masteredPct} />
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="text-[12px] font-medium tabular">
                          {s.mastered}
                          <span className="text-muted"> / {s.total}</span>
                        </div>
                        <div className="text-[10px] text-muted tabular">
                          {Math.round(masteredPct)}%
                        </div>
                      </div>
                      <span className="pl-3 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      )}

      {!stats && (
        <section>
          <div className="flex items-baseline justify-between mb-2">
            <p className="eyebrow">Contents</p>
            <p className="text-[11px] text-muted tabular">{total} cards · 65 subtopics</p>
          </div>
          <ol className="divide-y divide-border border-y border-border">
            {BLUEPRINT.map((sec) => (
              <li key={sec.code}>
                <Link
                  href={`/kb/${sec.slug}`}
                  className="group flex items-center py-3 hover:bg-bg-soft/60 px-2 -mx-2 rounded transition-colors"
                >
                  <span className="w-8 shrink-0 text-[12px] font-mono text-muted tabular">
                    {sec.code}.
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-[14px] font-medium tracking-tight group-hover:underline decoration-1 underline-offset-4">
                      {sec.title}
                    </div>
                    <div className="text-[11px] text-muted mt-0.5">
                      {SECTION_LABELS[sec.code]} · {sec.subtopics.length} subtopics
                    </div>
                  </div>
                  <span className="text-[11px] text-muted tabular pl-3 shrink-0">
                    {bySection.get(sec.code) ?? 0} cards
                  </span>
                  <span className="pl-3 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
                </Link>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}

function StatTile({ label, value, total }: { label: string; value: number; total?: number }) {
  return (
    <div className="sheet p-3">
      <p className="eyebrow">{label}</p>
      <div className="mt-1 flex items-baseline gap-1">
        <span className="text-2xl font-medium tracking-tightest tabular">{value}</span>
        {total !== undefined && (
          <span className="text-[12px] text-muted tabular">/ {total}</span>
        )}
      </div>
    </div>
  );
}

function ProgressBar({ seenPct, masteredPct }: { seenPct: number; masteredPct: number }) {
  // Three-segment bar:
  //   [0 → mastered%]        green   = mastered
  //   [mastered → seen%]     amber   = started, needs review
  //   [seen → 100%]          gray    = unseen
  const seen = Math.max(0, Math.min(100, seenPct));
  const mastered = Math.max(0, Math.min(100, masteredPct));
  return (
    <div className="relative h-[6px] rounded-full bg-border overflow-hidden">
      {/* amber fill up to seen% — the visible "needs review" band shows between mastered and seen */}
      <div
        className="absolute inset-y-0 left-0 bg-warning transition-all duration-500"
        style={{ width: `${seen}%` }}
      />
      {/* green fill up to mastered% */}
      <div
        className="absolute inset-y-0 left-0 bg-success transition-all duration-500"
        style={{ width: `${mastered}%` }}
      />
    </div>
  );
}

function LegendDot({ tone, children }: { tone: "success" | "warning" | "border"; children: React.ReactNode }) {
  const cls =
    tone === "success" ? "bg-success" : tone === "warning" ? "bg-warning" : "bg-border";
  return (
    <span className="flex items-center gap-1.5">
      <span className={`inline-block w-2 h-2 rounded-full ${cls}`} />
      <span>{children}</span>
    </span>
  );
}
