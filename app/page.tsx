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
    <div className="space-y-16">
      <section className="space-y-6 pt-4">
        <p className="eyebrow">ASCeXAM · Board Review</p>
        <h1 className="text-4xl sm:text-5xl font-medium tracking-tightest max-w-2xl leading-[1.05]">
          {stats
            ? "Pick up where you left off."
            : "A precise, no-nonsense knowledge base for the echo boards."}
        </h1>
        <p className="text-[15px] text-muted max-w-xl leading-relaxed">
          {stats ? (
            <>
              <span className="tabular">{stats.dueNow}</span> card{stats.dueNow === 1 ? "" : "s"} due,{" "}
              <span className="tabular">{stats.unseen}</span> unseen,{" "}
              <span className="tabular">{stats.struggling}</span> to shore up.
            </>
          ) : (
            <>
              {total} cards curated from the standard guideline literature and organized to the official ASE blueprint. FSRS scheduling — the cards you miss come back sooner, the ones you know drift further out.
            </>
          )}
        </p>
        <div className="flex flex-wrap items-center gap-3 pt-2">
          <Link
            href="/review"
            className="inline-flex items-center gap-2 rounded-md bg-fg text-accent-fg px-5 py-2.5 text-[14px] font-medium hover:opacity-90 transition-opacity"
          >
            {stats && stats.dueNow > 0 ? `Review ${stats.dueNow} due` : "Start reviewing"} →
          </Link>
          <Link
            href="/decks"
            className="inline-flex items-center gap-2 rounded-md border border-border px-5 py-2.5 text-[14px] font-medium hover:border-fg transition-colors"
          >
            Browse decks
          </Link>
        </div>
      </section>

      {stats && (
        <>
          {/* KPI row */}
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatTile label="Seen" value={stats.seen} total={total} />
            <StatTile label="Mastered" value={stats.mastered} total={total} />
            <StatTile label="Due" value={stats.dueNow} />
            <StatTile label="Struggling" value={stats.struggling} />
          </section>

          {/* Progress bar */}
          <section className="sheet p-6 sm:p-8 space-y-5">
            <div className="flex items-baseline justify-between">
              <p className="eyebrow">Progress</p>
              <p className="text-[12px] text-muted tabular">
                {stats.seen} of {total} cards seen
              </p>
            </div>
            <div>
              <ProgressBar seenPct={stats.coveragePct} masteredPct={stats.masteryPct} />
              <div className="flex items-center justify-between mt-3 text-[12px] text-muted">
                <span className="flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-fg" />
                  <span className="tabular">{stats.masteryPct}%</span> mastered
                </span>
                <span className="flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full border border-fg" />
                  <span className="tabular">{stats.coveragePct}%</span> seen
                </span>
              </div>
            </div>
          </section>

          {/* Per-section breakdown */}
          <section>
            <div className="flex items-baseline justify-between mb-4">
              <p className="eyebrow">By section</p>
              <p className="text-[12px] text-muted tabular">Mastery per domain</p>
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
                      className="group flex items-center py-4 px-2 -mx-2 rounded hover:bg-bg-soft/60 transition-colors"
                    >
                      <span className="w-10 shrink-0 text-[13px] font-mono text-muted tabular">
                        {sec.code}.
                      </span>
                      <div className="flex-1 min-w-0 pr-4">
                        <div className="text-[14px] font-medium tracking-tight truncate group-hover:underline underline-offset-4 decoration-1">
                          {sec.title}
                        </div>
                        <div className="mt-2 max-w-md">
                          <ProgressBar seenPct={seenPct} masteredPct={masteredPct} />
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="text-[13px] font-medium tabular">
                          {s.mastered}
                          <span className="text-muted"> / {s.total}</span>
                        </div>
                        <div className="text-[11px] text-muted mt-0.5 tabular">
                          {Math.round(masteredPct)}%
                        </div>
                      </div>
                      <span className="pl-4 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
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
          <div className="flex items-baseline justify-between mb-4">
            <p className="eyebrow">Contents</p>
            <p className="text-[12px] text-muted tabular">{total} cards · 65 subtopics</p>
          </div>
          <ol className="divide-y divide-border border-y border-border">
            {BLUEPRINT.map((sec) => (
              <li key={sec.code}>
                <Link
                  href={`/kb/${sec.slug}`}
                  className="group flex items-center py-5 hover:bg-bg-soft/60 px-2 -mx-2 rounded transition-colors"
                >
                  <span className="w-10 shrink-0 text-[13px] font-mono text-muted tabular">
                    {sec.code}.
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-[15px] font-medium tracking-tight group-hover:underline decoration-1 underline-offset-4">
                      {sec.title}
                    </div>
                    <div className="text-[12px] text-muted mt-1">
                      {SECTION_LABELS[sec.code]} · {sec.subtopics.length} subtopics
                    </div>
                  </div>
                  <span className="text-[12px] text-muted tabular pl-4 shrink-0">
                    {bySection.get(sec.code) ?? 0} cards
                  </span>
                  <span className="pl-4 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
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
    <div className="sheet p-4 sm:p-5">
      <p className="eyebrow">{label}</p>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className="text-3xl font-medium tracking-tightest tabular">{value}</span>
        {total !== undefined && (
          <span className="text-[13px] text-muted tabular">/ {total}</span>
        )}
      </div>
    </div>
  );
}

function ProgressBar({ seenPct, masteredPct }: { seenPct: number; masteredPct: number }) {
  // Two-layer bar: mastered (solid black) sits inside seen (mid-gray),
  // with a soft neutral track behind. Apple-esque, three shades of neutral.
  const seen = Math.max(0, Math.min(100, seenPct));
  const mastered = Math.max(0, Math.min(100, masteredPct));
  return (
    <div className="relative h-[6px] rounded-full bg-border overflow-hidden">
      <div
        className="absolute inset-y-0 left-0 bg-muted-soft transition-all duration-500"
        style={{ width: `${seen}%` }}
      />
      <div
        className="absolute inset-y-0 left-0 bg-fg transition-all duration-500"
        style={{ width: `${mastered}%` }}
      />
    </div>
  );
}
