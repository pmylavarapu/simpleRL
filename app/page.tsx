import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";

export default async function Home() {
  const session = await auth();
  const cards = loadAllCards();
  const total = cards.length;

  const bySection = new Map<string, number>();
  for (const c of cards) {
    const sec = c.topic.split(".")[0];
    bySection.set(sec, (bySection.get(sec) ?? 0) + 1);
  }

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
    <div>
      {/* Hero */}
      <section className="pt-16 sm:pt-28 pb-20 sm:pb-32 text-center">
        <h1 className="text-5xl sm:text-6xl md:text-7xl font-semibold tracking-tightest leading-[1.02]">
          {stats
            ? stats.dueNow > 0
              ? <>Ready when you are.</>
              : <>All caught up.</>
            : <>Echo KB.</>}
        </h1>
        <p className="mt-5 text-[17px] sm:text-[19px] text-muted max-w-xl mx-auto leading-relaxed">
          {stats ? (
            stats.dueNow > 0 ? (
              <>
                <span className="tabular text-fg font-medium">{stats.dueNow}</span> card{stats.dueNow === 1 ? "" : "s"} due today.
              </>
            ) : stats.unseen > 0 ? (
              <>
                No cards due. Learn <span className="tabular text-fg font-medium">{stats.unseen}</span> new one{stats.unseen === 1 ? "" : "s"}?
              </>
            ) : (
              <>Every card scheduled. See you tomorrow.</>
            )
          ) : (
            <>Board-review flashcards, spaced by FSRS.</>
          )}
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[15px]">
          <Link
            href="/review"
            className="text-fg font-medium hover:opacity-70 transition-opacity"
          >
            {stats && stats.dueNow > 0 ? "Review now" : "Start studying"} →
          </Link>
          <Link
            href={stats ? "/decks" : "/about"}
            className="text-muted hover:text-fg transition-colors"
          >
            {stats ? "Browse the knowledge base" : "Learn more"} →
          </Link>
        </div>
      </section>

      {/* Signed-in stats section */}
      {stats && (
        <section className="border-t border-border pt-14 sm:pt-20 pb-16">
          <div className="max-w-4xl mx-auto space-y-14">
            {/* Overall progress */}
            <div className="text-center space-y-5">
              <p className="eyebrow">Your progress</p>
              <div className="flex items-baseline justify-center gap-2">
                <span className="text-6xl sm:text-7xl font-semibold tracking-tightest tabular">{stats.masteryPct}</span>
                <span className="text-2xl text-muted tabular">%</span>
              </div>
              <p className="text-[14px] text-muted">
                mastered ·{" "}
                <span className="tabular">{stats.mastered}</span> of{" "}
                <span className="tabular">{total}</span> cards
              </p>
              <div className="max-w-lg mx-auto pt-3">
                <ProgressBar seenPct={stats.coveragePct} masteredPct={stats.masteryPct} />
                <div className="mt-4 flex flex-wrap items-center justify-center gap-x-6 gap-y-1 text-[12px] text-muted">
                  <LegendDot tone="success">
                    <span className="tabular">{stats.masteryPct}%</span> mastered
                  </LegendDot>
                  <LegendDot tone="warning">
                    <span className="tabular">{Math.max(0, stats.coveragePct - stats.masteryPct)}%</span> in progress
                  </LegendDot>
                  <LegendDot tone="border">
                    <span className="tabular">{Math.max(0, 100 - stats.coveragePct)}%</span> new
                  </LegendDot>
                </div>
              </div>
            </div>

            {/* Compact KPI row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-6 text-center pt-2">
              <Stat label="Seen" value={stats.seen} />
              <Stat label="Mastered" value={stats.mastered} />
              <Stat label="Due" value={stats.dueNow} />
              <Stat label="Struggling" value={stats.struggling} />
            </div>
          </div>
        </section>
      )}

      {/* By section */}
      <section className="border-t border-border pt-14 sm:pt-20 pb-24">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-10">
            <p className="eyebrow">The blueprint</p>
            <h2 className="mt-2 text-3xl sm:text-4xl font-semibold tracking-tightest">
              {stats ? "By section" : "Six domains, 857 cards."}
            </h2>
            {!stats && (
              <p className="mt-3 text-[15px] text-muted max-w-lg mx-auto">
                Every subtopic on the NBE&rsquo;s official ASCeXAM content outline, curated and reviewed.
              </p>
            )}
          </div>
          <ul className="divide-y divide-border border-y border-border">
            {BLUEPRINT.map((sec) => {
              const s = stats?.perSection.get(sec.code);
              const seenPct = s && s.total > 0 ? (s.seen / s.total) * 100 : 0;
              const masteredPct = s && s.total > 0 ? (s.mastered / s.total) * 100 : 0;
              return (
                <li key={sec.code}>
                  <Link
                    href={stats ? `/review?section=${sec.code}&status=smart&limit=30` : `/kb/${sec.slug}`}
                    className="group flex items-center py-4 px-2 -mx-2 rounded hover:bg-bg-soft/60 transition-colors"
                  >
                    <span className="w-10 shrink-0 text-[13px] font-mono text-muted tabular">
                      {sec.code}.
                    </span>
                    <div className="flex-1 min-w-0 pr-4">
                      <div className="text-[15px] font-medium tracking-tight truncate group-hover:underline underline-offset-4 decoration-1">
                        {sec.title}
                      </div>
                      {stats && s ? (
                        <div className="mt-2 max-w-md">
                          <ProgressBar seenPct={seenPct} masteredPct={masteredPct} />
                        </div>
                      ) : (
                        <div className="text-[12px] text-muted mt-0.5">
                          {sec.subtopics.length} subtopics
                        </div>
                      )}
                    </div>
                    <div className="text-right shrink-0">
                      {stats && s ? (
                        <>
                          <div className="text-[13px] font-medium tabular">
                            {s.mastered}
                            <span className="text-muted"> / {s.total}</span>
                          </div>
                          <div className="text-[11px] text-muted tabular">
                            {Math.round(masteredPct)}%
                          </div>
                        </>
                      ) : (
                        <div className="text-[13px] text-muted tabular">
                          {bySection.get(sec.code) ?? 0} cards
                        </div>
                      )}
                    </div>
                    <span className="pl-4 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-3xl sm:text-4xl font-semibold tabular tracking-tightest">{value}</div>
      <div className="mt-1 text-[12px] text-muted uppercase tracking-widest">{label}</div>
    </div>
  );
}

function ProgressBar({ seenPct, masteredPct }: { seenPct: number; masteredPct: number }) {
  const seen = Math.max(0, Math.min(100, seenPct));
  const mastered = Math.max(0, Math.min(100, masteredPct));
  return (
    <div className="relative h-[6px] rounded-full bg-border overflow-hidden">
      <div
        className="absolute inset-y-0 left-0 bg-warning transition-all duration-500"
        style={{ width: `${seen}%` }}
      />
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
