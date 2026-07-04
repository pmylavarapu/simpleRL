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
    reviewedToday: number;
    coveragePct: number;
    masteryPct: number;
    perSection: Map<string, { seen: number; mastered: number; total: number }>;
  } | null = null;

  if (session?.user) {
    const userId = (session.user as { id?: string }).id!;
    const now = new Date();
    // Midnight in the server's local time — Vercel runs UTC, so US users may
    // see the counter roll over a few hours before their local midnight.
    const todayStart = new Date(now);
    todayStart.setHours(0, 0, 0, 0);
    const [states, reviewedToday] = await Promise.all([
      prisma.reviewState.findMany({ where: { userId } }),
      prisma.reviewEvent.count({ where: { userId, reviewedAt: { gte: todayStart } } }),
    ]);
    const stateByCardId = new Map(states.map((s) => [s.cardId, s]));

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
      reviewedToday,
      coveragePct: total > 0 ? Math.round((seen / total) * 100) : 0,
      masteryPct: total > 0 ? Math.round((mastered / total) * 100) : 0,
      perSection,
    };
  }

  return (
    <div>
      {/* Hero */}
      <section className="bg-bg">
        <div className="mx-auto max-w-4xl px-6 py-10 sm:py-14 text-center">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tightest leading-[1.05]">
            {stats
              ? stats.dueNow > 0
                ? <>Ready when you are.</>
                : <>All caught up.</>
              : <>Echo KB.</>}
          </h1>
          <p className="mt-4 text-[16px] sm:text-[17px] text-muted max-w-xl mx-auto leading-relaxed">
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
              <>Board-review flashcards with spaced repetition.</>
            )}
          </p>
          <div className="mt-5 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[15px]">
            <Link href="/review" className="text-fg font-medium hover:opacity-70 transition-opacity">
              {stats && stats.dueNow > 0 ? "Review now" : "Start studying"} →
            </Link>
            <Link href={stats ? "/decks" : "/about"} className="text-muted hover:text-fg transition-colors">
              {stats ? "Browse the knowledge base" : "Learn more"} →
            </Link>
          </div>
        </div>
      </section>

      {/* Signed-in stats — contrasts against hero with a soft neutral */}
      {stats && (
        <section className="bg-bg-soft border-y border-border">
          <div className="mx-auto max-w-4xl px-6 py-8 sm:py-12 space-y-8">
            <div className="text-center space-y-4">
              <p className="eyebrow">Your progress</p>
              <div className="flex items-baseline justify-center gap-1.5">
                <span className="text-5xl sm:text-6xl font-semibold tracking-tightest tabular">{stats.masteryPct}</span>
                <span className="text-xl text-muted tabular">%</span>
              </div>
              <p className="text-[13px] text-muted">
                mastered · <span className="tabular">{stats.mastered}</span> of <span className="tabular">{total}</span> cards
              </p>
              <div className="max-w-lg mx-auto pt-2">
                <ProgressBar seenPct={stats.coveragePct} masteredPct={stats.masteryPct} />
                <div className="mt-3 flex flex-wrap items-center justify-center gap-x-5 gap-y-1 text-[12px] text-muted">
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

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-x-6 gap-y-5 text-center">
              <Stat label="Seen" value={stats.seen} />
              <Stat label="Mastered" value={stats.mastered} />
              <Stat label="Due" value={stats.dueNow} />
              <Stat label="Struggling" value={stats.struggling} />
              <Stat label="Today" value={stats.reviewedToday} />
            </div>
          </div>
        </section>
      )}

      {/* Blueprint — white again to contrast the soft stat section */}
      <section className="bg-bg">
        <div className="mx-auto max-w-4xl px-6 py-8 sm:py-12">
          <div className="text-center mb-6">
            <p className="eyebrow">The blueprint</p>
            <h2 className="mt-2 text-2xl sm:text-3xl font-semibold tracking-tightest">
              {stats ? "By section" : `Six domains, ${total} cards.`}
            </h2>
            {!stats && (
              <p className="mt-2 text-[14px] text-muted max-w-lg mx-auto">
                Every subtopic on the NBE&rsquo;s official ASCeXAM outline, curated and reviewed.
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
                    className="group flex items-center py-3 px-2 -mx-2 rounded hover:bg-bg-soft/60 transition-colors"
                  >
                    <span className="w-9 shrink-0 text-[12px] text-muted tabular">
                      {sec.code}.
                    </span>
                    <div className="flex-1 min-w-0 pr-3">
                      <div className="text-[14px] font-medium tracking-tight truncate group-hover:underline underline-offset-4 decoration-1">
                        {sec.title}
                      </div>
                      {stats && s ? (
                        <div className="mt-1.5 max-w-md">
                          <ProgressBar seenPct={seenPct} masteredPct={masteredPct} />
                        </div>
                      ) : (
                        <div className="text-[11px] text-muted mt-0.5">
                          {sec.subtopics.length} subtopics
                        </div>
                      )}
                    </div>
                    <div className="text-right shrink-0">
                      {stats && s ? (
                        <>
                          <div className="text-[12px] font-medium tabular">
                            {s.mastered}
                            <span className="text-muted"> / {s.total}</span>
                          </div>
                          <div className="text-[10px] text-muted tabular">
                            {Math.round(masteredPct)}%
                          </div>
                        </>
                      ) : (
                        <div className="text-[12px] text-muted tabular">
                          {bySection.get(sec.code) ?? 0} cards
                        </div>
                      )}
                    </div>
                    <span className="pl-3 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
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
      <div className="text-2xl sm:text-3xl font-semibold tabular tracking-tightest">{value}</div>
      <div className="mt-0.5 text-[11px] text-muted uppercase tracking-widest">{label}</div>
    </div>
  );
}

function ProgressBar({ seenPct, masteredPct }: { seenPct: number; masteredPct: number }) {
  const seen = Math.max(0, Math.min(100, seenPct));
  const mastered = Math.max(0, Math.min(100, masteredPct));
  return (
    <div className="relative h-[6px] rounded-full bg-border overflow-hidden">
      <div className="absolute inset-y-0 left-0 bg-warning transition-all duration-500" style={{ width: `${seen}%` }} />
      <div className="absolute inset-y-0 left-0 bg-success transition-all duration-500" style={{ width: `${mastered}%` }} />
    </div>
  );
}

function LegendDot({ tone, children }: { tone: "success" | "warning" | "border"; children: React.ReactNode }) {
  const cls = tone === "success" ? "bg-success" : tone === "warning" ? "bg-warning" : "bg-border";
  return (
    <span className="flex items-center gap-1.5">
      <span className={`inline-block w-2 h-2 rounded-full ${cls}`} />
      <span>{children}</span>
    </span>
  );
}
