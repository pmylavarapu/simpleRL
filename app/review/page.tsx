import Link from "next/link";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { loadAllCards } from "@/lib/cards";
import { BLUEPRINT } from "@/content/blueprint";
import { ReviewSession } from "@/components/ReviewSession";

type Status = "due" | "unseen" | "struggling" | "known" | "all";
type SectionCode = "all" | "I" | "II" | "III" | "IV" | "V" | "VI";

const STATUS_LABELS: Record<Status, string> = {
  due: "Due today",
  unseen: "New / unseen",
  struggling: "Struggling (recent Again / Hard)",
  known: "Known (stable)",
  all: "All cards",
};

const LIMIT_OPTIONS = [10, 20, 30, 50, 100];

function parseParam<T extends string>(v: string | string[] | undefined, allowed: T[], fallback: T): T {
  const s = Array.isArray(v) ? v[0] : v;
  if (s && (allowed as string[]).includes(s)) return s as T;
  return fallback;
}

export default async function ReviewPage({
  searchParams,
}: {
  searchParams: Promise<{
    start?: string;
    section?: string;
    status?: string;
    limit?: string;
  }>;
}) {
  const session = await auth();
  if (!session?.user) {
    return (
      <div className="max-w-md mx-auto space-y-3">
        <h1 className="text-2xl font-semibold">Sign in to review</h1>
        <p className="text-muted">Your FSRS progress is saved to your account.</p>
        <Link href="/signin" className="underline">Sign in with Google</Link>
      </div>
    );
  }

  const userId = (session.user as { id?: string }).id!;
  const params = await searchParams;
  const section = parseParam<SectionCode>(params.section, ["all", "I", "II", "III", "IV", "V", "VI"], "all");
  const status = parseParam<Status>(params.status, ["due", "unseen", "struggling", "known", "all"], "due");
  const limit = Math.min(Math.max(parseInt(params.limit ?? "30", 10) || 30, 1), 500);
  const start = params.start === "1";

  const allCards = loadAllCards();
  const now = new Date();
  const states = await prisma.reviewState.findMany({ where: { userId } });
  const stateByCardId = new Map(states.map((s) => [s.cardId, s]));

  // Filter by section
  const sectionCards = section === "all"
    ? allCards
    : allCards.filter((c) => c.topic.startsWith(section + "."));

  // Filter by status
  const filtered = sectionCards.filter((c) => {
    const s = stateByCardId.get(c.id);
    if (status === "unseen") return !s;
    if (status === "all") return true;
    if (!s) return status === "due"; // No state = eligible for "due" queue (new card)
    if (status === "due") return s.due.getTime() <= now.getTime();
    if (status === "struggling") return s.state === 1 || s.state === 3 || s.lapses >= 1;
    if (status === "known") return s.state === 2;
    return false;
  });

  // Counts for each combination (for the filter UI badges)
  const counts = {
    all: sectionCards.length,
    due: sectionCards.filter((c) => {
      const s = stateByCardId.get(c.id);
      if (!s) return true;
      return s.due.getTime() <= now.getTime();
    }).length,
    unseen: sectionCards.filter((c) => !stateByCardId.get(c.id)).length,
    struggling: sectionCards.filter((c) => {
      const s = stateByCardId.get(c.id);
      return !!s && (s.state === 1 || s.state === 3 || s.lapses >= 1);
    }).length,
    known: sectionCards.filter((c) => stateByCardId.get(c.id)?.state === 2).length,
  } satisfies Record<Status, number>;

  if (!start) {
    // Filter picker view
    return (
      <div className="max-w-3xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl font-semibold">Review</h1>
          <p className="text-muted">
            Choose what to review, then hit Start. FSRS scheduling runs underneath — cards you miss come back sooner.
          </p>
        </div>

        <form action="/review" method="GET" className="space-y-6 border border-border rounded-lg p-6">
          <input type="hidden" name="start" value="1" />

          <div>
            <div className="text-sm font-medium mb-2">Section</div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {(["all", "I", "II", "III", "IV", "V", "VI"] as SectionCode[]).map((s) => (
                <label
                  key={s}
                  className={`border border-border rounded px-3 py-2 text-sm cursor-pointer hover:border-accent ${s === section ? "border-accent bg-accent/10" : ""}`}
                >
                  <input
                    type="radio"
                    name="section"
                    value={s}
                    defaultChecked={s === section}
                    className="sr-only"
                  />
                  {s === "all" ? "All (I–VI)" : s === "I" ? "I · Physics" : s === "II" ? "II · Valvular" : s === "III" ? "III · Chambers" : s === "IV" ? "IV · Congenital" : s === "V" ? "V · Masses/Peri" : "VI · Misc"}
                </label>
              ))}
            </div>
          </div>

          <div>
            <div className="text-sm font-medium mb-2">Filter</div>
            <div className="space-y-1">
              {(Object.entries(STATUS_LABELS) as [Status, string][]).map(([s, label]) => (
                <label
                  key={s}
                  className={`flex items-center justify-between border border-border rounded px-3 py-2 text-sm cursor-pointer hover:border-accent ${s === status ? "border-accent bg-accent/10" : ""}`}
                >
                  <span>
                    <input
                      type="radio"
                      name="status"
                      value={s}
                      defaultChecked={s === status}
                      className="mr-2"
                    />
                    {label}
                  </span>
                  <span className="text-xs text-muted">{counts[s]} cards</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <div className="text-sm font-medium mb-2">Session size</div>
            <div className="flex flex-wrap gap-2">
              {LIMIT_OPTIONS.map((n) => (
                <label
                  key={n}
                  className={`border border-border rounded px-3 py-2 text-sm cursor-pointer hover:border-accent ${n === limit ? "border-accent bg-accent/10" : ""}`}
                >
                  <input
                    type="radio"
                    name="limit"
                    value={n}
                    defaultChecked={n === limit}
                    className="sr-only"
                  />
                  {n}
                </label>
              ))}
              <label
                className={`border border-border rounded px-3 py-2 text-sm cursor-pointer hover:border-accent ${limit >= 500 ? "border-accent bg-accent/10" : ""}`}
              >
                <input
                  type="radio"
                  name="limit"
                  value="500"
                  defaultChecked={limit >= 500}
                  className="sr-only"
                />
                Unlimited
              </label>
            </div>
          </div>

          <button
            type="submit"
            disabled={filtered.length === 0}
            className="w-full rounded-md bg-accent text-white py-2 font-medium hover:opacity-90 disabled:opacity-50"
          >
            Start review · {Math.min(filtered.length, limit)} card{Math.min(filtered.length, limit) === 1 ? "" : "s"}
          </button>
        </form>

        <details className="text-sm text-muted">
          <summary className="cursor-pointer">What each filter means</summary>
          <ul className="list-disc pl-5 mt-2 space-y-1">
            <li><strong>Due today</strong> — cards the FSRS scheduler says are ready for review, plus never-seen cards.</li>
            <li><strong>New / unseen</strong> — cards you've never reviewed.</li>
            <li><strong>Struggling</strong> — cards you recently rated Again or Hard, or that have lapsed at least once.</li>
            <li><strong>Known</strong> — cards in the FSRS Review state (stable).</li>
            <li><strong>All</strong> — every card in the selected section.</li>
          </ul>
        </details>

        <div className="text-sm">
          <Link href="/kb" className="underline">Browse the knowledge base</Link>
          {" · "}
          <Link href="/decks" className="underline">Deck breakdown</Link>
        </div>
      </div>
    );
  }

  // Session view — order queue: due first (oldest first), then unseen (blueprint order), then struggling, then rest
  const dueIds = filtered
    .filter((c) => {
      const s = stateByCardId.get(c.id);
      return !!s && s.due.getTime() <= now.getTime();
    })
    .sort((a, b) => {
      const sa = stateByCardId.get(a.id)!;
      const sb = stateByCardId.get(b.id)!;
      return sa.due.getTime() - sb.due.getTime();
    })
    .map((c) => c.id);
  const unseenIds = filtered.filter((c) => !stateByCardId.get(c.id)).map((c) => c.id);
  const otherIds = filtered.filter((c) => {
    const s = stateByCardId.get(c.id);
    return !!s && s.due.getTime() > now.getTime();
  }).map((c) => c.id);

  const queueIds = [...dueIds, ...unseenIds, ...otherIds].slice(0, limit);
  const queueCards = queueIds
    .map((id) => allCards.find((c) => c.id === id))
    .filter((c): c is NonNullable<typeof c> => Boolean(c));

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Review</h1>
        <Link href="/review" className="text-sm underline text-muted">Change filters</Link>
      </div>
      <div className="text-sm text-muted">
        {section === "all" ? "All sections" : `Section ${section}`} · {STATUS_LABELS[status]} · {queueCards.length} card{queueCards.length === 1 ? "" : "s"}
      </div>
      {queueCards.length === 0 ? (
        <div className="border border-dashed border-border rounded p-6 text-center text-muted">
          No cards match this filter right now.
          <div className="mt-3">
            <Link href="/review" className="underline">Change filters</Link>
          </div>
        </div>
      ) : (
        <ReviewSession
          initialQueue={queueCards.map((c) => ({
            id: c.id,
            type: c.type,
            front: c.front,
            back: c.back ?? "",
            topic: c.topic,
          }))}
        />
      )}
    </div>
  );
}
