import Link from "next/link";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { loadAllCards } from "@/lib/cards";
import { ReviewSession } from "@/components/ReviewSession";
import { ReviewFilterForm, type SectionCounts } from "@/components/ReviewFilterForm";

type Status = "due" | "unseen" | "struggling" | "known" | "all";
type SectionCode = "all" | "I" | "II" | "III" | "IV" | "V" | "VI";

const STATUS_LABELS: Record<Status, string> = {
  due: "Due today",
  unseen: "New / unseen",
  struggling: "Struggling",
  known: "Known",
  all: "All",
};

function parseParam<T extends string>(v: string | string[] | undefined, allowed: readonly T[], fallback: T): T {
  const s = Array.isArray(v) ? v[0] : v;
  if (s && (allowed as readonly string[]).includes(s)) return s as T;
  return fallback;
}

const SECTIONS: readonly SectionCode[] = ["all", "I", "II", "III", "IV", "V", "VI"];
const STATUSES: readonly Status[] = ["due", "unseen", "struggling", "known", "all"];

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
  const section = parseParam<SectionCode>(params.section, SECTIONS, "all");
  const status = parseParam<Status>(params.status, STATUSES, "due");
  const limit = Math.min(Math.max(parseInt(params.limit ?? "30", 10) || 30, 1), 500);
  const start = params.start === "1";

  const allCards = loadAllCards();
  const now = new Date();
  const states = await prisma.reviewState.findMany({ where: { userId } });
  const stateByCardId = new Map(states.map((s) => [s.cardId, s]));

  function cardsInSection(sec: SectionCode) {
    return sec === "all" ? allCards : allCards.filter((c) => c.topic.startsWith(sec + "."));
  }

  function matchesStatus(cardId: string, st: Status): boolean {
    const s = stateByCardId.get(cardId);
    if (st === "unseen") return !s;
    if (st === "all") return true;
    if (!s) return st === "due";
    if (st === "due") return s.due.getTime() <= now.getTime();
    if (st === "struggling") return s.state === 1 || s.state === 3 || s.lapses >= 1;
    if (st === "known") return s.state === 2;
    return false;
  }

  // Build counts for every (section × status) combination for the client component.
  const countsBySection = Object.fromEntries(
    SECTIONS.map((sec) => {
      const inSec = cardsInSection(sec);
      const byStatus = Object.fromEntries(
        STATUSES.map((st) => [st, inSec.filter((c) => matchesStatus(c.id, st)).length]),
      ) as Record<Status, number>;
      return [sec, byStatus];
    }),
  ) as SectionCounts;

  if (!start) {
    return (
      <div className="max-w-3xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl font-semibold">Review</h1>
          <p className="text-muted">
            Pick a section and filter, then start. FSRS runs underneath — cards you miss come back sooner, cards you know drift further out.
          </p>
        </div>

        <ReviewFilterForm
          initial={{ section, status, limit }}
          countsBySection={countsBySection}
        />

        <details className="text-sm text-muted">
          <summary className="cursor-pointer">What each filter means</summary>
          <ul className="list-disc pl-5 mt-2 space-y-1">
            <li><strong>Due today</strong> — cards FSRS says are ready, plus never-seen cards.</li>
            <li><strong>New / unseen</strong> — cards you've never reviewed.</li>
            <li><strong>Struggling</strong> — cards you recently rated Again or Hard, or that have lapsed.</li>
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

  // Build the session queue for the selected filters.
  const filtered = cardsInSection(section).filter((c) => matchesStatus(c.id, status));

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
