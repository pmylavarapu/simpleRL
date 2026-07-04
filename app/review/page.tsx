import Link from "next/link";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { loadAllCards } from "@/lib/cards";
import { ReviewSession } from "@/components/ReviewSession";
import { ReviewFilterForm, type SectionCounts } from "@/components/ReviewFilterForm";

type Status = "smart" | "new" | "incorrect" | "all";
type SectionCode = "all" | "I" | "II" | "III" | "IV" | "V" | "VI";

const STATUS_LABELS: Record<Status, string> = {
  smart: "Spaced repetition",
  new: "New cards",
  incorrect: "Incorrect only",
  all: "All cards",
};

function parseParam<T extends string>(v: string | string[] | undefined, allowed: readonly T[], fallback: T): T {
  const s = Array.isArray(v) ? v[0] : v;
  if (s && (allowed as readonly string[]).includes(s)) return s as T;
  return fallback;
}

const SECTIONS: readonly SectionCode[] = ["all", "I", "II", "III", "IV", "V", "VI"];
const STATUSES: readonly Status[] = ["smart", "new", "incorrect", "all"];

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
      <div className="max-w-md mx-auto sheet p-8 text-center space-y-3 mx-6 my-10">
        <p className="eyebrow">Restricted</p>
        <h1 className="text-2xl font-medium tracking-tight">Sign in to review</h1>
        <p className="text-[13px] text-muted">Your progress is saved to your account.</p>
        <div className="pt-2">
          <Link href="/signin" className="inline-flex items-center gap-2 rounded-md bg-fg text-accent-fg px-5 py-2.5 text-[14px] font-medium hover:opacity-90 transition-opacity">
            Sign in with Google →
          </Link>
        </div>
      </div>
    );
  }

  const userId = (session.user as { id?: string }).id!;
  const params = await searchParams;
  const section = parseParam<SectionCode>(params.section, SECTIONS, "all");
  const status = parseParam<Status>(params.status, STATUSES, "smart");
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
    if (st === "all") return true;
    if (st === "new") return !s;
    if (st === "incorrect") return !!s && (s.state === 1 || s.state === 3 || s.lapses >= 1);
    // "smart" = FSRS mix: due-scheduled cards + new (never-seen) cards.
    if (st === "smart") {
      if (!s) return true;
      return s.due.getTime() <= now.getTime();
    }
    return false;
  }

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
      <div className="max-w-4xl mx-auto space-y-8 px-6 py-10">
        <div className="space-y-3">
          <p className="eyebrow">Review</p>
          <h1 className="text-3xl font-medium tracking-tightest">
            Pick your set.
          </h1>
          <p className="text-[14px] text-muted max-w-xl leading-relaxed">
            Choose a section, filter, and session size. Spaced repetition runs underneath — miss a card and it comes back sooner.
          </p>
        </div>

        <ReviewFilterForm
          initial={{ section, status, limit }}
          countsBySection={countsBySection}
        />

        <div className="text-[13px] text-muted flex items-center gap-6">
          <Link href="/kb" className="hover:text-fg transition-colors">Browse the KB →</Link>
          <Link href="/decks" className="hover:text-fg transition-colors">Deck breakdown →</Link>
        </div>
      </div>
    );
  }

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
    <div className="max-w-4xl mx-auto space-y-6 px-6 py-10">
      <div className="flex items-baseline justify-between">
        <div>
          <p className="eyebrow">
            {section === "all" ? "All sections" : `Section ${section}`} · {STATUS_LABELS[status]}
          </p>
          <div className="text-[13px] text-muted mt-1 tabular">
            {queueCards.length} card{queueCards.length === 1 ? "" : "s"} in this session
          </div>
        </div>
        <Link href="/review" className="text-[12px] text-muted hover:text-fg transition-colors">← Change filters</Link>
      </div>
      {queueCards.length === 0 ? (
        <div className="sheet p-10 text-center text-[13px] text-muted">
          No cards match this filter right now.
          <div className="mt-4">
            <Link href="/review" className="underline hover:no-underline">Change filters</Link>
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
