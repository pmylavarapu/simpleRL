import Link from "next/link";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { loadAllCards } from "@/lib/cards";
import { ReviewSession } from "@/components/ReviewSession";

// Simple queue policy: due-or-new, sorted by dueness. Cap per session for a focused block.
const SESSION_LIMIT = 30;

export default async function ReviewPage() {
  const session = await auth();
  if (!session?.user) {
    return (
      <div className="max-w-md">
        <h1 className="text-xl font-semibold mb-2">Sign in to review</h1>
        <p className="text-muted mb-4">Your FSRS progress is saved to your account.</p>
        <Link href="/signin" className="underline">Sign in with Google</Link>
      </div>
    );
  }

  const userId = (session.user as { id?: string }).id!;
  const allCards = loadAllCards();
  const now = new Date();

  const states = await prisma.reviewState.findMany({ where: { userId } });
  const stateByCardId = new Map(states.map((s) => [s.cardId, s]));

  // Build the due queue: existing rows with due <= now, plus cards without a row yet (new).
  const dueRows = states.filter((s) => s.due.getTime() <= now.getTime());
  const seenIds = new Set(states.map((s) => s.cardId));
  const newCards = allCards.filter((c) => !seenIds.has(c.id));

  // Order: due first (oldest overdue first), then new cards in blueprint order.
  const queue = [
    ...dueRows
      .sort((a, b) => a.due.getTime() - b.due.getTime())
      .map((r) => r.cardId),
    ...newCards.map((c) => c.id),
  ].slice(0, SESSION_LIMIT);

  const queueCards = queue
    .map((id) => allCards.find((c) => c.id === id))
    .filter((c): c is NonNullable<typeof c> => Boolean(c));

  const dueCount = dueRows.length;
  const newCount = newCards.length;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Review</h1>
        <div className="text-sm text-muted">
          {dueCount} due · {newCount} new
        </div>
      </div>

      {queueCards.length === 0 ? (
        <div className="border border-dashed border-border rounded p-6 text-center text-muted">
          Nothing due right now. Come back later, or explore the{" "}
          <Link href="/kb" className="underline">knowledge base</Link>.
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
