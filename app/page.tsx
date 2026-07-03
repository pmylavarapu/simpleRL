import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

export default function Home() {
  const cards = loadAllCards();
  const total = cards.length;

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold">ASCeXAM Board Review</h1>
        <p className="text-muted max-w-2xl">
          A knowledge base and spaced-repetition review system organized around the ASE Board Exam blueprint.
          Cards use FSRS scheduling — cards you miss come back sooner, cards you know drift further out.
        </p>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        <Link href="/review" className="border border-border rounded-lg p-4 hover:border-accent">
          <div className="text-sm text-muted">Start</div>
          <div className="text-lg font-medium">Review due cards</div>
        </Link>
        <Link href="/kb" className="border border-border rounded-lg p-4 hover:border-accent">
          <div className="text-sm text-muted">Study</div>
          <div className="text-lg font-medium">Browse knowledge base</div>
        </Link>
        <Link href="/decks" className="border border-border rounded-lg p-4 hover:border-accent">
          <div className="text-sm text-muted">Explore</div>
          <div className="text-lg font-medium">{total} cards across {BLUEPRINT.length} domains</div>
        </Link>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Blueprint</h2>
        <ol className="space-y-2">
          {BLUEPRINT.map((sec) => (
            <li key={sec.code} className="border border-border rounded-md p-3">
              <Link href={`/kb/${sec.slug}`} className="font-medium hover:underline">
                {sec.code}. {sec.title}
              </Link>
              <div className="text-sm text-muted mt-1">
                {sec.subtopics.length} subtopics
              </div>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
