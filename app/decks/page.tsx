import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

export default function DecksPage() {
  const cards = loadAllCards();
  const bySection = new Map<string, number>();
  for (const c of cards) {
    const secCode = c.topic.split(".")[0];
    bySection.set(secCode, (bySection.get(secCode) ?? 0) + 1);
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Decks</h1>
        <p className="text-muted">{cards.length} cards total.</p>
      </div>
      <ul className="grid gap-2 sm:grid-cols-2">
        {BLUEPRINT.map((sec) => (
          <li key={sec.code} className="border border-border rounded p-3">
            <Link href={`/kb/${sec.slug}`} className="font-medium hover:underline">
              {sec.code}. {sec.title}
            </Link>
            <div className="text-xs text-muted mt-1">{bySection.get(sec.code) ?? 0} cards</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
