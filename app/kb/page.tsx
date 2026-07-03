import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

export default function KBIndex() {
  const cards = loadAllCards();
  const countByTopic = new Map<string, number>();
  for (const c of cards) countByTopic.set(c.topic, (countByTopic.get(c.topic) ?? 0) + 1);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Knowledge base</h1>
        <p className="text-muted">Notes and cards organized by the ASCeXAM content outline.</p>
      </div>
      <div className="space-y-6">
        {BLUEPRINT.map((sec) => (
          <section key={sec.code}>
            <h2 className="text-lg font-semibold mb-2">
              <Link href={`/kb/${sec.slug}`} className="hover:underline">
                {sec.code}. {sec.title}
              </Link>
            </h2>
            <ul className="grid gap-1 sm:grid-cols-2">
              {sec.subtopics.map((st) => {
                const n = countByTopic.get(st.code) ?? 0;
                return (
                  <li key={st.code}>
                    <Link
                      href={`/kb/${sec.slug}/${st.slug}`}
                      className="flex items-center justify-between border border-border rounded px-3 py-2 hover:border-accent"
                    >
                      <span>
                        <span className="text-muted mr-2">{st.code}</span>
                        {st.title}
                      </span>
                      <span className="text-xs text-muted">{n} cards</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
