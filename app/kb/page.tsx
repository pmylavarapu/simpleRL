import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

export default function KBIndex() {
  const cards = loadAllCards();
  const countByTopic = new Map<string, number>();
  for (const c of cards) countByTopic.set(c.topic, (countByTopic.get(c.topic) ?? 0) + 1);

  return (
    <div className="max-w-4xl mx-auto space-y-10 px-6 py-10">
      <div className="space-y-3">
        <p className="eyebrow">Knowledge base</p>
        <h1 className="text-3xl font-medium tracking-tightest">The ASE blueprint, in one place.</h1>
        <p className="text-[14px] text-muted max-w-xl leading-relaxed">
          Notes and cards for each of the {BLUEPRINT.flatMap((s) => s.subtopics).length} subtopics defined by the National Board of Echocardiography.
        </p>
      </div>

      <div className="space-y-12">
        {BLUEPRINT.map((sec) => (
          <section key={sec.code}>
            <div className="flex items-baseline gap-3 mb-4">
              <span className="eyebrow tabular">Section {sec.code}</span>
              <span className="text-[11px] text-muted-soft">·</span>
              <h2 className="text-[15px] font-semibold tracking-tight">
                <Link href={`/kb/${sec.slug}`} className="hover:underline underline-offset-4 decoration-1">
                  {sec.title}
                </Link>
              </h2>
            </div>
            <ul className="divide-y divide-border border-y border-border">
              {sec.subtopics.map((st) => {
                const n = countByTopic.get(st.code) ?? 0;
                return (
                  <li key={st.code}>
                    <Link
                      href={`/kb/${sec.slug}/${st.slug}`}
                      className="group flex items-center py-3.5 px-2 -mx-2 rounded hover:bg-bg-soft/60 transition-colors"
                    >
                      <span className="w-14 shrink-0 text-[12px] text-muted tabular">
                        {st.code}
                      </span>
                      <span className="flex-1 text-[14px] group-hover:underline underline-offset-4 decoration-1">
                        {st.title}
                      </span>
                      <span className="text-[12px] text-muted tabular shrink-0 pl-4">
                        {n} cards
                      </span>
                      <span className="pl-4 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
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
