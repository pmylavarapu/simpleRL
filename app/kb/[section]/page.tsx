import Link from "next/link";
import { notFound } from "next/navigation";
import { BLUEPRINT, findSection } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

export function generateStaticParams() {
  return BLUEPRINT.map((s) => ({ section: s.slug }));
}

export default async function SectionPage({ params }: { params: Promise<{ section: string }> }) {
  const { section: sectionSlug } = await params;
  const section = findSection(sectionSlug);
  if (!section) notFound();

  const cards = loadAllCards();
  const countByTopic = new Map<string, number>();
  for (const c of cards) countByTopic.set(c.topic, (countByTopic.get(c.topic) ?? 0) + 1);

  const totalCards = section.subtopics.reduce((sum, st) => sum + (countByTopic.get(st.code) ?? 0), 0);

  return (
    <div className="max-w-4xl mx-auto space-y-10">
      <div className="space-y-3">
        <Link href="/kb" className="eyebrow hover:text-fg transition-colors inline-block">← Knowledge base</Link>
        <div className="flex items-baseline gap-3 pt-1">
          <span className="text-[13px] font-mono text-muted tabular">{section.code}.</span>
          <h1 className="text-3xl font-medium tracking-tightest">{section.title}</h1>
        </div>
        <p className="text-[13px] text-muted">
          {section.subtopics.length} subtopics · {totalCards} cards
        </p>
      </div>

      <ul className="divide-y divide-border border-y border-border">
        {section.subtopics.map((st) => {
          const n = countByTopic.get(st.code) ?? 0;
          return (
            <li key={st.code}>
              <Link
                href={`/kb/${section.slug}/${st.slug}`}
                className="group flex items-center py-4 px-2 -mx-2 rounded hover:bg-bg-soft/60 transition-colors"
              >
                <span className="w-14 shrink-0 text-[12px] font-mono text-muted tabular">
                  {st.code}
                </span>
                <span className="flex-1 text-[15px] font-medium tracking-tight group-hover:underline underline-offset-4 decoration-1">
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
    </div>
  );
}
