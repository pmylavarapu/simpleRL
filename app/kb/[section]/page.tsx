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

  return (
    <div className="space-y-6">
      <div>
        <Link href="/kb" className="text-sm text-muted hover:text-fg">← All sections</Link>
        <h1 className="text-2xl font-semibold mt-2">
          {section.code}. {section.title}
        </h1>
      </div>
      <ul className="space-y-2">
        {section.subtopics.map((st) => {
          const n = countByTopic.get(st.code) ?? 0;
          return (
            <li key={st.code}>
              <Link
                href={`/kb/${section.slug}/${st.slug}`}
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
    </div>
  );
}
