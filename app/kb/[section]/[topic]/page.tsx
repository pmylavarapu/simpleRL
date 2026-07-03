import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BLUEPRINT, findSubtopic } from "@/content/blueprint";
import { cardsForTopic } from "@/lib/cards";
import { loadNote } from "@/lib/notes";

export function generateStaticParams() {
  return BLUEPRINT.flatMap((s) => s.subtopics.map((st) => ({ section: s.slug, topic: st.slug })));
}

export default async function TopicPage({
  params,
}: {
  params: Promise<{ section: string; topic: string }>;
}) {
  const { section: sectionSlug, topic: topicSlug } = await params;
  const found = findSubtopic(sectionSlug, topicSlug);
  if (!found) notFound();
  const { section, subtopic } = found;

  const note = loadNote(section.slug, subtopic.slug);
  const cards = cardsForTopic(subtopic.code);

  return (
    <div className="space-y-8">
      <div>
        <Link href={`/kb/${section.slug}`} className="text-sm text-muted hover:text-fg">
          ← {section.code}. {section.title}
        </Link>
        <h1 className="text-2xl font-semibold mt-2">
          {subtopic.code} — {subtopic.title}
        </h1>
      </div>

      <section>
        <h2 className="text-lg font-semibold mb-2">Notes</h2>
        {note.exists ? (
          <article className="prose prose-neutral dark:prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{note.content}</ReactMarkdown>
          </article>
        ) : (
          <div className="text-sm text-muted border border-dashed border-border rounded p-4">
            No notes yet for this subtopic. Notes will be added as sources arrive.
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-2">Cards ({cards.length})</h2>
        {cards.length === 0 ? (
          <div className="text-sm text-muted border border-dashed border-border rounded p-4">
            No cards yet for this subtopic.
          </div>
        ) : (
          <ul className="space-y-2">
            {cards.map((c) => (
              <li key={c.id} className="border border-border rounded p-3">
                <div className="text-xs text-muted mb-1">
                  {c.type.toUpperCase()} · {c.id}
                </div>
                <div className="text-sm">{c.front}</div>
                {c.back && <div className="text-sm text-muted mt-1">→ {c.back}</div>}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
