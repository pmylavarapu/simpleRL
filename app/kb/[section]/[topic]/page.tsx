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
    <div className="max-w-3xl mx-auto space-y-12">
      <header className="space-y-3">
        <Link href={`/kb/${section.slug}`} className="eyebrow hover:text-fg transition-colors inline-block">
          ← Section {section.code} · {section.title}
        </Link>
        <div className="flex items-baseline gap-3 pt-1">
          <span className="text-[13px] font-mono text-muted tabular">{subtopic.code}</span>
          <h1 className="text-3xl font-medium tracking-tightest">{subtopic.title}</h1>
        </div>
        <p className="text-[13px] text-muted tabular">
          {cards.length} card{cards.length === 1 ? "" : "s"}
        </p>
      </header>

      <section>
        <p className="eyebrow mb-4">Notes</p>
        {note.exists ? (
          <article className="prose prose-neutral max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{note.content}</ReactMarkdown>
          </article>
        ) : (
          <div className="sheet p-8 text-center text-[13px] text-muted">
            No notes yet for this subtopic.
          </div>
        )}
      </section>

      <section>
        <p className="eyebrow mb-4">Cards</p>
        {cards.length === 0 ? (
          <div className="sheet p-8 text-center text-[13px] text-muted">
            No cards yet for this subtopic.
          </div>
        ) : (
          <ul className="space-y-2">
            {cards.map((c) => (
              <li key={c.id} className="sheet p-4">
                <div className="flex items-center gap-3 mb-2">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider font-medium ${c.type === "cloze" ? "bg-fg text-accent-fg" : "border border-border text-muted"}`}>
                    {c.type}
                  </span>
                  <span className="text-[11px] text-muted font-mono tabular">{c.id}</span>
                </div>
                <div className="text-[14px] leading-relaxed">{c.front}</div>
                {c.back && <div className="text-[14px] text-muted mt-2 leading-relaxed">→ {c.back}</div>}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
