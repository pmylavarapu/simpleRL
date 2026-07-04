import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

const SECTION_LABELS: Record<string, string> = {
  I: "Physics · Instrumentation",
  II: "Valvular Heart Disease",
  III: "Chamber Size & Function",
  IV: "Congenital Heart Disease",
  V: "Masses · Pericardial · Contrast",
  VI: "Miscellaneous (Role of Echo)",
};

export default function Home() {
  const cards = loadAllCards();
  const total = cards.length;

  const bySection = new Map<string, number>();
  for (const c of cards) {
    const sec = c.topic.split(".")[0];
    bySection.set(sec, (bySection.get(sec) ?? 0) + 1);
  }

  return (
    <div className="space-y-16">
      <section className="space-y-6 pt-4">
        <p className="eyebrow">ASCeXAM · Board Review</p>
        <h1 className="text-4xl sm:text-5xl font-medium tracking-tightest max-w-2xl leading-[1.05]">
          A precise, no-nonsense knowledge base for the echo boards.
        </h1>
        <p className="text-[15px] text-muted max-w-xl leading-relaxed">
          {total} cards curated from the standard guideline literature and organized to the official ASE blueprint. FSRS scheduling — the cards you miss come back sooner, the ones you know drift further out.
        </p>
        <div className="flex flex-wrap items-center gap-3 pt-2">
          <Link
            href="/review"
            className="inline-flex items-center gap-2 rounded-md bg-fg text-accent-fg px-5 py-2.5 text-[14px] font-medium hover:opacity-90 transition-opacity"
          >
            Start reviewing →
          </Link>
          <Link
            href="/kb"
            className="inline-flex items-center gap-2 rounded-md border border-border px-5 py-2.5 text-[14px] font-medium hover:border-fg transition-colors"
          >
            Browse the KB
          </Link>
        </div>
      </section>

      <section>
        <div className="flex items-baseline justify-between mb-4">
          <p className="eyebrow">Contents</p>
          <p className="text-[12px] text-muted tabular">{total} cards · 65 subtopics</p>
        </div>
        <ol className="divide-y divide-border border-y border-border">
          {BLUEPRINT.map((sec) => (
            <li key={sec.code}>
              <Link
                href={`/kb/${sec.slug}`}
                className="group flex items-center py-5 hover:bg-bg-soft/60 px-2 -mx-2 rounded transition-colors"
              >
                <span className="w-10 shrink-0 text-[13px] font-mono text-muted tabular">
                  {sec.code}.
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-[15px] font-medium tracking-tight group-hover:underline decoration-1 underline-offset-4">
                    {sec.title}
                  </div>
                  <div className="text-[12px] text-muted mt-1">
                    {SECTION_LABELS[sec.code]} · {sec.subtopics.length} subtopics
                  </div>
                </div>
                <span className="text-[12px] text-muted tabular pl-4 shrink-0">
                  {bySection.get(sec.code) ?? 0} cards
                </span>
                <span className="pl-4 text-muted-soft group-hover:text-fg transition-colors" aria-hidden="true">→</span>
              </Link>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
