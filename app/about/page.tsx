import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

export default function AboutPage() {
  const cards = loadAllCards();
  const subtopics = BLUEPRINT.flatMap((s) => s.subtopics).length;

  return (
    <div className="max-w-4xl mx-auto space-y-12 px-6 py-10">
      <header className="space-y-3">
        <p className="eyebrow">About</p>
        <h1 className="text-3xl sm:text-4xl font-medium tracking-tightest leading-[1.1]">
          A quiet study tool for the ASE echo boards.
        </h1>
        <p className="text-[15px] text-muted leading-relaxed">
          {cards.length} cards across {subtopics} subtopics, indexed to the National Board of Echocardiography's official ASCeXAM blueprint. Built for focus.
        </p>
      </header>

      <section className="space-y-3">
        <p className="eyebrow">What it is</p>
        <div className="text-[15px] leading-relaxed space-y-3">
          <p>
            A spaced-repetition review deck and companion knowledge base for the Adult Comprehensive Echocardiography exam. Cards distill high-yield facts from board-review study guides and guideline documents; each links back to a note page for context.
          </p>
          <p>
            For a walkthrough of the daily loop, review modes, and keyboard shortcuts, see the{" "}
            <Link href="/how-to-use" className="underline underline-offset-4 decoration-1">How to use</Link> page.
          </p>
        </div>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Sources</p>
        <p className="text-[15px] leading-relaxed">
          Knowledge Base and flashcards were curated from notes shared by Cardiovascular Disease fellows at UCSD. Notes were then cross referenced with American Society of Echocardiography (ASE) guidelines using Claude Opus 4.7 to ensure accuracy. There may still be mistakes. Please let us know if you find any!
        </p>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Privacy</p>
        <p className="text-[15px] leading-relaxed">
          Sign in is Google-only. The only data stored is your per-card scheduling state. No trackers.
        </p>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Blueprint</p>
        <p className="text-[14px] text-muted leading-relaxed">
          The ASCeXAM covers six domains. Card counts weight roughly to what the exam tests.
        </p>
        <ul className="divide-y divide-border border-y border-border">
          {BLUEPRINT.map((sec) => {
            const n = cards.filter((c) => c.topic.startsWith(sec.code + ".")).length;
            return (
              <li key={sec.code}>
                <Link
                  href={`/kb/${sec.slug}`}
                  className="group flex items-center py-3 px-2 -mx-2 rounded hover:bg-bg-soft/60 transition-colors"
                >
                  <span className="w-10 shrink-0 text-[12px] text-muted tabular">
                    {sec.code}.
                  </span>
                  <span className="flex-1 text-[14px] group-hover:underline underline-offset-4 decoration-1">
                    {sec.title}
                  </span>
                  <span className="text-[12px] text-muted tabular shrink-0 pl-4">{n}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </section>

      <footer className="pt-4 border-t border-border">
        <p className="text-[12px] text-muted">
          Not affiliated with the American Society of Echocardiography or the National Board of Echocardiography.
        </p>
      </footer>
    </div>
  );
}
