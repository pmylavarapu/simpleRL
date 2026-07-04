import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

export default function AboutPage() {
  const cards = loadAllCards();
  const subtopics = BLUEPRINT.flatMap((s) => s.subtopics).length;

  return (
    <div className="max-w-2xl mx-auto space-y-14 py-6">
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
            Nothing here is proprietary content. Cards were curated from open study material and cross-checked against published guidelines (ASE 2016 Nagueh, ASE 2017 Zoghbi, 2020 ACC/AHA VHD, 2020 AHA/ACC HCM, ASE 2010 Rudski, 2016 Baumgartner/ASE AS, Duke IE, and others).
          </p>
        </div>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">How review works</p>
        <div className="text-[15px] leading-relaxed space-y-3">
          <p>
            Reviews are scheduled with <span className="font-medium">FSRS</span> — the algorithm behind modern Anki. After each card you rate it Again, Hard, Good, or Easy; the scheduler adapts the next interval to your recall.
          </p>
          <p>
            Cards you miss come back sooner. Cards you know drift further out. Over a few sessions the mix shifts to what you need to work on.
          </p>
        </div>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Filters</p>
        <ul className="text-[15px] leading-relaxed space-y-2 list-none">
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">Due today</span>
            <span className="flex-1">Cards FSRS says are ready, plus new ones you haven't seen.</span>
          </li>
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">Unseen</span>
            <span className="flex-1">Cards you've never reviewed.</span>
          </li>
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">Struggling</span>
            <span className="flex-1">Recent Again/Hard grades, or cards that have lapsed at least once.</span>
          </li>
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">Known</span>
            <span className="flex-1">Cards that have stabilized in FSRS's Review state.</span>
          </li>
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">All</span>
            <span className="flex-1">Every card in the selected section.</span>
          </li>
        </ul>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Keyboard</p>
        <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-[14px] tabular">
          <div className="flex justify-between border-b border-border py-1.5"><span className="text-muted">Reveal</span><span className="font-mono">SPACE · ENTER</span></div>
          <div className="flex justify-between border-b border-border py-1.5"><span className="text-muted">Again</span><span className="font-mono">1</span></div>
          <div className="flex justify-between border-b border-border py-1.5"><span className="text-muted">Hard</span><span className="font-mono">2</span></div>
          <div className="flex justify-between border-b border-border py-1.5"><span className="text-muted">Good</span><span className="font-mono">3</span></div>
          <div className="flex justify-between border-b border-border py-1.5"><span className="text-muted">Easy</span><span className="font-mono">4</span></div>
        </div>
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
                  <span className="w-10 shrink-0 text-[12px] font-mono text-muted tabular">
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
