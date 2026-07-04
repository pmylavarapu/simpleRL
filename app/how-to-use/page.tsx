import Link from "next/link";

export default function HowToUsePage() {
  return (
    <div className="max-w-2xl mx-auto space-y-14 py-6">
      <header className="space-y-3">
        <p className="eyebrow">How to use</p>
        <h1 className="text-3xl sm:text-4xl font-medium tracking-tightest leading-[1.1]">
          A short guide to studying here.
        </h1>
        <p className="text-[15px] text-muted leading-relaxed">
          Ten minutes a day, guided by the FSRS algorithm, will move you steadily toward mastery of the ASCeXAM blueprint.
        </p>
      </header>

      <section className="space-y-3">
        <p className="eyebrow">The daily loop</p>
        <ol className="text-[15px] leading-relaxed space-y-3 list-decimal pl-5">
          <li>
            Open the home page. Your <strong>Due</strong> count tells you what the scheduler wants you to review right now. Hit the primary button — it starts an FSRS session with those cards.
          </li>
          <li>
            Read the front. When you're ready, press <span className="font-mono text-[13px] bg-bg-soft border border-border rounded px-1.5 py-0.5">SPACE</span> to reveal the answer.
          </li>
          <li>
            Rate your recall with <span className="font-mono text-[13px]">1</span> (Again), <span className="font-mono text-[13px]">2</span> (Hard), <span className="font-mono text-[13px]">3</span> (Good), or <span className="font-mono text-[13px]">4</span> (Easy). Honesty here is the point — the algorithm needs a signal it can trust.
          </li>
          <li>
            Do 20–50 cards, take a break, come back tomorrow.
          </li>
        </ol>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">FSRS in one paragraph</p>
        <p className="text-[15px] leading-relaxed">
          FSRS (Free Spaced Repetition Scheduler) estimates two things per card: how stable your memory of it is, and how difficult it is for you. After each grade it updates both estimates and schedules the next review at the point where your recall probability is about 90 %. That means cards you miss come back within a day; cards you crush drift out weeks or months. The system pushes you into the zone where you're forgetting <em>just</em> enough to make each review count.
        </p>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Review modes</p>
        <p className="text-[14px] text-muted leading-relaxed">
          Four modes, chosen when you start a session. The defaults get you moving; the others are for targeted work.
        </p>
        <ul className="text-[15px] leading-relaxed space-y-3 list-none">
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">FSRS</span>
            <span className="flex-1">The recommended default. Due-scheduled cards first, new cards mixed in as capacity allows. This is what you'll want most days.</span>
          </li>
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">New</span>
            <span className="flex-1">Only cards you've never reviewed. Useful when you want to seed a new section before the scheduler starts folding it in.</span>
          </li>
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">Incorrect</span>
            <span className="flex-1">Only cards you've missed or that have lapsed. A quick cleanup pass before an exam block.</span>
          </li>
          <li className="flex gap-3">
            <span className="text-muted font-mono text-[12px] w-28 shrink-0 pt-1 tabular">All</span>
            <span className="flex-1">Every card in the selected section, regardless of state. Handy for skimming a whole domain.</span>
          </li>
        </ul>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Progress bar segments</p>
        <p className="text-[15px] leading-relaxed">
          The three-tone bar on the home page decomposes coverage into what you actually know versus what still needs work.
        </p>
        <ul className="text-[14px] leading-relaxed space-y-2 list-none">
          <li className="flex gap-3 items-center">
            <span className="inline-block w-3 h-3 rounded-full bg-success shrink-0" />
            <span><strong>Mastered</strong> — in FSRS's Review state with zero lapses. You've reliably recalled these.</span>
          </li>
          <li className="flex gap-3 items-center">
            <span className="inline-block w-3 h-3 rounded-full bg-warning shrink-0" />
            <span><strong>Needs review</strong> — seen but not yet stable. Learning, relearning, or previously lapsed cards.</span>
          </li>
          <li className="flex gap-3 items-center">
            <span className="inline-block w-3 h-3 rounded-full bg-border border border-border-strong shrink-0" />
            <span><strong>New</strong> — cards you haven't touched yet.</span>
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
        <p className="text-[12px] text-muted mt-2">Works anywhere on the review page — no need to click into the card first.</p>
      </section>

      <section className="space-y-3">
        <p className="eyebrow">Study tips</p>
        <ul className="text-[15px] leading-relaxed space-y-3 list-disc pl-5">
          <li>
            Prefer <strong>Good</strong> over <strong>Easy</strong> for most cards. Save Easy for cards you'd bet on getting right in a year.
          </li>
          <li>
            If you're uncertain, hit <strong>Hard</strong>, not Good — the algorithm needs to know when a card is on the edge.
          </li>
          <li>
            Read the linked note on the <Link href="/decks" className="underline underline-offset-4 decoration-1">Knowledge base</Link> page when a card confuses you. Cards are the drill; notes are the context.
          </li>
          <li>
            Come back daily. Skipping a day isn't a disaster, but the schedule assumes roughly consistent effort.
          </li>
        </ul>
      </section>

      <footer className="pt-4 border-t border-border">
        <p className="text-[12px] text-muted">
          Questions or feedback? Check the <Link href="/about" className="underline underline-offset-4 decoration-1">About</Link> page for context on how this was built.
        </p>
      </footer>
    </div>
  );
}
