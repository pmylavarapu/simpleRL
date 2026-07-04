import Link from "next/link";
import { BLUEPRINT } from "@/content/blueprint";
import { loadAllCards } from "@/lib/cards";

const SECTION_ONE_LINER: Record<string, string> = {
  I: "Physics, Doppler, artifacts, TEE, M-mode, 3D",
  II: "AS, AR, MS, MR, prosthetics, endocarditis, PHT",
  III: "Chamber quant, cardiomyopathies, diastolic, RV, stress",
  IV: "TOF, TGA, ASD, VSD, Fontan, Ebstein, TAPVR",
  V: "Pericardial, tumors, contrast, rhythm, transplant",
  VI: "HF, embolic sources, systemic disease, AF, TAVR",
};

export default function DecksPage() {
  const cards = loadAllCards();
  const bySection = new Map<string, number>();
  for (const c of cards) {
    const secCode = c.topic.split(".")[0];
    bySection.set(secCode, (bySection.get(secCode) ?? 0) + 1);
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="space-y-3">
        <p className="eyebrow">Decks</p>
        <h1 className="text-3xl font-medium tracking-tightest">{cards.length} cards, 6 domains.</h1>
        <p className="text-[14px] text-muted max-w-xl leading-relaxed">
          Each domain of the ASCeXAM blueprint is a deck. Click through to browse subtopics, or start a filtered review.
        </p>
      </div>

      <ul className="grid gap-3 sm:grid-cols-2">
        {BLUEPRINT.map((sec) => {
          const n = bySection.get(sec.code) ?? 0;
          return (
            <li key={sec.code}>
              <Link
                href={`/kb/${sec.slug}`}
                className="group block sheet p-5 hover:border-fg transition-colors"
              >
                <div className="flex items-baseline gap-2">
                  <span className="eyebrow tabular">Section {sec.code}</span>
                </div>
                <div className="mt-2 text-[16px] font-medium tracking-tight group-hover:underline underline-offset-4 decoration-1">
                  {sec.title}
                </div>
                <div className="text-[12px] text-muted mt-1">
                  {SECTION_ONE_LINER[sec.code]}
                </div>
                <div className="flex items-center justify-between mt-4 pt-3 border-t border-border">
                  <span className="text-[12px] text-muted">{sec.subtopics.length} subtopics</span>
                  <span className="text-[12px] tabular">{n} cards →</span>
                </div>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
