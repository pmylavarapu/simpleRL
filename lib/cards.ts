import fs from "node:fs";
import path from "node:path";
import { z } from "zod";
import { ALL_SUBTOPICS, findByCode } from "@/content/blueprint";

// Card content is git-tracked JSON, grouped one file per subtopic under /content/cards/<code>.json
// where <code> is the blueprint code with the dot replaced by "_" (e.g. "II_A.json") so it's filesystem-safe.

export const CardSchema = z.object({
  id: z.string().min(3),                 // globally unique, stable
  topic: z.string(),                     // blueprint code, e.g. "II.A"
  type: z.enum(["basic", "cloze"]),
  front: z.string(),                     // for cloze: text containing {{c1::answer}} markers
  back: z.string().optional(),           // for basic
  source: z.string().optional(),         // free-text source reference
  tags: z.array(z.string()).optional(),
});

export type Card = z.infer<typeof CardSchema>;

const CARDS_DIR = path.join(process.cwd(), "content", "cards");

function codePrefix(code: string): string {
  return code.replace(/\./g, "_");
}

let cache: Card[] | null = null;

export function loadAllCards(): Card[] {
  if (cache) return cache;
  const out: Card[] = [];
  const seen = new Set<string>();
  if (!fs.existsSync(CARDS_DIR)) {
    cache = out;
    return out;
  }
  for (const st of ALL_SUBTOPICS) {
    const prefix = codePrefix(st.code);
    // Load both the primary file (II_A.json) and any supplementary files
    // (II_A_extra.json, II_A_2.json, etc.) so cards can be added incrementally.
    const files = fs
      .readdirSync(CARDS_DIR)
      .filter((f) => f === `${prefix}.json` || f.startsWith(`${prefix}_`))
      .map((f) => path.join(CARDS_DIR, f))
      .sort();
    for (const file of files) {
      const raw = JSON.parse(fs.readFileSync(file, "utf8"));
      if (!Array.isArray(raw)) throw new Error(`${file} must be a JSON array`);
      for (const item of raw) {
        const card = CardSchema.parse(item);
        if (seen.has(card.id)) continue;
        seen.add(card.id);
        out.push(card);
      }
    }
  }
  cache = out;
  return out;
}

export function cardsForTopic(code: string): Card[] {
  return loadAllCards().filter((c) => c.topic === code);
}

export function getCard(id: string): Card | undefined {
  return loadAllCards().find((c) => c.id === id);
}

export function topicOfCard(card: Card): { sectionTitle: string; subtopicTitle: string; sectionSlug: string; subtopicSlug: string } | undefined {
  const found = findByCode(card.topic);
  if (!found) return undefined;
  return {
    sectionTitle: found.section.title,
    subtopicTitle: found.subtopic.title,
    sectionSlug: found.section.slug,
    subtopicSlug: found.subtopic.slug,
  };
}
