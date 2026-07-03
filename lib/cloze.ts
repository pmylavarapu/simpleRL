// Minimal cloze renderer.
// Card front contains one or more markers of the form {{c1::hidden text}} or {{c1::hidden text::hint}}.
// For a given cloze number N, all matching markers are hidden on the "question" side and shown on "answer" side.

export type ClozeSegment =
  | { kind: "text"; value: string }
  | { kind: "cloze"; index: number; answer: string; hint?: string };

const RE = /\{\{c(\d+)::([^}]+?)(?:::([^}]+?))?\}\}/g;

export function parseCloze(input: string): ClozeSegment[] {
  const out: ClozeSegment[] = [];
  let lastIdx = 0;
  for (const match of input.matchAll(RE)) {
    const start = match.index ?? 0;
    if (start > lastIdx) out.push({ kind: "text", value: input.slice(lastIdx, start) });
    out.push({
      kind: "cloze",
      index: Number(match[1]),
      answer: match[2],
      hint: match[3],
    });
    lastIdx = start + match[0].length;
  }
  if (lastIdx < input.length) out.push({ kind: "text", value: input.slice(lastIdx) });
  return out;
}

export function clozeIndices(input: string): number[] {
  const idx = new Set<number>();
  for (const match of input.matchAll(RE)) idx.add(Number(match[1]));
  return [...idx].sort((a, b) => a - b);
}
