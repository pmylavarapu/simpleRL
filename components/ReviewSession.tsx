"use client";

import { useState, useMemo } from "react";
import { parseCloze, clozeIndices } from "@/lib/cloze";

type QueueCard = {
  id: string;
  type: "basic" | "cloze";
  front: string;
  back: string;
  topic: string;
};

type Grade = "again" | "hard" | "good" | "easy";

const GRADE_LABEL: Record<Grade, string> = {
  again: "Again",
  hard: "Hard",
  good: "Good",
  easy: "Easy",
};

const GRADE_KEY: Record<string, Grade> = {
  "1": "again",
  "2": "hard",
  "3": "good",
  "4": "easy",
};

export function ReviewSession({ initialQueue }: { initialQueue: QueueCard[] }) {
  const [queue, setQueue] = useState(initialQueue);
  const [idx, setIdx] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);
  const [stats, setStats] = useState<Record<Grade, number>>({
    again: 0,
    hard: 0,
    good: 0,
    easy: 0,
  });

  const card = queue[idx];
  const clozeIdx = useMemo(() => {
    if (!card) return 1;
    const idxs = clozeIndices(card.front);
    return idxs[0] ?? 1;
  }, [card]);

  if (done || !card) {
    return (
      <div className="border border-border rounded-lg p-6 text-center space-y-4">
        <div className="text-lg font-medium">Session complete</div>
        <div className="text-sm text-muted">
          Again {stats.again} · Hard {stats.hard} · Good {stats.good} · Easy {stats.easy}
        </div>
        <a href="/review" className="inline-block underline">Start another session</a>
      </div>
    );
  }

  async function submitGrade(grade: Grade) {
    if (submitting || !revealed) return;
    setSubmitting(true);
    try {
      const res = await fetch("/api/review", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ cardId: card.id, grade }),
      });
      if (!res.ok) throw new Error(`grade failed: ${res.status}`);
      setStats((s) => ({ ...s, [grade]: s[grade] + 1 }));
      if (idx + 1 >= queue.length) {
        setDone(true);
      } else {
        setIdx(idx + 1);
        setRevealed(false);
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="space-y-4"
      onKeyDown={(e) => {
        if (!revealed && (e.key === " " || e.key === "Enter")) {
          e.preventDefault();
          setRevealed(true);
          return;
        }
        if (revealed && GRADE_KEY[e.key]) {
          e.preventDefault();
          void submitGrade(GRADE_KEY[e.key]);
        }
      }}
      tabIndex={0}
    >
      <div className="text-xs text-muted text-right">
        {idx + 1} / {queue.length} · {card.topic}
      </div>

      <div className="border border-border rounded-lg p-6 min-h-[220px] flex items-center justify-center text-center">
        {card.type === "basic" ? (
          <div className="text-lg leading-relaxed">
            {!revealed ? card.front : (
              <>
                <div className="mb-3">{card.front}</div>
                <div className="border-t border-border pt-3 mt-3 text-fg">{card.back}</div>
              </>
            )}
          </div>
        ) : (
          <ClozeView front={card.front} clozeIdx={clozeIdx} revealed={revealed} />
        )}
      </div>

      {!revealed ? (
        <button
          className="w-full rounded-md bg-accent text-white py-2 font-medium hover:opacity-90"
          onClick={() => setRevealed(true)}
        >
          Show answer (Space)
        </button>
      ) : (
        <div className="grid grid-cols-4 gap-2">
          {(["again", "hard", "good", "easy"] as Grade[]).map((g, i) => (
            <button
              key={g}
              disabled={submitting}
              onClick={() => submitGrade(g)}
              className="rounded-md border border-border py-2 text-sm hover:border-accent disabled:opacity-50"
            >
              <div className="font-medium">{GRADE_LABEL[g]}</div>
              <div className="text-[10px] text-muted">{i + 1}</div>
            </button>
          ))}
        </div>
      )}

      <div className="text-xs text-muted text-center">
        Keys: Space/Enter reveal · 1 Again · 2 Hard · 3 Good · 4 Easy
      </div>
    </div>
  );
}

function ClozeView({
  front,
  clozeIdx,
  revealed,
}: {
  front: string;
  clozeIdx: number;
  revealed: boolean;
}) {
  const segs = parseCloze(front);
  return (
    <div className="text-lg leading-relaxed">
      {segs.map((seg, i) => {
        if (seg.kind === "text") return <span key={i}>{seg.value}</span>;
        const active = seg.index === clozeIdx;
        if (active && !revealed) {
          return (
            <span
              key={i}
              className="inline-block px-2 py-0.5 rounded bg-border/60 text-muted italic"
            >
              {seg.hint ? `[${seg.hint}]` : "[…]"}
            </span>
          );
        }
        if (active && revealed) {
          return (
            <span key={i} className="font-semibold text-accent">
              {seg.answer}
            </span>
          );
        }
        // Other cloze numbers are shown as normal text.
        return <span key={i}>{seg.answer}</span>;
      })}
    </div>
  );
}
