"use client";

import { useState, useMemo, useEffect, useRef, useCallback } from "react";
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
  const [queue] = useState(initialQueue);
  const [idx, setIdx] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);
  const [flash, setFlash] = useState<Grade | null>(null);
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

  // Refs that always mirror latest state, so the window listener sees fresh values.
  const submittingRef = useRef(false);
  const revealedRef = useRef(revealed);
  const cardRef = useRef(card);
  const idxRef = useRef(idx);
  const queueLenRef = useRef(queue.length);
  const doneRef = useRef(done);
  useEffect(() => { revealedRef.current = revealed; }, [revealed]);
  useEffect(() => { cardRef.current = card; }, [card]);
  useEffect(() => { idxRef.current = idx; }, [idx]);
  useEffect(() => { queueLenRef.current = queue.length; }, [queue.length]);
  useEffect(() => { doneRef.current = done; }, [done]);

  const submitGrade = useCallback(async (grade: Grade) => {
    if (doneRef.current) return;
    if (submittingRef.current || !revealedRef.current) return;
    const currentCard = cardRef.current;
    if (!currentCard) return;
    submittingRef.current = true;
    setSubmitting(true);
    setFlash(grade);
    const flashDelay = new Promise<void>((resolve) => setTimeout(resolve, 320));
    try {
      const res = await fetch("/api/review", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ cardId: currentCard.id, grade }),
      });
      if (!res.ok) throw new Error(`grade failed: ${res.status}`);
      await flashDelay;
      setStats((s) => ({ ...s, [grade]: s[grade] + 1 }));
      if (idxRef.current + 1 >= queueLenRef.current) {
        setDone(true);
      } else {
        setIdx(idxRef.current + 1);
        setRevealed(false);
      }
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
      setFlash(null);
    }
  }, []);

  // Global keyboard shortcuts — always listens at document level, no focus needed.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (doneRef.current) return;
      // Skip when typing in a form field.
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      // Ignore modifier combos so we don't hijack browser shortcuts.
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      if (!revealedRef.current) {
        if (e.key === " " || e.code === "Space" || e.key === "Enter") {
          e.preventDefault();
          setRevealed(true);
        }
        return;
      }
      const grade = GRADE_KEY[e.key];
      if (grade) {
        e.preventDefault();
        void submitGrade(grade);
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [submitGrade]);

  if (done || !card) {
    const totalGraded = stats.again + stats.hard + stats.good + stats.easy;
    const correct = stats.good + stats.easy;
    const pctCorrect = totalGraded ? Math.round((correct / totalGraded) * 100) : 0;
    return (
      <div className="sheet p-10 text-center space-y-6">
        <p className="eyebrow">Session complete</p>
        <div className="text-5xl font-medium tracking-tightest tabular">{pctCorrect}%</div>
        <p className="text-[13px] text-muted">
          {correct} of {totalGraded} rated Good or Easy
        </p>
        <div className="grid grid-cols-4 gap-2 pt-2">
          {(["again", "hard", "good", "easy"] as Grade[]).map((g) => (
            <div key={g} className="border border-border rounded-md py-3">
              <div className="text-[11px] text-muted uppercase tracking-wider">{g}</div>
              <div className="text-lg font-medium tabular mt-1">{stats[g]}</div>
            </div>
          ))}
        </div>
        <div className="pt-4">
          <a href="/review" className="inline-flex items-center gap-2 rounded-md bg-fg text-accent-fg px-5 py-2.5 text-[14px] font-medium hover:opacity-90 transition-opacity">
            Start another session →
          </a>
        </div>
      </div>
    );
  }

  // Prevent Space from also triggering focused button click, which would fight the window listener.
  function stopButtonSpace(e: React.KeyboardEvent<HTMLButtonElement>) {
    if (e.key === " " || e.code === "Space") e.preventDefault();
  }

  const progressPct = (idx / queue.length) * 100;

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="flex-1 h-[3px] bg-border rounded-full overflow-hidden">
          <div
            className="h-full bg-fg transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
        <div className="text-[11px] text-muted tabular shrink-0">
          {idx + 1} / {queue.length}
        </div>
      </div>

      <div className={`sheet p-8 sm:p-10 min-h-[280px] flex flex-col justify-center transition-colors duration-200 ${flashClass(flash)}`}>
        <div className="text-center">
          <p className="eyebrow mb-6">{card.topic}</p>
          {card.type === "basic" ? (
            <div className="text-[18px] sm:text-[19px] leading-relaxed tracking-tight text-fg-strong">
              {!revealed ? (
                card.front
              ) : (
                <>
                  <div className="mb-6">{card.front}</div>
                  <div className="pt-6 border-t border-border text-fg text-left sm:text-center">
                    {card.back}
                  </div>
                </>
              )}
            </div>
          ) : (
            <ClozeView front={card.front} clozeIdx={clozeIdx} revealed={revealed} />
          )}
        </div>
      </div>

      {!revealed ? (
        <button
          type="button"
          onKeyDown={stopButtonSpace}
          className="w-full rounded-md bg-fg text-accent-fg py-3.5 text-[14px] font-medium hover:opacity-90 transition-opacity"
          onClick={() => setRevealed(true)}
        >
          Show answer
          <span className="ml-2 opacity-60 text-[11px] tabular">SPACE</span>
        </button>
      ) : (
        <div className="grid grid-cols-4 gap-2">
          {(["again", "hard", "good", "easy"] as Grade[]).map((g, i) => (
            <button
              key={g}
              type="button"
              disabled={submitting}
              onKeyDown={stopButtonSpace}
              onClick={() => submitGrade(g)}
              className="rounded-md border border-border py-3 text-[13px] hover:border-fg disabled:opacity-40 transition-colors"
            >
              <div className="font-medium">{GRADE_LABEL[g]}</div>
              <div className="text-[10px] text-muted tabular mt-0.5">{i + 1}</div>
            </button>
          ))}
        </div>
      )}

      <div className="text-[11px] text-muted text-center tabular">
        SPACE reveal · 1 again · 2 hard · 3 good · 4 easy
      </div>
    </div>
  );
}

function flashClass(grade: Grade | null): string {
  // Skip red on Again — flash amber for both Again and Hard so the user
  // gets a "needs review" signal without a harsh incorrect-answer feel.
  // Good / Easy land on green (correct → on the way to mastered).
  switch (grade) {
    case "again":
    case "hard":
      return "bg-amber-50 border-amber-400";
    case "good":
      return "bg-emerald-50 border-emerald-400";
    case "easy":
      return "bg-emerald-100 border-emerald-500";
    default:
      return "";
  }
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
