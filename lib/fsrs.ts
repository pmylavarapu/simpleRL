import { FSRS, generatorParameters, Rating, State, createEmptyCard, type Card as FSRSCard, type Grade as FSRSGrade } from "ts-fsrs";
import type { ReviewState } from "@prisma/client";

// Standard FSRS-4.5 params, target retention 0.9.
const fsrs = new FSRS(generatorParameters({ enable_fuzz: true }));

export type Grade = "again" | "hard" | "good" | "easy";

const gradeToRating: Record<Grade, FSRSGrade> = {
  again: Rating.Again,
  hard: Rating.Hard,
  good: Rating.Good,
  easy: Rating.Easy,
};

// Turn a DB row (or null for a brand-new card) into the ts-fsrs Card shape.
export function toFSRSCard(row: ReviewState | null): FSRSCard {
  if (!row) return createEmptyCard(new Date());
  return {
    due: row.due,
    stability: row.stability,
    difficulty: row.difficulty,
    elapsed_days: row.elapsedDays,
    scheduled_days: row.scheduledDays,
    reps: row.reps,
    lapses: row.lapses,
    state: row.state as State,
    last_review: row.lastReview ?? undefined,
  };
}

export type ScheduledUpdate = {
  due: Date;
  stability: number;
  difficulty: number;
  elapsedDays: number;
  scheduledDays: number;
  reps: number;
  lapses: number;
  state: number;
  lastReview: Date;
};

export function schedule(row: ReviewState | null, grade: Grade, now: Date = new Date()): ScheduledUpdate {
  const card = toFSRSCard(row);
  const result = fsrs.next(card, now, gradeToRating[grade]);
  const c = result.card;
  return {
    due: c.due,
    stability: c.stability,
    difficulty: c.difficulty,
    elapsedDays: c.elapsed_days,
    scheduledDays: c.scheduled_days,
    reps: c.reps,
    lapses: c.lapses,
    state: c.state,
    lastReview: c.last_review ?? now,
  };
}

// For UI: preview when each grade would push the next review to.
export function previewAll(row: ReviewState | null, now: Date = new Date()): Record<Grade, Date> {
  const card = toFSRSCard(row);
  const rec = fsrs.repeat(card, now) as Record<FSRSGrade, { card: FSRSCard }>;
  return {
    again: rec[Rating.Again].card.due,
    hard: rec[Rating.Hard].card.due,
    good: rec[Rating.Good].card.due,
    easy: rec[Rating.Easy].card.due,
  };
}
