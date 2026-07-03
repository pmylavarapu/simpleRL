# ASE Echo KB

A knowledge base and FSRS spaced-repetition review app for the ASE Board Exam (ASCeXAM), organized around the official NBE content outline.

## Stack

- **Next.js 15** (App Router) + React 19 + TypeScript
- **Tailwind CSS**
- **NextAuth v5** (Google provider only)
- **Prisma** + **Postgres** (Vercel Postgres / Neon in production)
- **ts-fsrs** for FSRS-4.5 scheduling
- Cards are git-tracked JSON in `/content/cards`; notes are Markdown in `/content/notes`. Only per-user review state lives in the DB.

## Structure

```
app/
  page.tsx                landing
  kb/                     browsable knowledge base (blueprint-driven)
  review/                 FSRS review session
  decks/                  section overview
  signin/                 Google sign-in
  api/
    auth/[...nextauth]    NextAuth handlers
    review/               POST rating → FSRS update
components/
  ReviewSession.tsx       review UI (basic + cloze, keyboard 1/2/3/4)
content/
  blueprint.ts            the 6-domain / 65-subtopic ASE outline
  notes/<section>/<topic>.md   curated notes (added as sources arrive)
  cards/<code>.json       cards per subtopic (basic + cloze only)
lib/
  auth.ts                 NextAuth config
  db.ts                   Prisma singleton
  fsrs.ts                 ts-fsrs wrapper
  cards.ts                content loader + zod schema
  cloze.ts                {{c1::…}} parser
  notes.ts                markdown loader
prisma/
  schema.prisma           User (NextAuth) + ReviewState
```

## Local setup

```bash
npm install
cp .env.example .env.local          # fill in values
npm run db:push                     # push schema to your Postgres
npm run dev
```

Env vars (`.env.local`):

- `DATABASE_URL` — Postgres connection string
- `AUTH_SECRET` — `openssl rand -base64 32`
- `AUTH_URL` — e.g. `http://localhost:3000` (Vercel sets automatically in prod)
- `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` — from Google Cloud Console → Credentials → OAuth client ID (Web application). Add authorized redirect URIs:
  - `http://localhost:3000/api/auth/callback/google`
  - `https://<your-vercel-domain>/api/auth/callback/google`

## Deploy to Vercel

1. Push branch to GitHub.
2. Import repository in Vercel.
3. Add **Vercel Postgres** integration (this sets `DATABASE_URL` automatically).
4. Add env vars for Google + `AUTH_SECRET` (Vercel handles `AUTH_URL` and `NEXTAUTH_URL`).
5. Deploy. First deploy: run `npx prisma db push` locally against the production DB (or add a build step), or use `prisma migrate deploy` once migrations exist.

## Card authoring format

`content/cards/<blueprint-code>.json` is an array. Codes use underscore instead of dot (e.g. `II_A.json` for blueprint code `II.A`).

```json
[
  {
    "id": "II.A-001",
    "topic": "II.A",
    "type": "basic",
    "front": "…question…",
    "back": "…answer…",
    "source": "Otto ch. 12"
  },
  {
    "id": "II.A-002",
    "topic": "II.A",
    "type": "cloze",
    "front": "Severe AS: peak velocity {{c1::≥ 4.0 m/s}} with mean gradient {{c2::≥ 40 mmHg}}.",
    "source": "2020 ACC/AHA"
  }
]
```

Rules:

- `id` must be globally unique and stable — it keys the user's FSRS state.
- `topic` must match a blueprint code exactly.
- `type` is `basic` (front/back) or `cloze` (markers in `front`, no `back`).
- Cloze markers: `{{cN::answer}}` or `{{cN::answer::hint}}`. Multiple `N` values in one card create separate reviewable variants (currently only the first index is shown per card view — will iterate on this if you want per-cloze-index reviews later).

## Status

**Scaffold only.** Two placeholder cards exist so the review loop can be verified end-to-end. Real content will be curated from the sources you provide, one at a time, and appended to the appropriate `content/cards/<code>.json` files and `content/notes/<section>/<topic>.md` files.
