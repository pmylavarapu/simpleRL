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
npx prisma migrate deploy           # apply schema to your Postgres
npm run dev
```

Env vars (`.env.local`):

- `POSTGRES_PRISMA_URL` — pooled Postgres URL (app runtime queries)
- `POSTGRES_URL_NON_POOLING` — direct Postgres URL (migrations)
- `AUTH_SECRET` — `openssl rand -base64 32`
- `AUTH_URL` — e.g. `http://localhost:3000` (Vercel sets automatically in prod)
- `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` — from Google Cloud Console → Credentials → OAuth 2.0 Client ID (Web application). Add authorized redirect URIs:
  - `http://localhost:3000/api/auth/callback/google`
  - `https://<your-vercel-domain>/api/auth/callback/google`

## Deploy to Vercel

**1. Import the repo.** Go to [vercel.com/new](https://vercel.com/new) → select `pmylavarapu/simpleRL` → set branch to `claude/ase-echo-knowledge-base-asuyl9` for the first preview (or merge to `main` first if you prefer prod). Framework auto-detects as Next.js.

**2. Provision a Postgres database.** In the new Vercel project → Storage tab → Create Database → Neon (Vercel's managed Postgres). Attach it to the project. This auto-injects `POSTGRES_PRISMA_URL` and `POSTGRES_URL_NON_POOLING` into all environments.

**3. Set up Google OAuth.** At [console.cloud.google.com](https://console.cloud.google.com) → APIs & Services → Credentials:
   - Create OAuth consent screen (External, add your email as a test user)
   - Create Credentials → OAuth 2.0 Client ID → Web application
   - Authorized redirect URIs (add both):
     - `http://localhost:3000/api/auth/callback/google`
     - `https://<your-vercel-domain>/api/auth/callback/google`  ← add after first Vercel deploy assigns a domain
   - Copy Client ID and Client Secret.

**4. Add remaining env vars in Vercel** (Project → Settings → Environment Variables):
   - `AUTH_SECRET` = output of `openssl rand -base64 32`
   - `AUTH_GOOGLE_ID` = Google client ID
   - `AUTH_GOOGLE_SECRET` = Google client secret

**5. Deploy.** Click "Redeploy" so the build picks up the new env vars. The build script runs `prisma migrate deploy` and applies the initial migration to your Neon DB automatically.

**6. Update Google OAuth redirect URI** with your final Vercel URL, then redeploy once more if needed.

**7. Sign in and test.** Visit `/signin`, sign in with Google, then `/review` — grade both placeholder cards. Confirm a `ReviewState` row appears (Vercel → Storage → Database → Data tab).

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
