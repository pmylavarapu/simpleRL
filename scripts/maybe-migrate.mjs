// Run `prisma migrate deploy` only if the DB URL is configured.
// Lets the first Vercel import build succeed before the Postgres integration is attached.
import { spawnSync } from "node:child_process";

const url = process.env.POSTGRES_URL_NON_POOLING || process.env.POSTGRES_PRISMA_URL;
if (!url) {
  console.log("[maybe-migrate] POSTGRES_URL_NON_POOLING not set — skipping migrate deploy.");
  process.exit(0);
}

const result = spawnSync("npx", ["prisma", "migrate", "deploy"], {
  stdio: "inherit",
  shell: process.platform === "win32",
});
process.exit(result.status ?? 1);
