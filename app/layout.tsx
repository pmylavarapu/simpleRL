import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import { auth, signOut } from "@/lib/auth";

export const metadata: Metadata = {
  title: "ASE Echo KB",
  description: "ASCeXAM board-review knowledge base with FSRS spaced repetition.",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b border-border">
          <div className="mx-auto max-w-6xl px-4 py-3 flex items-center gap-6">
            <Link href="/" className="font-semibold">ASE Echo KB</Link>
            <nav className="flex items-center gap-4 text-sm text-muted">
              <Link href="/kb" className="hover:text-fg">Knowledge base</Link>
              <Link href="/review" className="hover:text-fg">Review</Link>
              <Link href="/decks" className="hover:text-fg">Decks</Link>
            </nav>
            <div className="ml-auto text-sm">
              {session?.user ? (
                <form
                  action={async () => {
                    "use server";
                    await signOut({ redirectTo: "/" });
                  }}
                  className="flex items-center gap-3"
                >
                  <span className="text-muted">{session.user.email}</span>
                  <button className="underline hover:no-underline">Sign out</button>
                </form>
              ) : (
                <Link href="/signin" className="underline">Sign in</Link>
              )}
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
