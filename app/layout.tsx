import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import { Inter, JetBrains_Mono } from "next/font/google";
import { auth, signOut } from "@/lib/auth";
import { NavLinks } from "@/components/NavLinks";

const sans = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
  weight: ["400", "500", "600", "700"],
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Echo KB",
  description: "ASCeXAM board-review knowledge base with FSRS spaced repetition.",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-screen antialiased">
        <header className="sticky top-0 z-40 border-b border-border bg-bg/80 backdrop-blur">
          <div className="mx-auto max-w-6xl px-6 h-14 flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
              <span className="inline-block w-2 h-2 rounded-full bg-fg" aria-hidden="true" />
              Echo KB
            </Link>
            <NavLinks />
            <div className="ml-auto text-[13px]">
              {session?.user ? (
                <form
                  action={async () => {
                    "use server";
                    await signOut({ redirectTo: "/" });
                  }}
                  className="flex items-center gap-4"
                >
                  <span className="text-muted hidden md:inline">{session.user.email}</span>
                  <button className="text-fg hover:text-muted transition-colors">Sign out</button>
                </form>
              ) : (
                <Link href="/signin" className="text-fg hover:text-muted transition-colors">Sign in →</Link>
              )}
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-6">{children}</main>
        <footer className="border-t border-border mt-10">
          <div className="mx-auto max-w-6xl px-6 py-6 flex items-center justify-between text-[12px] text-muted">
            <span>Echo KB · Board review with FSRS</span>
            <span className="tabular">v0.1</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
