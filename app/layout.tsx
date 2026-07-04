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
            <Link
              href="/"
              className="flex items-center text-xl leading-none hover:opacity-70 transition-opacity"
              aria-label="Echo KB — Home"
            >
              <span role="img" aria-label="Anatomical heart">🫀</span>
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
        <main>{children}</main>
        <footer className="border-t border-border bg-bg-soft">
          <div className="mx-auto max-w-6xl px-6 py-5 flex items-center justify-between text-[12px] text-muted">
            <span>Echo KB · Board review with FSRS</span>
            <span className="tabular">v0.1</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
