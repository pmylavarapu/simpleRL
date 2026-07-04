"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type Item = { href: string; label: string; match: (path: string) => boolean };

const ITEMS: Item[] = [
  { href: "/", label: "Home", match: (p) => p === "/" },
  { href: "/about", label: "About", match: (p) => p === "/about" },
  { href: "/how-to-use", label: "How to use", match: (p) => p === "/how-to-use" },
  {
    href: "/decks",
    label: "Knowledge Base",
    match: (p) => p === "/decks" || p.startsWith("/kb"),
  },
];

export function NavLinks() {
  const pathname = usePathname() ?? "/";
  return (
    <nav className="hidden sm:flex items-center gap-1 text-[13px]">
      {ITEMS.map((item) => {
        const active = item.match(pathname);
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={`px-2.5 py-1 rounded-md border transition-colors ${
              active
                ? "text-fg border-border-strong bg-bg-soft"
                : "text-muted border-transparent hover:text-fg hover:border-border"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
