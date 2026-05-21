import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Outside Records Summary",
  description: "One-page synthesis with click-to-source",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
