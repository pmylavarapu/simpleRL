import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--bg) / <alpha-value>)",
        "bg-soft": "rgb(var(--bg-soft) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        "fg-strong": "rgb(var(--fg-strong) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        "muted-soft": "rgb(var(--muted-soft) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        "border-strong": "rgb(var(--border-strong) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-fg": "rgb(var(--accent-fg) / <alpha-value>)",
        success: "rgb(var(--success) / <alpha-value>)",
        "success-soft": "rgb(var(--success-soft) / <alpha-value>)",
        warning: "rgb(var(--warning) / <alpha-value>)",
        "warning-soft": "rgb(var(--warning-soft) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular"],
      },
      letterSpacing: {
        tightest: "-0.03em",
        tighter: "-0.02em",
      },
      boxShadow: {
        "sm-soft": "0 1px 0 rgb(0 0 0 / 0.02), 0 1px 2px rgb(0 0 0 / 0.04)",
        "elev": "0 1px 0 rgb(0 0 0 / 0.04), 0 8px 24px -8px rgb(0 0 0 / 0.08)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
