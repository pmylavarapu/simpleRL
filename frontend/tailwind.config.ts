import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0b1220",
        clinical: {
          50: "#f4f7fb",
          100: "#e6edf6",
          200: "#c9d6e8",
          300: "#9fb5d3",
          500: "#4870a1",
          700: "#1a3a6b",
          900: "#0e2244",
        },
        evidence: "#fff3a3",
        priority: {
          1: "#b91c1c",
          2: "#b45309",
          3: "#475569",
        },
      },
    },
  },
  plugins: [],
};
export default config;
