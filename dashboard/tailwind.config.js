/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
      keyframes: { flash: { "0%,100%": { backgroundColor: "rgba(239,68,68,0)" }, "50%": { backgroundColor: "rgba(239,68,68,0.35)" } } },
      animation: { flash: "flash 0.9s ease-in-out 3" },
    },
  },
  plugins: [],
};
