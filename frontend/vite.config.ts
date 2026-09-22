import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // In local dev (npm run dev), use empty string so Vite proxy handles /api → localhost:8000
  // In production (Vercel build), VITE_API_BASE_URL is either set in env vars or hardcoded in api.ts
  define: mode === "development" ? { "import.meta.env.VITE_API_BASE_URL": '""' } : {},
  server: {
    port: 4173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
}));