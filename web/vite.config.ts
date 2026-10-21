// Development: `npm run dev` serves the interface at http://localhost:5173 and forwards /api and /health
// to the FastAPI backend (http://localhost:8000). Production: `npm run build` writes dist/, which the
// backend serves at http://localhost:8000.
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const API = process.env.API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API, changeOrigin: true },
      "/health": { target: API, changeOrigin: true },
      "/docs": { target: API, changeOrigin: true },
      "/openapi.json": { target: API, changeOrigin: true },
    },
    fs: { allow: [".."] }, // the Run guide page shows ../docs/USER_GUIDE.md
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
          flow: ["@xyflow/react"],
          markdown: ["react-markdown", "remark-gfm"],
        },
      },
    },
  },
});
